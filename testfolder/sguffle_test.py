"""
=============================================================
SHUFFLE TEST - does the HMM regime label add anything?
=============================================================
Standalone. Does NOT import your other files.

    pip install MetaTrader5 pandas numpy hmmlearn scikit-learn scipy

Question answered:
    Keep the strategy, entries, exits and costs identical.
    Only change WHICH BARS are allowed to trade:

      ALWAYS  : no regime gate at all (baseline)
      CLOCK   : simple time-of-day rule, no HMM
      REAL    : causal HMM regime label (forward filter, no lookahead)
      SHUFFLE : the REAL labels permuted among bars of the SAME hour
                of day, repeated N times (the null distribution)

    If REAL does not clearly beat the SHUFFLE distribution, the HMM
    carries no information beyond time of day. Retire it.

Fixes vs your old backtests:
    - regime labels are causal (forward filter, not predict())
    - one trade at a time per symbol
    - spread paid on every trade (per-bar spread from MT5)
    - entry at NEXT bar open, not the signal bar close
    - no window truncation: trades run up to max_hold_bars
    - results in R multiples (no lot-size / pip-value distortion)

Same SL/TP rules as before: SL = 1.5 ATR (min 5 pips),
TP = 1.5 x SL (the old min_rr rule made this true for every trade).
=============================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import MetaTrader5 as mt5

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG - edit here
# ─────────────────────────────────────────────

CFG = {
    "terminal_path": r"C:\Program Files\MetaTrader 5\terminal64.exe",
    "terminal_path_2": r"C:\Program Files\EGM Securities MetaTrader 5 Terminal\terminal64.exe",
    "timeframe":     mt5.TIMEFRAME_M15,
    "bar_counts":    (50000, 30000, 20000),   # tried in order until one works

    # HMM - same as your validation file
    "train_bars": 400,
    "test_bars":  80,
    "n_states":   2,
    "n_iter":     200,
    "hmm_features": ["realized_vol", "atr_pct", "vol_ratio",
                     "bar_range", "volume_zscore"],

    # Trade rules
    "sl_atr_mult":     1.5,
    "min_rr":          1.5,
    "min_sl_pips":     5.0,
    "max_spread_frac": 0.10,    # skip if spread > 10% of SL
    "max_hold_bars":   96,      # 1 day of M15, then close at market

    # Test
    "n_shuffles": 200,
    "seed":       7,
    # CLOCK rule: trade "high_vol" only in these broker-server hours.
    # From your crosstab, high_vol share is >50% from 15:00 to 21:59.
    "clock_high_hours": list(range(15, 22)),

    "cache_dir": "shuffle_cache",
}

PIP = {"EURUSD": 0.0001, "GBPUSD": 0.0001, "EURGBP": 0.0001,
       "EURCAD": 0.0001, "GBPCAD": 0.0001, "AUDUSD": 0.0001,
       "USDCAD": 0.0001}

# Each case = one (symbol, strategy, regime it trades in).
# These are the configs from your validation files. Add/remove freely.
CASES = [
    {"symbol": "EURUSD", "strategy": "bollinger", "regime": "high_vol",
     "params": {"bb_period": 20, "bb_std": 3.0}},
    {"symbol": "GBPUSD", "strategy": "ema_flipped", "regime": "low_vol",
     "params": {"ema_fast": 10, "ema_slow": 50}},
    {"symbol": "GBPCAD", "strategy": "bollinger", "regime": "low_vol",
     "params": {"bb_period": 50, "bb_std": 2.0}},
    {"symbol": "EURCAD", "strategy": "rsi", "regime": "high_vol",
     "params": {"rsi_period": 21, "rsi_oversold": 30, "rsi_overbought": 70}},
]


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def load_bars(symbol):
    """Returns (DataFrame, point) or (None, None)."""
    if not mt5.symbol_select(symbol, True):
        print(f"  [WARN] cannot select {symbol}")
        return None, None
    rates = None
    for n in CFG["bar_counts"]:
        rates = mt5.copy_rates_from_pos(symbol, CFG["timeframe"], 0, n)
        if rates is not None and len(rates) > 0:
            break
        print(f"  n_bars={n} failed: {mt5.last_error()}")
    if rates is None or len(rates) == 0:
        return None, None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df.drop(columns=["real_volume"], errors="ignore", inplace=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)

    info = mt5.symbol_info(symbol)
    point = info.point if info is not None else 0.00001
    print(f"  {symbol}: {len(df):,} bars | "
          f"{df.index[0].date()} to {df.index[-1].date()}")
    return df, point


def add_features(raw):
    f = raw.copy()
    lr = np.log(f["close"] / f["close"].shift(1))
    f["realized_vol"] = lr.rolling(20).std() * np.sqrt(8064)
    f["vol_ratio"] = (lr.rolling(5).std() /
                      lr.rolling(20).std().replace(0, np.nan))
    tr = pd.concat([f["high"] - f["low"],
                    (f["high"] - f["close"].shift(1)).abs(),
                    (f["low"] - f["close"].shift(1)).abs()],
                   axis=1).max(axis=1)
    f["atr"] = tr.rolling(14).mean()
    f["atr_pct"] = f["atr"] / f["close"]
    f["bar_range"] = (f["high"] - f["low"]) / f["close"]
    vm = f["volume"].rolling(20).mean()
    vs = f["volume"].rolling(20).std().replace(0, np.nan)
    f["volume_zscore"] = (f["volume"] - vm) / vs
    f = f.dropna(subset=CFG["hmm_features"] + ["atr"])
    return f


# ─────────────────────────────────────────────
# CAUSAL HMM LABELS
# ─────────────────────────────────────────────

def fit_hmm(train_df):
    feats = CFG["hmm_features"]
    X_raw = train_df[feats].values
    if len(X_raw) < 50:
        return None
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    try:
        model = GaussianHMM(n_components=CFG["n_states"],
                            covariance_type="full",
                            n_iter=CFG["n_iter"], random_state=42)
        model.fit(X)
    except Exception:
        return None
    vol_idx = feats.index("realized_vol")
    high_state = int(np.argmax(model.means_[:, vol_idx]))
    return model, scaler, high_state


def forward_filter(model, X):
    """P(state_t | x_1..x_t). Uses past data only, unlike predict()."""
    K, d = model.n_components, X.shape[1]
    logB = np.column_stack([
        multivariate_normal.logpdf(
            X, mean=model.means_[k],
            cov=model.covars_[k] + 1e-6 * np.eye(d),
            allow_singular=True)
        for k in range(K)])
    B = np.exp(logB - logB.max(axis=1, keepdims=True))
    alpha = np.zeros_like(B)
    a = model.startprob_ * B[0]
    alpha[0] = a / a.sum()
    for t in range(1, len(X)):
        a = (alpha[t - 1] @ model.transmat_) * B[t]
        s = a.sum()
        alpha[t] = a / s if s > 0 else alpha[t - 1]
    return alpha


def causal_labels(df, symbol):
    """Returns (high: bool[], valid: bool[]) aligned to df."""
    os.makedirs(CFG["cache_dir"], exist_ok=True)
    path = os.path.join(CFG["cache_dir"], f"{symbol}_labels.csv")
    if os.path.exists(path):
        c = pd.read_csv(path, index_col=0, parse_dates=True)
        if len(c) == len(df) and c.index[0] == df.index[0] \
                and c.index[-1] == df.index[-1]:
            print(f"  [{symbol}] labels loaded from cache")
            return c["high"].values.astype(bool), c["valid"].values.astype(bool)

    tr, te = CFG["train_bars"], CFG["test_bars"]
    feats = CFG["hmm_features"]
    n = len(df)
    high = np.zeros(n, dtype=bool)
    valid = np.zeros(n, dtype=bool)
    starts = list(range(tr, n - te + 1, te))
    print(f"  [{symbol}] fitting {len(starts)} HMM windows...", flush=True)

    for k, s in enumerate(starts):
        fitted = fit_hmm(df.iloc[s - tr:s])
        if fitted is None:
            continue
        model, scaler, high_state = fitted
        X = scaler.transform(df[feats].iloc[s - tr:s + te].values)
        alpha = forward_filter(model, X)[-te:]
        high[s:s + te] = alpha.argmax(axis=1) == high_state
        valid[s:s + te] = True
        if (k + 1) % 100 == 0:
            print(f"    {k + 1}/{len(starts)}", flush=True)

    pd.DataFrame({"high": high, "valid": valid}, index=df.index).to_csv(path)
    return high, valid


# ─────────────────────────────────────────────
# SIGNALS (+1 buy, -1 sell, 0 none) - decided at bar close
# ─────────────────────────────────────────────

def make_signal(df, strat, p):
    c = df["close"]
    sig = np.zeros(len(df), dtype=int)
    if strat == "bollinger":
        mid = c.rolling(p["bb_period"]).mean()
        sd = c.rolling(p["bb_period"]).std()
        sig[(c < mid - p["bb_std"] * sd).values] = 1
        sig[(c > mid + p["bb_std"] * sd).values] = -1
    elif strat == "rsi":
        d = c.diff()
        g = d.clip(lower=0).rolling(p["rsi_period"]).mean()
        l = (-d.clip(upper=0)).rolling(p["rsi_period"]).mean()
        rsi = 100 - 100 / (1 + g / l.replace(0, np.nan))
        sig[(rsi < p["rsi_oversold"]).values] = 1
        sig[(rsi > p["rsi_overbought"]).values] = -1
    elif strat == "ema_flipped":
        f = c.ewm(span=p["ema_fast"], adjust=False).mean()
        s = c.ewm(span=p["ema_slow"], adjust=False).mean()
        up = (f.shift(1) <= s.shift(1)) & (f > s)
        dn = (f.shift(1) >= s.shift(1)) & (f < s)
        sig[up.values] = -1     # flipped: fade the breakout
        sig[dn.values] = 1
    else:
        raise ValueError(strat)
    return sig


# ─────────────────────────────────────────────
# TRADE OUTCOMES - computed once per case, regime-independent
# ─────────────────────────────────────────────

def precompute_trades(df, sig, pip, point):
    O = df["open"].values
    H = df["high"].values
    L = df["low"].values
    C = df["close"].values
    S = df["spread"].values * point            # spread in price units
    ATR = df["atr"].values
    n = len(df)
    max_hold = CFG["max_hold_bars"]

    sig_idx, exit_idx, R = [], [], []
    for i in np.flatnonzero(sig != 0):
        e_i = i + 1
        if e_i >= n:
            continue
        d = sig[i]
        atr_pips = ATR[i] / pip
        sl_pips = max(CFG["sl_atr_mult"] * atr_pips, CFG["min_sl_pips"])
        tp_pips = max(CFG["min_rr"] * sl_pips, CFG["min_sl_pips"])
        if S[e_i] / pip > CFG["max_spread_frac"] * sl_pips:
            continue

        j_end = min(e_i + max_hold, n)
        hi, lo, sp = H[e_i:j_end], L[e_i:j_end], S[e_i:j_end]

        if d == 1:                              # buy at ask, exit at bid
            e = O[e_i] + S[e_i]
            sl, tp = e - sl_pips * pip, e + tp_pips * pip
            sl_hit, tp_hit = lo <= sl, hi >= tp
        else:                                   # sell at bid, exit at ask
            e = O[e_i]
            sl, tp = e + sl_pips * pip, e - tp_pips * pip
            sl_hit, tp_hit = hi + sp >= sl, lo + sp <= tp

        big = len(hi) + 1
        sl_k = int(np.argmax(sl_hit)) if sl_hit.any() else big
        tp_k = int(np.argmax(tp_hit)) if tp_hit.any() else big

        if sl_k < big and sl_k <= tp_k:         # SL wins ties (conservative)
            pnl_pips, x_i = -sl_pips, e_i + sl_k
        elif tp_k < big:
            pnl_pips, x_i = tp_pips, e_i + tp_k
        else:
            last = j_end - 1
            x_px = C[last] if d == 1 else C[last] + S[last]
            pnl_pips = d * (x_px - e) / pip
            x_i = last

        sig_idx.append(i)
        exit_idx.append(x_i)
        R.append(pnl_pips / sl_pips)

    return np.array(sig_idx), np.array(exit_idx), np.array(R)


def run_arm(sig_idx, exit_idx, R, allowed):
    """One trade at a time; a signal is taken only if its bar is allowed."""
    taken, last_exit = [], -1
    for k in range(len(sig_idx)):
        i = sig_idx[k]
        if i < last_exit or not allowed[i]:
            continue
        taken.append(R[k])
        last_exit = exit_idx[k]
    return np.array(taken)


def stats(rs):
    n = len(rs)
    if n == 0:
        return {"n": 0, "total": 0.0, "mean": 0.0, "win": 0.0, "t": 0.0}
    sd = rs.std(ddof=1) if n > 1 else 0.0
    return {"n": n, "total": rs.sum(), "mean": rs.mean(),
            "win": (rs > 0).mean(),
            "t": rs.mean() / (sd / np.sqrt(n)) if sd > 0 else 0.0}


def shuffle_within_hour(high, valid, hours, rng):
    out = high.copy()
    for h in range(24):
        idx = np.flatnonzero(valid & (hours == h))
        if len(idx) > 1:
            out[idx] = rng.permutation(high[idx])
    return out


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def fmt(name, s):
    return (f"  {name:<9}{s['n']:>7}{s['total']:>+10.1f}"
            f"{s['mean']:>+10.3f}{s['win']:>8.1%}{s['t']:>8.2f}")


def run():
    print("=" * 66)
    print("  SHUFFLE TEST: HMM regime vs time-of-day-matched shuffle")
    print("=" * 66)

    if not mt5.initialize(path=CFG["terminal_path"]):
        print(f"[FATAL] MT5 init failed: {mt5.last_error()}")
        return

    rng = np.random.default_rng(CFG["seed"])
    data = {}
    summary = []

    for case in CASES:
        sym = case["symbol"]
        if sym not in data:
            print(f"\n[{sym}] loading")
            raw, point = load_bars(sym)
            if raw is None:
                data[sym] = None
                continue
            df = add_features(raw)
            high, valid = causal_labels(df, sym)
            data[sym] = (df, point, high, valid)
        if data[sym] is None:
            continue
        df, point, high, valid = data[sym]

        pip = PIP.get(sym, 0.0001)
        hours = df.index.hour.values
        want_high = case["regime"] == "high_vol"

        sig = make_signal(df, case["strategy"], case["params"])
        si, xi, R = precompute_trades(df, sig, pip, point)

        allow_always = valid.copy()
        allow_clock = valid & (np.isin(hours, CFG["clock_high_hours"])
                               == want_high)
        allow_real = valid & (high == want_high)

        s_always = stats(run_arm(si, xi, R, allow_always))
        s_clock = stats(run_arm(si, xi, R, allow_clock))
        s_real = stats(run_arm(si, xi, R, allow_real))

        sh_total, sh_mean = [], []
        for _ in range(CFG["n_shuffles"]):
            lab = shuffle_within_hour(high, valid, hours, rng)
            st = stats(run_arm(si, xi, R, valid & (lab == want_high)))
            sh_total.append(st["total"])
            sh_mean.append(st["mean"])
        sh_total, sh_mean = np.array(sh_total), np.array(sh_mean)

        pct = (sh_total < s_real["total"]).mean() * 100
        p_val = (sh_total >= s_real["total"]).mean()
        p95 = np.percentile(sh_total, 95)

        print(f"\n{'-' * 66}")
        print(f"  {sym} | {case['strategy']} {case['params']} | "
              f"trades only in {case['regime']}")
        print(f"{'-' * 66}")
        print(f"  {'arm':<9}{'trades':>7}{'total R':>10}"
              f"{'mean R':>10}{'win%':>8}{'t-stat':>8}")
        print(fmt("ALWAYS", s_always))
        print(fmt("CLOCK", s_clock))
        print(fmt("REAL", s_real))
        print(f"  SHUFFLE x{CFG['n_shuffles']}: total R median "
              f"{np.median(sh_total):+.1f}, 5-95% "
              f"[{np.percentile(sh_total, 5):+.1f}, {p95:+.1f}], "
              f"mean R median {np.median(sh_mean):+.3f}")
        print(f"  REAL beats {pct:.0f}% of shuffles  (p = {p_val:.3f})")

        summary.append({
            "case": f"{sym} {case['strategy']} {case['regime']}",
            "real": s_real["total"], "shuf_p95": p95, "pct": pct,
            "clock": s_clock["total"], "always": s_always["total"],
            "beats_shuffle": s_real["total"] > p95,
            "beats_clock": s_real["total"] > s_clock["total"],
        })

    mt5.shutdown()
    if not summary:
        print("\nNo cases ran.")
        return

    print(f"\n{'=' * 66}")
    print("  SUMMARY (total R, higher is better)")
    print(f"{'=' * 66}")
    print(f"  {'case':<30}{'REAL':>8}{'p95shuf':>9}{'CLOCK':>8}"
          f"{'ALWAYS':>8}{'pctile':>8}")
    for s in summary:
        print(f"  {s['case']:<30}{s['real']:>+8.1f}{s['shuf_p95']:>+9.1f}"
              f"{s['clock']:>+8.1f}{s['always']:>+8.1f}{s['pct']:>7.0f}%")

    k = sum(s["beats_shuffle"] for s in summary)
    c = sum(s["beats_clock"] for s in summary)
    n = len(summary)
    print(f"\n  REAL beats the 95th pct of shuffles in {k}/{n} cases")
    print(f"  REAL beats the plain CLOCK rule in     {c}/{n} cases")
    print("\n  Reading it:")
    print("   - ~5% of cases beating p95 is what pure chance gives.")
    print("   - Verdict 'HMM adds nothing' = few/no cases beat p95 AND")
    print("     REAL is not reliably above CLOCK.")
    print("   - If ALWAYS is about as good as REAL, the gate is irrelevant.")
    print("\n[DONE]")


if __name__ == "__main__":
    run()