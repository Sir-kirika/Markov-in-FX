"""
=============================================================
LAYER 5 — OPTIMISATION RUNNER (v4 — exhaustive)
Tests EVERY strategy in EVERY regime combination.
=============================================================
Requirements:
    pip install MetaTrader5 pandas numpy hmmlearn scikit-learn joblib

What changed from v3:
    - Every strategy is tested in both low_vol AND high_vol regime
    - Setups are auto-generated from strategy x params combinations
    - No manually disabled regimes — let the data decide
    - Still has resume support, vectorised SL/TP, step=400, n_iter=50

Setup generation:
    For each strategy + params combination, two setups are created:
        {strategy}_low   : strategy in low_vol,  high_vol disabled
        {strategy}_high  : strategy in high_vol, low_vol  disabled
    Plus combo setups pairing best candidates across regimes.

Total runs: ~(strategies x params x 2 regimes x symbols)
            = significant — expect 2-4 hours with 7 symbols
=============================================================
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import os
import json
import hashlib
import warnings
from datetime import datetime
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# BASE CONFIG
# ─────────────────────────────────────────────

BASE_CONFIG = {
    "terminal_path": r"C:\Program Files\EGM Securities MetaTrader 5 Terminal\terminal64.exe",

    "symbols": [
        "EURUSD",
        "GBPUSD",
        "EURGBP",
        "EURCAD",
        "GBPCAD",
        "AUDUSD",
        "USDCAD",
        "XAUUSD",
    ],

    "timeframe": mt5.TIMEFRAME_M15,
    "n_bars":    175000,

    # "train_bars": 400,
    # "test_bars":  80,
    # "step_bars":  400,
    
    # ── FIX 3: larger windows ──────────────────
    "train_bars": 2000,   # ~3 weeks M15  (was 400)
    "test_bars":  200,    # ~2.5 days     (was 80)
    "step_bars":  200,    # roll ~daily   (was 400)

    "n_states":   2,
    "n_iter":     50,

    "hmm_features": [
        "realized_vol",
        "atr_pct",
        "vol_ratio",
        "bar_range",
        "volume_zscore",
    ],

    "min_confidence":  0.65,
    "max_switch_prob": 0.15,

    "starting_balance": 10000.0,
    "base_risk_pct":    0.01,
    "sl_atr_mult":      1.5,
    "min_rr":           1.5,

    "pip_size": {
        "EURUSD": 0.0001, "GBPUSD": 0.0001,
        "EURGBP": 0.0001, "EURCAD": 0.0001,
        "GBPCAD": 0.0001, "AUDUSD": 0.0001,
        "USDCAD": 0.0001, "XAUUSD": .1,
    },
    "pip_value_per_lot": {
        "EURUSD": 10.0, "GBPUSD": 10.0,
        "EURGBP": 12.5, "EURCAD": 7.5,
        "GBPCAD": 7.5,  "AUDUSD": 10.0,
        "USDCAD": 7.5,  "XAUUSD": 10.0,
    },

    "results_dir":   "backtest_results/optimisation_v5231/",
    "progress_file": "backtest_results/optimisation_v5231/progress.csv",
}


# ─────────────────────────────────────────────
# SETUP GENERATION
# Every strategy tested in BOTH regimes.
# ─────────────────────────────────────────────

def build_setups() -> list:
    """
    Auto-generate all setups by testing every strategy
    in both low_vol and high_vol regime independently.

    Structure per setup:
        low_vol_strategy  : strategy active in low_vol regime
        high_vol_strategy : strategy active in high_vol regime
        params            : indicator parameters

    For each strategy+params we create TWO setups:
        name_LOW  → strategy in low_vol,  disabled in high_vol
        name_HIGH → strategy in high_vol, disabled in low_vol

    Then COMBO setups pair strategies across regimes.
    """

    # ── Strategy + parameter variants ────────
    # Each entry: (strategy_name, params_dict)
    strategy_variants = [

        # Bollinger Band
        ("bollinger", {"bb_period": 10, "bb_std": 2.0}),
        ("bollinger", {"bb_period": 20, "bb_std": 2.0}),
        ("bollinger", {"bb_period": 20, "bb_std": 2.5}),
        ("bollinger", {"bb_period": 20, "bb_std": 3.0}),
        ("bollinger", {"bb_period": 50, "bb_std": 2.0}),
        ("bollinger", {"bb_period": 50, "bb_std": 2.5}),

        # EMA flipped (fade the breakout)
        ("ema_flipped", {"ema_fast": 5,  "ema_slow": 20}),
        ("ema_flipped", {"ema_fast": 10, "ema_slow": 50}),
        ("ema_flipped", {"ema_fast": 10, "ema_slow": 200}),
        ("ema_flipped", {"ema_fast": 20, "ema_slow": 100}),

        # Standard EMA crossover (trend following)
        ("ema", {"ema_fast": 5,  "ema_slow": 20}),
        ("ema", {"ema_fast": 10, "ema_slow": 50}),
        ("ema", {"ema_fast": 20, "ema_slow": 100}),

        # RSI mean reversion
        ("rsi", {"rsi_period": 7,  "rsi_oversold": 25, "rsi_overbought": 75}),
        ("rsi", {"rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70}),
        ("rsi", {"rsi_period": 14, "rsi_oversold": 20, "rsi_overbought": 80}),
        ("rsi", {"rsi_period": 21, "rsi_oversold": 30, "rsi_overbought": 70}),
        ("rsi", {"rsi_period": 21, "rsi_oversold": 20, "rsi_overbought": 80}),

        # ATR channel breakout
        ("atr_breakout", {"atr_channel_period": 10, "atr_break_mult": 0.5}),
        ("atr_breakout", {"atr_channel_period": 20, "atr_break_mult": 0.5}),
        ("atr_breakout", {"atr_channel_period": 20, "atr_break_mult": 1.0}),

        # VWAP reversion
        ("vwap", {"vwap_threshold": 0.001}),
        ("vwap", {"vwap_threshold": 0.002}),
        ("vwap", {"vwap_threshold": 0.003}),
    ]

    setups = []

    # ── Single strategy in each regime ────────
    for strategy, params in strategy_variants:

        # Short param string for naming
        param_str = "_".join(str(v) for v in params.values())

        # Test in LOW_VOL regime
        setups.append({
            "name":              f"{strategy}_{param_str}_LOW",
            "low_vol_strategy":  strategy,
            "high_vol_strategy": "disabled",
            "params":            params.copy(),
        })

        # Test in HIGH_VOL regime
        setups.append({
            "name":              f"{strategy}_{param_str}_HIGH",
            "low_vol_strategy":  "disabled",
            "high_vol_strategy": strategy,
            "params":            params.copy(),
        })

    # ── Combo setups: pair across regimes ─────
    # Best candidates from prior runs — test combinations
    combos = [
        # BB high + EMA_flip low (current winning config)
        ("COMBO_BB20_3_EMAf20_100",
         "ema_flipped", "bollinger",
         {"bb_period": 20, "bb_std": 3.0, "ema_fast": 20, "ema_slow": 100}),
        ("COMBO_BB20_25_EMAf20_100",
         "ema_flipped", "bollinger",
         {"bb_period": 20, "bb_std": 2.5, "ema_fast": 20, "ema_slow": 100}),
        ("COMBO_BB10_2_EMAf20_100",
         "ema_flipped", "bollinger",
         {"bb_period": 10, "bb_std": 2.0, "ema_fast": 20, "ema_slow": 100}),
        ("COMBO_BB20_3_EMAf10_50",
         "ema_flipped", "bollinger",
         {"bb_period": 20, "bb_std": 3.0, "ema_fast": 10, "ema_slow": 50}),
        ("COMBO_BB20_3_EMAf5_20",
         "ema_flipped", "bollinger",
         {"bb_period": 20, "bb_std": 3.0, "ema_fast": 5, "ema_slow": 20}),

        # RSI high + EMA_flip low
        ("COMBO_RSI21_2080_EMAf20_100",
         "ema_flipped", "rsi",
         {"rsi_period": 21, "rsi_oversold": 20, "rsi_overbought": 80,
          "ema_fast": 20, "ema_slow": 100}),
        ("COMBO_RSI14_2080_EMAf20_100",
         "ema_flipped", "rsi",
         {"rsi_period": 14, "rsi_oversold": 20, "rsi_overbought": 80,
          "ema_fast": 20, "ema_slow": 100}),
        ("COMBO_RSI21_2080_EMAf10_50",
         "ema_flipped", "rsi",
         {"rsi_period": 21, "rsi_oversold": 20, "rsi_overbought": 80,
          "ema_fast": 10, "ema_slow": 50}),
        ("COMBO_RSI14_3070_EMAf10_50",
         "ema_flipped", "rsi",
         {"rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70,
          "ema_fast": 10, "ema_slow": 50}),

        # VWAP high + EMA_flip low
        ("COMBO_VWAP003_EMAf20_100",
         "ema_flipped", "vwap",
         {"vwap_threshold": 0.003, "ema_fast": 20, "ema_slow": 100}),
        ("COMBO_VWAP002_EMAf20_100",
         "ema_flipped", "vwap",
         {"vwap_threshold": 0.002, "ema_fast": 20, "ema_slow": 100}),

        # BB in both regimes (different params per regime)
        ("COMBO_BB10_LOW_BB20_3_HIGH",
         "bollinger", "bollinger",
         {"bb_period": 10, "bb_std": 2.0}),  # same params both regimes

        # RSI in both regimes
        ("COMBO_RSI21_BOTH",
         "rsi", "rsi",
         {"rsi_period": 21, "rsi_oversold": 20, "rsi_overbought": 80}),

        # EMA standard high + EMA flipped low
        ("COMBO_EMA_flip_LOW_EMA_HIGH",
         "ema_flipped", "ema",
         {"ema_fast": 20, "ema_slow": 100}),
    ]

    for name, lv, hv, params in combos:
        setups.append({
            "name":              name,
            "low_vol_strategy":  lv,
            "high_vol_strategy": hv,
            "params":            params,
        })

    return setups


# Build setups at module load time
SETUPS = build_setups()


# ─────────────────────────────────────────────
# RESUME SYSTEM
# ─────────────────────────────────────────────

def build_fingerprint() -> str:
    sig = {
        "n_bars":     BASE_CONFIG["n_bars"],
        "train_bars": BASE_CONFIG["train_bars"],
        "test_bars":  BASE_CONFIG["test_bars"],
        "step_bars":  BASE_CONFIG["step_bars"],
        "n_iter":     BASE_CONFIG["n_iter"],
        "symbols":    sorted(BASE_CONFIG["symbols"]),
        "setups":     [{"name": s["name"], "params": s["params"],
                        "lv": s["low_vol_strategy"],
                        "hv": s["high_vol_strategy"]}
                       for s in SETUPS],
    }
    raw = json.dumps(sig, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def load_progress() -> tuple:
    path        = BASE_CONFIG["progress_file"]
    fingerprint = build_fingerprint()

    if not os.path.exists(path):
        return set(), [], fingerprint

    df = pd.read_csv(path)

    if ("fingerprint" not in df.columns or
            df["fingerprint"].iloc[0] != fingerprint):
        print("  [RESUME] Settings changed — starting fresh")
        return set(), [], fingerprint

    completed = set(zip(df["symbol"], df["setup"]))
    results   = df.drop(columns=["fingerprint"]).to_dict("records")
    print(f"  [RESUME] {len(completed)} runs already done — skipping")
    return completed, results, fingerprint


def save_progress(result: dict, fingerprint: str):
    path = BASE_CONFIG["progress_file"]
    os.makedirs(os.path.dirname(path), exist_ok=True)

    row                = result.copy()
    row["fingerprint"] = fingerprint

    df_row       = pd.DataFrame([row])
    write_header = not os.path.exists(path)
    df_row.to_csv(path, mode="a", header=write_header, index=False)


# ─────────────────────────────────────────────
# DATA FETCH
# ─────────────────────────────────────────────

def fetch_history(symbol: str) -> pd.DataFrame | None:
    if not mt5.symbol_select(symbol, True):
        print(f"  [WARN] Cannot select {symbol} — skipping")
        return None

    rates = mt5.copy_rates_from_pos(
        symbol, BASE_CONFIG["timeframe"], 0, BASE_CONFIG["n_bars"]
    )
    if rates is None or len(rates) == 0:
        print(f"  [WARN] No data for {symbol} — skipping")
        return None
    
    

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df.drop(columns=["real_volume"], errors="ignore", inplace=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)

    print(f"  {symbol}: {len(df):,} bars | "
          f"{df.index[0].date()} to {df.index[-1].date()}")
    return df


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    feat = df.copy()

    feat["log_return"]   = np.log(feat["close"] / feat["close"].shift(1))
    feat["realized_vol"] = feat["log_return"].rolling(20).std() * np.sqrt(8064)
    feat["vol_ratio"]    = (
        feat["log_return"].rolling(5).std() /
        feat["log_return"].rolling(20).std().replace(0, np.nan)
    )

    hl  = feat["high"] - feat["low"]
    hc  = (feat["high"] - feat["close"].shift(1)).abs()
    lc  = (feat["low"]  - feat["close"].shift(1)).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    feat["atr"]       = tr.rolling(14).mean()
    feat["atr_pct"]   = feat["atr"] / feat["close"]
    feat["bar_range"] = (feat["high"] - feat["low"]) / feat["close"]

    vm = feat["volume"].rolling(20).mean()
    vs = feat["volume"].rolling(20).std().replace(0, np.nan)
    feat["volume_zscore"] = (feat["volume"] - vm) / vs

    # EMA
    fast = params.get("ema_fast", 10)
    slow = params.get("ema_slow", 50)
    feat["ema_fast"] = feat["close"].ewm(span=fast, adjust=False).mean()
    feat["ema_slow"] = feat["close"].ewm(span=slow, adjust=False).mean()

    # Bollinger Bands
    bb_p             = params.get("bb_period", 20)
    bb_s             = params.get("bb_std", 2.0)
    bb_mid           = feat["close"].rolling(bb_p).mean()
    bb_std           = feat["close"].rolling(bb_p).std()
    feat["bb_upper"] = bb_mid + bb_s * bb_std
    feat["bb_lower"] = bb_mid - bb_s * bb_std

    # RSI
    rsi_p  = params.get("rsi_period", 14)
    delta  = feat["close"].diff()
    gain   = delta.clip(lower=0).rolling(rsi_p).mean()
    loss   = (-delta.clip(upper=0)).rolling(rsi_p).mean()
    rs     = gain / loss.replace(0, np.nan)
    feat["rsi"] = 100 - (100 / (1 + rs))

    # ATR channel
    atr_cp               = params.get("atr_channel_period", 20)
    feat["highest_high"] = feat["high"].rolling(atr_cp).max()
    feat["lowest_low"]   = feat["low"].rolling(atr_cp).min()

    # VWAP
    feat["typical_price"] = (feat["high"] + feat["low"] + feat["close"]) / 3
    feat["tp_volume"]     = feat["typical_price"] * feat["volume"]
    feat["date"]          = feat.index.date
    feat["cum_tp_vol"]    = feat.groupby("date")["tp_volume"].cumsum()
    feat["cum_vol"]       = feat.groupby("date")["volume"].cumsum()
    feat["vwap"]          = (feat["cum_tp_vol"] /
                             feat["cum_vol"].replace(0, np.nan))
    feat.drop(columns=["date", "cum_tp_vol", "cum_vol",
                        "typical_price", "tp_volume"], inplace=True)

    feat.dropna(inplace=True)
    return feat


# ─────────────────────────────────────────────
# HMM FIT + CLASSIFY
# ─────────────────────────────────────────────

def fit_hmm(train_df: pd.DataFrame):
    features = BASE_CONFIG["hmm_features"]
    X_raw    = train_df[features].values
    if len(X_raw) < 50:
        return None, None, None

    scaler = StandardScaler()
    X      = scaler.fit_transform(X_raw)

    try:
        model = GaussianHMM(
            n_components=BASE_CONFIG["n_states"],
            covariance_type="full",
            n_iter=BASE_CONFIG["n_iter"],
            random_state=42,
            verbose=False,
        )
        model.fit(X)
    except Exception:
        return None, None, None

    vol_idx       = features.index("realized_vol")
    state_vols    = {s: model.means_[s][vol_idx]
                     for s in range(BASE_CONFIG["n_states"])}
    sorted_states = sorted(state_vols, key=state_vols.get)
    regime_map    = {sorted_states[0]: "low_vol",
                     sorted_states[1]: "high_vol"}

    return model, scaler, regime_map


def classify_window(model, scaler, regime_map,
                    test_df: pd.DataFrame) -> pd.DataFrame:
    features    = BASE_CONFIG["hmm_features"]
    X           = scaler.transform(test_df[features].values)
    state_seq   = model.predict(X)
    state_probs = model.predict_proba(X)

    result                = test_df.copy()
    result["regime"]      = [regime_map[s] for s in state_seq]
    result["confidence"]  = state_probs.max(axis=1)
    result["switch_prob"] = [1 - model.transmat_[s][s] for s in state_seq]
    return result


# ─────────────────────────────────────────────
# SIGNAL FUNCTIONS
# ─────────────────────────────────────────────

def signal_bollinger(row: pd.Series) -> tuple:
    price = row["close"]
    if price < row["bb_lower"]:
        return "buy",  "bollinger"
    elif price > row["bb_upper"]:
        return "sell", "bollinger"
    return "flat", "bollinger"


def signal_ema(row: pd.Series, prev_row: pd.Series,
               flipped: bool = False) -> tuple:
    bullish = (prev_row["ema_fast"] <= prev_row["ema_slow"] and
               row["ema_fast"]      >  row["ema_slow"])
    bearish = (prev_row["ema_fast"] >= prev_row["ema_slow"] and
               row["ema_fast"]      <  row["ema_slow"])
    name = "ema_flipped" if flipped else "ema"
    if bullish:
        return ("sell" if flipped else "buy"),  name
    elif bearish:
        return ("buy"  if flipped else "sell"), name
    return "flat", name


def signal_rsi(row: pd.Series, params: dict) -> tuple:
    rsi        = row["rsi"]
    oversold   = params.get("rsi_oversold", 30)
    overbought = params.get("rsi_overbought", 70)
    if rsi < oversold:
        return "buy",  "rsi"
    elif rsi > overbought:
        return "sell", "rsi"
    return "flat", "rsi"


def signal_atr_breakout(row: pd.Series, params: dict) -> tuple:
    price    = row["close"]
    mult     = params.get("atr_break_mult", 0.5)
    upper_ch = row["highest_high"] + mult * row["atr"]
    lower_ch = row["lowest_low"]   - mult * row["atr"]
    if price > upper_ch:
        return "buy",  "atr_breakout"
    elif price < lower_ch:
        return "sell", "atr_breakout"
    return "flat", "atr_breakout"


def signal_vwap(row: pd.Series, params: dict) -> tuple:
    price     = row["close"]
    vwap      = row.get("vwap", 0)
    threshold = params.get("vwap_threshold", 0.002)
    if vwap == 0 or np.isnan(vwap):
        return "flat", "vwap"
    deviation = (price - vwap) / vwap
    if deviation < -threshold:
        return "buy",  "vwap"
    elif deviation > threshold:
        return "sell", "vwap"
    return "flat", "vwap"


def get_signal(row: pd.Series, prev_row: pd.Series,
               setup: dict) -> tuple:
    regime      = row["regime"]
    confidence  = row["confidence"]
    switch_prob = row["switch_prob"]
    params      = setup["params"]

    if confidence  < BASE_CONFIG["min_confidence"]:
        return "flat", "filtered"
    if switch_prob > BASE_CONFIG["max_switch_prob"]:
        return "flat", "filtered"

    strategy_type = (setup["low_vol_strategy"]  if regime == "low_vol"
                     else setup["high_vol_strategy"])

    if strategy_type == "disabled":
        return "flat", "disabled"
    if strategy_type == "bollinger":
        return signal_bollinger(row)
    elif strategy_type == "ema":
        return signal_ema(row, prev_row, flipped=False)
    elif strategy_type == "ema_flipped":
        return signal_ema(row, prev_row, flipped=True)
    elif strategy_type == "rsi":
        return signal_rsi(row, params)
    elif strategy_type == "atr_breakout":
        return signal_atr_breakout(row, params)
    elif strategy_type == "vwap":
        return signal_vwap(row, params)
    return "flat", "unknown"


# ─────────────────────────────────────────────
# TRADE SIMULATION — vectorised
# ─────────────────────────────────────────────

def simulate_trade(entry_bar: pd.Series, subsequent_bars: pd.DataFrame,
                   direction: str, strategy: str, regime: str,
                   symbol: str, balance: float,
                   confidence: float) -> dict:
    pip_size  = BASE_CONFIG["pip_size"].get(symbol, 0.0001)
    pip_value = BASE_CONFIG["pip_value_per_lot"].get(symbol, 10.0)

    atr_pips = entry_bar["atr"] / pip_size
    sl_pips  = max(round(atr_pips * BASE_CONFIG["sl_atr_mult"], 1), 5.0)

    trend_strategies = {"ema", "ema_flipped", "atr_breakout"}
    tp_mult = 2.0 if strategy in trend_strategies else 1.0
    tp_pips = max(round(atr_pips * tp_mult, 1),
                  sl_pips * BASE_CONFIG["min_rr"], 5.0)

    entry_price = entry_bar["close"]
    if direction == "buy":
        sl_price = entry_price - sl_pips * pip_size
        tp_price = entry_price + tp_pips * pip_size
    else:
        sl_price = entry_price + sl_pips * pip_size
        tp_price = entry_price - tp_pips * pip_size

    highs = subsequent_bars["high"].values
    lows  = subsequent_bars["low"].values

    if direction == "buy":
        sl_hit = lows  <= sl_price
        tp_hit = highs >= tp_price
    else:
        sl_hit = highs >= sl_price
        tp_hit = lows  <= tp_price

    sl_idx = int(np.argmax(sl_hit)) if sl_hit.any() else len(subsequent_bars)
    tp_idx = int(np.argmax(tp_hit)) if tp_hit.any() else len(subsequent_bars)

    if sl_hit.any() and sl_idx <= tp_idx:
        outcome    = "sl"
        exit_price = sl_price
    elif tp_hit.any() and tp_idx < sl_idx:
        outcome    = "tp"
        exit_price = tp_price
    else:
        outcome    = "timeout"
        exit_price = subsequent_bars["close"].iloc[-1]

    min_conf    = BASE_CONFIG["min_confidence"]
    conf_scalar = max(0.5, min(1.0,
        0.5 + 0.5 * ((confidence - min_conf) / (1.0 - min_conf))
    ))
    risk_amount = balance * BASE_CONFIG["base_risk_pct"] * conf_scalar
    lot_size    = max(round(risk_amount / (sl_pips * pip_value), 2), 0.01)

    pnl_pips = ((exit_price - entry_price) / pip_size if direction == "buy"
                else (entry_price - exit_price) / pip_size)
    pnl_usd  = round(pnl_pips * pip_value * lot_size, 2)

    return {
        "direction":  direction,  "regime":  regime,
        "strategy":   strategy,   "confidence": round(confidence, 4),
        "sl_pips":    sl_pips,    "tp_pips": tp_pips,
        "lot_size":   lot_size,   "outcome": outcome,
        "pnl_pips":   round(pnl_pips, 2),
        "pnl_usd":    pnl_usd,   "won": pnl_usd > 0,
    }


# ─────────────────────────────────────────────
# WALK-FORWARD
# ─────────────────────────────────────────────

def walk_forward(df: pd.DataFrame, symbol: str,
                 setup: dict) -> pd.DataFrame:
    train_bars = BASE_CONFIG["train_bars"]
    test_bars  = BASE_CONFIG["test_bars"]
    step_bars  = BASE_CONFIG["step_bars"]

    trades  = []
    balance = BASE_CONFIG["starting_balance"]
    n       = len(df)
    start   = train_bars

    while start + test_bars <= n:
        train_df = df.iloc[start - train_bars : start]
        test_df  = df.iloc[start : start + test_bars]

        model, scaler, regime_map = fit_hmm(train_df)
        if model is None:
            start += step_bars
            continue

        classified = classify_window(model, scaler, regime_map, test_df)

        for i in range(1, len(classified)):
            row      = classified.iloc[i]
            prev_row = classified.iloc[i - 1]

            direction, strategy = get_signal(row, prev_row, setup)
            if direction == "flat":
                continue

            remaining = classified.iloc[i + 1:]
            if len(remaining) == 0:
                continue

            result = simulate_trade(
                entry_bar       = row,
                subsequent_bars = remaining,
                direction       = direction,
                strategy        = strategy,
                regime          = row["regime"],
                symbol          = symbol,
                balance         = balance,
                confidence      = float(row["confidence"]),
            )

            result["entry_time"]     = classified.index[i]
            result["balance_before"] = balance
            balance                 += result["pnl_usd"]
            result["balance_after"]  = balance
            trades.append(result)

        start += step_bars

    return pd.DataFrame(trades)


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def compute_metrics(trades: pd.DataFrame,
                    symbol: str, setup_name: str) -> dict:
    base = {
        "setup": setup_name, "symbol": symbol,
        "total_trades": 0, "win_rate": 0.0, "total_pnl": 0.0,
        "profit_factor": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
        "max_drawdown": 0.0, "sharpe": 0.0,
        "final_balance": BASE_CONFIG["starting_balance"],
        "expectancy": 0.0,
    }
    if trades.empty:
        return base

    total   = len(trades)
    winning = trades[trades["won"] == True]
    losing  = trades[trades["won"] == False]

    win_rate      = len(winning) / total
    total_pnl     = trades["pnl_usd"].sum()
    avg_win       = winning["pnl_usd"].mean() if len(winning) > 0 else 0
    avg_loss      = losing["pnl_usd"].mean()  if len(losing)  > 0 else 0
    gross_profit  = winning["pnl_usd"].sum()
    gross_loss    = abs(losing["pnl_usd"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    equity = trades["balance_after"].values
    peak   = np.maximum.accumulate(equity)
    max_dd = ((equity - peak) / peak).min()

    returns    = trades["pnl_usd"] / trades["balance_before"]
    sharpe     = (returns.mean() / returns.std() * np.sqrt(252)
                  if returns.std() > 0 else 0)
    expectancy = total_pnl / total

    return {
        "setup":          setup_name,
        "symbol":         symbol,
        "total_trades":   total,
        "win_rate":       round(win_rate, 3),
        "total_pnl":      round(total_pnl, 2),
        "profit_factor":  round(profit_factor, 3),
        "avg_win":        round(avg_win, 2),
        "avg_loss":       round(avg_loss, 2),
        "max_drawdown":   round(max_dd, 4),
        "sharpe":         round(sharpe, 3),
        "final_balance":  round(equity[-1], 2),
        "expectancy":     round(expectancy, 2),
    }


# ─────────────────────────────────────────────
# PRINT RANKED TABLE
# ─────────────────────────────────────────────

def print_comparison_table(results: list, symbol: str):
    df = pd.DataFrame(results)
    df = df[df["symbol"] == symbol].copy()
    if df.empty:
        return

    df.sort_values("sharpe", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"\n{'='*100}")
    print(f"  RESULTS — {symbol}  (sorted by Sharpe, top 20)")
    print(f"{'='*100}")
    print(f"  {'#':<3} {'Setup':<38} {'Trades':>6} {'WinRate':>7} "
          f"{'PnL':>10} {'PF':>5} {'MaxDD':>7} {'Sharpe':>7} {'Expect':>8}")
    print(f"  {'-'*96}")

    # Print top 20 only to keep output manageable
    for i, row in df.head(20).iterrows():
        marker = " **" if i < 3 else "   "
        print(
            f"  {i+1:<3} {row['setup']:<38} {row['total_trades']:>6} "
            f"{row['win_rate']:>6.1%} "
            f"${row['total_pnl']:>9,.0f} "
            f"{row['profit_factor']:>5.2f} "
            f"{row['max_drawdown']:>6.1%} "
            f"{row['sharpe']:>7.2f} "
            f"${row['expectancy']:>7.2f}{marker}"
        )

    if len(df) > 20:
        print(f"  ... and {len(df) - 20} more (see CSV for full results)")

    print(f"\n  -- Best per metric --")
    print(f"  Highest Sharpe    : {df.iloc[0]['setup']} "
          f"({df.iloc[0]['sharpe']:.2f})")
    best_dd = df.loc[df['max_drawdown'].idxmax()]
    print(f"  Lowest drawdown   : {best_dd['setup']} "
          f"({best_dd['max_drawdown']:.1%})")
    best_pnl = df.loc[df['total_pnl'].idxmax()]
    print(f"  Highest P&L       : {best_pnl['setup']} "
          f"(${best_pnl['total_pnl']:,.0f})")
    best_pf = df.loc[df['profit_factor'].idxmax()]
    print(f"  Best profit factor: {best_pf['setup']} "
          f"({best_pf['profit_factor']:.2f})")
    best_ex = df.loc[df['expectancy'].idxmax()]
    print(f"  Best expectancy   : {best_ex['setup']} "
          f"(${best_ex['expectancy']:.2f}/trade)")


# ─────────────────────────────────────────────
# CROSS-SYMBOL CONSISTENCY
# ─────────────────────────────────────────────

def print_cross_symbol_table(all_results: list):
    results_df = pd.DataFrame(all_results)

    print(f"\n{'='*80}")
    print("  CROSS-SYMBOL CONSISTENCY")
    print(f"{'='*80}")

    rows = []
    for setup_name in results_df["setup"].unique():
        subset       = results_df[results_df["setup"] == setup_name]
        n_profitable = int((subset["total_pnl"] > 0).sum())
        n_tested     = int(len(subset))
        rows.append({
            "setup":      setup_name,
            "profitable": n_profitable,
            "tested":     n_tested,
            "avg_sharpe": round(subset["sharpe"].mean(), 3),
            "min_sharpe": round(subset["sharpe"].min(), 3),
            "avg_dd":     round(subset["max_drawdown"].mean(), 4),
            "total_pnl":  round(subset["total_pnl"].sum(), 2),
        })

    rows_df = pd.DataFrame(rows).sort_values(
        ["profitable", "avg_sharpe"], ascending=[False, False]
    )

    n_symbols = len(results_df["symbol"].unique())

    for threshold, label in [(n_symbols, "ALL"), (n_symbols - 1, f"{n_symbols-1}/{n_symbols}")]:
        subset = rows_df[rows_df["profitable"] >= threshold]
        if subset.empty:
            continue
        print(f"\n  Profitable on {label} symbols "
              f"(top 15 by avg Sharpe):")
        print(f"  {'Setup':<40} {'N':>5} {'AvgSharpe':>9} "
              f"{'MinSharpe':>9} {'AvgDD':>7} {'TotalPnL':>10}")
        print(f"  {'-'*82}")
        for _, r in subset.head(15).iterrows():
            print(
                f"  {r['setup']:<40} "
                f"{r['profitable']}/{r['tested']} "
                f"{r['avg_sharpe']:>9.2f} "
                f"{r['min_sharpe']:>9.2f} "
                f"{r['avg_dd']:>6.1%} "
                f"${r['total_pnl']:>9,.0f}"
            )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_optimisation():
    n_symbols  = len(BASE_CONFIG["symbols"])
    n_setups   = len(SETUPS)
    total_runs = n_symbols * n_setups

    print("=" * 60)
    print("  LAYER 5 — OPTIMISATION RUNNER v4 (exhaustive)")
    print(f"  {n_setups} setups x {n_symbols} symbols = {total_runs} total runs")
    print(f"  Strategies tested in BOTH regimes — no disabled combos")
    print(f"  Data: ~8 years M15 | step=400 | n_iter=50")
    est_min = total_runs * 0.4
    est_max = total_runs * 0.7
    print(f"  Estimated time: {est_min:.0f}–{est_max:.0f} minutes")
    print("=" * 60)

    completed, all_results, fingerprint = load_progress()
    if completed:
        done = len(completed)
        print(f"  Resuming — {done} done, "
              f"{total_runs - done} remaining")

    if not mt5.initialize(path=BASE_CONFIG["terminal_path"]):
        print(f"[FATAL] MT5 init failed: {mt5.last_error()}")
        return

    print("\nFetching historical data...")
    raw_data = {}
    for symbol in BASE_CONFIG["symbols"]:
        raw = fetch_history(symbol)
        if raw is not None:
            raw_data[symbol] = raw

    mt5.shutdown()

    if not raw_data:
        print("[FATAL] No data fetched.")
        return

    actual_symbols = list(raw_data.keys())
    print(f"\n  Fetched {len(actual_symbols)} symbols: {actual_symbols}")

    os.makedirs(BASE_CONFIG["results_dir"], exist_ok=True)

    run_num    = 0
    start_time = datetime.now()

    for symbol, raw in raw_data.items():
        print(f"\n{'='*60}")
        print(f"  {symbol} — {n_setups} setups")
        print(f"{'='*60}")

        for setup in SETUPS:
            run_num += 1
            name    = setup["name"]
            key     = (symbol, name)

            if key in completed:
                continue   # silent skip for cleaner output

            print(f"  [{run_num:>3}/{total_runs}] {name:<40}",
                  end="", flush=True)

            df = engineer_features(raw, setup["params"])

            if len(df) < BASE_CONFIG["train_bars"] + BASE_CONFIG["test_bars"]:
                print(" SKIP")
                continue

            trades  = walk_forward(df, symbol, setup)
            metrics = compute_metrics(trades, symbol, name)
            all_results.append(metrics)
            save_progress(metrics, fingerprint)

            if not trades.empty:
                trades.to_csv(
                    os.path.join(
                        BASE_CONFIG["results_dir"],
                        f"{symbol}_{name}_trades.csv"
                    ),
                    index=False
                )

            # ETA calculation
            elapsed     = (datetime.now() - start_time).seconds
            done_so_far = sum(
                1 for r in all_results
                if r["symbol"] == symbol or
                list(raw_data.keys()).index(r["symbol"]) <
                list(raw_data.keys()).index(symbol)
            )
            avg_secs = elapsed / max(done_so_far, 1)
            eta_secs = int(avg_secs * (total_runs - run_num))
            eta_str  = f"{eta_secs//60}m{eta_secs%60:02d}s"

            pnl_sign = "+" if metrics["total_pnl"] >= 0 else ""
            print(
                f" {metrics['total_trades']:>5} trades | "
                f"pnl={pnl_sign}${metrics['total_pnl']:>7,.0f} | "
                f"sharpe={metrics['sharpe']:>5.2f} | "
                f"dd={metrics['max_drawdown']:>6.1%} | "
                f"ETA {eta_str}"
            )

    # Save full summary
    results_df   = pd.DataFrame(all_results)
    results_path = os.path.join(
        BASE_CONFIG["results_dir"], "optimisation_summary.csv"
    )
    results_df.to_csv(results_path, index=False)
    print(f"\n[SAVED] -> {results_path}")

    # Print tables
    for symbol in actual_symbols:
        print_comparison_table(all_results, symbol)

    print_cross_symbol_table(all_results)

    total_time = (datetime.now() - start_time).seconds
    print(f"\n[DONE] {total_time//60}m {total_time%60:02d}s total")
    print(f"       Full results -> {BASE_CONFIG['results_dir']}")


if __name__ == "__main__":
    print(f"\nSetup count: {len(SETUPS)} setups")
    print(f"Total runs : {len(SETUPS) * len(BASE_CONFIG['symbols'])}")
    print()
    run_optimisation()