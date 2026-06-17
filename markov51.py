"""
=============================================================
LAYER 5 — WALK-FORWARD BACKTESTER (v2)
=============================================================
Requirements:
    pip install MetaTrader5 pandas numpy hmmlearn scikit-learn joblib matplotlib

Changes from v1:
    - Per-symbol strategy config mirrors Layer 3 exactly
    - EURUSD: Bollinger high_vol only
    - GBPUSD: Flipped EMA low_vol + Bollinger high_vol
    - USDJPY: Fully disabled — no edge in either regime
    - Convergence warnings suppressed
=============================================================
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", message="Model is not converging")
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

CONFIG = {
    "terminal_path" : r"C:\Program Files\EGM Securities MetaTrader 5 Terminal\terminal64.exe",
    #"terminal_path": r"C:\Program Files\MetaTrader 5\terminal64.exe",
    "symbols":       ["EURUSD", "GBPUSD", "USDJPY"],
    "timeframe":     mt5.TIMEFRAME_M15,
    "n_bars":        35000,

    # Walk-forward parameters
    "train_bars": 400,
    "test_bars":  80,
    "step_bars":  80,

    # HMM
    "n_states": 2,
    "n_iter":   200,
    "hmm_features": [
        "realized_vol",
        "atr_pct",
        "vol_ratio",
        "bar_range",
        "volume_zscore",
    ],

    # Strategy parameters — must match Layer 3
    "ema_fast":  10,
    "ema_slow":  50,
    "bb_period": 20,
    "bb_std":    2.0,

    # Risk — must match Layer 3
    "starting_balance": 10000.0,
    "base_risk_pct":    0.01,
    "min_confidence":   0.65,
    "max_switch_prob":  0.15,

    "pip_size": {
        "EURUSD": 0.0001,
        "GBPUSD": 0.0001,
        "USDJPY": 0.01,
    },
    "pip_value_per_lot": {
        "EURUSD": 10.0,
        "GBPUSD": 10.0,
        "USDJPY": 9.0,
    },

    "sl_atr_mult":       1.5,
    "tp_atr_mult_trend": 2.0,
    "tp_atr_mult_mean":  1.0,
    "min_rr":            1.5,

    # ── Per-symbol strategy — mirrors Layer 3 exactly ──
    # Options: "bollinger" | "ema" | "ema_flipped" | "disabled"
    "symbol_strategy": {
        "EURUSD": {"low_vol": "disabled",    "high_vol": "bollinger"},
        "GBPUSD": {"low_vol": "ema_flipped", "high_vol": "bollinger"},
        "USDJPY": {"low_vol": "disabled",    "high_vol": "disabled"},
    },

    "results_dir": "backtest_results/",
}


# ─────────────────────────────────────────────
# DATA FETCH
# ─────────────────────────────────────────────

def fetch_history(symbol: str) -> pd.DataFrame | None:
    if not mt5.symbol_select(symbol, True):
        print(f"[WARN] Cannot select {symbol}")
        return None

    rates = mt5.copy_rates_from_pos(
        symbol, CONFIG["timeframe"], 0, CONFIG["n_bars"]
    )
    if rates is None or len(rates) == 0:
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df.drop(columns=["real_volume"], errors="ignore", inplace=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)

    print(f"  {symbol}: {len(df)} bars | "
          f"{df.index[0].date()} to {df.index[-1].date()}")
    return df


# ─────────────────────────────────────────────
# FEATURE ENGINEERING — identical to Layer 1
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()

    feat["log_return"]    = np.log(feat["close"] / feat["close"].shift(1))
    annualise             = np.sqrt(8064)
    feat["realized_vol"]  = feat["log_return"].rolling(20).std() * annualise
    feat["vol_ratio"]     = (
        feat["log_return"].rolling(5).std() /
        feat["log_return"].rolling(20).std().replace(0, np.nan)
    )

    high_low   = feat["high"] - feat["low"]
    high_close = (feat["high"] - feat["close"].shift(1)).abs()
    low_close  = (feat["low"]  - feat["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    feat["atr"]     = true_range.rolling(14).mean()
    feat["atr_pct"] = feat["atr"] / feat["close"]
    feat["bar_range"] = (feat["high"] - feat["low"]) / feat["close"]

    vol_mean = feat["volume"].rolling(20).mean()
    vol_std  = feat["volume"].rolling(20).std().replace(0, np.nan)
    feat["volume_zscore"] = (feat["volume"] - vol_mean) / vol_std

    ema_fast = feat["close"].ewm(span=CONFIG["ema_fast"], adjust=False).mean()
    ema_slow = feat["close"].ewm(span=CONFIG["ema_slow"], adjust=False).mean()
    feat["ema_fast"] = ema_fast
    feat["ema_slow"] = ema_slow

    middle = feat["close"].rolling(CONFIG["bb_period"]).mean()
    std    = feat["close"].rolling(CONFIG["bb_period"]).std()
    feat["bb_upper"] = middle + CONFIG["bb_std"] * std
    feat["bb_lower"] = middle - CONFIG["bb_std"] * std

    feat.dropna(inplace=True)
    return feat


# ─────────────────────────────────────────────
# HMM FIT
# ─────────────────────────────────────────────

def fit_hmm_window(train_df: pd.DataFrame):
    """Fit HMM on a single training window."""
    features = CONFIG["hmm_features"]
    X_raw    = train_df[features].values

    if len(X_raw) < 50:
        return None, None, None

    scaler = StandardScaler()
    X      = scaler.fit_transform(X_raw)

    try:
        model = GaussianHMM(
            n_components=CONFIG["n_states"],
            covariance_type="full",
            n_iter=CONFIG["n_iter"],
            random_state=42,
            verbose=False,
        )
        model.fit(X)
    except Exception:
        return None, None, None

    # Map states by realized_vol mean — lowest = low_vol, highest = high_vol
    vol_idx       = features.index("realized_vol")
    state_vols    = {s: model.means_[s][vol_idx] for s in range(CONFIG["n_states"])}
    sorted_states = sorted(state_vols, key=state_vols.get)
    regime_map    = {sorted_states[0]: "low_vol", sorted_states[1]: "high_vol"}

    return model, scaler, regime_map


# ─────────────────────────────────────────────
# CLASSIFY WINDOW
# ─────────────────────────────────────────────

def classify_window(model, scaler, regime_map,
                    test_df: pd.DataFrame) -> pd.DataFrame:
    """Apply fitted HMM to test window — no refitting."""
    features    = CONFIG["hmm_features"]
    X           = scaler.transform(test_df[features].values)
    state_seq   = model.predict(X)
    state_probs = model.predict_proba(X)

    result              = test_df.copy()
    result["regime"]    = [regime_map[s] for s in state_seq]
    result["confidence"] = state_probs.max(axis=1)
    result["switch_prob"] = [
        1 - model.transmat_[s][s] for s in state_seq
    ]
    return result


# ─────────────────────────────────────────────
# SIGNAL GENERATION — mirrors Layer 3 exactly
# ─────────────────────────────────────────────

def get_signal_for_bar(row: pd.Series, prev_row: pd.Series,
                       symbol: str) -> tuple:
    """
    Generate signal for a single bar using per-symbol strategy config.
    Mirrors Layer 3's generate_signal() logic exactly.
    Returns (direction, strategy_name).
    """
    regime      = row["regime"]
    confidence  = row["confidence"]
    switch_prob = row["switch_prob"]

    # Regime filter
    if confidence < CONFIG["min_confidence"]:
        return "flat", "filtered_confidence"
    if switch_prob > CONFIG["max_switch_prob"]:
        return "flat", "filtered_switch"

    # Look up strategy for this symbol + regime
    symbol_cfg    = CONFIG["symbol_strategy"].get(symbol, {})
    strategy_type = symbol_cfg.get(regime, "disabled")

    if strategy_type == "disabled":
        return "flat", "disabled"

    # ── Bollinger reversion ───────────────────
    if strategy_type == "bollinger":
        price = row["close"]
        if price < row["bb_lower"]:
            return "buy",  "bollinger_reversion"
        elif price > row["bb_upper"]:
            return "sell", "bollinger_reversion"
        else:
            return "flat", "bollinger_reversion"

    # ── Standard EMA crossover ────────────────
    elif strategy_type == "ema":
        bullish = (prev_row["ema_fast"] <= prev_row["ema_slow"] and
                   row["ema_fast"]      >  row["ema_slow"])
        bearish = (prev_row["ema_fast"] >= prev_row["ema_slow"] and
                   row["ema_fast"]      <  row["ema_slow"])
        if bullish:
            return "buy",  "ema_crossover"
        elif bearish:
            return "sell", "ema_crossover"
        else:
            return "flat", "ema_crossover"

    # ── Flipped EMA — fade the breakout ───────
    elif strategy_type == "ema_flipped":
        bullish = (prev_row["ema_fast"] <= prev_row["ema_slow"] and
                   row["ema_fast"]      >  row["ema_slow"])
        bearish = (prev_row["ema_fast"] >= prev_row["ema_slow"] and
                   row["ema_fast"]      <  row["ema_slow"])
        if bullish:
            return "sell", "ema_flipped"   # sell the breakout
        elif bearish:
            return "buy",  "ema_flipped"   # buy the breakdown
        else:
            return "flat", "ema_flipped"

    return "flat", "unknown"


# ─────────────────────────────────────────────
# TRADE SIMULATION
# ─────────────────────────────────────────────

def simulate_trade(entry_bar: pd.Series, subsequent_bars: pd.DataFrame,
                   direction: str, regime: str, strategy: str,
                   symbol: str, balance: float,
                   confidence: float) -> dict:
    """
    Simulate a single trade through subsequent bars.
    Checks high/low each bar for SL or TP hit.
    """
    pip_size  = CONFIG["pip_size"].get(symbol, 0.0001)
    pip_value = CONFIG["pip_value_per_lot"].get(symbol, 10.0)

    atr_pips = entry_bar["atr"] / pip_size
    sl_pips  = round(atr_pips * CONFIG["sl_atr_mult"], 1)

    # TP depends on strategy type
    if strategy in ("ema_crossover", "ema_flipped"):
        tp_pips = round(atr_pips * CONFIG["tp_atr_mult_trend"], 1)
    else:
        tp_pips = round(atr_pips * CONFIG["tp_atr_mult_mean"], 1)

    # Enforce minimum reward:risk
    if tp_pips < sl_pips * CONFIG["min_rr"]:
        tp_pips = round(sl_pips * CONFIG["min_rr"], 1)

    sl_pips = max(sl_pips, 5.0)
    tp_pips = max(tp_pips, 5.0)

    entry_price = entry_bar["close"]

    if direction == "buy":
        sl_price = entry_price - sl_pips * pip_size
        tp_price = entry_price + tp_pips * pip_size
    else:
        sl_price = entry_price + sl_pips * pip_size
        tp_price = entry_price - tp_pips * pip_size

    # Confidence-scaled position sizing
    min_conf    = CONFIG["min_confidence"]
    conf_scalar = 0.5 + 0.5 * ((confidence - min_conf) / (1.0 - min_conf))
    conf_scalar = max(0.5, min(1.0, conf_scalar))
    risk_amount = balance * CONFIG["base_risk_pct"] * conf_scalar
    lot_size    = max(round(risk_amount / (sl_pips * pip_value), 2), 0.01)

    # Simulate bar by bar
    outcome    = "timeout"
    exit_price = subsequent_bars["close"].iloc[-1]

    for _, bar in subsequent_bars.iterrows():
        if direction == "buy":
            if bar["low"] <= sl_price:
                outcome    = "sl"
                exit_price = sl_price
                break
            if bar["high"] >= tp_price:
                outcome    = "tp"
                exit_price = tp_price
                break
        else:
            if bar["high"] >= sl_price:
                outcome    = "sl"
                exit_price = sl_price
                break
            if bar["low"] <= tp_price:
                outcome    = "tp"
                exit_price = tp_price
                break

    pnl_pips = ((exit_price - entry_price) / pip_size if direction == "buy"
                else (entry_price - exit_price) / pip_size)
    pnl_usd  = round(pnl_pips * pip_value * lot_size, 2)

    return {
        "direction":   direction,
        "regime":      regime,
        "strategy":    strategy,
        "confidence":  round(confidence, 4),
        "entry_price": entry_price,
        "exit_price":  exit_price,
        "sl_pips":     sl_pips,
        "tp_pips":     tp_pips,
        "lot_size":    lot_size,
        "outcome":     outcome,
        "pnl_pips":    round(pnl_pips, 2),
        "pnl_usd":     pnl_usd,
        "won":         pnl_usd > 0,
    }


# ─────────────────────────────────────────────
# WALK-FORWARD ENGINE
# ─────────────────────────────────────────────

def walk_forward(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Slide a train/test window across the full history.
    HMM is refit on each training window.
    Signals generated on test window only — no lookahead.
    """
    train_bars = CONFIG["train_bars"]
    test_bars  = CONFIG["test_bars"]
    step_bars  = CONFIG["step_bars"]

    trades  = []
    balance = CONFIG["starting_balance"]
    n       = len(df)
    start   = train_bars
    window  = 0

    while start + test_bars <= n:
        window += 1
        train_df = df.iloc[start - train_bars : start]
        test_df  = df.iloc[start : start + test_bars]

        model, scaler, regime_map = fit_hmm_window(train_df)
        if model is None:
            start += step_bars
            continue

        classified = classify_window(model, scaler, regime_map, test_df)

        for i in range(1, len(classified)):
            row      = classified.iloc[i]
            prev_row = classified.iloc[i - 1]

            direction, strategy = get_signal_for_bar(row, prev_row, symbol)

            if direction == "flat":
                continue

            remaining = classified.iloc[i + 1:]
            if len(remaining) == 0:
                continue

            result = simulate_trade(
                entry_bar       = row,
                subsequent_bars = remaining,
                direction       = direction,
                regime          = row["regime"],
                strategy        = strategy,
                symbol          = symbol,
                balance         = balance,
                confidence      = float(row["confidence"]),
            )

            result["symbol"]         = symbol
            result["entry_time"]     = classified.index[i]
            result["window"]         = window
            result["balance_before"] = balance
            balance                 += result["pnl_usd"]
            result["balance_after"]  = balance
            trades.append(result)

        start += step_bars

    print(f"  {symbol}: {window} windows | {len(trades)} trades simulated")
    return pd.DataFrame(trades)


# ─────────────────────────────────────────────
# PERFORMANCE METRICS
# ─────────────────────────────────────────────

def compute_metrics(trades: pd.DataFrame, symbol: str) -> dict:
    if trades.empty:
        return {"symbol": symbol, "error": "No trades generated"}

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

    returns = trades["pnl_usd"] / trades["balance_before"]
    sharpe  = (returns.mean() / returns.std() * np.sqrt(252)
               if returns.std() > 0 else 0)

    # Per-regime breakdown
    regime_stats = {}
    for regime in ["low_vol", "high_vol"]:
        r = trades[trades["regime"] == regime]
        if len(r) > 0:
            regime_stats[regime] = {
                "trades":   len(r),
                "win_rate": round(len(r[r["won"]]) / len(r), 3),
                "pnl":      round(r["pnl_usd"].sum(), 2),
            }

    # Per-strategy breakdown
    strategy_stats = {}
    for strat in trades["strategy"].dropna().unique():
        s = trades[trades["strategy"] == strat]
        strategy_stats[strat] = {
            "trades":   len(s),
            "win_rate": round(len(s[s["won"]]) / len(s), 3),
            "pnl":      round(s["pnl_usd"].sum(), 2),
        }

    return {
        "symbol":        symbol,
        "total_trades":  total,
        "win_rate":      round(win_rate, 3),
        "total_pnl":     round(total_pnl, 2),
        "profit_factor": round(profit_factor, 3),
        "avg_win":       round(avg_win, 2),
        "avg_loss":      round(avg_loss, 2),
        "max_drawdown":  round(max_dd, 4),
        "sharpe":        round(sharpe, 3),
        "final_balance": round(equity[-1], 2),
        "regime_stats":  regime_stats,
        "strategy_stats": strategy_stats,
    }


# ─────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────

def print_report(metrics: dict):
    print(f"\n{'='*60}")
    print(f"  BACKTEST REPORT — {metrics['symbol']}")
    print(f"{'='*60}")

    if "error" in metrics:
        print(f"  ERROR: {metrics['error']}")
        return

    print(f"  Total trades    : {metrics['total_trades']}")
    print(f"  Win rate        : {metrics['win_rate']:.1%}")
    print(f"  Total P&L       : ${metrics['total_pnl']:,.2f}")
    print(f"  Profit factor   : {metrics['profit_factor']:.2f}")
    print(f"  Avg win         : ${metrics['avg_win']:.2f}")
    print(f"  Avg loss        : ${metrics['avg_loss']:.2f}")
    print(f"  Max drawdown    : {metrics['max_drawdown']:.1%}")
    print(f"  Sharpe ratio    : {metrics['sharpe']:.2f}")
    print(f"  Final balance   : ${metrics['final_balance']:,.2f}")

    print(f"\n  -- By Regime --")
    for regime, s in metrics["regime_stats"].items():
        print(f"  {regime:12} | trades={s['trades']:4} | "
              f"win={s['win_rate']:.0%} | pnl=${s['pnl']:,.2f}")

    print(f"\n  -- By Strategy --")
    for strat, s in metrics["strategy_stats"].items():
        print(f"  {strat:22} | trades={s['trades']:4} | "
              f"win={s['win_rate']:.0%} | pnl=${s['pnl']:,.2f}")


# ─────────────────────────────────────────────
# SAVE + PLOT
# ─────────────────────────────────────────────

def save_results(trades: pd.DataFrame, metrics: dict, symbol: str):
    os.makedirs(CONFIG["results_dir"], exist_ok=True)

    trades.to_csv(
        os.path.join(CONFIG["results_dir"], f"{symbol}_trades.csv"),
        index=False
    )
    flat = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
    pd.DataFrame([flat]).to_csv(
        os.path.join(CONFIG["results_dir"], f"{symbol}_metrics.csv"),
        index=False
    )
    print(f"  Results saved -> {CONFIG['results_dir']}")


def plot_equity_curve(trades: pd.DataFrame, symbol: str):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  [PLOT] pip install matplotlib to enable charts")
        return

    if trades.empty:
        return

    equity = trades["balance_after"].values
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"Walk-Forward Backtest — {symbol}", fontsize=14)

    for i in range(len(trades) - 1):
        colour = "#1D9E75" if trades.iloc[i]["regime"] == "low_vol" else "#D85A30"
        ax1.plot([i, i+1], [equity[i], equity[i+1]],
                 color=colour, linewidth=1.2)

    ax1.axhline(CONFIG["starting_balance"], color="gray",
                linestyle="--", linewidth=0.8)
    ax1.set_ylabel("Account Balance ($)")
    ax1.grid(True, alpha=0.3)

    low_p  = mpatches.Patch(color="#1D9E75", label="low_vol")
    high_p = mpatches.Patch(color="#D85A30", label="high_vol")
    ax1.legend(handles=[low_p, high_p], loc="upper left")

    roll_wr = trades["won"].astype(int).rolling(20).mean()
    ax2.plot(roll_wr.values, color="#378ADD", linewidth=1.2)
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("Win Rate (rolling 20)")
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Trade number")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(CONFIG["results_dir"], f"{symbol}_equity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved -> {path}")
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_backtest():
    print("=" * 60)
    print("  LAYER 5 — WALK-FORWARD BACKTESTER (v2)")
    print("=" * 60)

    if not mt5.initialize(path=CONFIG["terminal_path"]):
        print(f"[FATAL] MT5 init failed: {mt5.last_error()}")
        return

    print(f"\nFetching {CONFIG['n_bars']} bars of M15 history...")

    all_metrics = []

    for symbol in CONFIG["symbols"]:
        print(f"\n[{symbol}]")

        # Check if symbol has any active strategy at all
        sym_cfg = CONFIG["symbol_strategy"].get(symbol, {})
        if all(v == "disabled" for v in sym_cfg.values()):
            print(f"  All strategies disabled for {symbol} — skipping")
            continue

        raw = fetch_history(symbol)
        if raw is None:
            continue

        df = engineer_features(raw)
        print(f"  Feature matrix: {len(df)} bars x "
              f"{len(CONFIG['hmm_features'])} features")

        if len(df) < CONFIG["train_bars"] + CONFIG["test_bars"]:
            print(f"  [SKIP] Not enough bars")
            continue

        print(f"  Running walk-forward "
              f"(train={CONFIG['train_bars']} test={CONFIG['test_bars']})...")
        trades = walk_forward(df, symbol)

        if trades.empty:
            print(f"  [SKIP] No trades generated")
            continue

        metrics = compute_metrics(trades, symbol)
        all_metrics.append(metrics)
        print_report(metrics)
        save_results(trades, metrics, symbol)
        plot_equity_curve(trades, symbol)

    # Combined summary
    if len(all_metrics) > 1:
        print(f"\n{'='*60}")
        print("  COMBINED SUMMARY")
        print(f"{'='*60}")
        total_pnl    = sum(m.get("total_pnl", 0) for m in all_metrics)
        total_trades = sum(m.get("total_trades", 0) for m in all_metrics)
        print(f"  Total trades : {total_trades}")
        print(f"  Combined P&L : ${total_pnl:,.2f}")

    mt5.shutdown()
    print("\n[DONE] Backtest complete.")


if __name__ == "__main__":
    run_backtest()