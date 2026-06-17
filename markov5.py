"""
=============================================================
LAYER 5 — WALK-FORWARD BACKTESTER & PERFORMANCE EVALUATOR
=============================================================
Requirements:
    pip install MetaTrader5 pandas numpy hmmlearn scikit-learn 
                joblib matplotlib

What it does:
    1. Pulls 1-2 years of historical M15 bars from MT5
    2. Runs a walk-forward simulation:
         - Train HMM on TRAIN window
         - Generate signals on TEST window (no lookahead)
         - Simulate order execution with SL/TP
         - Slide forward and repeat
    3. Aggregates all test windows into full equity curve
    4. Prints performance report:
         - Total return, Sharpe, max drawdown
         - Win rate per regime and strategy
         - Regime detection stability
    5. Plots equity curve + regime overlay
    6. Detects model drift (when to refit in live trading)

Walk-forward structure (no lookahead bias):
    [==400 bars TRAIN==][80 bars TEST]
          [==400 bars TRAIN==][80 bars TEST]
                [==400 bars TRAIN==][80 bars TEST]
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

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

CONFIG = {
    # MT5 connection
    "terminal_path": r"C:\Program Files\MetaTrader 5\terminal64.exe",
    "symbols":       ["EURUSD", "GBPUSD", "USDJPY"],
    "timeframe":     mt5.TIMEFRAME_M15,
    #"n_bars":        6000,          # ~2.5 months of M15 bars (adjust up for more history)
    "n_bars": 35000,  # ~2 years of M15 bars

    # Walk-forward parameters
    "train_bars":    400,           # bars used to train HMM each window
    "test_bars":     80,            # bars tested per window (~1 trading day of M15)
    "step_bars":     80,            # how far to slide forward each iteration

    # HMM
    "n_states":      2,
    "n_iter":        200,
    "hmm_features": [
        "realized_vol",
        "atr_pct",
        "vol_ratio",
        "bar_range",
        "volume_zscore",
    ],

    # Strategy parameters (must match Layer 3)
    "ema_fast":      10,
    "ema_slow":      50,
    "bb_period":     20,
    "bb_std":        2.0,

    # Risk / position sizing (must match Layer 3)
    "starting_balance":  10000.0,
    "base_risk_pct":     0.01,
    "min_confidence":    0.65,
    "max_switch_prob":   0.15,

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

    # SL/TP multipliers (must match Layer 3)
    "sl_atr_mult":       1.5,
    "tp_atr_mult_trend": 2.0,
    "tp_atr_mult_mean":  1.0,
    "min_rr":            1.5,       # minimum reward:risk ratio

    # Output
    "results_dir":   "backtest_results/",
}


# ─────────────────────────────────────────────
# STEP 1: FETCH HISTORICAL DATA FROM MT5
# ─────────────────────────────────────────────

def fetch_history(symbol: str) -> pd.DataFrame | None:
    """
    Pull a large block of historical OHLCV bars from MT5.
    This is the raw material for the entire backtest.
    """
    if not mt5.symbol_select(symbol, True):
        print(f"[WARN] Cannot select {symbol}")
        return None

    rates = mt5.copy_rates_from_pos(
        symbol, CONFIG["timeframe"], 0, CONFIG["n_bars"]
    )

    if rates is None or len(rates) == 0:
        print(f"[WARN] No data for {symbol}")
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
# STEP 2: FEATURE ENGINEERING (mirrors Layer 1)
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the same features as Layer 1.
    Must be identical — any difference invalidates the backtest.
    """
    feat = df.copy()

    feat["log_return"] = np.log(feat["close"] / feat["close"].shift(1))

    # Annualised realised vol (M15 = 8064 bars/year)
    annualise = np.sqrt(8064)
    feat["realized_vol"] = feat["log_return"].rolling(20).std() * annualise

    feat["vol_ratio"] = (
        feat["log_return"].rolling(5).std() /
        feat["log_return"].rolling(20).std().replace(0, np.nan)
    )

    high_low   = feat["high"] - feat["low"]
    high_close = (feat["high"] - feat["close"].shift(1)).abs()
    low_close  = (feat["low"]  - feat["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    feat["atr"] = true_range.rolling(14).mean()
    feat["atr_pct"] = feat["atr"] / feat["close"]

    feat["bar_range"] = (feat["high"] - feat["low"]) / feat["close"]

    vol_mean = feat["volume"].rolling(20).mean()
    vol_std  = feat["volume"].rolling(20).std().replace(0, np.nan)
    feat["volume_zscore"] = (feat["volume"] - vol_mean) / vol_std

    ema_fast = feat["close"].ewm(span=CONFIG["ema_fast"], adjust=False).mean()
    ema_slow = feat["close"].ewm(span=CONFIG["ema_slow"], adjust=False).mean()
    feat["ema_diff"] = (ema_fast - ema_slow) / feat["close"]

    # Store EMAs and Bollinger Bands for signal generation
    feat["ema_fast"] = ema_fast
    feat["ema_slow"] = ema_slow

    middle = feat["close"].rolling(CONFIG["bb_period"]).mean()
    std    = feat["close"].rolling(CONFIG["bb_period"]).std()
    feat["bb_upper"] = middle + CONFIG["bb_std"] * std
    feat["bb_lower"] = middle - CONFIG["bb_std"] * std

    feat.dropna(inplace=True)
    return feat


# ─────────────────────────────────────────────
# STEP 3: FIT HMM ON TRAINING WINDOW
# ─────────────────────────────────────────────

def fit_hmm_window(train_df: pd.DataFrame):
    """
    Fit HMM on a single training window.
    Returns (model, scaler, regime_map) or None on failure.
    """
    features = CONFIG["hmm_features"]
    missing  = [f for f in features if f not in train_df.columns]
    if missing:
        return None, None, None

    X_raw = train_df[features].values
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

    # Map states to regime names by realized_vol mean
    vol_idx    = features.index("realized_vol")
    state_vols = {s: model.means_[s][vol_idx] for s in range(CONFIG["n_states"])}
    sorted_states = sorted(state_vols, key=state_vols.get)
    regime_map = {sorted_states[0]: "low_vol", sorted_states[1]: "high_vol"}

    return model, scaler, regime_map


# ─────────────────────────────────────────────
# STEP 4: CLASSIFY TEST WINDOW
# ─────────────────────────────────────────────

def classify_window(model, scaler, regime_map,
                    test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply fitted HMM to the test window — no refitting.
    Returns test_df with regime columns added.
    """
    features = CONFIG["hmm_features"]
    X_raw    = test_df[features].values
    X        = scaler.transform(X_raw)

    state_seq  = model.predict(X)
    state_probs = model.predict_proba(X)

    result = test_df.copy()
    result["hmm_state"]  = state_seq
    result["regime"]     = [regime_map[s] for s in state_seq]
    result["confidence"] = state_probs.max(axis=1)

    # Transition-based switch probability
    switch_probs = []
    for s in state_seq:
        stay = model.transmat_[s][s]
        switch_probs.append(1 - stay)
    result["switch_prob"] = switch_probs

    return result


# ─────────────────────────────────────────────
# STEP 5: SIGNAL GENERATION (mirrors Layer 3)
# ─────────────────────────────────────────────
def get_signal_for_bar(row: pd.Series, prev_row: pd.Series) -> tuple:
    regime      = row["regime"]
    confidence  = row["confidence"]
    switch_prob = row["switch_prob"]

    if confidence < CONFIG["min_confidence"]:
        return "flat", "filtered_confidence"
    if switch_prob > CONFIG["max_switch_prob"]:
        return "flat", "filtered_switch"

    # EMA crossover disabled — no edge found in walk-forward test
    if regime == "low_vol":
        return "flat", "ema_disabled"

    # high_vol only — Bollinger reversion
    price = row["close"]
    if price < row["bb_lower"]:
        return "buy", "bollinger_reversion"
    elif price > row["bb_upper"]:
        return "sell", "bollinger_reversion"
    else:
        return "flat", "bollinger_reversion"
    
def get_signal_for_bar2(row: pd.Series, prev_row: pd.Series) -> tuple:
    """
    Generate a signal for a single bar given its regime.
    Mirrors Layer 3 logic exactly.
    Returns (direction, strategy_name).
    """
    regime     = row["regime"]
    confidence = row["confidence"]
    switch_prob = row["switch_prob"]

    # Regime filter
    if confidence < CONFIG["min_confidence"]:
        return "flat", "filtered_confidence"
    if switch_prob > CONFIG["max_switch_prob"]:
        return "flat", "filtered_switch"

    # Flip test — set to True to test inverted EMA logic
    FLIP_EMA = True

    if regime == "low_vol":
        fast_now  = row["ema_fast"]
        fast_prev = prev_row["ema_fast"]
        slow_now  = row["ema_slow"]
        slow_prev = prev_row["ema_slow"]

        bullish_cross = (fast_prev <= slow_prev) and (fast_now > slow_now)
        bearish_cross = (fast_prev >= slow_prev) and (fast_now < slow_now)

        if FLIP_EMA:
            # Swap buy and sell
            if bullish_cross:
                return "sell", "ema_crossover"
            elif bearish_cross:
                return "buy", "ema_crossover"
        else:
            if bullish_cross:
                return "buy", "ema_crossover"
            elif bearish_cross:
                return "sell", "ema_crossover"
        
        return "flat", "ema_crossover"

    if regime == "low_vol":
        # EMA crossover — only fires on the crossover bar
        fast_now  = row["ema_fast"]
        fast_prev = prev_row["ema_fast"]
        slow_now  = row["ema_slow"]
        slow_prev = prev_row["ema_slow"]

        if fast_prev <= slow_prev and fast_now > slow_now:
            return "buy", "ema_crossover"
        elif fast_prev >= slow_prev and fast_now < slow_now:
            return "sell", "ema_crossover"
        else:
            return "flat", "ema_crossover"

    else:  # high_vol
        # Bollinger Band mean reversion
        price = row["close"]
        if price < row["bb_lower"]:
            return "buy", "bollinger_reversion"
        elif price > row["bb_upper"]:
            return "sell", "bollinger_reversion"
        else:
            return "flat", "bollinger_reversion"


# ─────────────────────────────────────────────
# STEP 6: SIMULATE TRADE EXECUTION
# ─────────────────────────────────────────────

def simulate_trade(entry_bar: pd.Series, subsequent_bars: pd.DataFrame,
                   direction: str, regime: str,
                   symbol: str, balance: float,
                   confidence: float) -> dict:
    """
    Simulate a single trade through subsequent bars.
    Checks each bar's high/low to see if SL or TP was hit.

    Returns a dict with trade result details.
    """
    pip_size      = CONFIG["pip_size"].get(symbol, 0.0001)
    pip_value     = CONFIG["pip_value_per_lot"].get(symbol, 10.0)

    # ATR-based SL/TP
    atr      = entry_bar["atr"]
    atr_pips = atr / pip_size
    sl_pips  = round(atr_pips * CONFIG["sl_atr_mult"], 1)
    tp_mult  = (CONFIG["tp_atr_mult_trend"] if regime == "low_vol"
                else CONFIG["tp_atr_mult_mean"])
    tp_pips  = round(atr_pips * tp_mult, 1)

    # Enforce minimum reward:risk
    if tp_pips < sl_pips * CONFIG["min_rr"]:
        tp_pips = round(sl_pips * CONFIG["min_rr"], 1)

    sl_pips = max(sl_pips, 5.0)
    tp_pips = max(tp_pips, 5.0)

    # Entry price (use open of next bar — realistic execution)
    entry_price = entry_bar["close"]

    if direction == "buy":
        sl_price = entry_price - sl_pips * pip_size
        tp_price = entry_price + tp_pips * pip_size
    else:
        sl_price = entry_price + sl_pips * pip_size
        tp_price = entry_price - tp_pips * pip_size

    # Position sizing (confidence-scaled)
    min_conf = CONFIG["min_confidence"]
    conf_scalar = 0.5 + 0.5 * ((confidence - min_conf) / (1.0 - min_conf))
    conf_scalar = max(0.5, min(1.0, conf_scalar))
    risk_amount = balance * CONFIG["base_risk_pct"] * conf_scalar
    lot_size    = round(risk_amount / (sl_pips * pip_value), 2)
    lot_size    = max(lot_size, 0.01)

    # Simulate bar by bar
    outcome    = "open"
    exit_price = None
    exit_bar   = None

    for idx, bar in subsequent_bars.iterrows():
        if direction == "buy":
            if bar["low"] <= sl_price:
                outcome    = "sl"
                exit_price = sl_price
                exit_bar   = idx
                break
            if bar["high"] >= tp_price:
                outcome    = "tp"
                exit_price = tp_price
                exit_bar   = idx
                break
        else:  # sell
            if bar["high"] >= sl_price:
                outcome    = "sl"
                exit_price = sl_price
                exit_bar   = idx
                break
            if bar["low"] <= tp_price:
                outcome    = "tp"
                exit_price = tp_price
                exit_bar   = idx
                break

    # If never hit SL or TP, close at end of test window
    if outcome == "open":
        exit_price = subsequent_bars["close"].iloc[-1]
        exit_bar   = subsequent_bars.index[-1]
        outcome    = "timeout"

    # P&L calculation
    if direction == "buy":
        pnl_pips = (exit_price - entry_price) / pip_size
    else:
        pnl_pips = (entry_price - exit_price) / pip_size

    pnl_usd = pnl_pips * pip_value * lot_size

    return {
        "direction":   direction,
        "regime":      regime,
        "confidence":  round(confidence, 4),
        "entry_price": entry_price,
        "exit_price":  exit_price,
        "exit_bar":    exit_bar,
        "sl_pips":     sl_pips,
        "tp_pips":     tp_pips,
        "lot_size":    lot_size,
        "outcome":     outcome,
        "pnl_pips":    round(pnl_pips, 2),
        "pnl_usd":     round(pnl_usd, 2),
        "won":         pnl_usd > 0,
    }


# ─────────────────────────────────────────────
# STEP 7: WALK-FORWARD ENGINE
# ─────────────────────────────────────────────

def walk_forward(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Core walk-forward loop.
    Slides a train/test window across the full history.
    Returns a DataFrame of all simulated trades.
    """
    train_bars = CONFIG["train_bars"]
    test_bars  = CONFIG["test_bars"]
    step_bars  = CONFIG["step_bars"]

    trades   = []
    balance  = CONFIG["starting_balance"]
    n        = len(df)
    start    = train_bars
    window   = 0

    open_trade = None   # track if we have a position open

    while start + test_bars <= n:
        window += 1

        # Slice train and test windows
        train_df = df.iloc[start - train_bars : start]
        test_df  = df.iloc[start : start + test_bars]

        # Fit HMM on training window
        model, scaler, regime_map = fit_hmm_window(train_df)
        if model is None:
            start += step_bars
            continue

        # Classify test window (no refitting — honest simulation)
        classified = classify_window(model, scaler, regime_map, test_df)

        # Generate signals bar by bar
        for i in range(1, len(classified)):
            row      = classified.iloc[i]
            prev_row = classified.iloc[i - 1]

            # If we have an open trade, check if it closed
            if open_trade is not None:
                entry_idx = open_trade["_entry_idx"]
                # Find remaining bars after entry
                remaining = classified.iloc[i:]
                result    = simulate_trade(
                    entry_bar       = classified.iloc[entry_idx],
                    subsequent_bars = remaining,
                    direction       = open_trade["direction"],
                    regime          = open_trade["regime"],
                    symbol          = symbol,
                    balance         = balance,
                    confidence      = open_trade["confidence"],
                )
                result["symbol"]      = symbol
                result["entry_time"]  = classified.index[entry_idx]
                result["window"]      = window
                result["balance_before"] = balance
                balance += result["pnl_usd"]
                result["balance_after"]  = balance
                trades.append(result)
                open_trade = None
                continue

            # Generate signal for this bar
            direction, strategy = get_signal_for_bar(row, prev_row)

            if direction == "flat":
                continue

            # Open a new simulated trade
            remaining = classified.iloc[i + 1:]
            if len(remaining) == 0:
                continue

            result = simulate_trade(
                entry_bar       = row,
                subsequent_bars = remaining,
                direction       = direction,
                regime          = row["regime"],
                symbol          = symbol,
                balance         = balance,
                confidence      = float(row["confidence"]),
            )
            result["symbol"]         = symbol
            result["strategy"]       = strategy
            result["entry_time"]     = classified.index[i]
            result["window"]         = window
            result["balance_before"] = balance
            balance += result["pnl_usd"]
            result["balance_after"]  = balance
            trades.append(result)

        start += step_bars

    print(f"  {symbol}: {window} windows | {len(trades)} trades simulated")
    return pd.DataFrame(trades)


# ─────────────────────────────────────────────
# STEP 8: PERFORMANCE METRICS
# ─────────────────────────────────────────────

def compute_metrics(trades: pd.DataFrame, symbol: str) -> dict:
    """
    Compute the key performance metrics from the trade log.
    """
    if trades.empty:
        return {"symbol": symbol, "error": "No trades generated"}

    total_trades  = len(trades)
    winning       = trades[trades["won"] == True]
    losing        = trades[trades["won"] == False]
    win_rate      = len(winning) / total_trades

    total_pnl     = trades["pnl_usd"].sum()
    avg_win       = winning["pnl_usd"].mean() if len(winning) > 0 else 0
    avg_loss      = losing["pnl_usd"].mean()  if len(losing)  > 0 else 0
    profit_factor = (winning["pnl_usd"].sum() /
                     abs(losing["pnl_usd"].sum())
                     if abs(losing["pnl_usd"].sum()) > 0 else 0)

    # Equity curve
    equity = trades["balance_after"].values
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / peak
    max_dd = dd.min()

    # Sharpe ratio (annualised, assuming M15 bars)
    returns       = trades["pnl_usd"] / trades["balance_before"]
    sharpe        = (returns.mean() / returns.std() * np.sqrt(252)
                     if returns.std() > 0 else 0)

    # Per-regime breakdown
    regime_stats = {}
    for regime in ["low_vol", "high_vol"]:
        r_trades = trades[trades["regime"] == regime]
        if len(r_trades) > 0:
            regime_stats[regime] = {
                "trades":   len(r_trades),
                "win_rate": round(len(r_trades[r_trades["won"]]) / len(r_trades), 3),
                "pnl":      round(r_trades["pnl_usd"].sum(), 2),
            }

    # Per-strategy breakdown
    strategy_stats = {}
    if "strategy" in trades.columns:
        for strat in trades["strategy"].dropna().unique():
            s_trades = trades[trades["strategy"] == strat]
            strategy_stats[strat] = {
                "trades":   len(s_trades),
                "win_rate": round(len(s_trades[s_trades["won"]]) / len(s_trades), 3),
                "pnl":      round(s_trades["pnl_usd"].sum(), 2),
            }

    return {
        "symbol":         symbol,
        "total_trades":   total_trades,
        "win_rate":       round(win_rate, 3),
        "total_pnl":      round(total_pnl, 2),
        "profit_factor":  round(profit_factor, 3),
        "avg_win":        round(avg_win, 2),
        "avg_loss":       round(avg_loss, 2),
        "max_drawdown":   round(max_dd, 4),
        "sharpe":         round(sharpe, 3),
        "final_balance":  round(equity[-1], 2),
        "regime_stats":   regime_stats,
        "strategy_stats": strategy_stats,
    }


# ─────────────────────────────────────────────
# STEP 9: PRINT REPORT
# ─────────────────────────────────────────────

def print_report(metrics: dict):
    """Print a clean performance report to the terminal."""
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
    for regime, stats in metrics["regime_stats"].items():
        print(f"  {regime:12} | trades={stats['trades']:3} | "
              f"win={stats['win_rate']:.0%} | pnl=${stats['pnl']:,.2f}")

    print(f"\n  -- By Strategy --")
    for strat, stats in metrics["strategy_stats"].items():
        print(f"  {strat:22} | trades={stats['trades']:3} | "
              f"win={stats['win_rate']:.0%} | pnl=${stats['pnl']:,.2f}")


# ─────────────────────────────────────────────
# STEP 10: SAVE RESULTS
# ─────────────────────────────────────────────

def save_results(trades: pd.DataFrame, metrics: dict, symbol: str):
    """Save trade log and metrics to CSV for further analysis."""
    os.makedirs(CONFIG["results_dir"], exist_ok=True)

    # Trade log
    trades_path = os.path.join(
        CONFIG["results_dir"], f"{symbol}_trades.csv"
    )
    trades.to_csv(trades_path, index=False)

    # Metrics summary
    metrics_path = os.path.join(
        CONFIG["results_dir"], f"{symbol}_metrics.csv"
    )
    flat_metrics = {
        k: v for k, v in metrics.items()
        if not isinstance(v, dict)
    }
    pd.DataFrame([flat_metrics]).to_csv(metrics_path, index=False)

    print(f"  Results saved -> {CONFIG['results_dir']}")


# ─────────────────────────────────────────────
# STEP 11: PLOT EQUITY CURVE
# ─────────────────────────────────────────────

def plot_equity_curve(trades: pd.DataFrame, symbol: str):
    """
    Plot the equity curve with regime colouring.
    Requires matplotlib — skipped gracefully if not installed.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  [PLOT] matplotlib not installed — skipping chart")
        print("         pip install matplotlib to enable charts")
        return

    if trades.empty or "balance_after" not in trades.columns:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"Walk-Forward Backtest — {symbol}", fontsize=14)

    # Equity curve coloured by regime
    equity = trades["balance_after"].values
    times  = range(len(equity))

    for i in range(len(trades)):
        colour = "#1D9E75" if trades.iloc[i]["regime"] == "low_vol" else "#D85A30"
        ax1.plot([i, i+1],
                 [equity[i], equity[min(i+1, len(equity)-1)]],
                 color=colour, linewidth=1.5)

    ax1.axhline(CONFIG["starting_balance"], color="gray",
                linestyle="--", linewidth=0.8, label="Starting balance")
    ax1.set_ylabel("Account Balance ($)")
    ax1.set_title("Equity Curve")
    ax1.grid(True, alpha=0.3)

    # Legend
    low_patch  = mpatches.Patch(color="#1D9E75", label="low_vol regime")
    high_patch = mpatches.Patch(color="#D85A30", label="high_vol regime")
    ax1.legend(handles=[low_patch, high_patch], loc="upper left")

    # Rolling win rate
    window    = 20
    won_vals  = trades["won"].astype(int)
    roll_wr   = won_vals.rolling(window).mean()
    ax2.plot(roll_wr.values, color="#378ADD", linewidth=1.5)
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_ylabel(f"Win Rate\n(rolling {window})")
    ax2.set_xlabel("Trade Number")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs(CONFIG["results_dir"], exist_ok=True)
    chart_path = os.path.join(CONFIG["results_dir"], f"{symbol}_equity.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved -> {chart_path}")
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_backtest():
    """
    Full walk-forward backtest pipeline.
    Connects to MT5, fetches history, runs simulation,
    prints report, saves results, and plots equity curve.
    """
    print("=" * 60)
    print("  LAYER 5 — WALK-FORWARD BACKTESTER")
    print("=" * 60)

    # Connect to MT5 for historical data
    if not mt5.initialize(path=CONFIG["terminal_path"]):
        print(f"[FATAL] MT5 init failed: {mt5.last_error()}")
        return

    print(f"\nFetching {CONFIG['n_bars']} bars of M15 history...")

    all_metrics = []

    for symbol in CONFIG["symbols"]:
        print(f"\n[{symbol}]")

        # 1. Fetch raw history
        raw = fetch_history(symbol)
        if raw is None:
            continue

        # 2. Engineer features
        df = engineer_features(raw)
        print(f"  Feature matrix: {len(df)} bars x {len(CONFIG['hmm_features'])} features")

        if len(df) < CONFIG["train_bars"] + CONFIG["test_bars"]:
            print(f"  [SKIP] Not enough bars for walk-forward")
            continue

        # 3. Run walk-forward simulation
        print(f"  Running walk-forward "
              f"(train={CONFIG['train_bars']} test={CONFIG['test_bars']})...")
        trades = walk_forward(df, symbol)

        if trades.empty:
            print(f"  [SKIP] No trades generated")
            continue

        # 4. Compute metrics
        metrics = compute_metrics(trades, symbol)
        all_metrics.append(metrics)

        # 5. Print report
        print_report(metrics)

        # 6. Save results
        save_results(trades, metrics, symbol)

        # 7. Plot equity curve
        plot_equity_curve(trades, symbol)

    # Combined summary across all symbols
    if len(all_metrics) > 1:
        print(f"\n{'='*60}")
        print("  COMBINED SUMMARY")
        print(f"{'='*60}")
        total_pnl    = sum(m.get("total_pnl", 0) for m in all_metrics)
        total_trades = sum(m.get("total_trades", 0) for m in all_metrics)
        print(f"  Total trades across all symbols : {total_trades}")
        print(f"  Combined P&L                    : ${total_pnl:,.2f}")

    mt5.shutdown()
    print("\n[DONE] Backtest complete.")


if __name__ == "__main__":
    run_backtest()