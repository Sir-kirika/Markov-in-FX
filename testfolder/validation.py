"""
=============================================================
LAYER 5 — VALIDATION BACKTEST
Runs the exact winning configuration from optimisation results.
Edit STRATEGY_CONFIG and PARAMS below to test any setup.
=============================================================
Requirements:
    pip install MetaTrader5 pandas numpy hmmlearn scikit-learn joblib matplotlib

Standalone — does NOT require Layers 1-4 to be running.
Only needs MT5 terminal open for historical data.

Current config based on optimisation winners:
    EURUSD high_vol : Bollinger BB_20_3.0  (Sharpe 2.10, DD -11.3%)
    GBPUSD low_vol  : EMA_flip_10_50       (Sharpe 1.63, DD -14.9%)
    USDJPY          : disabled             (no edge found)
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
# STRATEGY CONFIG
# Edit this section to test different setups.
# Options per regime: "bollinger" | "ema" | "ema_flipped" | "rsi" | "disabled"
# ─────────────────────────────────────────────

STRATEGY_CONFIG = {
    "EURUSD": {
        "low_vol":  "disabled",
        "high_vol": "bollinger",    # BB_20_3.0 — Sharpe 2.10, DD -11.3%
    },
    "GBPUSD": {
        "low_vol":  "ema_flipped",  # EMA_flip_10_50 — Sharpe 1.63, DD -14.9%
        "high_vol": "disabled",
    },
    "USDJPY": {
        "low_vol":  "disabled",
        "high_vol": "disabled",
    },
}

# ─────────────────────────────────────────────
# PARAMS
# ─────────────────────────────────────────────

PARAMS = {
    "bb_period":          20,
    "bb_std":             3.0,
    "ema_fast":           10,
    "ema_slow":           50,
    "rsi_period":         21,
    "rsi_oversold":       20,
    "rsi_overbought":     80,
    "atr_channel_period": 20,
    "atr_break_mult":     0.5,
    "vwap_threshold":     0.002,
}


# ─────────────────────────────────────────────
# BASE CONFIG
# ─────────────────────────────────────────────

CONFIG = {
    "terminal_path": r"C:\Program Files\MetaTrader 5\terminal64.exe",
    "terminal_path_2": r"C:\Program Files\EGM Securities MetaTrader 5 Terminal\terminal64.exe",
    "symbols":       ["EURUSD", "GBPUSD", "USDJPY"],
    "timeframe":     mt5.TIMEFRAME_M15,
    "n_bars":        170000,

    "train_bars": 400,
    "test_bars":  80,
    "step_bars":  80,

    "n_states": 2,
    "n_iter":   200,
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
        "EURUSD": 0.0001,
        "GBPUSD": 0.0001,
        "USDJPY": 0.01,
    },
    "pip_value_per_lot": {
        "EURUSD": 10.0,
        "GBPUSD": 10.0,
        "USDJPY": 9.0,
    },

    "results_dir": "backtest_results/validation/",
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
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
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

    feat["ema_fast"] = feat["close"].ewm(span=PARAMS["ema_fast"], adjust=False).mean()
    feat["ema_slow"] = feat["close"].ewm(span=PARAMS["ema_slow"], adjust=False).mean()

    bb_mid           = feat["close"].rolling(PARAMS["bb_period"]).mean()
    bb_std           = feat["close"].rolling(PARAMS["bb_period"]).std()
    feat["bb_upper"] = bb_mid + PARAMS["bb_std"] * bb_std
    feat["bb_lower"] = bb_mid - PARAMS["bb_std"] * bb_std
    feat["bb_mid"]   = bb_mid

    delta          = feat["close"].diff()
    gain           = delta.clip(lower=0).rolling(PARAMS["rsi_period"]).mean()
    loss           = (-delta.clip(upper=0)).rolling(PARAMS["rsi_period"]).mean()
    rs             = gain / loss.replace(0, np.nan)
    feat["rsi"]    = 100 - (100 / (1 + rs))

    cp = PARAMS["atr_channel_period"]
    feat["highest_high"] = feat["high"].rolling(cp).max()
    feat["lowest_low"]   = feat["low"].rolling(cp).min()

    feat["typical_price"] = (feat["high"] + feat["low"] + feat["close"]) / 3
    feat["tp_volume"]     = feat["typical_price"] * feat["volume"]
    feat["date"]          = feat.index.date
    feat["cum_tp_vol"]    = feat.groupby("date")["tp_volume"].cumsum()
    feat["cum_vol"]       = feat.groupby("date")["volume"].cumsum()
    feat["vwap"]          = feat["cum_tp_vol"] / feat["cum_vol"].replace(0, np.nan)
    feat.drop(columns=["date", "cum_tp_vol", "cum_vol",
                        "typical_price", "tp_volume"], inplace=True)

    feat.dropna(inplace=True)
    return feat


# ─────────────────────────────────────────────
# HMM FIT + CLASSIFY
# ─────────────────────────────────────────────

def fit_hmm(train_df: pd.DataFrame):
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

    vol_idx       = features.index("realized_vol")
    state_vols    = {s: model.means_[s][vol_idx] for s in range(CONFIG["n_states"])}
    sorted_states = sorted(state_vols, key=state_vols.get)
    regime_map    = {sorted_states[0]: "low_vol", sorted_states[1]: "high_vol"}

    return model, scaler, regime_map


def classify_window(model, scaler, regime_map, test_df: pd.DataFrame) -> pd.DataFrame:
    features    = CONFIG["hmm_features"]
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


def signal_ema(row: pd.Series, prev_row: pd.Series, flipped: bool = False) -> tuple:
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


def signal_rsi(row: pd.Series) -> tuple:
    rsi = row["rsi"]
    if rsi < PARAMS["rsi_oversold"]:
        return "buy",  "rsi"
    elif rsi > PARAMS["rsi_overbought"]:
        return "sell", "rsi"
    return "flat", "rsi"


def signal_atr_breakout(row: pd.Series) -> tuple:
    price    = row["close"]
    mult     = PARAMS["atr_break_mult"]
    upper_ch = row["highest_high"] + mult * row["atr"]
    lower_ch = row["lowest_low"]   - mult * row["atr"]
    if price > upper_ch:
        return "buy",  "atr_breakout"
    elif price < lower_ch:
        return "sell", "atr_breakout"
    return "flat", "atr_breakout"


def signal_vwap(row: pd.Series) -> tuple:
    price     = row["close"]
    vwap      = row.get("vwap", 0)
    threshold = PARAMS["vwap_threshold"]
    if vwap == 0 or np.isnan(vwap):
        return "flat", "vwap"
    deviation = (price - vwap) / vwap
    if deviation < -threshold:
        return "buy",  "vwap"
    elif deviation > threshold:
        return "sell", "vwap"
    return "flat", "vwap"


def get_signal(row: pd.Series, prev_row: pd.Series, symbol: str) -> tuple:
    regime      = row["regime"]
    confidence  = row["confidence"]
    switch_prob = row["switch_prob"]

    if confidence  < CONFIG["min_confidence"]:
        return "flat", "filtered_confidence"
    if switch_prob > CONFIG["max_switch_prob"]:
        return "flat", "filtered_switch"

    sym_cfg       = STRATEGY_CONFIG.get(symbol, {})
    strategy_type = sym_cfg.get(regime, "disabled")

    if strategy_type == "disabled":
        return "flat", "disabled"
    if strategy_type == "bollinger":
        return signal_bollinger(row)
    elif strategy_type == "ema":
        return signal_ema(row, prev_row, flipped=False)
    elif strategy_type == "ema_flipped":
        return signal_ema(row, prev_row, flipped=True)
    elif strategy_type == "rsi":
        return signal_rsi(row)
    elif strategy_type == "atr_breakout":
        return signal_atr_breakout(row)
    elif strategy_type == "vwap":
        return signal_vwap(row)

    return "flat", "unknown"


# ─────────────────────────────────────────────
# TRADE SIMULATION
# ─────────────────────────────────────────────

def simulate_trade(entry_bar: pd.Series, subsequent_bars: pd.DataFrame,
                   direction: str, strategy: str, regime: str,
                   symbol: str, balance: float, confidence: float) -> dict:
    pip_size  = CONFIG["pip_size"].get(symbol, 0.0001)
    pip_value = CONFIG["pip_value_per_lot"].get(symbol, 10.0)

    atr_pips = entry_bar["atr"] / pip_size
    sl_pips  = max(round(atr_pips * CONFIG["sl_atr_mult"], 1), 5.0)

    trend_strategies = {"ema", "ema_flipped", "atr_breakout"}
    tp_mult = 2.0 if strategy in trend_strategies else 1.0
    tp_pips = max(round(atr_pips * tp_mult, 1), sl_pips * CONFIG["min_rr"], 5.0)

    entry_price = entry_bar["close"]
    if direction == "buy":
        sl_price = entry_price - sl_pips * pip_size
        tp_price = entry_price + tp_pips * pip_size
    else:
        sl_price = entry_price + sl_pips * pip_size
        tp_price = entry_price - tp_pips * pip_size

    min_conf    = CONFIG["min_confidence"]
    conf_scalar = max(0.5, min(1.0,
        0.5 + 0.5 * ((confidence - min_conf) / (1.0 - min_conf))
    ))
    risk_amount = balance * CONFIG["base_risk_pct"] * conf_scalar
    lot_size    = max(round(risk_amount / (sl_pips * pip_value), 2), 0.01)

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
        "direction":  direction,
        "regime":     regime,
        "strategy":   strategy,
        "confidence": round(confidence, 4),
        "sl_pips":    sl_pips,
        "tp_pips":    tp_pips,
        "lot_size":   lot_size,
        "outcome":    outcome,
        "pnl_pips":   round(pnl_pips, 2),
        "pnl_usd":    pnl_usd,
        "won":        pnl_usd > 0,
    }


# ─────────────────────────────────────────────
# WALK-FORWARD ENGINE
# ─────────────────────────────────────────────

def walk_forward(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    train_bars = CONFIG["train_bars"]
    test_bars  = CONFIG["test_bars"]
    step_bars  = CONFIG["step_bars"]

    trades  = []
    balance = CONFIG["starting_balance"]
    n       = len(df)
    start   = train_bars
    windows = 0

    while start + test_bars <= n:
        windows += 1
        train_df   = df.iloc[start - train_bars : start]
        test_df    = df.iloc[start : start + test_bars]

        model, scaler, regime_map = fit_hmm(train_df)
        if model is None:
            start += step_bars
            continue

        classified = classify_window(model, scaler, regime_map, test_df)

        for i in range(1, len(classified)):
            row      = classified.iloc[i]
            prev_row = classified.iloc[i - 1]

            direction, strategy = get_signal(row, prev_row, symbol)

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

            result["symbol"]         = symbol
            result["entry_time"]     = classified.index[i]
            result["window"]         = windows
            result["balance_before"] = balance
            balance                 += result["pnl_usd"]
            result["balance_after"]  = balance
            trades.append(result)

        start += step_bars

    print(f"  {symbol}: {windows} windows | {len(trades)} trades simulated")
    return pd.DataFrame(trades)


# ─────────────────────────────────────────────
# METRICS
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

    returns    = trades["pnl_usd"] / trades["balance_before"]
    sharpe     = (returns.mean() / returns.std() * np.sqrt(252)
                  if returns.std() > 0 else 0)
    expectancy = total_pnl / total

    regime_stats = {}
    for regime in ["low_vol", "high_vol"]:
        r = trades[trades["regime"] == regime]
        if len(r) > 0:
            regime_stats[regime] = {
                "trades":   len(r),
                "win_rate": round(len(r[r["won"]]) / len(r), 3),
                "pnl":      round(r["pnl_usd"].sum(), 2),
            }

    strategy_stats = {}
    for strat in trades["strategy"].dropna().unique():
        s = trades[trades["strategy"] == strat]
        strategy_stats[strat] = {
            "trades":   len(s),
            "win_rate": round(len(s[s["won"]]) / len(s), 3),
            "pnl":      round(s["pnl_usd"].sum(), 2),
        }

    return {
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
        "regime_stats":   regime_stats,
        "strategy_stats": strategy_stats,
        "_equity":        equity,           # stored for plotting, not saved to CSV
        "_trades":        trades,           # stored for plotting
    }


# ─────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────

def print_report(metrics: dict):
    print(f"\n{'='*60}")
    print(f"  VALIDATION REPORT — {metrics['symbol']}")
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
    print(f"  Expectancy      : ${metrics['expectancy']:.2f} per trade")
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
# SAVE
# ─────────────────────────────────────────────

def save_results(trades: pd.DataFrame, metrics: dict, symbol: str):
    os.makedirs(CONFIG["results_dir"], exist_ok=True)

    trades.to_csv(
        os.path.join(CONFIG["results_dir"], f"{symbol}_trades.csv"),
        index=False
    )

    # Exclude private keys (_equity, _trades) from CSV
    flat = {k: v for k, v in metrics.items()
            if not isinstance(v, dict) and not k.startswith("_")}
    pd.DataFrame([flat]).to_csv(
        os.path.join(CONFIG["results_dir"], f"{symbol}_metrics.csv"),
        index=False
    )
    print(f"  Results saved -> {CONFIG['results_dir']}")


# ─────────────────────────────────────────────
# EQUITY CURVE PLOT
# ─────────────────────────────────────────────

def plot_equity_curve(metrics: dict, symbol: str):
    """
    Plot equity curve with regime colouring + rolling win rate.
    Green segments = low_vol regime trades.
    Orange/red segments = high_vol regime trades.
    Saves PNG to results_dir and shows interactive window.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.lines as mlines
    except ImportError:
        print("  [PLOT] matplotlib not installed — run: pip install matplotlib")
        return

    trades = metrics.get("_trades")
    equity = metrics.get("_equity")

    if trades is None or equity is None or len(equity) == 0:
        print("  [PLOT] No data to plot")
        return

    # ── Colour palette ────────────────────────
    colour_low  = "#1D9E75"   # teal  — low_vol regime
    colour_high = "#D85A30"   # coral — high_vol regime
    colour_line = "#378ADD"   # blue  — rolling win rate

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 10),
        gridspec_kw={"height_ratios": [4, 1.5, 1.5]},
        sharex=True
    )
    fig.suptitle(
        f"Walk-Forward Validation — {symbol}   "
        f"(Sharpe {metrics['sharpe']:.2f} | "
        f"DD {metrics['max_drawdown']:.1%} | "
        f"PF {metrics['profit_factor']:.2f})",
        fontsize=13, fontweight="normal"
    )

    # ── Panel 1: Equity curve ─────────────────
    for i in range(len(trades) - 1):
        colour = colour_low if trades.iloc[i]["regime"] == "low_vol" else colour_high
        ax1.plot(
            [i, i + 1],
            [equity[i], equity[i + 1]],
            color=colour, linewidth=1.2, solid_capstyle="round"
        )

    # Starting balance reference line
    ax1.axhline(
        CONFIG["starting_balance"], color="gray",
        linestyle="--", linewidth=0.8, alpha=0.6, label="Starting balance"
    )

    # Peak equity reference
    peak_val = equity.max()
    peak_idx = equity.argmax()
    ax1.annotate(
        f"Peak ${peak_val:,.0f}",
        xy=(peak_idx, peak_val),
        xytext=(peak_idx + len(trades) * 0.02, peak_val),
        fontsize=8, color="gray",
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5)
    )

    ax1.set_ylabel("Account Balance ($)", fontsize=10)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    ax1.grid(True, alpha=0.25, linestyle=":")

    low_patch  = mpatches.Patch(color=colour_low,  label="low_vol regime")
    high_patch = mpatches.Patch(color=colour_high, label="high_vol regime")
    ax1.legend(handles=[low_patch, high_patch], loc="upper left", fontsize=9)

    # ── Panel 2: Drawdown ─────────────────────
    peak_series = np.maximum.accumulate(equity)
    dd_series   = (equity - peak_series) / peak_series * 100   # in %

    ax2.fill_between(
        range(len(dd_series)), dd_series, 0,
        color=colour_high, alpha=0.4, linewidth=0
    )
    ax2.plot(dd_series, color=colour_high, linewidth=0.8)
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.set_ylabel("Drawdown (%)", fontsize=10)
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0f}%")
    )
    ax2.grid(True, alpha=0.25, linestyle=":")

    # Annotate max drawdown
    max_dd_idx = dd_series.argmin()
    ax2.annotate(
        f"Max DD {dd_series.min():.1f}%",
        xy=(max_dd_idx, dd_series.min()),
        xytext=(max_dd_idx + len(trades) * 0.02, dd_series.min() * 0.7),
        fontsize=8, color=colour_high,
        arrowprops=dict(arrowstyle="-", color=colour_high, lw=0.5)
    )

    # ── Panel 3: Rolling win rate ─────────────
    window    = 20
    roll_wr   = trades["won"].astype(int).rolling(window).mean() * 100

    ax3.plot(roll_wr.values, color=colour_line, linewidth=1.2)
    ax3.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax3.fill_between(
        range(len(roll_wr)),
        roll_wr.values, 50,
        where=(roll_wr.values >= 50),
        color=colour_low, alpha=0.25, linewidth=0
    )
    ax3.fill_between(
        range(len(roll_wr)),
        roll_wr.values, 50,
        where=(roll_wr.values < 50),
        color=colour_high, alpha=0.25, linewidth=0
    )
    ax3.set_ylabel(f"Win Rate %\n(rolling {window})", fontsize=10)
    ax3.set_xlabel("Trade Number", fontsize=10)
    ax3.set_ylim(0, 100)
    ax3.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0f}%")
    )
    ax3.grid(True, alpha=0.25, linestyle=":")

    plt.tight_layout()

    os.makedirs(CONFIG["results_dir"], exist_ok=True)
    chart_path = os.path.join(CONFIG["results_dir"], f"{symbol}_equity.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved -> {chart_path}")
    plt.show()


# ─────────────────────────────────────────────
# PRINT ACTIVE CONFIG
# ─────────────────────────────────────────────

def print_active_config():
    print("\n  Active strategy config:")
    for symbol, regimes in STRATEGY_CONFIG.items():
        for regime, strategy in regimes.items():
            if strategy != "disabled":
                print(f"    {symbol} {regime:8} -> {strategy}")

    print("\n  Active parameters:")
    for k, v in PARAMS.items():
        print(f"    {k:22} : {v}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_validation():
    print("=" * 60)
    print("  LAYER 5 — VALIDATION BACKTEST")
    print("=" * 60)
    print_active_config()

    if not mt5.initialize(path=CONFIG["terminal_path"]):
        print(f"\n[FATAL] MT5 init failed: {mt5.last_error()}")
        return

    print(f"\nFetching {CONFIG['n_bars']} bars of M15 history...")

    all_metrics = []

    for symbol in CONFIG["symbols"]:
        sym_cfg = STRATEGY_CONFIG.get(symbol, {})
        if all(v == "disabled" for v in sym_cfg.values()):
            print(f"\n[{symbol}] All strategies disabled — skipping")
            continue

        print(f"\n[{symbol}]")
        raw = fetch_history(symbol)
        if raw is None:
            continue

        df = engineer_features(raw)
        print(f"  Feature matrix: {len(df)} bars x "
              f"{len(CONFIG['hmm_features'])} features")

        if len(df) < CONFIG["train_bars"] + CONFIG["test_bars"]:
            print(f"  [SKIP] Not enough bars")
            continue

        trades = walk_forward(df, symbol)

        if trades.empty:
            print(f"  [SKIP] No trades generated")
            continue

        metrics = compute_metrics(trades, symbol)
        all_metrics.append(metrics)
        print_report(metrics)
        save_results(trades, metrics, symbol)
        plot_equity_curve(metrics, symbol)   # show chart per symbol

    # Combined summary
    if len(all_metrics) >= 1:
        print(f"\n{'='*60}")
        print("  COMBINED SUMMARY")
        print(f"{'='*60}")
        total_pnl    = sum(m.get("total_pnl", 0) for m in all_metrics)
        total_trades = sum(m.get("total_trades", 0) for m in all_metrics)
        avg_sharpe   = np.mean([m.get("sharpe", 0) for m in all_metrics])
        worst_dd     = min(m.get("max_drawdown", 0) for m in all_metrics)
        print(f"  Symbols traded  : {len(all_metrics)}")
        print(f"  Total trades    : {total_trades}")
        print(f"  Combined P&L    : ${total_pnl:,.2f}")
        print(f"  Avg Sharpe      : {avg_sharpe:.2f}")
        print(f"  Worst drawdown  : {worst_dd:.1%}")

    mt5.shutdown()
    print("\n[DONE] Validation complete.")


if __name__ == "__main__":
    run_validation()