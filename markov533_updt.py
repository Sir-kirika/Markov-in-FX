"""
=============================================================
LAYER 5 — VALIDATION BACKTEST (v3 — exhaustive optimisation config)
=============================================================
Requirements:
    pip install MetaTrader5 pandas numpy hmmlearn scikit-learn matplotlib

Mirrors Layer 3 SYMBOL_CONFIG exactly.
Edit SYMBOL_CONFIG below to test different configurations.

Current config based on 8-year exhaustive optimisation:
    EURUSD : rsi_21_20_80   low_vol | bollinger_20_3.0  high_vol
    GBPUSD : bollinger_20_3.0 low_vol | rsi_21_20_80    high_vol
    EURGBP : rsi_14_20_80   low_vol  | bollinger_20_2.5 high_vol
    EURCAD : rsi_21_20_80   low_vol  | disabled
    GBPCAD : bollinger_20_3.0 low_vol | rsi_14_20_80    high_vol
    AUDUSD : bollinger_20_2.5 low_vol | ema_flipped_20_100 high_vol
    USDCAD : bollinger_20_3.0 low_vol | ema_flipped_20_100 high_vol
    XAUUSD : disabled                 | ema_flipped_10_200 high_vol
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
# PER-SYMBOL STRATEGY CONFIG
# Must mirror Layer 3 SYMBOL_CONFIG exactly.
# ─────────────────────────────────────────────

SYMBOL_CONFIG = {
    "EURUSD": {
        "low_vol":  "disabled",#"vwap",
        "high_vol": "disabled",#
        "params": {
            "rsi_period": 21, "rsi_oversold": 20, "rsi_overbought": 80,
            "bb_period":  20, "bb_std": 3.0,"vwap_threshold": 0.002,
        },
    },
    "GBPUSD": {
        "low_vol":  "disabled",#"bollinger",
        "high_vol": "disabled",#"rsi",
        "params": {
            "bb_period":  20, "bb_std": 3.0,
            "rsi_period": 21, "rsi_oversold": 20, "rsi_overbought": 80,
        },
    },
    "EURGBP": {
        "low_vol":  "disabled",#"rsi",
        "high_vol": "disabled",#"bollinger",
        "params": {
            "rsi_period": 14, "rsi_oversold": 20, "rsi_overbought": 80,
            "bb_period":  20, "bb_std": 2.5,
        },
    },
    "EURCAD": {
        "low_vol":  "disabled",#"bollinger",
        "high_vol": "rsi",
        "params": {            
            "bb_period":  10, "bb_std": 2.0,
            "rsi_period": 21, "rsi_oversold": 30, "rsi_overbought": 70,
        },
    },
    "GBPCAD": {
        "low_vol":  "bollinger",
        "high_vol": "disabled",#"rsi",
        "params": {
            "bb_period":  50, "bb_std": 2.0,
            "rsi_period": 14, "rsi_oversold": 20, "rsi_overbought": 80,
        },
    },
    "AUDUSD": {
        "low_vol":  "disabled",#"bollinger",
        "high_vol": "disabled",#"ema_flipped",
        "params": {
            "bb_period": 20, "bb_std": 2.5,
            "ema_fast":  20, "ema_slow": 100,
        },
    },
    "USDCAD": {
        "low_vol":  "disabled",#"ema_flipped",
        "high_vol": "disabled",
        "params": {
            "bb_period": 20, "bb_std": 3.0,
            "ema_fast":  10, "ema_slow": 200,
        },
    },
    "XAUUSD": {
        "low_vol":  "disabled",
        "high_vol": "disabled",
        "params": {
            "ema_fast": 10, "ema_slow": 200,
        },
    },
}


# ─────────────────────────────────────────────
# BASE CONFIG
# ─────────────────────────────────────────────

CONFIG = {
    "terminal_path": r"C:\Program Files\EGM Securities MetaTrader 5 Terminal\terminal64.exe",
    "timeframe":     mt5.TIMEFRAME_H1,
    "n_bars":        175000,

    "train_bars": 2000,
    "test_bars":  200,
    "step_bars":  200,
    "n_states":   2,
    "n_iter":     50,

    "hmm_features": [
        "realized_vol", "atr_pct", "vol_ratio",
        "bar_range",    "volume_zscore",
    ],

    "min_confidence":  0.65,
    "max_switch_prob": 0.15,

    "starting_balance": 10000.0,
    "base_risk_pct":    0.01,
    "sl_atr_mult":      1.5,
    "min_rr":           1.5,
    "max_spread_to_sl_pct": 10,

    "pip_size": {
        "EURUSD": 0.0001, "GBPUSD": 0.0001,
        "EURGBP": 0.0001, "EURCAD": 0.0001,
        "GBPCAD": 0.0001, "AUDUSD": 0.0001,
        "USDCAD": 0.0001, "XAUUSD": 0.1,
    },
    "pip_value_per_lot": {
        "EURUSD": 10.0,  "GBPUSD": 10.0,
        "EURGBP": 12.5,  "EURCAD": 7.5,
        "GBPCAD": 7.5,   "AUDUSD": 10.0,
        "USDCAD": 7.5,   "XAUUSD": 10.0,
    },

    "results_dir": "backtest_results/validation_v533_updt/",
}


# ─────────────────────────────────────────────
# DATA FETCH
# ─────────────────────────────────────────────

def fetch_history(symbol: str) -> pd.DataFrame | None:
    if not mt5.symbol_select(symbol, True):
        print(f"  [WARN] Cannot select {symbol}")
        return None

    rates = mt5.copy_rates_from_pos(
        symbol, CONFIG["timeframe"], 0, CONFIG["n_bars"]
    )
    if rates is None or len(rates) == 0:
        return None
    

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    print(f"copied rates:{len(rates)} instead of {CONFIG['n_bars']} on {symbol} from {df['time'].iloc[0]}")
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
    state_vols    = {s: model.means_[s][vol_idx]
                     for s in range(CONFIG["n_states"])}
    sorted_states = sorted(state_vols, key=state_vols.get)
    regime_map    = {sorted_states[0]: "low_vol",
                     sorted_states[1]: "high_vol"}

    return model, scaler, regime_map


def classify_window(model, scaler, regime_map,
                    test_df: pd.DataFrame) -> pd.DataFrame:
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
    oversold   = params.get("rsi_oversold", 20)
    overbought = params.get("rsi_overbought", 80)
    if rsi < oversold:
        return "buy",  "rsi"
    elif rsi > overbought:
        return "sell", "rsi"
    return "flat", "rsi"


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
               symbol: str) -> tuple:
    regime      = row["regime"]
    confidence  = row["confidence"]
    switch_prob = row["switch_prob"]

    if confidence  < CONFIG["min_confidence"]:
        return "flat", "filtered"
    if switch_prob > CONFIG["max_switch_prob"]:
        return "flat", "filtered"

    sym_cfg       = SYMBOL_CONFIG.get(symbol, {})
    strategy_type = sym_cfg.get(regime, "disabled")
    params        = sym_cfg.get("params", {})

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
    pip_size  = CONFIG["pip_size"].get(symbol, 0.0001)    
    pip_value = CONFIG["pip_value_per_lot"].get(symbol, 10.0)

    atr_pips = entry_bar["atr"] / pip_size
    sl_pips  = max(round(atr_pips * CONFIG["sl_atr_mult"], 1), 5.0)
    spread_pips = entry_bar["spread"]/10
    if(spread_pips>sl_pips*(CONFIG["max_spread_to_sl_pct"]/100)):        
        #print(f"{symbol} {entry_bar.name} SKIPPING spread is {spread_pips} VS min required {sl_pips*(CONFIG["max_spread_to_sl_pct"]/100)}")
        return None
    
    spread_val = spread_pips*CONFIG["pip_size"].get(symbol,0.0001)

    trend_strategies = {"ema", "ema_flipped"}
    tp_mult = 2.0 if strategy in trend_strategies else 1.0
    tp_pips = max(round(atr_pips * tp_mult, 1),
                  sl_pips * CONFIG["min_rr"], 5.0)

    entry_price = entry_bar["close"]
    if direction == "buy":
        entry_price = entry_bar["close"] + spread_val
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
        sl_hit = highs+spread_val >= sl_price
        tp_hit = lows+spread_val  <= tp_price

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
        if direction == "buy" and exit_price > entry_price:
            tp_pips = (exit_price - entry_price)/pip_size
        elif direction == "buy" and exit_price < entry_price:
            sl_pips = abs(exit_price - entry_price)/pip_size
        elif direction == "sell" and exit_price > entry_price:
            sl_pips = abs(exit_price - entry_price)/pip_size
        elif direction == "sell" and exit_price < entry_price:
            tp_pips = (exit_price - entry_price)/pip_size

    min_conf    = CONFIG["min_confidence"]
    conf_scalar = max(0.5, min(1.0,
        0.5 + 0.5 * ((confidence - min_conf) / (1.0 - min_conf))
    ))
    risk_amount = balance * CONFIG["base_risk_pct"] * conf_scalar
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

def walk_forward(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    train_bars = CONFIG["train_bars"]
    test_bars  = CONFIG["test_bars"]
    step_bars  = CONFIG["step_bars"]

    trades  = []
    balance = CONFIG["starting_balance"]
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

            if(result==None):
                continue# trade was skipped due to high spread to al ratio

            result["symbol"]         = symbol
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

def compute_metrics(trades: pd.DataFrame, symbol: str) -> dict:
    if trades.empty:
        return {"symbol": symbol, "error": "No trades"}

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

    strategy_stats = {}
    for strat in trades["strategy"].dropna().unique():
        s = trades[trades["strategy"] == strat]
        strategy_stats[strat] = {
            "trades":   len(s),
            "win_rate": round(len(s[s["won"]]) / len(s), 3),
            "pnl":      round(s["pnl_usd"].sum(), 2),
        }

    regime_stats = {}
    for regime in ["low_vol", "high_vol"]:
        r = trades[trades["regime"] == regime]
        if len(r) > 0:
            regime_stats[regime] = {
                "trades":   len(r),
                "win_rate": round(len(r[r["won"]]) / len(r), 3),
                "pnl":      round(r["pnl_usd"].sum(), 2),
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
        "strategy_stats": strategy_stats,
        "regime_stats":   regime_stats,
        "_equity":        equity,
        "_trades":        trades,
    }


# ─────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────

def print_report(metrics: dict):
    print(f"\n{'='*60}")
    print(f"  VALIDATION REPORT — {metrics['symbol']}")
    print(f"{'='*60}")

    if "error" in metrics:
        print(f"  {metrics['error']}")
        return

    sym_cfg = SYMBOL_CONFIG.get(metrics["symbol"], {})
    active  = []
    if sym_cfg.get("low_vol",  "disabled") != "disabled":
        active.append(f"low_vol={sym_cfg['low_vol']}")
    if sym_cfg.get("high_vol", "disabled") != "disabled":
        active.append(f"high_vol={sym_cfg['high_vol']}")

    print(f"  Strategy       : {' | '.join(active)}")
    print(f"  Params         : {sym_cfg.get('params', {})}")
    print(f"  Total trades   : {metrics['total_trades']:,}")
    print(f"  Win rate       : {metrics['win_rate']:.1%}")
    print(f"  Total P&L      : ${metrics['total_pnl']:,.2f}")
    print(f"  Profit factor  : {metrics['profit_factor']:.2f}")
    print(f"  Avg win        : ${metrics['avg_win']:.2f}")
    print(f"  Avg loss       : ${metrics['avg_loss']:.2f}")
    print(f"  Max drawdown   : {metrics['max_drawdown']:.1%}")
    print(f"  Sharpe ratio   : {metrics['sharpe']:.2f}")
    print(f"  Expectancy     : ${metrics['expectancy']:.2f} per trade")
    print(f"  Final balance  : ${metrics['final_balance']:,.2f}")

    print(f"\n  -- By Regime --")
    for regime, s in metrics.get("regime_stats", {}).items():
        print(f"  {regime:12} | trades={s['trades']:4} | "
              f"win={s['win_rate']:.0%} | pnl=${s['pnl']:,.2f}")

    print(f"\n  -- By Strategy --")
    for strat, s in metrics.get("strategy_stats", {}).items():
        print(f"  {strat:22} | trades={s['trades']:4} | "
              f"win={s['win_rate']:.0%} | pnl=${s['pnl']:,.2f}")


# ─────────────────────────────────────────────
# EQUITY CURVE PLOT
# ─────────────────────────────────────────────

def plot_equity_curve(metrics: dict, symbol: str):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  [PLOT] pip install matplotlib to enable charts")
        return

    trades = metrics.get("_trades")
    equity = metrics.get("_equity")
    if trades is None or equity is None or len(equity) == 0:
        return

    colour_low  = "#1D9E75"
    colour_high = "#D85A30"

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 10),
        gridspec_kw={"height_ratios": [4, 1.5, 1.5]},
        sharex=True
    )

    sym_cfg = SYMBOL_CONFIG.get(symbol, {})
    active  = []
    if sym_cfg.get("low_vol",  "disabled") != "disabled":
        active.append(f"low={sym_cfg['low_vol']}")
    if sym_cfg.get("high_vol", "disabled") != "disabled":
        active.append(f"high={sym_cfg['high_vol']}")

    fig.suptitle(
        f"Walk-Forward Validation — {symbol}  "
        f"({' | '.join(active)})  "
        f"Sharpe {metrics['sharpe']:.2f} | "
        f"DD {metrics['max_drawdown']:.1%} | "
        f"PF {metrics['profit_factor']:.2f}",
        fontsize=12
    )

    # Equity curve
    for i in range(len(trades) - 1):
        colour = (colour_low
                  if trades.iloc[i]["regime"] == "low_vol"
                  else colour_high)
        ax1.plot([i, i+1], [equity[i], equity[i+1]],
                 color=colour, linewidth=1.2)

    ax1.axhline(CONFIG["starting_balance"], color="gray",
                linestyle="--", linewidth=0.8, alpha=0.6)
    peak_val = equity.max()
    peak_idx = equity.argmax()
    ax1.annotate(
        f"Peak ${peak_val:,.0f}",
        xy=(peak_idx, peak_val),
        xytext=(peak_idx + len(trades) * 0.03, peak_val),
        fontsize=8, color="gray",
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5)
    )
    ax1.set_ylabel("Account Balance ($)")
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    ax1.grid(True, alpha=0.25, linestyle=":")
    low_p  = mpatches.Patch(color=colour_low,  label="low_vol")
    high_p = mpatches.Patch(color=colour_high, label="high_vol")
    ax1.legend(handles=[low_p, high_p], loc="upper left", fontsize=9)

    # Drawdown
    peak_series = np.maximum.accumulate(equity)
    dd_series   = (equity - peak_series) / peak_series * 100
    ax2.fill_between(range(len(dd_series)), dd_series, 0,
                     color=colour_high, alpha=0.4)
    ax2.plot(dd_series, color=colour_high, linewidth=0.8)
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.set_ylabel("Drawdown (%)")
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0f}%")
    )
    max_dd_idx = dd_series.argmin()
    ax2.annotate(
        f"Max DD {dd_series.min():.1f}%",
        xy=(max_dd_idx, dd_series.min()),
        xytext=(max_dd_idx + len(trades) * 0.03, dd_series.min() * 0.6),
        fontsize=8, color=colour_high,
        arrowprops=dict(arrowstyle="-", color=colour_high, lw=0.5)
    )
    ax2.grid(True, alpha=0.25, linestyle=":")

    # Rolling win rate
    roll_wr = trades["won"].astype(int).rolling(20).mean() * 100
    ax3.plot(roll_wr.values, color="#378ADD", linewidth=1.2)
    ax3.axhline(50, color="gray", linestyle="--", linewidth=0.8)
    ax3.fill_between(range(len(roll_wr)), roll_wr.values, 50,
                     where=(roll_wr.values >= 50),
                     color=colour_low, alpha=0.25)
    ax3.fill_between(range(len(roll_wr)), roll_wr.values, 50,
                     where=(roll_wr.values < 50),
                     color=colour_high, alpha=0.25)
    ax3.set_ylabel("Win Rate %\n(rolling 20)")
    ax3.set_xlabel("Trade Number")
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
# SAVE
# ─────────────────────────────────────────────

def save_results(trades: pd.DataFrame, metrics: dict, symbol: str):
    os.makedirs(CONFIG["results_dir"], exist_ok=True)
    trades.to_csv(
        os.path.join(CONFIG["results_dir"], f"{symbol}_trades.csv"),
        index=False
    )
    flat = {k: v for k, v in metrics.items()
            if not isinstance(v, dict) and not k.startswith("_")}
    pd.DataFrame([flat]).to_csv(
        os.path.join(CONFIG["results_dir"], f"{symbol}_metrics.csv"),
        index=False
    )
    print(f"  Results saved -> {CONFIG['results_dir']}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_validation():
    print("=" * 60)
    print("  LAYER 5 — VALIDATION BACKTEST (v3)")
    print("=" * 60)

    print("\n  Per-symbol strategy:")
    for sym, cfg in SYMBOL_CONFIG.items():
        lv = cfg.get("low_vol",  "disabled")
        hv = cfg.get("high_vol", "disabled")
        p  = cfg.get("params", {})
        active = []
        if lv != "disabled": active.append(f"low_vol={lv}")
        if hv != "disabled": active.append(f"high_vol={hv}")
        print(f"    {sym:8} | "
              f"{' | '.join(active) or 'disabled':45} | {p}")

    if not mt5.initialize(path=CONFIG["terminal_path"]):
        print(f"\n[FATAL] MT5 init failed: {mt5.last_error()}")
        return

    print(f"\nFetching {CONFIG['n_bars']:,} bars per symbol...")
    all_metrics = []
    start_time  = datetime.now()

    for symbol, sym_cfg in SYMBOL_CONFIG.items():
        if (sym_cfg.get("low_vol",  "disabled") == "disabled" and
                sym_cfg.get("high_vol", "disabled") == "disabled"):
            print(f"\n[{symbol}] All strategies disabled — skipping")
            continue

        print(f"\n[{symbol}]")
        raw = fetch_history(symbol)
        if raw is None:
            continue

        params = sym_cfg.get("params", {})
        df     = engineer_features(raw, params)
        print(f"  Feature matrix: {len(df):,} bars")

        if len(df) < CONFIG["train_bars"] + CONFIG["test_bars"]:
            print(f"  [SKIP] Not enough bars")
            continue

        print(f"  Running walk-forward...", end="", flush=True)
        trades = walk_forward(df, symbol)
        print(f" {len(trades):,} trades")

        if trades.empty:
            print(f"  [SKIP] No trades generated")
            continue

        metrics = compute_metrics(trades, symbol)
        all_metrics.append(metrics)
        print_report(metrics)
        save_results(trades, metrics, symbol)
        plot_equity_curve(metrics, symbol)

    # Combined summary
    valid = [m for m in all_metrics if "error" not in m]
    if valid:
        print(f"\n{'='*60}")
        print("  COMBINED SUMMARY — ALL SYMBOLS")
        print(f"{'='*60}")
        total_pnl    = sum(m["total_pnl"] for m in valid)
        total_trades = sum(m["total_trades"] for m in valid)
        avg_sharpe   = np.mean([m["sharpe"] for m in valid])
        worst_dd     = min(m["max_drawdown"] for m in valid)
        best_dd      = max(m["max_drawdown"] for m in valid)
        total_time   = (datetime.now() - start_time).seconds

        print(f"  Symbols traded  : {len(valid)}")
        print(f"  Total trades    : {total_trades:,}")
        print(f"  Combined P&L    : ${total_pnl:,.2f}")
        print(f"  Avg Sharpe      : {avg_sharpe:.2f}")
        print(f"  Worst drawdown  : {worst_dd:.1%}")
        print(f"  Best drawdown   : {best_dd:.1%}")
        print(f"  Runtime         : {total_time//60}m {total_time%60:02d}s")

        print(f"\n  {'Symbol':8} {'Trades':>7} {'Sharpe':>7} "
              f"{'MaxDD':>7} {'PF':>5} {'P&L':>10}")
        print(f"  {'-'*52}")
        for m in sorted(valid, key=lambda x: x["sharpe"], reverse=True):
            print(
                f"  {m['symbol']:8} {m['total_trades']:>7,} "
                f"{m['sharpe']:>7.2f} "
                f"{m['max_drawdown']:>6.1%} "
                f"{m['profit_factor']:>5.2f} "
                f"${m['total_pnl']:>9,.0f}"
            )

    mt5.shutdown()
    print("\n[DONE] Validation complete.")
    print(f"       Results -> {CONFIG['results_dir']}")


if __name__ == "__main__":
    run_validation()