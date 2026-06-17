"""
=============================================================
LAYER 3 — SIGNAL LAYER (v3 — exhaustive optimisation config)
Regime Filter -> Per-Symbol Strategy -> Position Sizing -> Signal
=============================================================
Requirements:
    pip install pandas numpy joblib

Configuration based on 8-year exhaustive walk-forward optimisation
testing every strategy in every regime across 8 symbols.

Key findings vs previous config:
    - Bollinger in LOW_VOL discovered as strongest signal
      (was previously only tested in high_vol)
    - RSI in LOW_VOL discovered — confirmed on multiple pairs
    - GBPCAD bollinger_20_3.0_LOW: Sharpe 5.64, DD -3.9%
    - EURGBP rsi_14_20_80_LOW: Sharpe 3.71, DD -27%
    - EURCAD rsi_21_20_80_LOW: Sharpe 4.39, DD -12%

Per-symbol config (each symbol uses its own strategy + params):
    EURUSD : rsi_21_20_80   low_vol  | bollinger_20_3.0 high_vol
    GBPUSD : bollinger_20_3.0 low_vol | rsi_21_20_80    high_vol
    EURGBP : rsi_14_20_80   low_vol  | bollinger_20_2.5 high_vol
    EURCAD : rsi_21_20_80   low_vol  | disabled
    GBPCAD : bollinger_20_3.0 low_vol | rsi_14_20_80    high_vol
    AUDUSD : bollinger_20_2.5 low_vol | ema_flipped_20_100 high_vol
    USDCAD : bollinger_20_3.0 low_vol | ema_flipped_20_100 high_vol
=============================================================
"""

import numpy as np
import pandas as pd
import joblib
import os
import warnings
from datetime import datetime
from dataclasses import dataclass

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# PER-SYMBOL CONFIGURATION
# Edit only this block to change strategy settings.
# Each symbol has its own low_vol, high_vol, and params.
#
# Strategy options:
#   "bollinger"   — Bollinger Band mean reversion
#   "ema_flipped" — Inverted EMA (fade the breakout)
#   "ema"         — Standard EMA crossover
#   "rsi"         — RSI mean reversion
#   "vwap"        — VWAP mean reversion
#   "disabled"    — no trading in this regime
# ─────────────────────────────────────────────

SYMBOL_CONFIG = {
    "EURUSD": {
        "low_vol":  "disabled",
        "high_vol": "disabled",
        "params": {
            "rsi_period":     21,
            "rsi_oversold":   20,
            "rsi_overbought": 80,
            "bb_period":      20,
            "bb_std":         3.0,
        },
    },
    "GBPUSD": {
        "low_vol":  "bollinger",
        "high_vol": "rsi",
        "params": {
            "bb_period":      20,
            "bb_std":         3.0,
            "rsi_period":     21,
            "rsi_oversold":   20,
            "rsi_overbought": 80,
        },
    },
    "EURGBP": {
        "low_vol":  "rsi",
        "high_vol": "bollinger",
        "params": {
            "rsi_period":     14,
            "rsi_oversold":   20,
            "rsi_overbought": 80,
            "bb_period":      20,
            "bb_std":         2.5,
        },
    },
    "EURCAD": {
        "low_vol":  "disabled",
        "high_vol": "disabled",
        "params": {
            "rsi_period":     21,
            "rsi_oversold":   20,
            "rsi_overbought": 80,
        },
    },
    "GBPCAD": {
        "low_vol":  "bollinger",
        "high_vol": "rsi",
        "params": {
            "bb_period":      20,
            "bb_std":         3.0,
            "rsi_period":     14,
            "rsi_oversold":   20,
            "rsi_overbought": 80,
        },
    },
    "AUDUSD": {
        "low_vol":  "bollinger",
        "high_vol": "ema_flipped",
        "params": {
            "bb_period":  20,
            "bb_std":     2.5,
            "ema_fast":   20,
            "ema_slow":   100,
        },
    },
    "USDCAD": {
        "low_vol":  "disabled",
        "high_vol": "disabled",
        "params": {
            "bb_period":  20,
            "bb_std":     3.0,
            "ema_fast":   20,
            "ema_slow":   100,
        },
    },
}


# ─────────────────────────────────────────────
# GLOBAL CONFIG
# ─────────────────────────────────────────────

CONFIG = {
    "symbols":   list(SYMBOL_CONFIG.keys()),
    "data_dir":  "data/",
    "model_dir": "models/",

    "min_confidence":  0.65,
    "max_switch_prob": 0.15,

    "base_risk_pct":   0.01,
    "max_risk_pct":    0.02,
    "account_balance": 10000,
    "default_sl_pips": 20,
    "sl_atr_mult":     1.5,
    "min_rr":          1.5,

    "pip_size": {
        "EURUSD": 0.0001, "GBPUSD": 0.0001,
        "EURGBP": 0.0001, "EURCAD": 0.0001,
        "GBPCAD": 0.0001, "AUDUSD": 0.0001,
        "USDCAD": 0.0001, 
    },
    "pip_value_per_lot": {
        "EURUSD": 10.0,  "GBPUSD": 10.0,
        "EURGBP": 12.5,  "EURCAD": 7.5,
        "GBPCAD": 7.5,   "AUDUSD": 10.0,
        "USDCAD": 7.5,   
    },
}


# ─────────────────────────────────────────────
# TRADE SIGNAL
# ─────────────────────────────────────────────

@dataclass
class TradeSignal:
    symbol:           str
    timestamp:        str
    direction:        str
    regime:           str
    confidence:       float
    strategy:         str
    lot_size:         float
    stop_loss_pips:   float
    take_profit_pips: float
    reason:           str


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_features(symbol: str) -> pd.DataFrame | None:
    path = os.path.join(CONFIG["data_dir"], f"{symbol}_features.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, index_col="time", parse_dates=True)


def load_model_bundle(symbol: str) -> dict | None:
    path = os.path.join(CONFIG["model_dir"], f"{symbol}_hmm.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


# ─────────────────────────────────────────────
# REGIME DETECTION
# ─────────────────────────────────────────────

def get_current_regime(symbol: str) -> dict | None:
    bundle = load_model_bundle(symbol)
    if bundle is None:
        return None

    model      = bundle["model"]
    scaler     = bundle["scaler"]
    regime_map = bundle["regime_map"]
    features   = bundle["features"]

    df = load_features(symbol)
    if df is None:
        return None

    X_raw = df[features].dropna().values
    X     = scaler.transform(X_raw)

    state_probs   = model.predict_proba(X)
    latest_probs  = state_probs[-1]
    current_state = int(np.argmax(latest_probs))
    confidence    = float(latest_probs[current_state])
    regime        = regime_map[current_state]
    switch_prob   = float(1 - model.transmat_[current_state][current_state])

    return {
        "regime":      regime,
        "confidence":  confidence,
        "switch_prob": switch_prob,
        "timestamp":   df.index[-1].isoformat(),
    }


def regime_is_tradeable(regime_info: dict) -> tuple:
    if regime_info["confidence"] < CONFIG["min_confidence"]:
        return False, (
            f"Regime uncertain: confidence={regime_info['confidence']:.2%} "
            f"< threshold={CONFIG['min_confidence']:.2%}"
        )
    if regime_info["switch_prob"] > CONFIG["max_switch_prob"]:
        return False, (
            f"Regime unstable: switch_prob={regime_info['switch_prob']:.2%} "
            f"> threshold={CONFIG['max_switch_prob']:.2%}"
        )
    return True, "Regime clear and stable"


# ─────────────────────────────────────────────
# STRATEGY: BOLLINGER BAND MEAN REVERSION
# ─────────────────────────────────────────────

def bollinger_reversion_signal(df: pd.DataFrame,
                                params: dict) -> tuple:
    """
    Bollinger Band mean reversion with per-symbol params.

    Key insight from exhaustive optimisation:
    BB_20_3.0 in LOW_VOL regime is the strongest cross-symbol signal.
    Wide bands in quiet regimes only fire on genuine extremes that always revert.
    Previously only tested in high_vol — exhaustive test revealed low_vol edge.

    GBPCAD low_vol: Sharpe 5.64, DD -3.9% (70 trades over 8 years)
    GBPUSD low_vol: Sharpe 2.59, DD -6.8%
    USDCAD low_vol: Sharpe 2.79, DD -6.4%
    """
    bb_period = params.get("bb_period", 20)
    bb_std    = params.get("bb_std", 3.0)

    close  = df["close"]
    middle = close.rolling(bb_period).mean()
    std    = close.rolling(bb_period).std()
    upper  = middle + bb_std * std
    lower  = middle - bb_std * std

    price_now = close.iloc[-1]
    upper_now = upper.iloc[-1]
    lower_now = lower.iloc[-1]

    if price_now < lower_now:
        return "buy",  (f"Price {price_now:.5f} below lower BB "
                        f"{lower_now:.5f} (std={bb_std})")
    elif price_now > upper_now:
        return "sell", (f"Price {price_now:.5f} above upper BB "
                        f"{upper_now:.5f} (std={bb_std})")
    else:
        return "flat", f"Price inside BB [{lower_now:.5f} - {upper_now:.5f}]"


# ─────────────────────────────────────────────
# STRATEGY: RSI MEAN REVERSION
# ─────────────────────────────────────────────

def rsi_signal(df: pd.DataFrame, params: dict) -> tuple:
    """
    RSI mean reversion with per-symbol params.

    Key insight from exhaustive optimisation:
    RSI in LOW_VOL regime works strongly on multiple pairs.
    In quiet trending markets RSI extremes signal genuine overextension.

    EURCAD low_vol RSI_21_20_80: Sharpe 4.39, DD -12%
    EURGBP low_vol RSI_14_20_80: Sharpe 3.71, DD -27%
    EURUSD low_vol RSI_21_20_80: Sharpe 2.41, DD -11%

    Extreme thresholds (20/80) confirmed — filters overtrading
    while capturing genuine momentum exhaustion signals.
    """
    rsi_period    = params.get("rsi_period", 14)
    rsi_oversold  = params.get("rsi_oversold", 20)
    rsi_overbought = params.get("rsi_overbought", 80)

    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(rsi_period).mean()
    loss  = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = (100 - (100 / (1 + rs))).iloc[-1]

    if rsi < rsi_oversold:
        return "buy",  (f"RSI({rsi_period}) {rsi:.1f} "
                        f"< oversold {rsi_oversold}")
    elif rsi > rsi_overbought:
        return "sell", (f"RSI({rsi_period}) {rsi:.1f} "
                        f"> overbought {rsi_overbought}")
    else:
        return "flat", f"RSI({rsi_period}) {rsi:.1f} neutral"


# ─────────────────────────────────────────────
# STRATEGY: EMA CROSSOVER (standard + flipped)
# ─────────────────────────────────────────────

def ema_crossover_signal(df: pd.DataFrame, params: dict,
                         flipped: bool = False) -> tuple:
    """
    EMA crossover with per-symbol params.
    flipped=True: sell upward cross, buy downward cross (fade breakout).

    AUDUSD high_vol ema_flipped_20_100: Sharpe 2.73, DD -7.5%
    USDCAD high_vol ema_flipped_20_100: Sharpe 1.59, DD -12%
    """
    ema_fast_p = params.get("ema_fast", 10)
    ema_slow_p = params.get("ema_slow", 50)

    ema_fast = df["close"].ewm(span=ema_fast_p, adjust=False).mean()
    ema_slow = df["close"].ewm(span=ema_slow_p, adjust=False).mean()

    fast_now  = ema_fast.iloc[-1]
    fast_prev = ema_fast.iloc[-2]
    slow_now  = ema_slow.iloc[-1]
    slow_prev = ema_slow.iloc[-2]

    bullish_cross = (fast_prev <= slow_prev) and (fast_now > slow_now)
    bearish_cross = (fast_prev >= slow_prev) and (fast_now < slow_now)

    name = "ema_flipped" if flipped else "ema_crossover"

    if bullish_cross:
        direction = "sell" if flipped else "buy"
        return direction, (f"{name}: EMA{ema_fast_p}/"
                           f"EMA{ema_slow_p} crossover")
    elif bearish_cross:
        direction = "buy" if flipped else "sell"
        return direction, (f"{name}: EMA{ema_fast_p}/"
                           f"EMA{ema_slow_p} crossover")
    else:
        return "flat", (f"{name}: no crossover "
                        f"(fast={fast_now:.5f} slow={slow_now:.5f})")


# ─────────────────────────────────────────────
# STRATEGY: VWAP MEAN REVERSION
# ─────────────────────────────────────────────

def vwap_signal(df: pd.DataFrame, params: dict) -> tuple:
    """
    VWAP mean reversion.
    BUY  when price is below VWAP by threshold %
    SELL when price is above VWAP by threshold %
    Requires 'vwap' column in features — computed in Layer 1.
    """
    if "vwap" not in df.columns:
        return "flat", "vwap column not available"

    price     = df["close"].iloc[-1]
    vwap      = df["vwap"].iloc[-1]
    threshold = params.get("vwap_threshold", 0.002)

    if vwap == 0 or np.isnan(vwap):
        return "flat", "vwap unavailable"

    deviation = (price - vwap) / vwap

    if deviation < -threshold:
        return "buy",  (f"Price {deviation:.3%} below VWAP "
                        f"(threshold={threshold:.3%})")
    elif deviation > threshold:
        return "sell", (f"Price {deviation:.3%} above VWAP "
                        f"(threshold={threshold:.3%})")
    else:
        return "flat", f"Price within VWAP band ({deviation:.3%})"


# ─────────────────────────────────────────────
# POSITION SIZING
# ─────────────────────────────────────────────

def calculate_lot_size(symbol: str, sl_pips: float,
                       confidence: float, balance: float) -> float:
    """
    Risk-based sizing scaled by regime confidence.
    confidence=1.00 -> 1.0x base risk
    confidence=0.65 -> 0.5x base risk
    """
    pip_value = CONFIG["pip_value_per_lot"].get(symbol, 10.0)
    min_conf  = CONFIG["min_confidence"]

    conf_scalar = 0.5 + 0.5 * ((confidence - min_conf) / (1.0 - min_conf))
    conf_scalar = max(0.5, min(1.0, conf_scalar))

    risk_amount = min(
        balance * CONFIG["base_risk_pct"] * conf_scalar,
        balance * CONFIG["max_risk_pct"]
    )

    if sl_pips <= 0:
        sl_pips = CONFIG["default_sl_pips"]

    lot_size = round(risk_amount / (sl_pips * pip_value), 2)
    return max(lot_size, 0.01)


def calculate_sl_tp(df: pd.DataFrame, strategy: str,
                    symbol: str) -> tuple:
    """
    ATR-based SL and TP.
    Trend strategies (EMA): 2.0x ATR TP
    Mean-reversion (BB, RSI, VWAP): 1.0x ATR TP
    Minimum 1.5:1 reward:risk always enforced.
    """
    atr      = df["atr"].iloc[-1]
    pip_size = CONFIG["pip_size"].get(symbol, 0.0001)
    atr_pips = atr / pip_size

    sl_pips = max(round(atr_pips * CONFIG["sl_atr_mult"], 1), 5.0)

    trend_strategies = {"ema_crossover", "ema_flipped"}
    tp_mult = 2.0 if strategy in trend_strategies else 1.0
    tp_pips = round(atr_pips * tp_mult, 1)

    if tp_pips < sl_pips * CONFIG["min_rr"]:
        tp_pips = round(sl_pips * CONFIG["min_rr"], 1)

    tp_pips = max(tp_pips, 5.0)
    return sl_pips, tp_pips


# ─────────────────────────────────────────────
# MASTER SIGNAL GENERATOR
# ─────────────────────────────────────────────

def generate_signal(symbol: str, balance: float = None) -> TradeSignal:
    """
    Full signal pipeline for one symbol.

    Flow:
        1. Load per-symbol strategy config and params
        2. Get current regime from HMM
        3. Regime filter (confidence + switch probability)
        4. Look up strategy for this symbol + regime
        5. Run that strategy with symbol-specific params
        6. Calculate SL/TP and lot size
        7. Return TradeSignal
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if balance is None:
        balance = CONFIG["account_balance"]

    # ── 1. Symbol config ──────────────────────
    sym_cfg = SYMBOL_CONFIG.get(symbol)
    if sym_cfg is None:
        return TradeSignal(
            symbol=symbol, timestamp=timestamp,
            direction="flat", regime="unknown", confidence=0.0,
            strategy="no_config", lot_size=0.0,
            stop_loss_pips=0.0, take_profit_pips=0.0,
            reason=f"No config for {symbol} — add to SYMBOL_CONFIG"
        )

    params = sym_cfg.get("params", {})

    # ── 2. Regime ─────────────────────────────
    regime_info = get_current_regime(symbol)
    if regime_info is None:
        return TradeSignal(
            symbol=symbol, timestamp=timestamp,
            direction="flat", regime="unknown", confidence=0.0,
            strategy="none", lot_size=0.0,
            stop_loss_pips=0.0, take_profit_pips=0.0,
            reason="No model or features — run Layer 2 first"
        )

    regime     = regime_info["regime"]
    confidence = regime_info["confidence"]

    # ── 3. Regime filter ──────────────────────
    tradeable, filter_reason = regime_is_tradeable(regime_info)
    if not tradeable:
        return TradeSignal(
            symbol=symbol, timestamp=timestamp,
            direction="flat", regime=regime, confidence=confidence,
            strategy="none", lot_size=0.0,
            stop_loss_pips=0.0, take_profit_pips=0.0,
            reason=f"[FILTERED] {filter_reason}"
        )

    # ── 4. Strategy for this regime ───────────
    strategy_type = sym_cfg.get(regime, "disabled")

    if strategy_type == "disabled":
        return TradeSignal(
            symbol=symbol, timestamp=timestamp,
            direction="flat", regime=regime, confidence=confidence,
            strategy="disabled", lot_size=0.0,
            stop_loss_pips=0.0, take_profit_pips=0.0,
            reason=f"[DISABLED] No strategy for {symbol} {regime}"
        )

    # ── 5. Load features ──────────────────────
    df = load_features(symbol)
    if df is None or len(df) < 60:
        return TradeSignal(
            symbol=symbol, timestamp=timestamp,
            direction="flat", regime=regime, confidence=confidence,
            strategy="none", lot_size=0.0,
            stop_loss_pips=0.0, take_profit_pips=0.0,
            reason="Insufficient feature data"
        )

    # ── 6. Run strategy ───────────────────────
    if strategy_type == "bollinger":
        direction, strat_reason = bollinger_reversion_signal(df, params)
        strategy_name = "bollinger"

    elif strategy_type == "rsi":
        direction, strat_reason = rsi_signal(df, params)
        strategy_name = "rsi"

    elif strategy_type == "ema_flipped":
        direction, strat_reason = ema_crossover_signal(
            df, params, flipped=True)
        strategy_name = "ema_flipped"

    elif strategy_type == "ema":
        direction, strat_reason = ema_crossover_signal(
            df, params, flipped=False)
        strategy_name = "ema_crossover"

    elif strategy_type == "vwap":
        direction, strat_reason = vwap_signal(df, params)
        strategy_name = "vwap"

    else:
        return TradeSignal(
            symbol=symbol, timestamp=timestamp,
            direction="flat", regime=regime, confidence=confidence,
            strategy="unknown", lot_size=0.0,
            stop_loss_pips=0.0, take_profit_pips=0.0,
            reason=f"Unknown strategy: {strategy_type}"
        )

    # ── 7. No signal ──────────────────────────
    if direction == "flat":
        return TradeSignal(
            symbol=symbol, timestamp=timestamp,
            direction="flat", regime=regime, confidence=confidence,
            strategy=strategy_name, lot_size=0.0,
            stop_loss_pips=0.0, take_profit_pips=0.0,
            reason=strat_reason
        )

    # ── 8. Size the trade ─────────────────────
    sl_pips, tp_pips = calculate_sl_tp(df, strategy_name, symbol)
    lot_size         = calculate_lot_size(
        symbol, sl_pips, confidence, balance)

    reason = (
        f"[{regime.upper()}] {strat_reason} | "
        f"strategy={strategy_name} | "
        f"confidence={confidence:.2%} | "
        f"switch_prob={regime_info['switch_prob']:.2%}"
    )

    return TradeSignal(
        symbol=symbol, timestamp=timestamp,
        direction=direction, regime=regime, confidence=confidence,
        strategy=strategy_name, lot_size=lot_size,
        stop_loss_pips=sl_pips, take_profit_pips=tp_pips,
        reason=reason
    )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_signal_layer():
    print("=" * 60)
    print("  LAYER 3 — SIGNAL LAYER (v3)")
    print("=" * 60)

    print("\n  Per-symbol config:")
    for sym, cfg in SYMBOL_CONFIG.items():
        lv = cfg["low_vol"]
        hv = cfg["high_vol"]
        p  = cfg["params"]
        active = []
        if lv != "disabled":
            active.append(f"low_vol={lv}")
        if hv != "disabled":
            active.append(f"high_vol={hv}")
        print(f"    {sym:8} | "
              f"{' | '.join(active) or 'disabled':40} | {p}")

    print()
    for symbol in CONFIG["symbols"]:
        print(f"\n[{symbol}]")
        signal = generate_signal(symbol)
        print(f"  Direction  : {signal.direction.upper()}")
        print(f"  Regime     : {signal.regime} ({signal.confidence:.2%})")
        print(f"  Strategy   : {signal.strategy}")
        print(f"  Lot size   : {signal.lot_size}")
        print(f"  SL / TP    : {signal.stop_loss_pips} / "
              f"{signal.take_profit_pips} pips")
        print(f"  Reason     : {signal.reason}")

    print("\n[DONE] Layer 3 complete.")


if __name__ == "__main__":
    run_signal_layer()