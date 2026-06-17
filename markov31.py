"""
=============================================================
LAYER 3 — SIGNAL LAYER (v2)
Regime Filter → Per-Symbol Strategy → Position Sizing → Trade Signal
=============================================================
Requirements:
    pip install pandas numpy joblib

Changes from v1:
    - Per-symbol strategy config based on walk-forward backtest evidence
    - EURUSD: Bollinger reversion in high_vol only
    - GBPUSD: Flipped EMA in low_vol + Bollinger in high_vol
    - USDJPY: Disabled — no consistent edge found in either regime
    - Convergence warnings suppressed

Reads from:
    data/{symbol}_features.csv   (Layer 1)
    models/{symbol}_hmm.pkl      (Layer 2)
=============================================================
"""

import numpy as np
import pandas as pd
import joblib
import os
import warnings
from datetime import datetime
from dataclasses import dataclass

warnings.filterwarnings("ignore", message="Model is not converging")


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

CONFIG = {
    "symbols":   ["EURUSD", "GBPUSD", "USDJPY"],
    "data_dir":  "data/",
    "model_dir": "models/",

    # ── Regime filter thresholds ──────────────
    "min_confidence":  0.65,
    "max_switch_prob": 0.15,

    # ── EMA settings ──────────────────────────
    "ema_fast": 10,
    "ema_slow": 50,

    # ── Bollinger Band settings ───────────────
    "bb_period": 20,
    "bb_std":    2.0,

    # ── Position sizing ───────────────────────
    "base_risk_pct":   0.01,
    "max_risk_pct":    0.02,
    "account_balance": 10000,
    "default_sl_pips": 20,

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

    # ── Per-symbol strategy selection ─────────
    # Based on 17-month walk-forward backtest results
    #
    # Options per regime slot:
    #   "bollinger"    — Bollinger Band mean reversion
    #   "ema"          — standard EMA crossover (buy cross up, sell cross down)
    #   "ema_flipped"  — inverted EMA (sell cross up, buy cross down)
    #   "disabled"     — no trading in this regime
    #
    # Evidence summary:
    #   EURUSD low_vol  EMA: -$3,753  → disabled
    #   EURUSD high_vol BB:  +$3,891  → bollinger
    #   GBPUSD low_vol  EMA: -$7,202  → flipped to +$9,697 → ema_flipped
    #   GBPUSD high_vol BB:  +$5,511  → bollinger
    #   USDJPY low_vol  EMA: -$791    → disabled
    #   USDJPY high_vol BB:  -$2,827  → disabled (no edge)
    "symbol_strategy": {
        "EURUSD": {"low_vol": "disabled",    "high_vol": "bollinger"},
        "GBPUSD": {"low_vol": "ema_flipped", "high_vol": "bollinger"},
        "USDJPY": {"low_vol": "disabled",    "high_vol": "disabled"},
    },
}


# ─────────────────────────────────────────────
# TRADE SIGNAL
# ─────────────────────────────────────────────

@dataclass
class TradeSignal:
    """
    Everything Layer 4 needs to execute a trade.
    direction = 'buy', 'sell', or 'flat'.
    """
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
        print(f"[WARN] No features for {symbol}")
        return None
    return pd.read_csv(path, index_col="time", parse_dates=True)


def load_model_bundle(symbol: str) -> dict | None:
    path = os.path.join(CONFIG["model_dir"], f"{symbol}_hmm.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


# ─────────────────────────────────────────────
# REGIME FILTER
# ─────────────────────────────────────────────

def get_current_regime(symbol: str) -> dict | None:
    """
    Classify the current regime for a symbol using the saved HMM.
    Returns None if model or features are unavailable.
    """
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
    """
    Gate check — both conditions must pass before any strategy runs.
    Returns (tradeable: bool, reason: str).
    """
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
# STRATEGY: EMA CROSSOVER
# ─────────────────────────────────────────────

def ema_crossover_signal(df: pd.DataFrame,
                         flipped: bool = False) -> tuple:
    """
    EMA crossover signal.

    Standard (flipped=False):
        BUY  when fast crosses above slow — trend following
        SELL when fast crosses below slow

    Flipped (flipped=True):
        SELL when fast crosses above slow — fade the breakout
        BUY  when fast crosses below slow

    GBPUSD uses flipped=True based on backtest evidence:
        Standard EMA lost -$7,202 over 17 months
        Flipped EMA gained +$9,697 over the same period
        Indicates GBP low_vol regime has false breakout behaviour
    """
    fast = CONFIG["ema_fast"]
    slow = CONFIG["ema_slow"]

    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()

    fast_now  = ema_fast.iloc[-1]
    fast_prev = ema_fast.iloc[-2]
    slow_now  = ema_slow.iloc[-1]
    slow_prev = ema_slow.iloc[-2]

    bullish_cross = (fast_prev <= slow_prev) and (fast_now > slow_now)
    bearish_cross = (fast_prev >= slow_prev) and (fast_now < slow_now)

    strategy_name = "ema_flipped" if flipped else "ema_crossover"

    if bullish_cross:
        direction = "sell" if flipped else "buy"
        return direction, f"{strategy_name}: crossover signal"
    elif bearish_cross:
        direction = "buy" if flipped else "sell"
        return direction, f"{strategy_name}: crossover signal"
    else:
        return "flat", f"{strategy_name}: no crossover"


# ─────────────────────────────────────────────
# STRATEGY: BOLLINGER BAND MEAN REVERSION
# ─────────────────────────────────────────────

def bollinger_reversion_signal(df: pd.DataFrame) -> tuple:
    """
    Bollinger Band mean-reversion signal.
    BUY  when price closes below lower band — oversold
    SELL when price closes above upper band — overbought
    FLAT when price is inside the bands
    """
    period = CONFIG["bb_period"]
    n_std  = CONFIG["bb_std"]

    close  = df["close"]
    middle = close.rolling(period).mean()
    std    = close.rolling(period).std()
    upper  = middle + (n_std * std)
    lower  = middle - (n_std * std)

    price_now = close.iloc[-1]
    upper_now = upper.iloc[-1]
    lower_now = lower.iloc[-1]

    if price_now < lower_now:
        return "buy",  (f"Price {price_now:.5f} below lower BB {lower_now:.5f}")
    elif price_now > upper_now:
        return "sell", (f"Price {price_now:.5f} above upper BB {upper_now:.5f}")
    else:
        return "flat", (f"Price inside bands [{lower_now:.5f} - {upper_now:.5f}]")


# ─────────────────────────────────────────────
# POSITION SIZING
# ─────────────────────────────────────────────

def calculate_lot_size(symbol: str, sl_pips: float,
                       confidence: float, balance: float) -> float:
    """
    Risk-based position sizing scaled by regime confidence.
    Higher confidence → larger position, up to max_risk_pct.
    """
    pip_value = CONFIG["pip_value_per_lot"].get(symbol, 10.0)
    min_conf  = CONFIG["min_confidence"]

    # Linear scale: 0.5x at min_confidence, 1.0x at full confidence
    conf_scalar = 0.5 + 0.5 * ((confidence - min_conf) / (1.0 - min_conf))
    conf_scalar = max(0.5, min(1.0, conf_scalar))

    risk_amount = balance * CONFIG["base_risk_pct"] * conf_scalar
    max_risk    = balance * CONFIG["max_risk_pct"]
    risk_amount = min(risk_amount, max_risk)

    if sl_pips <= 0:
        sl_pips = CONFIG["default_sl_pips"]

    lot_size = round(risk_amount / (sl_pips * pip_value), 2)
    return max(lot_size, 0.01)


def calculate_sl_tp(df: pd.DataFrame, direction: str,
                    regime: str) -> tuple:
    """
    ATR-based SL and TP with minimum reward:risk enforcement.
    Trend trades: 2:1 TP ratio
    Mean-reversion trades: 1.5:1 TP ratio (tighter)
    """
    atr      = df["atr"].iloc[-1]
    pip_size = CONFIG["pip_size"].get(df.attrs.get("symbol", "EURUSD"), 0.0001)
    atr_pips = atr / pip_size

    sl_pips = round(atr_pips * 1.5, 1)
    tp_pips = round(atr_pips * 2.0 if regime == "low_vol" else atr_pips * 1.0, 1)

    # Enforce minimum reward:risk of 1.5:1
    if tp_pips < sl_pips * 1.5:
        tp_pips = round(sl_pips * 1.5, 1)

    sl_pips = max(sl_pips, 5.0)
    tp_pips = max(tp_pips, 5.0)

    return sl_pips, tp_pips


# ─────────────────────────────────────────────
# MASTER SIGNAL GENERATOR
# ─────────────────────────────────────────────

def generate_signal(symbol: str, balance: float = None) -> TradeSignal:
    """
    Full signal pipeline for one symbol.
    Applies per-symbol strategy config from backtest evidence.

    Flow:
        1. Check if symbol has any active strategies
        2. Get current regime from HMM
        3. Regime filter (confidence + switch probability)
        4. Look up which strategy is configured for this symbol + regime
        5. Run that strategy (or return flat if disabled)
        6. Calculate SL/TP and lot size
        7. Return TradeSignal
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if balance is None:
        balance = CONFIG["account_balance"]

    # ── 1. Check symbol has strategy config ──
    symbol_cfg = CONFIG["symbol_strategy"].get(symbol)
    if symbol_cfg is None:
        return TradeSignal(
            symbol=symbol, timestamp=timestamp,
            direction="flat", regime="unknown", confidence=0.0,
            strategy="no_config", lot_size=0.0,
            stop_loss_pips=0.0, take_profit_pips=0.0,
            reason=f"No strategy configured for {symbol}"
        )

    # ── 2. Get current regime ─────────────────
    regime_info = get_current_regime(symbol)
    if regime_info is None:
        return TradeSignal(
            symbol=symbol, timestamp=timestamp,
            direction="flat", regime="unknown", confidence=0.0,
            strategy="none", lot_size=0.0,
            stop_loss_pips=0.0, take_profit_pips=0.0,
            reason="No model or features available"
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

    # ── 4. Look up strategy for this regime ──
    strategy_type = symbol_cfg.get(regime, "disabled")

    if strategy_type == "disabled":
        return TradeSignal(
            symbol=symbol, timestamp=timestamp,
            direction="flat", regime=regime, confidence=confidence,
            strategy="disabled", lot_size=0.0,
            stop_loss_pips=0.0, take_profit_pips=0.0,
            reason=f"[DISABLED] No edge in {regime} regime for {symbol}"
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

    df.attrs["symbol"] = symbol

    # ── 6. Run the configured strategy ────────
    if strategy_type == "bollinger":
        direction, strat_reason = bollinger_reversion_signal(df)
        strategy_name = "bollinger_reversion"

    elif strategy_type == "ema":
        direction, strat_reason = ema_crossover_signal(df, flipped=False)
        strategy_name = "ema_crossover"

    elif strategy_type == "ema_flipped":
        direction, strat_reason = ema_crossover_signal(df, flipped=True)
        strategy_name = "ema_flipped"

    else:
        return TradeSignal(
            symbol=symbol, timestamp=timestamp,
            direction="flat", regime=regime, confidence=confidence,
            strategy="unknown", lot_size=0.0,
            stop_loss_pips=0.0, take_profit_pips=0.0,
            reason=f"Unknown strategy type: {strategy_type}"
        )

    # ── 7. No signal this bar ─────────────────
    if direction == "flat":
        return TradeSignal(
            symbol=symbol, timestamp=timestamp,
            direction="flat", regime=regime, confidence=confidence,
            strategy=strategy_name, lot_size=0.0,
            stop_loss_pips=0.0, take_profit_pips=0.0,
            reason=strat_reason
        )

    # ── 8. Size the trade ─────────────────────
    sl_pips, tp_pips = calculate_sl_tp(df, direction, regime)
    lot_size         = calculate_lot_size(symbol, sl_pips, confidence, balance)

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
    print("  LAYER 3 — SIGNAL LAYER")
    print("=" * 60)

    for symbol in CONFIG["symbols"]:
        print(f"\n[{symbol}]")
        signal = generate_signal(symbol)
        print(f"  Direction  : {signal.direction.upper()}")
        print(f"  Regime     : {signal.regime} ({signal.confidence:.2%})")
        print(f"  Strategy   : {signal.strategy}")
        print(f"  Lot size   : {signal.lot_size}")
        print(f"  SL / TP    : {signal.stop_loss_pips} / {signal.take_profit_pips} pips")
        print(f"  Reason     : {signal.reason}")

    print("\n[DONE] Layer 3 complete.")


if __name__ == "__main__":
    run_signal_layer()