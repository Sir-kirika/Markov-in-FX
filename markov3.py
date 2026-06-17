"""
=============================================================
LAYER 3 — SIGNAL LAYER (final)
Regime Filter -> Per-Symbol Strategy -> Position Sizing -> Trade Signal
=============================================================
Requirements:
    pip install pandas numpy joblib

VALIDATED CONFIGURATION (17-month walk-forward backtest):
    EURUSD high_vol : Bollinger BB_20_3.0
        Sharpe 2.10 | Drawdown -11.3% | 202 trades/17mo
    GBPUSD low_vol  : EMA flipped 10/50
        Sharpe 1.63 | Drawdown -14.9% | 544 trades/17mo
    Combined avg Sharpe 1.86 | Worst drawdown -14.9%

Strategy logic:
    EURUSD low_vol  -> disabled (no edge confirmed)
    EURUSD high_vol -> Bollinger Band mean reversion (bb_std=3.0)
    GBPUSD low_vol  -> Flipped EMA crossover (fade the breakout)
    GBPUSD high_vol -> disabled (RSI no edge at safe thresholds)
    USDJPY          -> disabled (no edge in either regime)
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
# CONFIGURATION
# ─────────────────────────────────────────────

CONFIG = {
    "symbols":   ["EURUSD", "GBPUSD"],
    "data_dir":  "data/",
    "model_dir": "models/",

    # ── Regime filter ─────────────────────────
    "min_confidence":  0.65,
    "max_switch_prob": 0.15,

    # ── Strategy parameters (validated) ───────
    "ema_fast":       10,
    "ema_slow":       50,
    "bb_period":      20,
    "bb_std":         3.0,     # wider bands — fires on genuine extremes only
    "rsi_period":     21,      # kept for future use
    "rsi_oversold":   20,
    "rsi_overbought": 80,

    # ── Position sizing ───────────────────────
    "base_risk_pct":   0.01,
    "max_risk_pct":    0.02,
    "account_balance": 10000,  # fallback — Layer 4 passes live balance
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

    # ── Per-symbol strategy ───────────────────
    # Based on 17-month walk-forward validation.
    # Options: "bollinger" | "ema" | "ema_flipped" | "rsi" | "disabled"
    #
    # Evidence:
    #   EURUSD high_vol BB_20_3.0  : Sharpe 2.10, DD -11.3%  -> bollinger
    #   EURUSD low_vol  EMA        : no consistent edge       -> disabled
    #   GBPUSD low_vol  EMA_flip   : Sharpe 1.63, DD -14.9%  -> ema_flipped
    #   GBPUSD high_vol RSI 20/80  : flat, no edge            -> disabled
    #   USDJPY all                 : losses in both regimes   -> disabled
    "symbol_strategy": {
        "EURUSD": {"low_vol": "disabled",    "high_vol": "bollinger"},
        "GBPUSD": {"low_vol": "ema_flipped", "high_vol": "disabled"},
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
    Classify the current regime using the saved HMM.
    Uses scaler.transform() — never refit here.
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

    X_raw         = df[features].dropna().values
    X             = scaler.transform(X_raw)
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
# STRATEGY: BOLLINGER BAND MEAN REVERSION
# EURUSD high_vol — validated Sharpe 2.10
# ─────────────────────────────────────────────

def bollinger_reversion_signal(df: pd.DataFrame) -> tuple:
    """
    Bollinger Band mean reversion with bb_std=3.0.
    Wide bands ensure only genuine statistical extremes trigger.

    BUY  when price closes below lower band — oversold extreme
    SELL when price closes above upper band — overbought extreme
    FLAT when price is inside bands

    Validated on EURUSD high_vol: Sharpe 2.10, DD -11.3%
    """
    period = CONFIG["bb_period"]
    n_std  = CONFIG["bb_std"]

    close  = df["close"]
    middle = close.rolling(period).mean()
    std    = close.rolling(period).std()
    upper  = middle + n_std * std
    lower  = middle - n_std * std

    price = close.iloc[-1]
    u     = upper.iloc[-1]
    l     = lower.iloc[-1]

    if price < l:
        return "buy",  f"Price {price:.5f} below lower BB {l:.5f}"
    elif price > u:
        return "sell", f"Price {price:.5f} above upper BB {u:.5f}"
    else:
        return "flat", f"Price inside bands [{l:.5f} - {u:.5f}]"


# ─────────────────────────────────────────────
# STRATEGY: EMA CROSSOVER (FLIPPED)
# GBPUSD low_vol — validated Sharpe 1.63
# ─────────────────────────────────────────────

def ema_flipped_signal(df: pd.DataFrame) -> tuple:
    """
    Inverted EMA crossover — fade the breakout.

    GBPUSD low_vol showed false breakout behaviour:
        Standard EMA (buy cross up): -$7,202 over 17 months
        Flipped EMA (sell cross up): +$9,697 over 17 months

    Logic:
        SELL when fast EMA crosses ABOVE slow EMA (fade the upside breakout)
        BUY  when fast EMA crosses BELOW slow EMA (fade the downside breakdown)
        FLAT when no crossover on this bar

    Only fires on the actual crossover bar — not on every bar
    where fast > slow, which would massively overtrade.

    Validated on GBPUSD low_vol: Sharpe 1.63, DD -14.9%
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

    if bullish_cross:
        return "sell", f"EMA{fast} crossed above EMA{slow} — fading upside breakout"
    elif bearish_cross:
        return "buy",  f"EMA{fast} crossed below EMA{slow} — fading downside breakdown"
    else:
        return "flat", f"No EMA crossover (fast={fast_now:.5f}, slow={slow_now:.5f})"


# ─────────────────────────────────────────────
# STRATEGY: RSI MEAN REVERSION
# Available but not active in current config.
# Re-enable in symbol_strategy if needed.
# ─────────────────────────────────────────────

def rsi_signal(df: pd.DataFrame) -> tuple:
    """
    RSI mean reversion.
    BUY  when RSI < rsi_oversold  (20) — momentum exhausted downside
    SELL when RSI > rsi_overbought (80) — momentum exhausted upside

    Tested on GBPUSD high_vol at 20/80 thresholds:
    401 trades, pnl=-$45 — effectively flat, disabled in current config.
    Re-enable here and in symbol_strategy if market conditions change.
    """
    if "rsi" not in df.columns:
        return "flat", "RSI not available in features"

    rsi = df["rsi"].iloc[-1]
    if rsi < CONFIG["rsi_oversold"]:
        return "buy",  f"RSI {rsi:.1f} below oversold threshold {CONFIG['rsi_oversold']}"
    elif rsi > CONFIG["rsi_overbought"]:
        return "sell", f"RSI {rsi:.1f} above overbought threshold {CONFIG['rsi_overbought']}"
    return "flat", f"RSI {rsi:.1f} within normal range"


# ─────────────────────────────────────────────
# POSITION SIZING
# ─────────────────────────────────────────────

def calculate_lot_size(symbol: str, sl_pips: float,
                       confidence: float, balance: float) -> float:
    """
    Risk-based position sizing scaled by regime confidence.

    confidence=1.00 -> 1.0x base risk (full size)
    confidence=0.65 -> 0.5x base risk (half size)
    Linear between min_confidence and 1.0.
    """
    pip_value   = CONFIG["pip_value_per_lot"].get(symbol, 10.0)
    min_conf    = CONFIG["min_confidence"]
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


def calculate_sl_tp(df: pd.DataFrame, direction: str,
                    regime: str, strategy: str) -> tuple:
    """
    ATR-based SL and TP.

    SL = 1.5 x ATR
    TP = 2.0 x ATR for trend strategies (ema, ema_flipped, atr_breakout)
    TP = 1.0 x ATR for mean-reversion strategies (bollinger, rsi, vwap)
    Minimum reward:risk = 1.5:1 enforced.
    """
    atr      = df["atr"].iloc[-1]
    pip_size = CONFIG["pip_size"].get(df.attrs.get("symbol", "EURUSD"), 0.0001)
    atr_pips = atr / pip_size

    sl_pips  = max(round(atr_pips * 1.5, 1), 5.0)

    trend_strategies = {"ema", "ema_flipped", "atr_breakout"}
    tp_mult  = 2.0 if strategy in trend_strategies else 1.0
    tp_pips  = round(atr_pips * tp_mult, 1)

    # Enforce minimum 1.5:1 reward:risk
    if tp_pips < sl_pips * 1.5:
        tp_pips = round(sl_pips * 1.5, 1)

    tp_pips = max(tp_pips, 5.0)
    return sl_pips, tp_pips


# ─────────────────────────────────────────────
# MASTER SIGNAL GENERATOR
# ─────────────────────────────────────────────

def generate_signal(symbol: str, balance: float = None) -> TradeSignal:
    """
    Full signal pipeline for one symbol.

    Flow:
        1. Check symbol has an active strategy configured
        2. Get current regime from saved HMM
        3. Regime filter (confidence + switch probability)
        4. Look up strategy for this symbol + regime
        5. Run that strategy
        6. Calculate SL/TP and lot size
        7. Return TradeSignal
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if balance is None:
        balance = CONFIG["account_balance"]

    # ── 1. Check symbol config ────────────────
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
            reason=f"[DISABLED] No edge confirmed in {regime} for {symbol}"
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

    # ── 6. Run strategy ───────────────────────
    if strategy_type == "bollinger":
        direction, strat_reason = bollinger_reversion_signal(df)
        strategy_name = "bollinger"

    elif strategy_type == "ema_flipped":
        direction, strat_reason = ema_flipped_signal(df)
        strategy_name = "ema_flipped"

    elif strategy_type == "ema":
        # Standard (non-flipped) EMA — available but not active in current config
        from layer3_signal_layer import ema_flipped_signal as _ema
        ema_fast = df["close"].ewm(span=CONFIG["ema_fast"], adjust=False).mean()
        ema_slow = df["close"].ewm(span=CONFIG["ema_slow"], adjust=False).mean()
        fast_now  = ema_fast.iloc[-1]; fast_prev = ema_fast.iloc[-2]
        slow_now  = ema_slow.iloc[-1]; slow_prev = ema_slow.iloc[-2]
        if (fast_prev <= slow_prev) and (fast_now > slow_now):
            direction, strat_reason = "buy", "EMA bullish crossover"
        elif (fast_prev >= slow_prev) and (fast_now < slow_now):
            direction, strat_reason = "sell", "EMA bearish crossover"
        else:
            direction, strat_reason = "flat", "No EMA crossover"
        strategy_name = "ema"

    elif strategy_type == "rsi":
        direction, strat_reason = rsi_signal(df)
        strategy_name = "rsi"

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
    sl_pips, tp_pips = calculate_sl_tp(df, direction, regime, strategy_name)
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
    print("  Active strategies:")
    for sym, regimes in CONFIG["symbol_strategy"].items():
        for regime, strat in regimes.items():
            if strat != "disabled":
                print(f"    {sym} {regime:8} -> {strat}")

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