"""
=============================================================
LAYER 1 — DATA PIPELINE
MT5 Connection -> Raw Data -> Feature Engineering -> Feature Store
=============================================================
Requirements:
    pip install MetaTrader5 pandas numpy

Symbols: EURUSD, GBPUSD, EURGBP, EURCAD, GBPCAD, AUDUSD, USDCAD
All validated over 8 years of walk-forward backtesting.

Run this AFTER MT5 terminal is open and logged in.
Stop with Ctrl+C.
=============================================================
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

CONFIG = {
    "symbols": [
        "EURUSD",
        "GBPUSD",
        "EURGBP",
        "EURCAD",
        "GBPCAD",
        "AUDUSD",
        "USDCAD",
    ],
    "timeframe":  mt5.TIMEFRAME_M15,
    "lookback":   500,
    "loop_sleep": 60,
    "data_dir":   "data/",
}

TF_NAMES = {
    mt5.TIMEFRAME_M1:  "M1",
    mt5.TIMEFRAME_M5:  "M5",
    mt5.TIMEFRAME_M15: "M15",
    mt5.TIMEFRAME_M30: "M30",
    mt5.TIMEFRAME_H1:  "H1",
    mt5.TIMEFRAME_H4:  "H4",
    mt5.TIMEFRAME_D1:  "D1",
}


# ─────────────────────────────────────────────
# MT5 CONNECTION
# ─────────────────────────────────────────────

def connect_mt5() -> bool:
    terminal_path = r"C:\Program Files\EGM Securities MetaTrader 5 Terminal\terminal64.exe"
    if not mt5.initialize(path=terminal_path):
        print(f"[ERROR] MT5 initialize() failed: {mt5.last_error()}")
        return False

    info    = mt5.terminal_info()
    account = mt5.account_info()
    print(f"[OK] Connected to MT5 terminal")
    print(f"     Broker  : {info.company}")
    print(f"     Build   : {info.build}")
    if account:
        print(f"[OK] Account : {account.login} | {account.server} | "
              f"Balance: {account.balance} {account.currency}")
    return True


def disconnect_mt5():
    mt5.shutdown()
    print("[INFO] MT5 connection closed.")


# ─────────────────────────────────────────────
# MARKET OPEN CHECK
# ─────────────────────────────────────────────

def is_market_open(symbol: str) -> bool:
    """
    Check if the broker is accepting orders for this symbol.
    trade_mode > 0 means orders are accepted.
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        return False
    return info.trade_mode > 0


# ─────────────────────────────────────────────
# RAW DATA FETCH
# ─────────────────────────────────────────────

def fetch_ohlcv(symbol: str, timeframe: int,
                n_bars: int) -> pd.DataFrame | None:
    """Pull the last n_bars OHLCV candles for a symbol from MT5."""
    if not mt5.symbol_select(symbol, True):
        print(f"[WARN] Could not select symbol {symbol}")
        return None

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)

    if rates is None or len(rates) == 0:
        print(f"[WARN] No data for {symbol}: {mt5.last_error()}")
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df.drop(columns=["real_volume"], errors="ignore", inplace=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    return df


def fetch_spread(symbol: str) -> float | None:
    """Get the current spread for a symbol in points."""
    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if tick is None or info is None:
        return None
    spread = (tick.ask - tick.bid) / info.point
    return round(spread, 2)


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# Must stay identical to Layer 5 — any difference
# between training and live inference is a bug.
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Compute HMM regime detection features.
    All features are stationary and normalised so they work
    consistently across different currency pairs.

    Features:
        log_return     : bar-level log return
        realized_vol   : rolling 20-bar annualised volatility
        vol_ratio      : 5-bar vol / 20-bar vol (regime shift signal)
        atr            : Average True Range (14-bar)
        atr_pct        : ATR / close (normalised)
        bar_range      : (high - low) / close
        volume_zscore  : z-score of tick volume
        ema_diff       : (EMA10 - EMA50) / close (informational only)
    """
    feat = df.copy()

    feat["log_return"] = np.log(feat["close"] / feat["close"].shift(1))

    tf_key        = TF_NAMES.get(CONFIG["timeframe"], "M15")
    bars_per_year = {
        "M1": 96768, "M5": 24192, "M15": 8064,
        "M30": 4032, "H1": 2016,  "H4": 504, "D1": 252
    }
    annualise = np.sqrt(bars_per_year.get(tf_key, 8064))

    feat["realized_vol"] = feat["log_return"].rolling(20).std() * annualise
    feat["vol_ratio"]    = (
        feat["log_return"].rolling(5).std() /
        feat["log_return"].rolling(20).std().replace(0, np.nan)
    )

    high_low   = feat["high"] - feat["low"]
    high_close = (feat["high"] - feat["close"].shift(1)).abs()
    low_close  = (feat["low"]  - feat["close"].shift(1)).abs()
    true_range = pd.concat(
        [high_low, high_close, low_close], axis=1
    ).max(axis=1)
    feat["atr"]     = true_range.rolling(14).mean()
    feat["atr_pct"] = feat["atr"] / feat["close"]

    feat["bar_range"] = (feat["high"] - feat["low"]) / feat["close"]

    vol_mean = feat["volume"].rolling(20).mean()
    vol_std  = feat["volume"].rolling(20).std().replace(0, np.nan)
    feat["volume_zscore"] = (feat["volume"] - vol_mean) / vol_std

    # Informational only — not used by HMM
    ema_fast = feat["close"].ewm(span=10, adjust=False).mean()
    ema_slow = feat["close"].ewm(span=50, adjust=False).mean()
    feat["ema_diff"] = (ema_fast - ema_slow) / feat["close"]

    feat.dropna(inplace=True)
    return feat


# ─────────────────────────────────────────────
# FEATURE STORE
# ─────────────────────────────────────────────

def save_features(df: pd.DataFrame, symbol: str):
    os.makedirs(CONFIG["data_dir"], exist_ok=True)
    path = os.path.join(CONFIG["data_dir"], f"{symbol}_features.csv")
    df.to_csv(path)
    print(f"  [SAVED] {symbol} -> {path} ({len(df)} rows)")


def load_features(symbol: str) -> pd.DataFrame | None:
    path = os.path.join(CONFIG["data_dir"], f"{symbol}_features.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, index_col="time", parse_dates=True)


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

def run_data_pipeline():
    """
    Main loop — connects to MT5, fetches OHLCV for all symbols,
    engineers features, and saves to the feature store every cycle.
    Layer 2 reads from the feature store.
    Layer 4 triggers Layer 2 refit on schedule.
    Stop with Ctrl+C.
    """
    print("=" * 60)
    print("  LAYER 1 — DATA PIPELINE STARTING")
    print(f"  Symbols: {', '.join(CONFIG['symbols'])}")
    print("=" * 60)

    if not connect_mt5():
        print("[FATAL] Could not connect to MT5.")
        return

    try:
        cycle = 0
        while True:
            cycle    += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[CYCLE {cycle}] {timestamp}")
            print("-" * 40)

            for symbol in CONFIG["symbols"]:
                if not is_market_open(symbol):
                    print(f"  {symbol:8} | market closed — skipping")
                    continue

                raw = fetch_ohlcv(
                    symbol, CONFIG["timeframe"], CONFIG["lookback"]
                )
                if raw is None:
                    continue

                spread   = fetch_spread(symbol)
                features = engineer_features(raw, symbol)

                if spread is not None:
                    features["current_spread"] = spread

                save_features(features, symbol)

                latest = features.iloc[-1]
                print(f"  {symbol:8} | bars:{len(raw):4} | "
                      f"spread:{spread:5.1f}pts | "
                      f"vol={latest['realized_vol']:.4f} | "
                      f"atr={latest['atr_pct']:.5f} | "
                      f"vol_ratio={latest['vol_ratio']:.3f}")

            print(f"[SLEEP] Next pull in {CONFIG['loop_sleep']}s...")
            time.sleep(CONFIG["loop_sleep"])

    except KeyboardInterrupt:
        print("\n[STOPPED] Pipeline interrupted by user.")

    finally:
        disconnect_mt5()


if __name__ == "__main__":
    run_data_pipeline()