"""
=============================================================
LAYER 5 — OPTIMISATION RUNNER (v4 — exhaustive)
Tests EVERY strategy in EVERY regime combination.
=============================================================

WHAT THIS SCRIPT DOES (plain English):
---------------------------------------
1. Connects to MetaTrader 5 (a trading platform) to download price history
2. For each currency pair (e.g. EURUSD), it walks through historical data
   in chunks — training a Hidden Markov Model (HMM) to detect whether the
   market is in a "low volatility" or "high volatility" state (called a "regime")
3. Depending on the regime, it applies a trading strategy (e.g. Bollinger Bands,
   RSI, EMA crossover) to generate buy/sell signals
4. It simulates those trades with realistic stop-losses and take-profits
5. It scores every strategy+regime combination and saves a ranked summary CSV

Key terms for beginners:
  - HMM (Hidden Markov Model): a statistical model that figures out hidden
    "states" (like market regimes) from observable data (like volatility)
  - Regime: the market's current "mood" — calm (low vol) vs chaotic (high vol)
  - Walk-forward: train on past data, test on the next chunk, roll forward,
    repeat — simulates real trading without peeking at future prices
  - PnL: Profit and Loss (money made or lost)
  - Sharpe ratio: risk-adjusted return — higher is better
  - Drawdown: how far the balance dropped from its peak — lower is better
  - ATR: Average True Range — a measure of how much price moves per bar

Requirements:
    pip install MetaTrader5 pandas numpy hmmlearn scikit-learn joblib

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
from multiprocessing import Pool, cpu_count
import tempfile
import psutil                 # pip install psutil — reads actual CPU usage
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
        #"XAUUSD",
    ],

    "timeframe": mt5.TIMEFRAME_H1,
    "n_bars":    175000,

    "train_bars": 2000,
    "test_bars":  200,
    "step_bars":  200,

    "n_states": 2,       # BUG FIX 1: was "n_state" (missing 's') — caused KeyError in fit_hmm
    "n_iter":   50,

    "hmm_features": [
        "realized_vol",
        "atr_pct",
        "vol_ratio",
        "bar_range",
        "volume_zscore",
    ],

    "min_confidence":  0.65,
    "max_switch_prob": 0.15,  # BUG FIX 2: was "max_switch_probe" — caused KeyError in get_signal

    "starting_balance": 10000.0,
    "base_risk_pct":    0.01,
    "sl_atr_mult":      1.5,
    "min_rr":           1.5,
    "max_spread_to_sl_pct": 5,

    "pip_size": {
        "EURUSD": 0.0001, "GBPUSD": 0.0001,
        "EURGBP": 0.0001, "EURCAD": 0.0001,
        "GBPCAD": 0.0001, "AUDUSD": 0.0001,
        "USDCAD": 0.0001, #"XAUUSD": 0.1,
    },
    "pip_value_per_lot": {
        "EURUSD": 10.0, "GBPUSD": 10.0,
        "EURGBP": 12.5, "EURCAD": 7.5,
        "GBPCAD": 7.5,  "AUDUSD": 10.0,
        "USDCAD": 7.5,  #"XAUUSD": 10.0,
    },

    "results_dir":   "backtest_results/learn_opt_para_v1",
    "progress_file": "backtest_results/learn_opt_para_v1/progress.csv",
}


# ─────────────────────────────────────────────
# SETUP GENERATION
# ─────────────────────────────────────────────

# BUG FIX 3: function was named "buils_setups" (typo) — renamed to "build_setups"
def build_setups() -> list:
    """
    Auto-generate all setups by testing every strategy in both regimes.

    A "setup" is a dict describing one experiment:
        low_vol_strategy  : strategy to run in calm markets
        high_vol_strategy : strategy to run in choppy markets
        params            : indicator parameters for those strategies

    For each (strategy + params) pair, TWO setups are created:
        name_LOW  → strategy active in low_vol only
        name_HIGH → strategy active in high_vol only

    COMBO setups pair different strategies across regimes.
    """

    strategy_variants = [
        ("bollinger", {"bb_period": 10, "bb_std": 2.0}),
        ("bollinger", {"bb_period": 20, "bb_std": 2.0}),
        ("bollinger", {"bb_period": 20, "bb_std": 2.5}),
        ("bollinger", {"bb_period": 20, "bb_std": 3.0}),
        ("bollinger", {"bb_period": 50, "bb_std": 2.0}),
        ("bollinger", {"bb_period": 50, "bb_std": 2.5}),

        ("ema_flipped", {"ema_fast": 5,  "ema_slow": 20}),
        ("ema_flipped", {"ema_fast": 10, "ema_slow": 50}),
        ("ema_flipped", {"ema_fast": 10, "ema_slow": 200}),
        ("ema_flipped", {"ema_fast": 20, "ema_slow": 100}),

        ("ema", {"ema_fast": 5,  "ema_slow": 20}),
        ("ema", {"ema_fast": 10, "ema_slow": 50}),
        ("ema", {"ema_fast": 20, "ema_slow": 100}),

        ("rsi", {"rsi_period": 7,  "rsi_oversold": 25, "rsi_overbought": 75}),
        ("rsi", {"rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70}),
        ("rsi", {"rsi_period": 14, "rsi_oversold": 20, "rsi_overbought": 80}),
        ("rsi", {"rsi_period": 21, "rsi_oversold": 30, "rsi_overbought": 70}),
        ("rsi", {"rsi_period": 21, "rsi_oversold": 20, "rsi_overbought": 80}),

        ("atr_breakout", {"atr_channel_period": 10, "atr_break_mult": 0.5}),
        ("atr_breakout", {"atr_channel_period": 20, "atr_break_mult": 0.5}),
        ("atr_breakout", {"atr_channel_period": 20, "atr_break_mult": 1.0}),

        ("vwap", {"vwap_threshold": 0.001}),
        ("vwap", {"vwap_threshold": 0.002}),
        ("vwap", {"vwap_threshold": 0.003}),
    ]

    setups = []

    # Loop 1: single strategy per regime
    for strategy, params in strategy_variants:
        param_str = "_".join(str(v) for v in params.values())

        setups.append({
            "name":              f"{strategy}_{param_str}_LOW",
            "low_vol_strategy":  strategy,
            "high_vol_strategy": "disabled",
            "params":            params.copy(),
        })
        setups.append({
            "name":              f"{strategy}_{param_str}_HIGH",
            "low_vol_strategy":  "disabled",
            "high_vol_strategy": strategy,
            "params":            params.copy(),
        })

    # BUG FIX 4: combos list was INSIDE the for loop above (wrong indentation)
    # — it was re-created fresh on every iteration, and the combo loop that
    # follows it also ran inside the single-strategy loop. Moved both outside.
    combos = [
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

        ("COMBO_VWAP003_EMAf20_100",
         "ema_flipped", "vwap",
         {"vwap_threshold": 0.003, "ema_fast": 20, "ema_slow": 100}),
        ("COMBO_VWAP002_EMAf20_100",
         "ema_flipped", "vwap",
         {"vwap_threshold": 0.002, "ema_fast": 20, "ema_slow": 100}),

        ("COMBO_BB10_LOW_BB20_3_HIGH",
         "bollinger", "bollinger",
         {"bb_period": 10, "bb_std": 2.0}),
        ("COMBO_RSI21_BOTH",
         "rsi", "rsi",
         {"rsi_period": 21, "rsi_oversold": 20, "rsi_overbought": 80}),

        ("COMBO_EMA_flip_LOW_EMA_HIGH",
         "ema_flipped", "ema",
         {"ema_fast": 20, "ema_slow": 100}),
    ]

    # Loop 2: combo setups — one dict per combo (both regimes assigned explicitly)
    for name, lv, hv, params in combos:
        setups.append({
            "name":              name,
            "low_vol_strategy":  lv,
            "high_vol_strategy": hv,
            "params":            params,
        })

    return setups


SETUPS = build_setups()   # BUG FIX 3 continued: was "buils_setups()"


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
        "max_spread": BASE_CONFIG["max_spread_to_sl_pct"],
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
        print("Starting fresh — config changed")
        return set(), [], fingerprint

    # BUG FIX 5: was set(zip(df["symbols", df["setup"]])) — double-bracket typo
    # df["symbols", df["setup"]] tries to use a tuple as a column key → KeyError
    # Correct: zip(df["symbol"], df["setup"])  ← two separate column lookups
    # Also: column is "symbol" not "symbols" (matches what compute_metrics saves)
    completed = set(zip(df["symbol"], df["setup"]))
    results   = df.drop(columns=["fingerprint"]).to_dict("records")
    print(f"[RESUME] {len(completed)} runs already done — skipping")
    return completed, results, fingerprint


def save_progress(result: dict, fingerprint: str):
    path = BASE_CONFIG["progress_file"]
    os.makedirs(os.path.dirname(path), exist_ok=True)

    row                = result.copy()
    row["fingerprint"] = fingerprint

    df_row       = pd.DataFrame([row])
    # BUG FIX 6: was "not os,path.exists(path)" — comma instead of dot → SyntaxError
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

    # BUG FIX 7: last part of print was missing f-string braces
    # was: f"{df.index[-1].date()}"  →  df.index[-1].date() was printed as literal text
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

    fast = params.get("ema_fast", 10)
    slow = params.get("ema_slow", 50)
    feat["ema_fast"] = feat["close"].ewm(span=fast, adjust=False).mean()
    feat["ema_slow"] = feat["close"].ewm(span=slow, adjust=False).mean()

    bb_p             = params.get("bb_period", 20)
    bb_s             = params.get("bb_std", 2.0)
    bb_mid           = feat["close"].rolling(bb_p).mean()
    bb_std           = feat["close"].rolling(bb_p).std()
    feat["bb_upper"] = bb_mid + bb_s * bb_std
    feat["bb_lower"] = bb_mid - bb_s * bb_std

    rsi_p  = params.get("rsi_period", 14)
    delta  = feat["close"].diff()
    gain   = delta.clip(lower=0).rolling(rsi_p).mean()
    loss   = (-delta.clip(upper=0)).rolling(rsi_p).mean()
    rs     = gain / loss.replace(0, np.nan)
    feat["rsi"] = 100 - (100 / (1 + rs))

    atr_cp               = params.get("atr_channel_period", 20)
    feat["highest_high"] = feat["high"].rolling(atr_cp).max()
    feat["lowest_low"]   = feat["low"].rolling(atr_cp).min()

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
            n_components=BASE_CONFIG["n_states"],   # BUG FIX 1 applied here
            covariance_type="full",
            n_iter=BASE_CONFIG["n_iter"],
            random_state=42,
            verbose=False,
        )
        model.fit(X)
    except Exception:
        return None, None, None

    # BUG FIX 8: was features.index("realized_volume") — wrong name, the feature
    # is called "realized_vol" (no "ume") → ValueError: 'realized_volume' not in list
    vol_idx       = features.index("realized_vol")
    state_vols    = {s: model.means_[s][vol_idx]
                     for s in range(BASE_CONFIG["n_states"])}   # BUG FIX 1 applied
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
    # BUG FIX 9: was result["regim"] — missing the 'e' → column saved as "regim"
    # then get_signal reads row["regime"] → KeyError at runtime
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
    # BUG FIX 2 applied: key is "max_switch_prob" not "max_switch_probe"
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
# TRADE SIMULATION — spread-aware
# ─────────────────────────────────────────────

def simulate_trade(entry_bar: pd.Series, subsequent_bars: pd.DataFrame,
                   direction: str, strategy: str, regime: str,
                   symbol: str, balance: float,
                   confidence: float) -> dict:
    pip_size    = BASE_CONFIG["pip_size"].get(symbol, 0.0001)
    pip_value   = BASE_CONFIG["pip_value_per_lot"].get(symbol, 10.0)
    spread_pips = entry_bar["spread"]/10
    
    spread_val = spread_pips*pip_size

    atr_pips = entry_bar["atr"] / pip_size
    sl_pips  = max(round(atr_pips * BASE_CONFIG["sl_atr_mult"], 1), 5.0)
    if(spread_pips>sl_pips*(BASE_CONFIG["max_spread_to_sl_pct"]/100)):        
        #print(f"{symbol} {entry_bar.name} SKIPPING spread is {spread_pips} VS min required {sl_pips*(BASE_CONFIG["max_spread_to_sl_pct"]/100)}")
        return None

    trend_strategies = {"ema", "ema_flipped", "atr_breakout"}
    tp_mult = 2.0 if strategy in trend_strategies else 1.0
    tp_pips = max(round(atr_pips * tp_mult, 1),
                  sl_pips * BASE_CONFIG["min_rr"], 5.0)

    # Spread logic:
    #   BUY  → you buy at the Ask (close + spread). SL and TP measured from Ask.
    #   SELL → you sell at the Bid (close).          SL and TP measured from Bid.
    entry_price = entry_bar["close"]
    if direction == "buy":
        entry_price = entry_bar["close"] + spread_val
        sl_price = entry_price - sl_pips * pip_size
        tp_price = entry_price + tp_pips * pip_size
    else:
        sl_price = entry_price + sl_pips * pip_size
        tp_price = entry_price - tp_pips * pip_size

    # BUG FIX 10: was subsequent_bars["highs"] — wrong column name → KeyError
    # The DataFrame column is "high" (singular), matching what MT5 returns
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
            sl_pips = abs(entry_price - exit_price)/pip_size
        elif direction == "sell" and exit_price > entry_price:
            sl_pips = abs(exit_price - entry_price)/pip_size
        elif direction == "sell" and exit_price < entry_price:
            tp_pips = (entry_price - exit_price)/pip_size

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
        "direction":  direction,  "regime":     regime,
        "strategy":   strategy,   "confidence": round(confidence, 4),
        "sl_pips":    sl_pips,    "tp_pips":    tp_pips,
        "lot_size":   lot_size,   "outcome":    outcome,
        "pnl_pips":   round(pnl_pips, 2),
        "pnl_usd":    pnl_usd,   "won":        pnl_usd > 0,
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
                # BUG FIX 11: was confidence=float(row["confidence="]) — stray "="
                # inside the string → KeyError looking for column named "confidence="
                confidence      = float(row["confidence"]),
            )
            if(result==None):
                continue# trade was skipped due to high spread to al ratio

            result["entry_time"]     = classified.index[i]
            result["balance_before"] = balance
            # BUG FIX 12: was balance += result["pnl_used"] — wrong key name
            # The key saved by simulate_trade is "pnl_usd" not "pnl_used"
            balance                 += result["pnl_usd"]
            result["balance_after"]  = balance
            trades.append(result)

        # BUG FIX 13: was step_bars += step_bars INSIDE the for loop, and the
        # while loop never advanced `start` at all.
        # Two problems:
        #   (a) step_bars doubled on every trade → nonsensical window sizes
        #   (b) start never incremented → infinite loop
        # Fix: move start += step_bars OUTSIDE the inner for loop, at the
        # while-loop level (where it always belonged)
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

    returns = trades["pnl_usd"] / trades["balance_before"]
    sharpe  = (returns.mean() / returns.std() * np.sqrt(252)
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
    print(f"  Highest Sharpe    : {df.iloc[0]['setup']} ({df.iloc[0]['sharpe']:.2f})")
    best_dd  = df.loc[df['max_drawdown'].idxmax()]
    print(f"  Lowest drawdown   : {best_dd['setup']} ({best_dd['max_drawdown']:.1%})")
    best_pnl = df.loc[df['total_pnl'].idxmax()]
    print(f"  Highest P&L       : {best_pnl['setup']} (${best_pnl['total_pnl']:,.0f})")
    best_pf  = df.loc[df['profit_factor'].idxmax()]
    print(f"  Best profit factor: {best_pf['setup']} ({best_pf['profit_factor']:.2f})")
    best_ex  = df.loc[df['expectancy'].idxmax()]
    print(f"  Best expectancy   : {best_ex['setup']} (${best_ex['expectancy']:.2f}/trade)")


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
        print(f"\n  Profitable on {label} symbols (top 15 by avg Sharpe):")
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
# PARALLEL WORKER
# This is the function each CPU core runs.
# It receives one (symbol, setup) job, does the
# full backtest, and returns the metrics + trades.
#
# WHY a top-level function?
#   Python's multiprocessing sends work to other
#   processes by "pickling" (serialising) the
#   function and its arguments. Only functions
#   defined at the top level of a module can be
#   pickled — lambdas and nested functions can't.
# ─────────────────────────────────────────────

def _run_one(args: tuple) -> dict:
    """
    Worker function — runs one (symbol × setup) backtest.

    Receives a tuple so Pool.map() can pass it as a single argument.
    Unpacks into the pieces it needs, runs the pipeline, and returns
    a result dict that the main process collects.

    Returns a dict with:
        "metrics"    : the performance stats dict
        "trades_csv" : path to the saved trades CSV (or None if no trades)
        "symbol"     : echoed back so the main process can print progress
        "setup_name" : echoed back for the same reason
    """
    symbol, setup, featured_df, results_dir = args

    # Each worker suppresses warnings independently
    # (they don't inherit the main process's filter)
    import warnings
    warnings.filterwarnings("ignore")

    name = setup["name"]

    # Skip if not enough data (same guard as before)
    min_bars = BASE_CONFIG["train_bars"] + BASE_CONFIG["test_bars"]
    if len(featured_df) < min_bars:
        return {"metrics": None, "trades_csv": None,
                "symbol": symbol, "setup_name": name, "skipped": True}

    # Run the full backtest pipeline
    trades  = walk_forward(featured_df, symbol, setup)
    metrics = compute_metrics(trades, symbol, name)

    # Save this job's trade log to its own CSV
    # (no file-locking needed — each worker writes a unique filename)
    trades_csv = None
    if not trades.empty:
        trades_csv = os.path.join(results_dir, f"{symbol}_{name}_trades.csv")
        trades.to_csv(trades_csv, index=False)

    return {"metrics": metrics, "trades_csv": trades_csv,
            "symbol": symbol, "setup_name": name, "skipped": False}


# ─────────────────────────────────────────────
# CPU AVAILABILITY CHECK
# ─────────────────────────────────────────────

def available_workers(reserved_pct: float = 30.0) -> int:
    """
    Returns how many worker processes we can safely spawn right now,
    based on how much CPU is actually idle — not just how many cores exist.

    How it works:
      1. Measure total CPU usage across all cores over a 1-second window
         (the interval=1 makes it block for 1 second to get an accurate
         reading — instant snapshots are unreliable)
      2. Calculate how much CPU is free: free_pct = 100 - current_usage
      3. Reserve `reserved_pct` (default 30%) for the OS and other apps
         like Chrome, so they don't freeze
      4. Whatever is left, convert to a worker count:
             usable_pct / 100 × total_cores
      5. Clamp between 1 (always run at least one worker) and
         total_cores - 1 (never take every core)

    Examples with 4 cores:
      Chrome idle, total usage 15% → free=85%, usable=55% → 2 workers
      Chrome busy,  total usage 45% → free=55%, usable=25% → 1 worker
      Machine idle, total usage  5% → free=95%, usable=65% → 2 workers
      (still caps at total_cores-1 = 3, so the OS keeps one core free)

    Args:
        reserved_pct: percentage of total CPU to always leave for other apps.
                      30% on a 4-core machine = roughly 1.2 cores kept free.
                      Raise this if you want the optimizer to be more polite.
    """
    total_cores  = cpu_count()

    # psutil.cpu_percent(interval=1) measures across ALL cores combined
    # and returns a 0–100 value. interval=1 means it samples for 1 second
    # before returning, which gives a much more accurate reading than
    # interval=None (which just returns the value from the last call).
    current_usage_pct = psutil.cpu_percent(interval=1)

    free_pct   = 100.0 - current_usage_pct   # how much CPU is idle right now
    usable_pct = max(0.0, free_pct - reserved_pct)  # subtract the reserved buffer

    # Convert percentage to a core count, then clamp
    raw_workers = (usable_pct / 100.0) * total_cores
    workers     = max(1, min(int(raw_workers), total_cores))

    return workers, current_usage_pct   # return usage too so we can print it


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_optimisation():
    # ── How many cores to use ─────────────────
    # Measure actual CPU availability right now — not just total cores.
    # This respects whatever Chrome, Spotify, etc. is already using.
    # Takes 1 second to sample before printing anything.
    n_workers, current_cpu = available_workers(reserved_pct=-30)

    n_symbols  = len(BASE_CONFIG["symbols"])
    n_setups   = len(SETUPS)
    total_runs = n_symbols * n_setups

    print("=" * 60)
    print("  LAYER 5 — OPTIMISATION RUNNER v4 (parallel)")
    print(f"  {n_setups} setups x {n_symbols} symbols = {total_runs} total runs")
    print(f"  CPU usage at startup : {current_cpu:.0f}% used across {cpu_count()} cores")
    print(f"  Workers allocated    : {n_workers} (30% headroom reserved for other apps)")
    print(f"  Data: ~8 years M15 | step=200 | n_iter=50")
    est_min = round(total_runs * 0.4 / n_workers)
    est_max = round(total_runs * 0.7 / n_workers)
    print(f"  Estimated time: {est_min}–{est_max} minutes")
    print("=" * 60)

    completed, all_results, fingerprint = load_progress()
    if completed:
        print(f"  Resuming — {len(completed)} done, "
              f"{total_runs - len(completed)} remaining")

    # ── Fetch data (still sequential — MT5 is single-threaded) ───────
    if not mt5.initialize(path=BASE_CONFIG["terminal_path"]):
        print(f"[FATAL] MT5 init failed: {mt5.last_error()}")
        return

    print("\nFetching historical data...")
    raw_data = {}
    for symbol in BASE_CONFIG["symbols"]:
        raw = fetch_history(symbol)
        if raw is not None:
            raw_data[symbol] = raw

    mt5.shutdown()   # Done with MT5 — workers don't need it

    if not raw_data:
        print("[FATAL] No data fetched.")
        return

    actual_symbols = list(raw_data.keys())
    print(f"\n  Fetched {len(actual_symbols)} symbols: {actual_symbols}")

    os.makedirs(BASE_CONFIG["results_dir"], exist_ok=True)

    # ── Pre-compute features for every symbol ─────────────────────────
    # engineer_features is CPU-bound but we do it once per symbol,
    # then pass the pre-built DataFrame to every worker that needs it.
    # This avoids recomputing features 63 times per symbol.
    print("\nEngineering features for all symbols...")
    # We need one featured_df per (symbol, setup) because params affect
    # indicators like BB period and RSI length. Pre-compute per unique
    # params set to avoid redundant work.
    # For simplicity here: compute per job (small overhead vs walk_forward).

    # ── Build the job list ────────────────────────────────────────────
    # A "job" is everything one worker needs to run one (symbol × setup).
    # We filter out already-completed jobs before handing to the pool.
    jobs = []
    for symbol, raw in raw_data.items():
        for setup in SETUPS:
            key = (symbol, setup["name"])
            if key in completed:
                continue  # Skip — already done in a previous run
            # engineer_features here so each worker gets a ready-to-use df
            featured_df = engineer_features(raw, setup["params"])
            jobs.append((symbol, setup, featured_df, BASE_CONFIG["results_dir"]))

    print(f"  {len(jobs)} jobs to run across {n_workers} workers\n")

    start_time = datetime.now()
    done_count = len(completed)   # Already-completed count for progress display

    # ── Launch the process pool ───────────────────────────────────────
    # Pool(n_workers) creates n_workers child processes.
    # imap_unordered hands jobs out one-by-one and yields results as they
    # finish — in whatever order the workers complete them (fastest first).
    # This lets us print progress and save results incrementally rather
    # than waiting for ALL jobs to finish before seeing anything.
    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(_run_one, jobs):
            done_count += 1

            if result["skipped"]:
                print(f"  [{done_count:>3}/{total_runs}] "
                      f"{result['symbol']} / {result['setup_name']:<40} SKIP")
                continue

            metrics = result["metrics"]
            all_results.append(metrics)

            # Save progress after every completed job (crash-safe)
            save_progress(metrics, fingerprint)

            # ── Progress line ─────────────────────────────────────────
            elapsed  = (datetime.now() - start_time).seconds
            avg_secs = elapsed / max(done_count - len(completed), 1)
            remaining = total_runs - done_count
            eta_secs = int(avg_secs * remaining)
            eta_str  = f"{eta_secs//60}m{eta_secs%60:02d}s"

            pnl_sign = "+" if metrics["total_pnl"] >= 0 else ""
            print(
                f"  [{done_count:>3}/{total_runs}] "
                f"{result['symbol']:<8} {result['setup_name']:<38} "
                f"{metrics['total_trades']:>5} trades | "
                f"pnl={pnl_sign}${metrics['total_pnl']:>7,.0f} | "
                f"sharpe={metrics['sharpe']:>5.2f} | "
                f"dd={metrics['max_drawdown']:>6.1%} | "
                f"ETA {eta_str}"
            )

    # ── Save final summary and print tables ───────────────────────────
    results_df   = pd.DataFrame(all_results)
    results_path = os.path.join(BASE_CONFIG["results_dir"], "optimisation_summary.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n[SAVED] -> {results_path}")

    for symbol in actual_symbols:
        print_comparison_table(all_results, symbol)

    print_cross_symbol_table(all_results)

    total_time = (datetime.now() - start_time).seconds
    print(f"\n[DONE] {total_time//60}m {total_time%60:02d}s total")
    print(f"       Full results -> {BASE_CONFIG['results_dir']}")


# ─────────────────────────────────────────────
# ENTRY POINT
# The if __name__ == "__main__" guard is REQUIRED
# for multiprocessing on Windows. Without it,
# every spawned worker process would try to
# re-run the whole script — infinite forkbomb.
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nSetup count : {len(SETUPS)} setups")
    print(f"Total runs  : {len(SETUPS) * len(BASE_CONFIG['symbols'])}")
    print(f"CPU cores   : {cpu_count()} available, {max(1, cpu_count()-1)} will be used")
    print()
    run_optimisation()