"""
=============================================================
LAYER 2 — HMM REGIME DETECTOR (final)
Feature Store -> HMM -> Regime Label + Transition Probabilities
=============================================================
Requirements:
    pip install hmmlearn scikit-learn joblib pandas numpy

Changes from previous version:
    - USDJPY removed
    - Unicode arrows replaced with ASCII
    - Convergence warnings suppressed
=============================================================
"""

import numpy as np
import pandas as pd
import joblib
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
    "symbols":        ["EURUSD", "GBPUSD"],    # USDJPY removed
    "n_states":       2,
    "n_iter":         200,
    "min_train_bars": 200,
    "data_dir":       "data/",
    "model_dir":      "models/",
    "hmm_features": [
        "realized_vol",
        "atr_pct",
        "vol_ratio",
        "bar_range",
        "volume_zscore",
    ],
}

REGIME_NAMES = {0: "low_vol", 1: "high_vol"}


# ─────────────────────────────────────────────
# LOAD FEATURES
# ─────────────────────────────────────────────

def load_features(symbol: str) -> pd.DataFrame | None:
    path = os.path.join(CONFIG["data_dir"], f"{symbol}_features.csv")
    if not os.path.exists(path):
        print(f"[WARN] No feature file for {symbol}. Run Layer 1 first.")
        return None
    return pd.read_csv(path, index_col="time", parse_dates=True)


# ─────────────────────────────────────────────
# PREPARE HMM INPUT
# ─────────────────────────────────────────────

def prepare_hmm_input(df: pd.DataFrame):
    missing = [f for f in CONFIG["hmm_features"] if f not in df.columns]
    if missing:
        print(f"[ERROR] Missing features: {missing}")
        return None, None

    X_raw = df[CONFIG["hmm_features"]].dropna().values
    if len(X_raw) < CONFIG["min_train_bars"]:
        print(f"[WARN] Only {len(X_raw)} bars — need {CONFIG['min_train_bars']} to fit.")
        return None, None

    scaler = StandardScaler()
    X      = scaler.fit_transform(X_raw)
    return X, scaler


# ─────────────────────────────────────────────
# FIT HMM
# ─────────────────────────────────────────────

def fit_hmm(X: np.ndarray) -> GaussianHMM:
    model = GaussianHMM(
        n_components=CONFIG["n_states"],
        covariance_type="full",
        n_iter=CONFIG["n_iter"],
        random_state=42,
        verbose=False,
    )
    model.fit(X)
    score = model.score(X)
    print(f"  [HMM] Log-likelihood: {score:.2f} | Converged: {model.monitor_.converged}")
    return model


# ─────────────────────────────────────────────
# MAP STATES TO REGIMES
# ─────────────────────────────────────────────

def map_states_to_regimes(model: GaussianHMM, scaler: StandardScaler) -> dict:
    """
    Map HMM state numbers to regime names by comparing
    the realized_vol mean of each state's Gaussian.
    Lowest vol = low_vol, highest = high_vol.
    """
    vol_idx       = CONFIG["hmm_features"].index("realized_vol")
    state_vols    = {s: model.means_[s][vol_idx] for s in range(CONFIG["n_states"])}
    sorted_states = sorted(state_vols, key=state_vols.get)
    labels        = list(REGIME_NAMES.values())

    regime_map = {}
    for rank, state in enumerate(sorted_states):
        regime_map[state] = labels[min(rank, len(labels) - 1)]

    print(f"  [HMM] State -> Regime mapping: {regime_map}")
    return regime_map


# ─────────────────────────────────────────────
# DECODE REGIMES
# ─────────────────────────────────────────────

def decode_regimes(model: GaussianHMM, X: np.ndarray,
                   df: pd.DataFrame, regime_map: dict) -> pd.DataFrame:
    feature_index  = df[CONFIG["hmm_features"]].dropna().index
    state_sequence = model.predict(X)
    state_probs    = model.predict_proba(X)

    regimes = pd.DataFrame(index=feature_index)
    regimes["hmm_state"] = state_sequence
    regimes["regime"]    = [regime_map[s] for s in state_sequence]

    for state in range(CONFIG["n_states"]):
        name = regime_map.get(state, f"state_{state}")
        regimes[f"prob_{name}"] = state_probs[:, state]

    regimes["regime_confidence"] = state_probs.max(axis=1)
    return regimes


# ─────────────────────────────────────────────
# TRANSITION MATRIX
# ─────────────────────────────────────────────

def get_transition_matrix(model: GaussianHMM, regime_map: dict) -> pd.DataFrame:
    n            = CONFIG["n_states"]
    state_labels = [regime_map.get(i, f"state_{i}") for i in range(n)]
    return pd.DataFrame(
        model.transmat_,
        index=[f"from_{l}" for l in state_labels],
        columns=[f"to_{l}" for l in state_labels],
    )


# ─────────────────────────────────────────────
# SAVE / LOAD MODEL
# ─────────────────────────────────────────────

def save_model(model: GaussianHMM, scaler: StandardScaler,
               regime_map: dict, symbol: str):
    os.makedirs(CONFIG["model_dir"], exist_ok=True)
    bundle = {
        "model":      model,
        "scaler":     scaler,
        "regime_map": regime_map,
        "features":   CONFIG["hmm_features"],
        "fitted_at":  datetime.now().isoformat(),
    }
    path = os.path.join(CONFIG["model_dir"], f"{symbol}_hmm.pkl")
    joblib.dump(bundle, path)
    print(f"  [SAVED] Model -> {path}")


def load_model(symbol: str) -> dict | None:
    path = os.path.join(CONFIG["model_dir"], f"{symbol}_hmm.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


# ─────────────────────────────────────────────
# CLASSIFY LATEST BAR
# ─────────────────────────────────────────────

def classify_latest_bar(symbol: str) -> dict | None:
    bundle = load_model(symbol)
    if bundle is None:
        print(f"[WARN] No model for {symbol}. Fit first.")
        return None

    model      = bundle["model"]
    scaler     = bundle["scaler"]
    regime_map = bundle["regime_map"]
    features   = bundle["features"]

    df = load_features(symbol)
    if df is None:
        return None

    X_raw          = df[features].dropna().values
    X              = scaler.transform(X_raw)
    state_probs    = model.predict_proba(X)
    latest_probs   = state_probs[-1]
    current_state  = int(np.argmax(latest_probs))
    current_regime = regime_map[current_state]
    confidence     = float(latest_probs[current_state])
    transition_row = model.transmat_[current_state]

    next_regime_probs = {
        regime_map[i]: round(float(transition_row[i]), 4)
        for i in range(CONFIG["n_states"])
    }

    return {
        "symbol":             symbol,
        "timestamp":          df.index[-1].isoformat(),
        "current_regime":     current_regime,
        "confidence":         round(confidence, 4),
        "next_regime_probs":  next_regime_probs,
        "stay_probability":   round(float(transition_row[current_state]), 4),
        "switch_probability": round(1 - float(transition_row[current_state]), 4),
    }


# ─────────────────────────────────────────────
# SAVE REGIMES
# ─────────────────────────────────────────────

def save_regimes(regimes: pd.DataFrame, symbol: str):
    os.makedirs(CONFIG["data_dir"], exist_ok=True)
    path = os.path.join(CONFIG["data_dir"], f"{symbol}_regimes.csv")
    regimes.to_csv(path)
    print(f"  [SAVED] Regimes -> {path} ({len(regimes)} rows)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_hmm_pipeline():
    print("=" * 60)
    print("  LAYER 2 — HMM REGIME DETECTOR")
    print("=" * 60)

    for symbol in CONFIG["symbols"]:
        print(f"\n[{symbol}]")

        df = load_features(symbol)
        if df is None:
            continue

        X, scaler = prepare_hmm_input(df)
        if X is None:
            continue

        print(f"  Training on {len(X)} bars x {X.shape[1]} features")

        model      = fit_hmm(X)
        regime_map = map_states_to_regimes(model, scaler)
        regimes    = decode_regimes(model, X, df, regime_map)
        tmat       = get_transition_matrix(model, regime_map)

        print(f"\n  Transition matrix:")
        print(tmat.to_string())

        latest = regimes.iloc[-1]
        print(f"\n  Current regime : {latest['regime']}")
        print(f"  Confidence     : {latest['regime_confidence']:.2%}")

        save_model(model, scaler, regime_map, symbol)
        save_regimes(regimes, symbol)

    print("\n[DONE] Layer 2 complete. Layer 3 can now read regimes.")


if __name__ == "__main__":
    run_hmm_pipeline()

    print("\n" + "=" * 60)
    print("  LIVE REGIME SNAPSHOT")
    print("=" * 60)
    for symbol in CONFIG["symbols"]:
        result = classify_latest_bar(symbol)
        if result:
            print(f"\n{result['symbol']} @ {result['timestamp']}")
            print(f"  Regime     : {result['current_regime']} "
                  f"({result['confidence']:.2%} confident)")
            print(f"  Stay prob  : {result['stay_probability']:.2%}")
            print(f"  Switch prob: {result['switch_probability']:.2%}")
            print(f"  Next probs : {result['next_regime_probs']}")