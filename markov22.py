"""
=============================================================
LAYER 2 — HMM REGIME DETECTOR
Feature Store -> HMM -> Regime Labels + Transition Probabilities
=============================================================
Requirements:
    pip install hmmlearn scikit-learn joblib pandas numpy

Symbols: EURUSD, GBPUSD, EURGBP, EURCAD, GBPCAD, AUDUSD, USDCAD

Reads from : data/{symbol}_features.csv  (Layer 1)
Writes to  : models/{symbol}_hmm.pkl
             data/{symbol}_regimes.csv

Called automatically by Layer 4 every Monday at 6am.
Can also be run manually to force a refit.
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
    "symbols": [
        "EURUSD",
        "GBPUSD",
        "EURGBP",
        "EURCAD",
        "GBPCAD",
        "AUDUSD",
        "USDCAD",
    ],
    "n_states":       2,
    "n_iter":         50,          # sufficient — models converge before 200
    "min_train_bars": 200,
    "data_dir":       "data/",
    "model_dir":      "models/",

    # Features fed into the HMM.
    # Must be identical to what Layer 1 produces and Layer 5 uses.
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

def prepare_hmm_input(df: pd.DataFrame) -> tuple:
    """
    Extract and standardise the HMM feature columns.
    Standardisation is mandatory — HMMs are sensitive to feature scale.
    Returns (X: np.ndarray, scaler: StandardScaler) or (None, None).
    """
    missing = [f for f in CONFIG["hmm_features"] if f not in df.columns]
    if missing:
        print(f"[ERROR] Missing features: {missing}")
        return None, None

    X_raw = df[CONFIG["hmm_features"]].dropna().values

    if len(X_raw) < CONFIG["min_train_bars"]:
        print(f"[WARN] Only {len(X_raw)} bars — "
              f"need {CONFIG['min_train_bars']} to fit.")
        return None, None

    scaler = StandardScaler()
    X      = scaler.fit_transform(X_raw)
    return X, scaler


# ─────────────────────────────────────────────
# FIT HMM
# ─────────────────────────────────────────────

def fit_hmm(X: np.ndarray) -> GaussianHMM:
    """
    Fit a Gaussian HMM to the feature matrix.
    covariance_type='full' allows each state its own covariance matrix,
    capturing correlations between features per regime.
    n_iter=50 is sufficient — confirmed by convergence monitoring.
    """
    model = GaussianHMM(
        n_components=CONFIG["n_states"],
        covariance_type="full",
        n_iter=CONFIG["n_iter"],
        random_state=42,
        verbose=False,
    )
    model.fit(X)
    score = model.score(X)
    print(f"  [HMM] Log-likelihood: {score:.2f} | "
          f"Converged: {model.monitor_.converged}")
    return model


# ─────────────────────────────────────────────
# MAP STATES TO REGIME NAMES
# ─────────────────────────────────────────────

def map_states_to_regimes(model: GaussianHMM) -> dict:
    """
    HMM state numbers are arbitrary — map to regime names by sorting
    states on their realized_vol mean.
    Lowest vol = low_vol, highest = high_vol.
    Consistent across refits regardless of random initialisation.
    """
    vol_idx       = CONFIG["hmm_features"].index("realized_vol")
    state_vols    = {
        s: model.means_[s][vol_idx]
        for s in range(CONFIG["n_states"])
    }
    sorted_states = sorted(state_vols, key=state_vols.get)
    regime_labels = list(REGIME_NAMES.values())

    regime_map = {}
    for rank, state in enumerate(sorted_states):
        label_idx          = min(rank, len(regime_labels) - 1)
        regime_map[state]  = regime_labels[label_idx]

    print(f"  [HMM] State -> Regime: {regime_map}")
    return regime_map


# ─────────────────────────────────────────────
# DECODE REGIMES
# ─────────────────────────────────────────────

def decode_regimes(model: GaussianHMM, X: np.ndarray,
                   df: pd.DataFrame, regime_map: dict) -> pd.DataFrame:
    """
    Viterbi decoding for most likely state sequence.
    predict_proba() gives soft probabilities — confidence near 0.5
    means uncertain regime, which Layer 3 filters out.
    """
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

def get_transition_matrix(model: GaussianHMM,
                          regime_map: dict) -> pd.DataFrame:
    """
    The Markov transition matrix.
    Entry [i, j] = P(next state is j | current state is i).
    High diagonal values = sticky regimes = reliable signals.
    """
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
    """
    Save model bundle to disk.
    Layer 3 loads this to classify new bars without refitting.
    Bundle includes model, scaler, regime_map, feature list, and timestamp.
    """
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
# CLASSIFY LATEST BAR (INFERENCE)
# ─────────────────────────────────────────────

def classify_latest_bar(symbol: str) -> dict | None:
    """
    Classify the most recent bar using the saved model.
    Called by Layer 3 on every loop tick.

    Critical: uses scaler.transform() NOT fit_transform().
    The scaler must never be refit on new live data —
    only on the training data at model fit time.
    """
    bundle = load_model(symbol)
    if bundle is None:
        print(f"[WARN] No model for {symbol}. Run fit first.")
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

    state_probs    = model.predict_proba(X)
    latest_probs   = state_probs[-1]
    current_state  = int(np.argmax(latest_probs))
    current_regime = regime_map[current_state]
    confidence     = float(latest_probs[current_state])

    transition_row    = model.transmat_[current_state]
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
        "switch_probability": round(
            1 - float(transition_row[current_state]), 4
        ),
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
    """
    Fit (or refit) the HMM for all symbols and save outputs.
    Called automatically by Layer 4 every Monday at 6am.
    Can also be run manually to force an immediate refit.
    """
    print("=" * 60)
    print("  LAYER 2 — HMM REGIME DETECTOR")
    print(f"  Symbols: {', '.join(CONFIG['symbols'])}")
    print("=" * 60)

    for symbol in CONFIG["symbols"]:
        print(f"\n[{symbol}]")

        df = load_features(symbol)
        if df is None:
            continue

        X, scaler = prepare_hmm_input(df)
        if X is None:
            continue

        print(f"  Training on {len(X):,} bars x {X.shape[1]} features")

        model      = fit_hmm(X)
        regime_map = map_states_to_regimes(model)
        regimes    = decode_regimes(model, X, df, regime_map)
        tmat       = get_transition_matrix(model, regime_map)

        print(f"\n  Transition matrix:")
        print(tmat.to_string())

        latest = regimes.iloc[-1]
        print(f"\n  Current regime : {latest['regime']}")
        print(f"  Confidence     : {latest['regime_confidence']:.2%}")

        save_model(model, scaler, regime_map, symbol)
        save_regimes(regimes, symbol)

    print("\n[DONE] Layer 2 complete.")


if __name__ == "__main__":
    run_hmm_pipeline()

    # Show live regime snapshot for all symbols
    print("\n" + "=" * 60)
    print("  LIVE REGIME SNAPSHOT")
    print("=" * 60)
    for symbol in CONFIG["symbols"]:
        result = classify_latest_bar(symbol)
        if result:
            print(f"\n{result['symbol']:8} @ {result['timestamp']}")
            print(f"  Regime      : {result['current_regime']} "
                  f"({result['confidence']:.2%} confident)")
            print(f"  Stay prob   : {result['stay_probability']:.2%}")
            print(f"  Switch prob : {result['switch_probability']:.2%}")
            print(f"  Next probs  : {result['next_regime_probs']}")