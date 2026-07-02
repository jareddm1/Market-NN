"""
config.py
---------
Central configuration. Every path, column name, hyperparameter, and toggle
lives here so the rest of the codebase never hardcodes anything.

NOTE ON COLUMN NAMES
--------------------
These are best-guess names for the Hull Tactical schema. The moment you have
train.csv, run:

    import pandas as pd
    print(pd.read_csv("data/train.csv", nrows=5).columns.tolist())

and reconcile the four *_COL constants below. Everything downstream reads from
these constants, so fixing them here fixes the whole pipeline in one place.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[1]
DATA_DIR    = ROOT / "data"
SUBMISSIONS = ROOT / "submissions"
MODEL_DIR   = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS.mkdir(parents=True, exist_ok=True)

TRAIN_FILE  = DATA_DIR / "train.csv"
TEST_FILE   = DATA_DIR / "test.csv"

# ---------------------------------------------------------------------------
# Schema (BEST GUESS -- verify against train.columns)
# ---------------------------------------------------------------------------
DATE_COL    = "date_id"                       # integer day index
TARGET_COL  = "market_forward_excess_returns" # the excess return we regress on
FORWARD_COL = "forward_returns"               # realised next-day market return
RF_COL      = "risk_free_rate"                # daily risk-free rate
PRED_COL    = "prediction"                    # the column the API expects back

# Columns that must never be fed to a model as a feature (label leakage risk)
NON_FEATURE_COLS = [
    DATE_COL, TARGET_COL, FORWARD_COL, RF_COL, PRED_COL,
    "id", "row_id", "is_scored",
]

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
ROLLING_WINDOWS = [5, 21, 63]                 # 1w, 1m, 3m of trading days

# The Hull feature set is prefixed by type. We route each group to the
# transforms that make sense for it (see features.py). Prefixes are a guess;
# adjust the mapping after you see the real column names.
FEATURE_GROUP_PREFIXES = {
    "M": "technical",   # market dynamics
    "P": "technical",   # price / valuation
    "V": "volatility",  # volatility measures
    "S": "sentiment",   # sentiment
    "E": "macro",       # macro economic
    "I": "macro",       # interest-rate
    "D": "dummy",       # binary / regime dummies -- NO rate-of-change on these
}

# ---------------------------------------------------------------------------
# Temporal split: train -> tune -> test (NEVER shuffled)
# ---------------------------------------------------------------------------
# train : fit base models + meta-learner
# tune  : grid-search the prediction->allocation mapping against the real metric
# test  : single untouched final evaluation
TRAIN_FRAC = 0.70
TUNE_FRAC  = 0.15
TEST_FRAC  = 0.15
N_CV_SPLITS = 5                               # walk-forward folds inside train

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# --- MLP ---
MLP_HIDDEN_DIMS  = [256, 128, 64]
MLP_DROPOUT      = 0.3
MLP_LR           = 1e-3
MLP_WEIGHT_DECAY = 1e-4
MLP_BATCH_SIZE   = 512
MLP_EPOCHS       = 100
MLP_PATIENCE     = 15

# --- LSTM (now just one ensemble member, not the centerpiece) ---
LSTM_HIDDEN_DIM   = 96
LSTM_NUM_LAYERS   = 2
LSTM_DROPOUT      = 0.3
LSTM_SEQ_LEN      = 20
LSTM_LR           = 5e-4
LSTM_WEIGHT_DECAY = 1e-4
LSTM_BATCH_SIZE   = 256
LSTM_EPOCHS       = 80
LSTM_PATIENCE     = 12

# --- LightGBM ---
LGB_PARAMS = {
    "objective":        "regression",
    "metric":           "l2",
    "n_estimators":     2000,
    "learning_rate":    0.01,
    "num_leaves":       31,
    "max_depth":        -1,
    "min_child_samples":50,
    "subsample":        0.8,
    "subsample_freq":   1,
    "colsample_bytree": 0.6,
    "reg_alpha":        1.0,
    "reg_lambda":       1.0,
    "random_state":     42,
    "n_jobs":           -1,
    "verbose":          -1,
}
LGB_EARLY_STOPPING = 100

# --- ElasticNet (linear baseline member) ---
ENET_ALPHA    = 1e-3
ENET_L1_RATIO = 0.5

# --- Ridge meta-learner ---
ENSEMBLE_ALPHA = 1.0                          # higher reg: base preds correlate

# ---------------------------------------------------------------------------
# Strategy / allocation
# ---------------------------------------------------------------------------
ALLOC_MIN, ALLOC_MAX = 0.0, 2.0

# Allocation = clip(center + k * zscore(pred), 0, 2). Grid searched on tune set.
ALLOC_K_GRID      = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
ALLOC_CENTER_GRID = [0.8, 1.0, 1.2]

# Competition metric
VOL_PENALTY_THRESHOLD = 1.20                  # penalise vol > 120% of market
ANNUALISATION         = 252.0

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
