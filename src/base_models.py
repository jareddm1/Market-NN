"""
base_models.py
--------------
Non-neural ensemble members: a gradient-boosted tree model (LightGBM) and a
regularised linear model (ElasticNet).

WHY THESE ARE HERE
------------------
The v1 ensemble was MLP + LSTM -- two neural nets that tend to make correlated
errors and that, on ~9k rows of tabular daily data, rarely beat boosted trees.
Tabular finance competitions are usually won by tree models plus a strong
linear baseline, with neural nets as supporting members. Adding LightGBM and
ElasticNet gives the stacker genuinely diverse signals to blend.
"""

import warnings
import numpy as np
from sklearn.linear_model import ElasticNet

# LightGBM stores synthetic feature names when fit on a numpy array, then warns
# when predicting on a nameless numpy array. The warning is benign here.
warnings.filterwarnings("ignore", message="X does not have valid feature names")
from config import LGB_PARAMS, LGB_EARLY_STOPPING, ENET_ALPHA, ENET_L1_RATIO, SEED

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:                           # graceful fallback for portability
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAS_LGB = False


def train_lightgbm(X_tr, y_tr, X_val, y_val):
    """Train a boosted-tree regressor with early stopping on the val fold."""
    if HAS_LGB:
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(LGB_EARLY_STOPPING, verbose=False)],
        )
        return model
    # Fallback: sklearn's histogram GBM (no extra dependency)
    model = HistGradientBoostingRegressor(
        learning_rate=0.01, max_iter=2000, l2_regularization=1.0,
        max_leaf_nodes=31, early_stopping=True, random_state=SEED,
    )
    model.fit(X_tr, y_tr)
    return model


def train_elasticnet(X_tr, y_tr):
    """A regularised linear baseline. Fast, stable, hard to overfit."""
    model = ElasticNet(alpha=ENET_ALPHA, l1_ratio=ENET_L1_RATIO,
                       random_state=SEED, max_iter=10000)
    model.fit(X_tr, y_tr)
    return model


def predict_model(model, X):
    return np.asarray(model.predict(X))
