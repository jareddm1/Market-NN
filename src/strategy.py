"""
strategy.py
-----------
Maps model predictions to portfolio allocations and tunes that mapping against
the real competition metric.

THE v1 PROBLEM
--------------
The old default was  allocation = 2 * sigmoid(10 * pred).  Daily excess-return
predictions are tiny (~1e-3), so 10 * pred ~ 1e-2 and sigmoid(1e-2) ~ 0.5,
giving allocations stuck at ~1.0. The model's signal never reached the score:
it was effectively always-fully-invested.

THE FIX
-------
Standardise predictions to a z-score using statistics learned on training
predictions, then map linearly:

    allocation = clip(center + k * zscore(pred), 0, 2)

`center` sets the neutral allocation and `k` sets how aggressively signal
translates into position size. Both are grid-searched against the actual
metric (metric.competition_score) on the tune block -- not chosen by eye.
"""

import numpy as np
from itertools import product
from config import (
    ALLOC_MIN, ALLOC_MAX, ALLOC_K_GRID, ALLOC_CENTER_GRID,
)
from metric import competition_score


class AllocationMapper:
    """Holds the fitted z-score normaliser plus the chosen (k, center)."""

    def __init__(self, k: float = 1.0, center: float = 1.0):
        self.k = k
        self.center = center
        self.pred_mean_ = 0.0
        self.pred_std_  = 1.0

    def fit_normaliser(self, train_preds: np.ndarray) -> "AllocationMapper":
        """Learn the prediction mean/std from TRAINING predictions only."""
        self.pred_mean_ = float(np.mean(train_preds))
        self.pred_std_  = float(np.std(train_preds)) or 1.0
        return self

    def to_allocation(self, preds: np.ndarray) -> np.ndarray:
        z = (np.asarray(preds) - self.pred_mean_) / self.pred_std_
        alloc = self.center + self.k * z
        return np.clip(alloc, ALLOC_MIN, ALLOC_MAX)


def tune_allocation(train_preds, tune_preds, tune_forward, tune_rf,
                    k_grid=ALLOC_K_GRID, center_grid=ALLOC_CENTER_GRID):
    """
    Grid-search (k, center) to maximise the competition score on the tune set.

    Parameters
    ----------
    train_preds  : model predictions on the train block (to fit the normaliser)
    tune_preds   : model predictions on the tune block
    tune_forward : forward_returns on the tune block
    tune_rf      : risk_free_rate on the tune block

    Returns
    -------
    best_mapper : a fitted AllocationMapper with the winning (k, center)
    best_score  : its score on the tune block
    """
    best_mapper, best_score = None, -np.inf

    for k, center in product(k_grid, center_grid):
        mapper = AllocationMapper(k=k, center=center).fit_normaliser(train_preds)
        alloc  = mapper.to_allocation(tune_preds)
        score  = competition_score(alloc, tune_forward, tune_rf)
        if score > best_score:
            best_score, best_mapper = score, mapper

    print(f"[strategy] Best allocation: k={best_mapper.k}, "
          f"center={best_mapper.center} | tune score={best_score:.4f}")
    return best_mapper, best_score
