"""
features.py
-----------
Preprossesses data before it goes and gets uploaded, were using
a class here instead of functions to avoid leakage that the functions would
have. functions like impute_and_clip(df) would operate on the whole frame,
but that takes in data we dont necessarily want. a transform like we do here
is better as each statistic is learned in fit(train_df) and just applied to the transform. 
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from config import (
    ROLLING_WINDOWS, DATE_COL, NON_FEATURE_COLS, FEATURE_GROUP_PREFIXES,
)


def _group_of(col: str) -> str:
    """this maps a raw column to its transform group using its first letter prefix."""
    return FEATURE_GROUP_PREFIXES.get(col[:1].upper(), "technical")

def _black_scholes_greeks(S, K, T, r, sigma):
    """Greeks for a hypothetical ATM call..."""

class Preprocessor:
    """
    Learns imputation values, clip bounds, the engineered-feature recipe, and
    the scaler from training data only, then applies them to any frame.
    """

    def __init__(self, raw_feature_cols: list[str]):
        self.raw_feature_cols = raw_feature_cols
        self.impute_values_: dict[str, float] = {}
        self.clip_bounds_:   dict[str, tuple[float, float]] = {}
        self.scaler_: RobustScaler | None = None
        self.feature_cols_: list[str] | None = None   # final engineered columns

    # ----------------------------------------------------------------- fit
    def fit(self, train_df: pd.DataFrame) -> "Preprocessor":
        # 1. Imputation value per raw column = TRAIN median (not global)
        for col in self.raw_feature_cols:
            self.impute_values_[col] = float(train_df[col].median())

        # 2. Clip bounds = TRAIN mean +/- 5 std (winsorise outliers)
        for col in self.raw_feature_cols:
            mu, sigma = train_df[col].mean(), train_df[col].std()
            if sigma and not np.isnan(sigma):
                self.clip_bounds_[col] = (mu - 5 * sigma, mu + 5 * sigma)
            else:
                self.clip_bounds_[col] = (-np.inf, np.inf)

        # 3. Build engineered features on the imputed/clipped TRAIN frame to
        #    learn the final column list, then fit the scaler on it.
        clean = self._clean(train_df)
        engineered = self._engineer(clean)
        self.feature_cols_ = [c for c in engineered.columns if c not in NON_FEATURE_COLS]

        self.scaler_ = RobustScaler()           # median/IQR: robust to outliers
        self.scaler_.fit(engineered[self.feature_cols_].values)
        return self

    # ----------------------------------------------------------- transform
    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Returns
        -------
        X_scaled    : np.ndarray of engineered, scaled features
        engineered  : the engineered DataFrame (still holds metadata columns
                      like DATE_COL / target for downstream alignment)
        """
        clean      = self._clean(df)
        engineered = self._engineer(clean)

        # Guarantee every training-time feature exists (fill absent ones with 0)
        for col in self.feature_cols_:
            if col not in engineered.columns:
                engineered[col] = 0.0

        X = engineered[self.feature_cols_].values
        X_scaled = self.scaler_.transform(X)
        return X_scaled, engineered

    # --------------------------------------------------------- internals
    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute (train median) + clip (train bounds) + forward-fill only."""
        df = df.copy()
        for col in self.raw_feature_cols:
            if col not in df.columns:
                continue
            lo, hi = self.clip_bounds_[col]
            df[col] = df[col].clip(lo, hi)
            # Forward-fill carries the last *known* value forward (causal).
            # NO bfill -- that would carry future values backward.
            df[col] = df[col].ffill().fillna(self.impute_values_[col])
        return df

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._add_calendar(df)
        df = self._add_group_features(df)
        df = self._add_derived_features(df)
        df = self._add_greek_features(df)
        # Any NaNs from leading rolling windows -> 0 (no bfill)
        new_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
        df[new_cols] = df[new_cols].ffill().fillna(0.0)
        return df

    def _add_calendar(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if DATE_COL in df.columns:
            # date_id is an integer day index; derive a weekly + yearly cycle
            d = df[DATE_COL].astype(float)
            df["cal_dow_sin"]   = np.sin(2 * np.pi * (d % 5) / 5)
            df["cal_dow_cos"]   = np.cos(2 * np.pi * (d % 5) / 5)
            df["cal_year_sin"]  = np.sin(2 * np.pi * (d % 252) / 252)
            df["cal_year_cos"]  = np.cos(2 * np.pi * (d % 252) / 252)
        return df

    def _add_group_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Route each raw column to transforms appropriate for its type. This
        replaces the old `sorted(cols)[:20]` hack (which expanded whichever
        columns happened to sort first) and the unconditional rate-of-change
        (which exploded on near-zero / binary columns).
        """
        df = df.copy()
        new = {}
        for col in self.raw_feature_cols:
            if col not in df.columns:
                continue
            grp = _group_of(col)
            s = df[col]

            if grp == "dummy":
                # Binary / regime flags: no rolling, no ROC. Leave as-is.
                continue

            if grp in ("technical", "volatility"):
                for w in ROLLING_WINDOWS:
                    roll = s.rolling(w, min_periods=max(2, w // 2))
                    mu, sd = roll.mean(), roll.std()
                    new[f"{col}_rmean{w}"]  = mu
                    new[f"{col}_rstd{w}"]   = sd
                    new[f"{col}_z{w}"]      = (s - mu) / sd.replace(0, np.nan)

            if grp in ("technical", "sentiment"):
                # Momentum via differences (safe: no division by tiny numbers)
                for lag in (1, 5, 21):
                    new[f"{col}_diff{lag}"] = s - s.shift(lag)

            if grp in ("macro", "sentiment"):
                # Slow-moving: short rolling mean to denoise
                new[f"{col}_rmean21"] = s.rolling(21, min_periods=5).mean()

        if new:
            df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
        return df
def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:


 def _add_greek_features(self, df: pd.DataFrame) -> pd.DataFrame:    # <-- new method goes here
        df = df.copy()
        if not {"PRICE_COL", "VOL_COL"}.issubset(df.columns):
            return df
        S = df["PRICE_COL"]
        sigma = df["VOL_COL"] / 100
        K = S
        T = 21 / 252
        r = df["RATE_COL"] if "RATE_COL" in df.columns else 0.04

        greeks = _black_scholes_greeks(S, K, T, r, sigma)   # note: no self. -- it's a module function
        for name, vals in greeks.items():
            df[f"opt_{name}_proxy"] = vals
        return df
    
    """
    Cross-window and cross-feature combinations built from columns
    _add_group_features already produced. Free to compute, often carry
    more signal than any single rolling window alone.
    """
df = df.copy()
new = {}
    windows = sorted(ROLLING_WINDOWS)
    if len(windows) >= 2:
        fast, slow = windows[0], windows[-1]
        for col in self.raw_feature_cols:
            grp = _group_of(col)
            if grp not in ("technical", "volatility"):
                continue

            fast_col, slow_col = f"{col}_rmean{fast}", f"{col}_rmean{slow}"
            if fast_col in df.columns and slow_col in df.columns:
                # >1 = fast average above slow average = uptrend (classic MA crossover)
                new[f"{col}_trend_ratio"] = df[fast_col] / df[slow_col].replace(0, np.nan)

            z_col, diff_col = f"{col}_z{slow}", f"{col}_diff5"
            if z_col in df.columns and diff_col in df.columns:
                # Momentum scaled down when the series is already stretched
                # (high |z|) -- avoids chasing a move that's likely to mean-revert
                new[f"{col}_momentum_riskadj"] = df[diff_col] / (1 + df[z_col].abs())

    if new:
        df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    return df

# ---------------------------------------------------------------------------
# LSTM sequence builder (unchanged logic, kept here for reuse)
# ---------------------------------------------------------------------------

def make_sequences(X: np.ndarray, y: np.ndarray | None, seq_len: int):
    """
    Turn (N, F) into overlapping (N-seq_len+1, seq_len, F) windows. The label
    for each window is the value at its final timestep.
    """
    N, F = X.shape
    if N < seq_len:
        return np.empty((0, seq_len, F), dtype=X.dtype), (None if y is None else np.empty(0))
    n = N - seq_len + 1
    X_seq = np.lib.stride_tricks.sliding_window_view(X, (seq_len, F)).reshape(n, seq_len, F)
    if y is not None:
        return X_seq, y[seq_len - 1:]
    return X_seq, None
