"""
evaluate.py
-----------
Generate predictions for submission.

  --mode local : reads data/test.csv and writes a CSV (for offline sanity)
  --mode kaggle: starts the Kaggle inference server

KAGGLE API CAVEAT
-----------------
The exact inference-server class and the expected output column ship inside the
competition's bundled kaggle_evaluation/ directory and its official demo
notebook. The block below is a best-guess scaffold: it returns a single
position named PRED_COL per row, maintains a rolling history buffer so the LSTM
has lookback, and reuses the SAME Preprocessor + models + mapper saved by
train.py. Open the competition's demo notebook, confirm the server class name
and the return schema, and adjust `_predict_one` / the server wiring to match.
Everything else (preprocessing, model loading, allocation) stays identical.
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import joblib

from config import (
    DATE_COL, PRED_COL, MODEL_DIR, SUBMISSIONS, LSTM_SEQ_LEN,
)
from load_data import load_test, get_feature_columns
from validation import load_artifacts
from train import base_predict_all, stack


def _predict_frame(df: pd.DataFrame, pre, meta, mapper, models) -> np.ndarray:
    """Preprocess a frame and return one allocation per row."""
    X, _ = pre.transform(df)
    base = base_predict_all(models, X)
    ens  = stack(meta, base)
    return mapper.to_allocation(ens)


def local_evaluate():
    print("[evaluate] LOCAL mode")
    pre, meta, mapper, models = load_artifacts()
    test_df = load_test()

    alloc = _predict_frame(test_df, pre, meta, mapper, models)
    out = pd.DataFrame({DATE_COL: test_df.get(DATE_COL, np.arange(len(alloc))),
                        PRED_COL: alloc})
    path = SUBMISSIONS / "submission_local.csv"
    out.to_csv(path, index=False)
    print(f"[evaluate] Wrote {path}")
    print(f"[evaluate] alloc mean={alloc.mean():.3f} std={alloc.std():.3f} "
          f"min={alloc.min():.3f} max={alloc.max():.3f}")
    print(out.head().to_string(index=False))


def kaggle_evaluate():
    print("[evaluate] KAGGLE mode")
    pre, meta, mapper, models = load_artifacts()
    history = []

    def _predict_one(test: pd.DataFrame) -> pd.DataFrame:
        # Accumulate history so the LSTM has at least seq_len of lookback
        history.append(test)
        if len(history) > LSTM_SEQ_LEN * 4:
            del history[:-LSTM_SEQ_LEN * 4]
        ctx = pd.concat(history, ignore_index=True)

        alloc = _predict_frame(ctx, pre, meta, mapper, models)
        batch = alloc[-len(test):]
        return pd.DataFrame({PRED_COL: batch}, index=test.index)

    # --- Best-guess server wiring; confirm class name in the demo notebook ---
    try:
        import kaggle_evaluation.core.templates as templates  # type: ignore
        # Many 2025 comps expose a competition-specific gateway/server here.
        raise ImportError  # force the explicit message below until confirmed
    except ImportError:
        print("[evaluate] Confirm the inference-server import from the "
              "competition's kaggle_evaluation/ directory, then call its "
              ".serve() with _predict_one. The prediction logic above is ready.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "kaggle"], default="local")
    args = ap.parse_args()
    local_evaluate() if args.mode == "local" else kaggle_evaluate()
