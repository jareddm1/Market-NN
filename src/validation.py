"""
validation.py
-------------
Single, honest evaluation on the TEST block, which train.py never touched.
Also reports baselines so you can see whether the ensemble is worth its
complexity (the reviewer's step 4).

Usage:  python validation.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import joblib
import torch

from config import TARGET_COL, FORWARD_COL, RF_COL, MODEL_DIR, LSTM_SEQ_LEN
from load_data import load_train, get_feature_columns, three_way_split
from features  import Preprocessor
from model_torch import MLP, LSTMModel
from metric import competition_score
from train import base_predict_all, stack, BASE_MODELS, DEVICE


def load_artifacts():
    pre    = joblib.load(MODEL_DIR / "preprocessor.pkl")
    meta   = joblib.load(MODEL_DIR / "meta.pkl")
    mapper = joblib.load(MODEL_DIR / "mapper.pkl")
    dims   = joblib.load(MODEL_DIR / "dims.pkl")
    lgb    = joblib.load(MODEL_DIR / "lgb.pkl")
    enet   = joblib.load(MODEL_DIR / "enet.pkl")

    mlp = MLP(input_dim=dims["mlp_dim"]).to(DEVICE)
    mlp.load_state_dict(torch.load(MODEL_DIR / "mlp.pt", map_location=DEVICE)); mlp.eval()
    lstm = LSTMModel(input_dim=dims["lstm_dim"]).to(DEVICE)
    lstm.load_state_dict(torch.load(MODEL_DIR / "lstm.pt", map_location=DEVICE)); lstm.eval()

    models = {"mlp": mlp, "lstm": lstm, "lgb": lgb, "enet": enet}
    return pre, meta, mapper, models


def main():
    df = load_train()
    _ = get_feature_columns(df)
    _, _, test_df = three_way_split(df)

    pre, meta, mapper, models = load_artifacts()

    X_test, _ = pre.transform(test_df)
    fwd = test_df[FORWARD_COL].values
    rf  = test_df[RF_COL].values

    # Ensemble -> allocation
    base = base_predict_all(models, X_test)
    ens  = stack(meta, base)
    alloc = mapper.to_allocation(ens)

    print("\n" + "=" * 56)
    print("FINAL TEST-BLOCK EVALUATION (untouched during training)")
    print("=" * 56)

    comp = competition_score(alloc, fwd, rf, return_components=True)
    print("\nEnsemble strategy:")
    for k, v in comp.items():
        print(f"  {k:<20}: {v}")

    print(f"\n  allocation mean={alloc.mean():.3f} std={alloc.std():.3f} "
          f"min={alloc.min():.3f} max={alloc.max():.3f}")

    # --- Baselines ---
    print("\nBaselines (same test block):")
    for name, a in [
        ("buy & hold (=1.0)", np.ones_like(fwd)),
        ("defensive (=0.8)",  np.full_like(fwd, 0.8)),
        ("all cash (=0.0)",   np.zeros_like(fwd)),
    ]:
        s = competition_score(a, fwd, rf)
        print(f"  {name:<20}: {s:.4f}")

    # Each base model alone (mapped through the same allocation rule)
    print("\nSingle base models (through tuned mapping):")
    for m in BASE_MODELS:
        p = base[m].copy()
        p[np.isnan(p)] = np.nanmean(p)
        s = competition_score(mapper.to_allocation(p), fwd, rf)
        print(f"  {m:<20}: {s:.4f}")


if __name__ == "__main__":
    main()
