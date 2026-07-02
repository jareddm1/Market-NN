"""
train.py
--------
End-to-end training with strict temporal isolation.

PIPELINE
  1. Load + three-way temporal split (train / tune / test).
  2. Fit Preprocessor on TRAIN ONLY; transform all three blocks.
  3. Walk-forward out-of-fold (OOF) predictions for 4 base models, computed
     INSIDE the train block only (TimeSeriesSplit).
  4. Fit a Ridge meta-learner on those OOF predictions (stacking).
  5. Retrain each base model on the full train block.
  6. Use the TUNE block to grid-search the prediction->allocation mapping
     against the real competition metric.
  7. Save every artifact. test block is never touched here (see validation.py).

WHY THIS ORDER MATTERS
  Every statistic (impute medians, clip bounds, scaler, OOF preds, meta-learner
  weights, allocation params) is learned without ever seeing the test block.
  That is the difference between a backtest you can trust and one you can't.

Usage:  python train.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
import joblib

from config import (
    TARGET_COL, FORWARD_COL, RF_COL, N_CV_SPLITS, MODEL_DIR, SEED,
    LSTM_SEQ_LEN, ENSEMBLE_ALPHA,
    MLP_LR, MLP_WEIGHT_DECAY, MLP_BATCH_SIZE, MLP_EPOCHS, MLP_PATIENCE,
    LSTM_LR, LSTM_WEIGHT_DECAY, LSTM_BATCH_SIZE, LSTM_EPOCHS, LSTM_PATIENCE,
)
from load_data import load_train, get_feature_columns, three_way_split
from features  import Preprocessor, make_sequences
from model_torch import MLP, LSTMModel, EarlyStopper, train_epoch, eval_epoch, predict
from base_models import train_lightgbm, train_elasticnet, predict_model
from strategy import tune_allocation

torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[train] Device: {DEVICE}")

BASE_MODELS = ["mlp", "lstm", "lgb", "enet"]


# --------------------------------------------------------------------------- #
# Torch helpers
# --------------------------------------------------------------------------- #
def _loader(X, y=None, bs=512):
    tensors = [torch.tensor(X, dtype=torch.float32)]
    if y is not None:
        tensors.append(torch.tensor(y, dtype=torch.float32))
    return DataLoader(TensorDataset(*tensors), batch_size=bs, shuffle=False)


def fit_mlp(X_tr, y_tr, X_val, y_val):
    model = MLP(input_dim=X_tr.shape[1]).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MLP_EPOCHS)
    crit  = nn.HuberLoss()
    stop  = EarlyStopper(MLP_PATIENCE)
    tr, va = _loader(X_tr, y_tr, MLP_BATCH_SIZE), _loader(X_val, y_val, MLP_BATCH_SIZE * 2)
    for ep in range(MLP_EPOCHS):
        train_epoch(model, tr, opt, crit, DEVICE)
        if stop(eval_epoch(model, va, crit, DEVICE), model):
            break
        sched.step()
    stop.restore_best(model)
    return model


def fit_lstm(Xs_tr, ys_tr, Xs_val, ys_val):
    model = LSTMModel(input_dim=Xs_tr.shape[2]).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LSTM_LR, weight_decay=LSTM_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=LSTM_EPOCHS)
    crit  = nn.HuberLoss()
    stop  = EarlyStopper(LSTM_PATIENCE)
    tr, va = _loader(Xs_tr, ys_tr, LSTM_BATCH_SIZE), _loader(Xs_val, ys_val, LSTM_BATCH_SIZE * 2)
    for ep in range(LSTM_EPOCHS):
        train_epoch(model, tr, opt, crit, DEVICE)
        if stop(eval_epoch(model, va, crit, DEVICE), model):
            break
        sched.step()
    stop.restore_best(model)
    return model


# --------------------------------------------------------------------------- #
# Out-of-fold predictions (walk-forward, inside train only)
# --------------------------------------------------------------------------- #
def build_oof(X, y):
    """
    Returns a dict of OOF prediction arrays (length N, NaN where a model has no
    prediction) for each base model, using TimeSeriesSplit over the train rows.
    """
    N = len(X)
    oof = {m: np.full(N, np.nan) for m in BASE_MODELS}

    X_seq_all, y_seq_all = make_sequences(X, y, LSTM_SEQ_LEN)
    offset = LSTM_SEQ_LEN - 1                       # seq index j -> flat j+offset

    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
        print(f"[train]  OOF fold {fold}/{N_CV_SPLITS}")
        Xtr, Xva = X[tr_idx], X[val_idx]
        ytr, yva = y[tr_idx], y[val_idx]

        # internal early-stop split (tail of the fold's train)
        cut = int(len(tr_idx) * 0.85)
        Xtr_in, Xes = Xtr[:cut], Xtr[cut:]
        ytr_in, yes = ytr[:cut], ytr[cut:]

        # --- flat models ---
        mlp  = fit_mlp(Xtr_in, ytr_in, Xes, yes)
        oof["mlp"][val_idx]  = predict(mlp, _loader(Xva), DEVICE)

        lgb  = train_lightgbm(Xtr_in, ytr_in, Xes, yes)
        oof["lgb"][val_idx]  = predict_model(lgb, Xva)

        enet = train_elasticnet(Xtr, ytr)
        oof["enet"][val_idx] = predict_model(enet, Xva)

        # --- LSTM (sequence indices whose final timestep is in this fold) ---
        tr_seq  = tr_idx[tr_idx  >= offset] - offset
        val_seq = val_idx[val_idx >= offset] - offset
        if len(tr_seq) > LSTM_SEQ_LEN and len(val_seq) > 0:
            cut_s = int(len(tr_seq) * 0.85)
            lstm = fit_lstm(
                X_seq_all[tr_seq[:cut_s]], y_seq_all[tr_seq[:cut_s]],
                X_seq_all[tr_seq[cut_s:]], y_seq_all[tr_seq[cut_s:]],
            )
            preds = predict(lstm, _loader(X_seq_all[val_seq]), DEVICE)
            oof["lstm"][val_idx[val_idx >= offset]] = preds

    return oof


# --------------------------------------------------------------------------- #
# Final base-model training on full train block
# --------------------------------------------------------------------------- #
def fit_final_models(X, y):
    cut = int(len(X) * 0.85)
    Xtr, Xes, ytr, yes = X[:cut], X[cut:], y[:cut], y[cut:]

    print("[train] Final MLP ..."); mlp  = fit_mlp(Xtr, ytr, Xes, yes)
    print("[train] Final LGB ..."); lgb  = train_lightgbm(Xtr, ytr, Xes, yes)
    print("[train] Final ENet..."); enet = train_elasticnet(X, y)

    print("[train] Final LSTM...")
    Xs, ys = make_sequences(X, y, LSTM_SEQ_LEN)
    cut_s  = int(len(Xs) * 0.85)
    lstm   = fit_lstm(Xs[:cut_s], ys[:cut_s], Xs[cut_s:], ys[cut_s:])
    return {"mlp": mlp, "lstm": lstm, "lgb": lgb, "enet": enet}


def base_predict_all(models, X):
    """Predict every base model on a flat feature matrix X. LSTM rows that lack
    enough lookback are NaN-padded."""
    out = {}
    out["mlp"]  = predict(models["mlp"], _loader(X), DEVICE)
    out["lgb"]  = predict_model(models["lgb"], X)
    out["enet"] = predict_model(models["enet"], X)

    lstm_pred = np.full(len(X), np.nan)
    Xs, _ = make_sequences(X, None, LSTM_SEQ_LEN)
    if len(Xs):
        lstm_pred[LSTM_SEQ_LEN - 1:] = predict(models["lstm"], _loader(Xs), DEVICE)
    out["lstm"] = lstm_pred
    return out


def stack(meta_learner, base_preds):
    """Blend base predictions with the meta-learner where all are present;
    fall back to the simple mean elsewhere (early LSTM-less rows)."""
    cols = np.column_stack([base_preds[m] for m in BASE_MODELS])
    valid = ~np.isnan(cols).any(axis=1)
    out = np.nanmean(cols, axis=1)             # fallback
    if valid.any():
        out[valid] = meta_learner.predict(cols[valid])
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    df = load_train()
    raw_cols = get_feature_columns(df)
    train_df, tune_df, test_df = three_way_split(df)

    # Preprocessor sees TRAIN ONLY
    pre = Preprocessor(raw_cols).fit(train_df)
    Xtr, _   = pre.transform(train_df)
    Xtune, tune_eng = pre.transform(tune_df)

    ytr = train_df[TARGET_COL].values.astype("float32")

    # 1. OOF + meta-learner (stacking), all inside train
    print("[train] Building out-of-fold predictions...")
    oof = build_oof(Xtr, ytr)
    cols  = np.column_stack([oof[m] for m in BASE_MODELS])
    valid = ~np.isnan(cols).any(axis=1)
    meta  = Ridge(alpha=ENSEMBLE_ALPHA)
    meta.fit(cols[valid], ytr[valid])
    print("[train] Meta weights:",
          {m: round(float(w), 3) for m, w in zip(BASE_MODELS, meta.coef_)})

    # 4. Save artifacts
    torch.save(models["mlp"].state_dict(),  MODEL_DIR / "mlp.pt")
    torch.save(models["lstm"].state_dict(), MODEL_DIR / "lstm.pt")
    joblib.dump(models["lgb"],  MODEL_DIR / "lgb.pkl")
    joblib.dump(models["enet"], MODEL_DIR / "enet.pkl")
    joblib.dump(pre,    MODEL_DIR / "preprocessor.pkl")
    joblib.dump(meta,   MODEL_DIR / "meta.pkl")
    joblib.dump(mapper, MODEL_DIR / "mapper.pkl")
    joblib.dump({"mlp_dim": Xtr.shape[1], "lstm_dim": Xtr.shape[1],
                 "seq_len": LSTM_SEQ_LEN}, MODEL_DIR / "dims.pkl")
    print(f"[train] Saved artifacts to {MODEL_DIR}. Done.")


def oof_to_full(oof, Xtr, models):
    """The meta-normaliser wants ensemble preds over train. OOF already covers
    most train rows; the earliest rows (first fold's train portion) have no OOF,
    so fill those with final-model in-sample predictions."""
    filled = {m: oof[m].copy() for m in BASE_MODELS}
    insample = base_predict_all(models, Xtr)
    for m in BASE_MODELS:
        nan_mask = np.isnan(filled[m])
        filled[m][nan_mask] = insample[m][nan_mask]
    return filled


if __name__ == "__main__":
    main()
