"""
model_torch.py
--------------
PyTorch base learners: MLP and LSTM. Both output a scalar excess-return
prediction. The stacking/ensemble logic lives in train.py; the tree and linear
members live in base_models.py.

CHANGE FROM v1: the MLP uses LayerNorm, not BatchNorm. BatchNorm normalises
across the batch using running statistics that drift as the market regime
changes, and with chronological (non-shuffled) batches those statistics are
biased toward whatever regime a batch happens to fall in. LayerNorm normalises
each sample across its own features, so it is regime-agnostic and plays well
with the non-stationary, non-shuffled data here.
"""

import numpy as np
import torch
import torch.nn as nn
from config import (
    MLP_HIDDEN_DIMS, MLP_DROPOUT,
    LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, LSTM_DROPOUT, SEED,
)

torch.manual_seed(SEED)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=MLP_HIDDEN_DIMS, dropout=MLP_DROPOUT):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),            # <- was BatchNorm1d
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=LSTM_HIDDEN_DIM,
                 num_layers=LSTM_NUM_LAYERS, dropout=LSTM_DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0, batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class EarlyStopper:
    def __init__(self, patience=10, min_delta=1e-6):
        self.patience, self.min_delta = patience, min_delta
        self.counter, self.best = 0, np.inf
        self.best_weights = None

    def __call__(self, val_loss, model):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(yb)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        total += criterion(model(xb), yb).item() * len(yb)
    return total / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for batch in loader:
        xb = batch[0] if isinstance(batch, (list, tuple)) else batch
        out.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)
