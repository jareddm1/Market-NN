# Hull Tactical Market Prediction

PyTorch + LightGBM stacking ensemble for the [Hull Tactical Kaggle competition](https://www.kaggle.com/competitions/hull-tactical-market-prediction).

**Task:** predict daily S&P 500 excess returns and convert them into a daily portfolio allocation in `[0, 2]`, scored by an Adjusted Sharpe Ratio that penalises excess volatility and underperformance vs the market.

> **Repo description (paste into GitHub's About box):** Leakage-safe PyTorch + LightGBM stacking ensemble for the Hull Tactical market-prediction competition.

## Run it (in order)

```bash
pip install -r requirements.txt
cd src

# 1. (optional) sanity-check the whole pipeline on synthetic data
python make_mock_data.py     # writes mock train.csv / test.csv into ../data
python train.py              # trains + saves artifacts into ../models
python validation.py         # final score on the untouched test block + baselines
python evaluate.py --mode local   # writes ../submissions/submission_local.csv

# 2. when the real data arrives, drop train.csv/test.csv into data/, then:
python -c "import pandas as pd; print(pd.read_csv('../data/train.csv', nrows=5).columns.tolist())"
#   ^ reconcile these names with the *_COL constants in config.py, then rerun train/validation
```

All scripts are run from inside `src/`. Paths in `config.py` assume that location, so keep the files in `src/` rather than the repo root.

---

## Pipeline (leakage-safe by construction)

```
load_train()  →  three_way_split()  →  train | tune | test   (chronological, never shuffled)
                                          │
         Preprocessor.fit(train)  ←───────┘   (medians, clip bounds, scaler: TRAIN ONLY)
                  │
                  ▼
   ┌──────── base models (OOF via TimeSeriesSplit, inside train only) ────────┐
   │   MLP (LayerNorm)   LSTM   LightGBM   ElasticNet                          │
   └──────────────────────────────┬───────────────────────────────────────────┘
                                   ▼
                     Ridge meta-learner (stacking)
                                   │
                                   ▼
        AllocationMapper:  clip(center + k · zscore(pred), 0, 2)
        (k, center) grid-searched on the TUNE block against the real metric
                                   │
                                   ▼
              single untouched score on the TEST block
```

## File map

| File | Role |
|------|------|
| `config.py` | All paths, **schema column names**, hyperparameters |
| `metric.py` | The competition scorer — single source of truth for "good" |
| `load_data.py` | Loading + chronological three-way split (no preprocessing) |
| `features.py` | `Preprocessor` (fit/transform) + group-aware feature engineering |
| `model_torch.py` | MLP (LayerNorm) and LSTM + training helpers |
| `base_models.py` | LightGBM and ElasticNet ensemble members |
| `strategy.py` | z-score allocation mapping + metric-driven grid search |
| `train.py` | Orchestrates everything; saves artifacts |
| `validation.py` | Final test-block score + baselines + per-model breakdown |
| `evaluate.py` | Submission CSV (local) / Kaggle inference scaffold |
| `make_mock_data.py` | Synthetic competition-shaped data to test before real data arrives |

## Quickstart

```bash
pip install -r requirements.txt
cd src

# Optional: run the whole thing on synthetic data first
python make_mock_data.py
python train.py
python validation.py
python evaluate.py --mode local
```

Then drop the real `train.csv` / `test.csv` into `data/` and rerun.

## Two things to confirm with the real data

1. **Column names.** `config.py` guesses `date_id`, `forward_returns`, `risk_free_rate`, `market_forward_excess_returns`, `prediction`. Run `pd.read_csv("data/train.csv", nrows=5).columns.tolist()` and reconcile. Also check the `FEATURE_GROUP_PREFIXES` map against the real prefixes.
2. **Inference API.** `evaluate.py`'s Kaggle server wiring is a scaffold. Open the competition's demo notebook / `kaggle_evaluation/` directory, confirm the server class and the expected return column, and adjust the wiring. The prediction logic itself is done.

## Design notes

- **No leakage:** every statistic is fit inside `Preprocessor.fit(train_df)`; nothing is recomputed in `.transform`. Forward-fill only — no `bfill`. OOF and the meta-learner live entirely inside the train block; tuning lives in `tune`; `test` is scored once.
- **Real objective everywhere:** models train on HuberLoss for a stable gradient, but the allocation mapping and all reported scores use `metric.competition_score`, so what we optimise the *strategy* on matches what Kaggle grades.
- **Diverse ensemble:** trees (LightGBM) + linear (ElasticNet) + two neural nets, blended by a Ridge stacker. On tabular daily data the trees usually carry the most weight; the LSTM is a supporting member, not the centerpiece.
- **Allocation is where points live:** the mapping is standardised and tuned against the metric, so model signal actually reaches the score instead of collapsing to ~1.0.
