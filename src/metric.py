"""
metric.py
---------
Single source of truth for the objective we are actually being scored on.

train.py, strategy.py, and validation.py all import from here so that "good"
means the same thing everywhere. This is the most important conceptual fix in
the rewrite: previously the model trained on Huber loss and was evaluated on a
loosely-related Sharpe, so the number we optimised and the number we were
graded on could diverge.

IMPORTANT
---------
This is a faithful *reconstruction* of the competition's Adjusted Sharpe metric
based on its public description: a Sharpe-style ratio penalised when strategy
volatility exceeds ~120% of market volatility, and penalised when the strategy
underperforms the market return. The exact official metric code ships in the
competition's evaluation files -- once you have it, replace the body of
`competition_score` and keep the signature identical so nothing else changes.
"""

import numpy as np
from config import VOL_PENALTY_THRESHOLD, ANNUALISATION


def strategy_returns(position, forward_returns, risk_free_rate):
    """
    Convert a daily position (allocation in [0, 2]) into the realised daily
    return of the strategy.

    A position p means: p invested in the S&P 500, (1 - p) in the risk-free
    asset. With leverage (p > 1) the borrowed portion costs the risk-free rate,
    which the algebra below handles automatically:

        r_strategy = rf + p * (market_return - rf)
                   = rf + p * market_excess_return

    Parameters
    ----------
    position        : array, allocation per day, already clipped to [0, 2]
    forward_returns : array, realised next-day market return
    risk_free_rate  : array, daily risk-free rate
    """
    position        = np.asarray(position, dtype=float)
    forward_returns = np.asarray(forward_returns, dtype=float)
    risk_free_rate  = np.asarray(risk_free_rate, dtype=float)

    market_excess = forward_returns - risk_free_rate
    return risk_free_rate + position * market_excess


def competition_score(position, forward_returns, risk_free_rate,
                      vol_threshold=VOL_PENALTY_THRESHOLD,
                      annualisation=ANNUALISATION,
                      return_components=False):
    """
    Adjusted Sharpe Ratio (reconstruction).

    Steps:
      1. Build strategy returns from the position.
      2. Compute the strategy's annualised Sharpe (excess over rf / vol).
      3. Penalise if strategy vol > vol_threshold * market vol.
      4. Penalise if strategy return < market return (no reward for
         underperforming a passive buy-and-hold).

    Returns the scalar score, or a dict of components if return_components.
    """
    r_strat = strategy_returns(position, forward_returns, risk_free_rate)
    rf      = np.asarray(risk_free_rate, dtype=float)
    r_mkt   = np.asarray(forward_returns, dtype=float)

    strat_excess = r_strat - rf
    mkt_excess   = r_mkt - rf

    strat_vol = np.std(strat_excess, ddof=1) * np.sqrt(annualisation) + 1e-12
    mkt_vol   = np.std(mkt_excess,  ddof=1) * np.sqrt(annualisation) + 1e-12

    sharpe = (np.mean(strat_excess) * annualisation) / strat_vol

    # --- Volatility penalty ---
    vol_limit   = vol_threshold * mkt_vol
    vol_penalty = max(0.0, (strat_vol - vol_limit) / vol_limit)

    # --- Return-underperformance penalty ---
    strat_total = np.mean(r_strat) * annualisation
    mkt_total   = np.mean(r_mkt)   * annualisation
    underperf_penalty = max(0.0, (mkt_total - strat_total) / (abs(mkt_total) + 1e-12))

    adjusted = sharpe * (1.0 - vol_penalty) * (1.0 - underperf_penalty)

    if return_components:
        return {
            "adjusted_score":    round(float(adjusted), 4),
            "raw_sharpe":        round(float(sharpe), 4),
            "strat_vol_ann":     round(float(strat_vol), 4),
            "mkt_vol_ann":       round(float(mkt_vol), 4),
            "vol_penalty":       round(float(vol_penalty), 4),
            "underperf_penalty": round(float(underperf_penalty), 4),
            "strat_return_ann":  round(float(strat_total), 4),
            "mkt_return_ann":    round(float(mkt_total), 4),
        }
    return float(adjusted)
