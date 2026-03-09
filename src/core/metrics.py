from __future__ import annotations

import math

import numpy as np
import pandas as pd


def compute_max_recovery_days(equity: pd.Series) -> int:
    equity = equity.astype(float).dropna()
    if equity.empty:
        return 0

    running_peak = equity.cummax()
    underwater = equity < running_peak

    max_len = 0
    cur_len = 0
    for flag in underwater:
        if bool(flag):
            cur_len += 1
            max_len = max(max_len, cur_len)
        else:
            cur_len = 0

    return int(max_len)


def compute_metrics(equity: pd.Series) -> dict[str, float]:
    equity = equity.astype(float).dropna()

    if equity.empty:
        return {
            "cagr": np.nan,
            "mdd": np.nan,
            "max_recovery_days": 0,
            "seed_multiple": np.nan,
        }

    seed_multiple = float(equity.iloc[-1] / equity.iloc[0])
    peak = equity.cummax()
    dd = equity / peak - 1.0
    mdd = float(dd.min())

    years = (equity.index[-1] - equity.index[0]).days / 365.2425
    cagr = float(seed_multiple ** (1.0 / years) - 1.0) if years > 0 else np.nan
    max_recovery_days = compute_max_recovery_days(equity)

    return {
        "cagr": cagr,
        "mdd": mdd,
        "max_recovery_days": max_recovery_days,
        "seed_multiple": seed_multiple,
    }


def compute_trailing_10y_metrics(equity: pd.Series) -> dict[str, float]:
    equity = equity.astype(float).dropna()

    if equity.empty:
        return {
            "cagr": np.nan,
            "mdd": np.nan,
            "max_recovery_days": 0,
            "seed_multiple": np.nan,
        }

    end_dt = equity.index[-1]
    start_dt = end_dt - pd.DateOffset(years=10)
    sub = equity.loc[equity.index >= start_dt].copy()

    if sub.empty:
        sub = equity.copy()

    return compute_metrics(sub)


def compute_selection_score(
    cagr: float,
    mdd: float,
    max_recovery_days: int,
    *,
    mdd_penalty: float = 0.50,
    recovery_penalty: float = 0.15,
    trading_days_per_year: float = 252.0,
) -> float:
    """
    Higher is better.

    score = CAGR - mdd_penalty * |MDD| - recovery_penalty * (max_recovery_days / 252)
    """
    if any(math.isnan(float(x)) for x in [cagr, mdd]):
        return float("nan")

    recovery_years = float(max(max_recovery_days, 0)) / float(trading_days_per_year)
    return float(cagr) - float(mdd_penalty) * abs(float(mdd)) - float(recovery_penalty) * recovery_years
