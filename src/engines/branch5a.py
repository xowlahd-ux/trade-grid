from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Metrics:
    cagr: float
    mdd: float
    max_recovery_days: int
    seed_multiple: float


def weekly_rebalance_dates(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.resample("W-FRI").last().dropna().values)


def biweekly_rebalance_dates(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    weekly = weekly_rebalance_dates(idx)
    return weekly[::2]


def monthly_rebalance_dates(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.resample("ME").last().dropna().values)


def quarterly_rebalance_dates(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.resample("QE").last().dropna().values)


def get_rebalance_dates(idx: pd.DatetimeIndex, rebalance: str) -> pd.DatetimeIndex:
    mode = str(rebalance).lower().strip()
    if mode == "weekly":
        return weekly_rebalance_dates(idx)
    if mode == "biweekly":
        return biweekly_rebalance_dates(idx)
    if mode == "monthly":
        return monthly_rebalance_dates(idx)
    if mode == "quarterly":
        return quarterly_rebalance_dates(idx)
    raise ValueError(f"unsupported rebalance mode: {rebalance}")


def normalize_weights(w: dict[str, float]) -> dict[str, float]:
    s = float(sum(w.values()))
    if s <= 0:
        return {}
    return {k: float(v / s) for k, v in w.items()}


def turnover_cost_frac(
    w_prev: dict[str, float],
    w_new: dict[str, float],
    buy_cost: float,
    sell_cost: float,
) -> float:
    keys = set(w_prev.keys()) | set(w_new.keys())
    buy_turn = 0.0
    sell_turn = 0.0

    for k in keys:
        prev = float(w_prev.get(k, 0.0))
        new = float(w_new.get(k, 0.0))
        d = new - prev
        if d > 0:
            buy_turn += d
        elif d < 0:
            sell_turn += -d

    return float(buy_turn * buy_cost + sell_turn * sell_cost)


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


def compute_metrics(equity: pd.Series) -> Metrics:
    equity = equity.astype(float).dropna()
    if equity.empty:
        return Metrics(cagr=np.nan, mdd=np.nan, max_recovery_days=0, seed_multiple=np.nan)

    seed_multiple = float(equity.iloc[-1] / equity.iloc[0])
    peak = equity.cummax()
    dd = equity / peak - 1.0
    mdd = float(dd.min())
    years = (equity.index[-1] - equity.index[0]).days / 365.2425
    cagr = float(seed_multiple ** (1.0 / years) - 1.0) if years > 0 else np.nan
    max_recovery_days = compute_max_recovery_days(equity)

    return Metrics(
        cagr=cagr,
        mdd=mdd,
        max_recovery_days=max_recovery_days,
        seed_multiple=seed_multiple,
    )


def compute_recent_10y_metrics(equity: pd.Series) -> Metrics:
    if equity.empty:
        return Metrics(cagr=np.nan, mdd=np.nan, max_recovery_days=0, seed_multiple=np.nan)

    end_dt = equity.index[-1]
    start_dt = end_dt - pd.DateOffset(years=10)
    sub = equity.loc[equity.index >= start_dt].copy()
    if sub.empty:
        sub = equity.copy()
    return compute_metrics(sub)


def rolling_sharpe_scores(
    prices: pd.DataFrame,
    underlyings: list[str],
    lookback: int,
) -> pd.DataFrame:
    rets = prices[underlyings].pct_change()
    mu = rets.rolling(lookback, min_periods=lookback).mean()
    sigma = rets.rolling(lookback, min_periods=lookback).std(ddof=0)
    sharpe = mu / sigma.replace(0.0, np.nan)
    return sharpe.shift(1)


def build_branch5a_holdings(
    prices: pd.DataFrame,
    lookback: int,
    rebalance: str,
    top1_weight: float,
    buy_cost: float,
    sell_cost: float,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = prices.copy().sort_index()
    returns = prices.pct_change().fillna(0.0)

    underlyings = ["QQQ", "SPY", "SOXX"]
    trade_map = {
        "QQQ": "TQQQ_MIX",
        "SPY": "UPRO_MIX",
        "SOXX": "SOXL_MIX",
    }
    defensive = "SGOV_MIX"

    missing = [c for c in underlyings + list(trade_map.values()) + [defensive] if c not in prices.columns]
    if missing:
        raise ValueError(f"prices.csv에 필요한 컬럼이 없음: {missing}")

    if not (0.0 <= float(top1_weight) <= 1.0):
        raise ValueError("top1_weight must be between 0.0 and 1.0")

    top2_weight = 1.0 - float(top1_weight)

    scores = rolling_sharpe_scores(prices, underlyings, lookback)
    reb_dates = set(get_rebalance_dates(prices.index, rebalance))

    h_cur: dict[str, float] = {defensive: 1.0}
    equity = 1.0
    curve: list[float] = []

    holdings_rows: list[dict] = []
    rebalance_rows: list[dict] = []
    target_rows: list[dict] = []

    for i, dt in enumerate(prices.index):
        for t, w in h_cur.items():
            holdings_rows.append(
                {
                    "date": str(dt.date()),
                    "ticker": t,
                    "weight": float(w),
                }
            )

        daily_ret = 0.0
        for t, w in h_cur.items():
            daily_ret += float(returns.loc[dt, t]) * float(w)
        equity *= (1.0 + daily_ret)

        h_des = h_cur.copy()
        rebalance_today = dt in reb_dates

        if rebalance_today:
            s = scores.loc[dt].dropna()
            s = s[s > 0.0].sort_values(ascending=False)
            chosen = list(s.index[:2])

            if len(chosen) == 0:
                h_des = {defensive: 1.0}
            elif len(chosen) == 1:
                h_des = {trade_map[chosen[0]]: 1.0}
            else:
                h_des = {
                    trade_map[chosen[0]]: float(top1_weight),
                    trade_map[chosen[1]]: float(top2_weight),
                }

            h_des = normalize_weights(h_des)

            rebalance_rows.append(
                {
                    "date": str(dt.date()),
                    "rebalance": rebalance,
                    "lookback": int(lookback),
                    "top1_weight": float(top1_weight),
                    "top2_weight": float(top2_weight),
                    "rank1": chosen[0] if len(chosen) >= 1 else "",
                    "rank2": chosen[1] if len(chosen) >= 2 else "",
                    "score1": float(s.iloc[0]) if len(s) >= 1 else np.nan,
                    "score2": float(s.iloc[1]) if len(s) >= 2 else np.nan,
                    "target_holdings": json.dumps(h_des, ensure_ascii=False),
                }
            )

        target_rows.append(
            {
                "date": dt,
                "TQQQ_MIX": float(h_des.get("TQQQ_MIX", 0.0)),
                "UPRO_MIX": float(h_des.get("UPRO_MIX", 0.0)),
                "SOXL_MIX": float(h_des.get("SOXL_MIX", 0.0)),
                "SGOV_MIX": float(h_des.get("SGOV_MIX", 0.0)),
            }
        )

        if i < len(prices.index) - 1:
            cost_frac = turnover_cost_frac(h_cur, h_des, buy_cost, sell_cost)
            if cost_frac > 0:
                equity *= (1.0 - cost_frac)
            h_cur = h_des

        curve.append(equity)

    equity_series = pd.Series(curve, index=prices.index, name="equity")
    holdings_df = pd.DataFrame(holdings_rows)
    rebalance_df = pd.DataFrame(rebalance_rows)
    target_df = pd.DataFrame(target_rows)
    return equity_series, holdings_df, rebalance_df, target_df


def run_one(
    prices: pd.DataFrame,
    lookback: int,
    rebalance: str,
    top1_weight: float,
    buy_cost: float,
    sell_cost: float,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    equity, holdings_df, rebalance_df, target_df = build_branch5a_holdings(
        prices=prices,
        lookback=lookback,
        rebalance=rebalance,
        top1_weight=top1_weight,
        buy_cost=buy_cost,
        sell_cost=sell_cost,
    )

    full = compute_metrics(equity)
    recent10 = compute_recent_10y_metrics(equity)

    summary_row = {
        "strategy": "branch5a_sharpe",
        "lookback": int(lookback),
        "rebalance": str(rebalance),
        "top1_weight": float(top1_weight),
        "top2_weight": float(1.0 - top1_weight),
        "buy_cost": float(buy_cost),
        "sell_cost": float(sell_cost),
        "cagr": full.cagr,
        "mdd": full.mdd,
        "max_recovery_days": full.max_recovery_days,
        "seed_multiple": full.seed_multiple,
        "cagr_10y": recent10.cagr,
        "mdd_10y": recent10.mdd,
        "max_recovery_10y_days": recent10.max_recovery_days,
        "seed_multiple_10y": recent10.seed_multiple,
    }
    return equity, holdings_df, rebalance_df, target_df, summary_row


def build_branch5a_targets(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    lookback = int(cfg.get("lookback", 133))
    rebalance = str(cfg.get("rebalance", "weekly"))
    top1_weight = float(cfg.get("top1_weight", 0.80))

    _, _, _, target_df, _ = run_one(
        prices=prices,
        lookback=lookback,
        rebalance=rebalance,
        top1_weight=top1_weight,
        buy_cost=0.0,
        sell_cost=0.0,
    )
    return target_df.copy()