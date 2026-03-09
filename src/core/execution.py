from __future__ import annotations

import pandas as pd


TRADE_COLS = ["TQQQ_MIX", "UPRO_MIX", "SOXL_MIX", "SGOV_MIX"]
CASH = "CASH"


def normalize_target(row: pd.Series) -> dict[str, float]:
    tgt: dict[str, float] = {}
    for c in TRADE_COLS:
        v = float(row.get(c, 0.0) or 0.0)
        if v < 0:
            v = 0.0
        tgt[c] = v

    s = sum(tgt.values())
    if s > 1.000001:
        for k in tgt:
            tgt[k] /= s
        s = 1.0

    tgt[CASH] = max(0.0, 1.0 - s)
    return tgt


def apply_returns(weights: dict[str, float], returns_row: pd.Series) -> float:
    port_ret = 0.0
    for k, w in weights.items():
        if k == CASH:
            continue
        port_ret += float(w) * float(returns_row.get(k, 0.0))
    return port_ret


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


def rebalance_full(
    weights: dict[str, float],
    target: dict[str, float],
    buy_cost: float,
    sell_cost: float,
) -> tuple[dict[str, float], float]:
    cost = turnover_cost_frac(weights, target, buy_cost, sell_cost)
    return dict(target), float(cost)


def rebalance_sell_only(
    weights: dict[str, float],
    target: dict[str, float],
    sell_cost: float,
) -> tuple[dict[str, float], float]:
    out = dict(weights)
    out.setdefault(CASH, 0.0)
    sell_turn = 0.0

    for k in TRADE_COLS:
        cur = float(out.get(k, 0.0))
        tgt = float(target.get(k, 0.0))
        if cur > tgt:
            delta = cur - tgt
            out[k] = tgt
            out[CASH] += delta
            sell_turn += delta

    cost = sell_turn * sell_cost
    out[CASH] = max(0.0, out.get(CASH, 0.0) - cost)

    total = sum(out.values())
    if total > 0:
        for k in list(out.keys()):
            out[k] /= total

    return out, float(cost)


def rebalance_buy_only(
    weights: dict[str, float],
    target: dict[str, float],
    buy_cost: float,
) -> tuple[dict[str, float], float]:
    out = dict(weights)
    out.setdefault(CASH, 0.0)

    needs: dict[str, float] = {}
    need_total = 0.0
    for k in TRADE_COLS:
        cur = float(out.get(k, 0.0))
        tgt = float(target.get(k, 0.0))
        if tgt > cur:
            need = tgt - cur
            needs[k] = need
            need_total += need

    cash_avail = float(out.get(CASH, 0.0))
    if cash_avail <= 0 or need_total <= 0:
        return out, 0.0

    gross_buy_cap = cash_avail / (1.0 + buy_cost)
    buy_turn = min(need_total, gross_buy_cap)
    if buy_turn <= 0:
        return out, 0.0

    scale = buy_turn / need_total
    for k, need in needs.items():
        alloc = need * scale
        out[k] = float(out.get(k, 0.0)) + alloc

    cost = buy_turn * buy_cost
    out[CASH] = max(0.0, cash_avail - buy_turn - cost)

    total = sum(out.values())
    if total > 0:
        for k in list(out.keys()):
            out[k] /= total

    return out, float(cost)


def simulate_execution(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    buy_cost: float,
    sell_cost: float,
    mode: str = "mode1",
) -> pd.Series:
    prices = prices.copy().sort_index()
    prices.index = pd.to_datetime(prices.index)

    targets = targets.copy()
    if "date" in targets.columns:
        targets["date"] = pd.to_datetime(targets["date"])
        targets = targets.set_index("date")
    targets.index = pd.to_datetime(targets.index)
    targets = targets.sort_index()

    returns = prices[TRADE_COLS].pct_change().fillna(0.0)
    idx = prices.index.intersection(targets.index)

    if len(idx) == 0:
        raise ValueError("prices and targets have no overlapping dates")

    prices = prices.loc[idx]
    returns = returns.loc[idx]
    targets = targets.loc[idx, TRADE_COLS]

    first_target = normalize_target(targets.iloc[0])
    weights = dict(first_target)

    curve: list[float] = []

    mode = str(mode).lower().strip()

    for i, dt in enumerate(idx):
        daily_ret = apply_returns(weights, returns.loc[dt])
        equity_prev = curve[-1] if curve else 1.0
        equity = equity_prev * (1.0 + daily_ret)

        if mode == "mode1":
            if i >= 1:
                signal_dt = idx[i - 1]
                target = normalize_target(targets.loc[signal_dt])
                weights, cost = rebalance_full(weights, target, buy_cost, sell_cost)
                equity *= (1.0 - cost)

        elif mode == "mode2":
            # D일 판단 고정
            # D+1: 매도만
            if i >= 1:
                signal_dt_for_sell = idx[i - 1]
                target_sell = normalize_target(targets.loc[signal_dt_for_sell])
                weights, sell_cost_frac = rebalance_sell_only(weights, target_sell, sell_cost)
                equity *= (1.0 - sell_cost_frac)

            # D+2: 같은 D일 판단으로 매수만
            if i >= 2:
                signal_dt_for_buy = idx[i - 2]
                target_buy = normalize_target(targets.loc[signal_dt_for_buy])
                weights, buy_cost_frac = rebalance_buy_only(weights, target_buy, buy_cost)
                equity *= (1.0 - buy_cost_frac)

        else:
            raise ValueError(f"unsupported execution mode: {mode}")

        curve.append(equity)

    return pd.Series(curve, index=idx, name="equity")