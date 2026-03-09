from __future__ import annotations

import argparse
import json
import math
import time
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from src.core.execution import simulate_execution
from src.core.grid import expand_grid
from src.core.metrics import compute_metrics, compute_selection_score, compute_trailing_10y_metrics
from src.core.vol_targeting import apply_vol_targeting
from src.engines.branch5a import build_branch5a_targets
from src.engines.meta import run_meta_portfolio


AUTO_START_TOKENS = {"", "auto", "earliest", "first", "min"}
AUTO_END_TOKENS = {"", "auto", "latest", "today", "max"}


def serialize_param_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def shard_filter(total: int, shard_idx: int, shard_count: int) -> list[int]:
    if shard_count <= 0:
        raise ValueError(f"shard_count must be >= 1, got {shard_count}")
    if shard_idx < 0 or shard_idx >= shard_count:
        raise ValueError(
            f"shard_idx must satisfy 0 <= shard_idx < shard_count, "
            f"got shard_idx={shard_idx}, shard_count={shard_count}"
        )
    return [i for i in range(total) if i % shard_count == shard_idx]


def fmt_seconds(sec: float) -> str:
    if not math.isfinite(sec) or sec < 0:
        return "unknown"

    sec_i = int(round(sec))
    h = sec_i // 3600
    m = (sec_i % 3600) // 60
    s = sec_i % 60

    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def resolve_bound(value: str, idx: pd.DatetimeIndex, *, side: str) -> pd.Timestamp:
    text = str(value or "").strip().lower()

    if side == "start" and text in AUTO_START_TOKENS:
        return pd.Timestamp(idx.min())
    if side == "end" and text in AUTO_END_TOKENS:
        return pd.Timestamp(idx.max())

    try:
        return pd.Timestamp(value)
    except Exception as e:  # pragma: no cover - defensive error message
        raise ValueError(f"invalid {side}_date: {value!r}") from e


def resolve_price_window(
    idx: pd.DatetimeIndex,
    start_date: str,
    end_date: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if idx.empty:
        raise ValueError("price index is empty")

    start_dt = resolve_bound(start_date, idx, side="start")
    end_dt = resolve_bound(end_date, idx, side="end")

    if start_dt > end_dt:
        raise ValueError(
            f"resolved start_date is after end_date: start_date={start_dt.date()} end_date={end_dt.date()}"
        )

    return start_dt, end_dt


def apply_meta_cost_overrides(
    cfg: dict,
    param_map: dict,
    buy_cost: float,
    sell_cost: float,
) -> tuple[dict, dict]:
    out_cfg = deepcopy(cfg)
    out_params = dict(param_map)

    costs_cfg = dict(out_cfg.get("costs", {}) or {})
    costs_cfg["buy"] = float(buy_cost)
    costs_cfg["sell"] = float(sell_cost)
    out_cfg["costs"] = costs_cfg

    out_params["costs.buy"] = float(buy_cost)
    out_params["costs.sell"] = float(sell_cost)
    return out_cfg, out_params


def evaluate_meta(
    prices: pd.DataFrame,
    cfg: dict,
    execution_mode: str,
    buy_cost: float,
    sell_cost: float,
) -> tuple[pd.Series, pd.DataFrame]:
    raw_equity, engine_choice_log, picks_df, holdings_daily, holdings_weekly, targets = run_meta_portfolio(
        prices.copy(),
        cfg,
        buy_cost=buy_cost,
        sell_cost=sell_cost,
    )
    _ = raw_equity, engine_choice_log, picks_df, holdings_daily, holdings_weekly

    targets = targets.copy()
    if "date" in targets.columns:
        targets["date"] = pd.to_datetime(targets["date"])

    targets = apply_vol_targeting(prices=prices, targets=targets, cfg_or_block=cfg)

    equity = simulate_execution(
        prices=prices,
        targets=targets,
        buy_cost=buy_cost,
        sell_cost=sell_cost,
        mode=execution_mode,
    )
    return equity, targets


def evaluate_branch(
    prices: pd.DataFrame,
    cfg: dict,
    execution_mode: str,
    buy_cost: float,
    sell_cost: float,
) -> tuple[pd.Series, pd.DataFrame]:
    targets = build_branch5a_targets(prices.copy(), cfg).copy()
    if "date" in targets.columns:
        targets["date"] = pd.to_datetime(targets["date"])

    targets = apply_vol_targeting(prices=prices, targets=targets, cfg_or_block=cfg)

    equity = simulate_execution(
        prices=prices,
        targets=targets,
        buy_cost=buy_cost,
        sell_cost=sell_cost,
        mode=execution_mode,
    )
    return equity, targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, choices=["meta", "branch5a"])
    parser.add_argument("--grid-yaml", required=True)
    parser.add_argument("--prices-csv", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--execution-mode", required=True, choices=["mode1", "mode2"])
    parser.add_argument("--buy-cost", type=float, required=True)
    parser.add_argument("--sell-cost", type=float, required=True)
    parser.add_argument("--shard-idx", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    prices = pd.read_csv(args.prices_csv, index_col=0, parse_dates=True).sort_index()
    prices.index = pd.to_datetime(prices.index)

    start_dt, end_dt = resolve_price_window(prices.index, args.start_date, args.end_date)
    prices = prices.loc[(prices.index >= start_dt) & (prices.index <= end_dt)].copy()

    if prices.empty:
        raise ValueError("filtered prices are empty")

    grid_cfg = load_yaml(args.grid_yaml)
    combos = expand_grid(grid_cfg)
    selected_indices = shard_filter(len(combos), args.shard_idx, args.shard_count)

    rows: list[dict] = []

    total = len(selected_indices)
    started_at = time.perf_counter()

    print(
        json.dumps(
            {
                "engine": args.engine,
                "grid_yaml": args.grid_yaml,
                "total_param_sets": len(combos),
                "this_shard_param_sets": total,
                "shard_idx": args.shard_idx,
                "shard_count": args.shard_count,
                "execution_mode": args.execution_mode,
                "buy_cost": float(args.buy_cost),
                "sell_cost": float(args.sell_cost),
                "prices_rows": int(len(prices)),
                "start_date": str(prices.index.min().date()),
                "end_date": str(prices.index.max().date()),
            },
            ensure_ascii=False,
        )
    )

    if total == 0:
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        info = {
            "engine": args.engine,
            "grid_yaml": args.grid_yaml,
            "total_param_sets": len(combos),
            "this_shard_param_sets": 0,
            "shard_idx": args.shard_idx,
            "shard_count": args.shard_count,
            "execution_mode": args.execution_mode,
            "buy_cost": float(args.buy_cost),
            "sell_cost": float(args.sell_cost),
            "out_csv": str(out_csv),
            "elapsed_sec": 0.0,
        }
        with open(out_csv.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        print(json.dumps({"status": "empty_shard"}, ensure_ascii=False))
        return

    for done_idx, combo_idx in enumerate(selected_indices, start=1):
        one_cfg, param_map = combos[combo_idx]

        if args.engine == "meta":
            one_cfg, param_map = apply_meta_cost_overrides(
                one_cfg,
                param_map,
                buy_cost=args.buy_cost,
                sell_cost=args.sell_cost,
            )

        item_started = time.perf_counter()

        if args.engine == "meta":
            equity, targets = evaluate_meta(
                prices=prices.copy(),
                cfg=one_cfg,
                execution_mode=args.execution_mode,
                buy_cost=args.buy_cost,
                sell_cost=args.sell_cost,
            )
        else:
            equity, targets = evaluate_branch(
                prices=prices.copy(),
                cfg=one_cfg,
                execution_mode=args.execution_mode,
                buy_cost=args.buy_cost,
                sell_cost=args.sell_cost,
            )

        full = compute_metrics(equity)
        recent10 = compute_trailing_10y_metrics(equity)

        selection_score = compute_selection_score(
            cagr=full["cagr"],
            mdd=full["mdd"],
            max_recovery_days=full["max_recovery_days"],
        )
        recent10y_selection_score = compute_selection_score(
            cagr=recent10["cagr"],
            mdd=recent10["mdd"],
            max_recovery_days=recent10["max_recovery_days"],
        )

        row = {
            "engine": args.engine,
            "combo_idx": int(combo_idx),
            "execution_mode": args.execution_mode,
            "buy_cost": float(args.buy_cost),
            "sell_cost": float(args.sell_cost),
            "start_date": str(prices.index.min().date()),
            "end_date": str(prices.index.max().date()),
            "rows_prices": int(len(prices)),
            "rows_targets": int(len(targets)),
            "cagr": full["cagr"],
            "mdd": full["mdd"],
            "max_recovery_days": full["max_recovery_days"],
            "seed_multiple": full["seed_multiple"],
            "selection_score": selection_score,
            "recent10y_cagr": recent10["cagr"],
            "recent10y_mdd": recent10["mdd"],
            "recent10y_max_recovery_days": recent10["max_recovery_days"],
            "recent10y_seed_multiple": recent10["seed_multiple"],
            "recent10y_selection_score": recent10y_selection_score,
        }

        for k, v in param_map.items():
            row[f"param::{k}"] = serialize_param_value(v)

        rows.append(row)

        elapsed = time.perf_counter() - started_at
        item_elapsed = time.perf_counter() - item_started
        avg_sec = elapsed / done_idx
        remain = total - done_idx
        eta_sec = avg_sec * remain
        pct = 100.0 * done_idx / total

        print(
            f"[{args.engine} shard {args.shard_idx}/{args.shard_count}] "
            f"{done_idx}/{total} ({pct:.1f}%) "
            f"combo_idx={combo_idx} "
            f"item={fmt_seconds(item_elapsed)} "
            f"elapsed={fmt_seconds(elapsed)} "
            f"eta={fmt_seconds(eta_sec)} "
            f"score={selection_score:.6f} "
            f"cagr={full['cagr']:.6f} "
            f"mdd={full['mdd']:.6f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    elapsed_total = time.perf_counter() - started_at

    info = {
        "engine": args.engine,
        "grid_yaml": args.grid_yaml,
        "total_param_sets": len(combos),
        "this_shard_param_sets": len(selected_indices),
        "shard_idx": args.shard_idx,
        "shard_count": args.shard_count,
        "execution_mode": args.execution_mode,
        "buy_cost": float(args.buy_cost),
        "sell_cost": float(args.sell_cost),
        "out_csv": str(out_csv),
        "elapsed_sec": elapsed_total,
        "elapsed_hms": fmt_seconds(elapsed_total),
    }

    with open(out_csv.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(json.dumps(info, ensure_ascii=False))
    if not df.empty:
        print(df.sort_values(["selection_score", "cagr", "mdd"], ascending=[False, False, False]).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
