from __future__ import annotations

import argparse
import itertools
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pandas.errors import EmptyDataError

from src.core.execution import simulate_execution
from src.core.grid import expand_grid
from src.core.metrics import compute_metrics, compute_selection_score, compute_trailing_10y_metrics
from src.core.vol_targeting import apply_vol_targeting
from src.engines.branch5a import build_branch5a_targets
from src.engines.meta import run_meta_portfolio


TRADE_COLS = ["TQQQ_MIX", "UPRO_MIX", "SOXL_MIX", "SGOV_MIX"]


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all_results(input_dir: Path) -> pd.DataFrame:
    csvs = sorted(input_dir.rglob("*.csv"))
    frames: list[pd.DataFrame] = []

    for p in csvs:
        if not p.name.startswith("grid_"):
            continue

        try:
            if p.stat().st_size == 0:
                continue
        except FileNotFoundError:
            continue

        try:
            df = pd.read_csv(p)
        except EmptyDataError:
            continue

        if df is None or df.empty:
            continue

        required = {"engine", "combo_idx", "cagr", "mdd"}
        if not required.issubset(set(df.columns)):
            continue

        frames.append(df)

    if not frames:
        raise RuntimeError(f"no valid shard csvs found under {input_dir}")

    return pd.concat(frames, ignore_index=True)


def find_combo_by_index(grid_yaml: str, combo_idx: int) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = load_yaml(grid_yaml)
    combos = expand_grid(cfg)
    for idx, (one_cfg, params) in enumerate(combos):
        if idx == combo_idx:
            return one_cfg, params
    raise RuntimeError(f"combo_idx={combo_idx} not found in grid={grid_yaml}")


def save_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def serialize_param_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def normalize_trade_weights(row: pd.Series | dict[str, float]) -> dict[str, float]:
    out = {c: float(row.get(c, 0.0) or 0.0) for c in TRADE_COLS}
    for c in TRADE_COLS:
        if out[c] < 0:
            out[c] = 0.0
    s = sum(out.values())
    if s <= 0:
        return {"TQQQ_MIX": 0.0, "UPRO_MIX": 0.0, "SOXL_MIX": 0.0, "SGOV_MIX": 1.0}
    return {k: v / s for k, v in out.items()}


def align_target_df(targets: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    df = targets.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    for c in TRADE_COLS:
        if c not in df.columns:
            df[c] = 0.0

    df = df[TRADE_COLS].copy()
    df = df.reindex(idx).ffill().fillna(0.0)

    s = df.sum(axis=1).replace(0.0, pd.NA)
    df = df.div(s, axis=0).fillna(0.0)

    zero_rows = df.sum(axis=1) <= 0.0
    if zero_rows.any():
        df.loc[zero_rows, "SGOV_MIX"] = 1.0

    return df


def month_end_trading_dates(idx: pd.DatetimeIndex) -> set[pd.Timestamp]:
    s = pd.Series(idx, index=idx)
    return set(pd.DatetimeIndex(s.resample("ME").last().dropna().values))


def apply_meta_cost_overrides(
    cfg: dict[str, Any],
    params: dict[str, Any],
    buy_cost: float,
    sell_cost: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out_cfg = deepcopy(cfg)
    out_params = dict(params)

    costs_cfg = dict(out_cfg.get("costs", {}) or {})
    costs_cfg["buy"] = float(buy_cost)
    costs_cfg["sell"] = float(sell_cost)
    out_cfg["costs"] = costs_cfg

    out_params["costs.buy"] = float(buy_cost)
    out_params["costs.sell"] = float(sell_cost)
    return out_cfg, out_params


def _weight_token(value: float) -> str:
    pct = round(float(value) * 100.0, 4)
    text = f"{pct:.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "neg").replace(".", "p") or "0"


def hybrid_engine_name(core_weight: float, satellite_weight: float) -> str:
    return f"hybrid_{_weight_token(core_weight)}_{_weight_token(satellite_weight)}"


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def build_hybrid_vol_block_from_args(args: argparse.Namespace) -> dict[str, Any]:
    enabled = bool(args.hybrid_vol_enabled)
    if not enabled:
        return {
            "enabled": False,
            "lookback_days": int(args.hybrid_vol_lookback_days),
            "target_annual_vol": float(args.hybrid_vol_target_annual_vol),
            "min_scale": float(args.hybrid_vol_min_scale),
            "cash_asset": str(args.hybrid_vol_cash_asset),
            "annualization": int(args.hybrid_vol_annualization),
        }

    return {
        "enabled": True,
        "lookback_days": int(args.hybrid_vol_lookback_days),
        "target_annual_vol": float(args.hybrid_vol_target_annual_vol),
        "min_scale": float(args.hybrid_vol_min_scale),
        "cash_asset": str(args.hybrid_vol_cash_asset),
        "annualization": int(args.hybrid_vol_annualization),
    }


def build_meta_result(
    *,
    combo_idx: int,
    prices: pd.DataFrame,
    grid_yaml: str,
    execution_mode: str,
    buy_cost: float,
    sell_cost: float,
) -> dict[str, Any]:
    one_cfg, params = find_combo_by_index(grid_yaml, combo_idx)
    one_cfg, params = apply_meta_cost_overrides(one_cfg, params, buy_cost, sell_cost)

    raw_equity, engine_choice_log, picks_df, holdings_daily, holdings_weekly, target_df = run_meta_portfolio(
        prices.copy(),
        one_cfg,
        buy_cost=buy_cost,
        sell_cost=sell_cost,
    )

    target_aligned = align_target_df(target_df, prices.index)
    target_aligned = align_target_df(
        apply_vol_targeting(prices=prices, targets=target_aligned, cfg_or_block=one_cfg),
        prices.index,
    )

    exec_equity = simulate_execution(
        prices=prices,
        targets=target_aligned.reset_index().rename(columns={"index": "date"}),
        buy_cost=buy_cost,
        sell_cost=sell_cost,
        mode=execution_mode,
    )

    state_df = pd.DataFrame(engine_choice_log)
    state_df["date"] = pd.to_datetime(state_df["date"])
    meta_state = state_df.set_index("date")["state"].copy().reindex(exec_equity.index).ffill()

    return {
        "combo_idx": int(combo_idx),
        "config": one_cfg,
        "params": params,
        "raw_equity": raw_equity,
        "equity": exec_equity,
        "targets": target_aligned,
        "meta_state": meta_state,
        "engine_choice_log": pd.DataFrame(engine_choice_log),
        "picks_df": picks_df,
        "holdings_daily": holdings_daily,
        "holdings_weekly": holdings_weekly,
    }


def build_branch_result(
    *,
    combo_idx: int,
    prices: pd.DataFrame,
    grid_yaml: str,
    execution_mode: str,
    buy_cost: float,
    sell_cost: float,
) -> dict[str, Any]:
    one_cfg, params = find_combo_by_index(grid_yaml, combo_idx)

    targets = build_branch5a_targets(prices.copy(), one_cfg).copy()
    target_aligned = align_target_df(targets, prices.index)
    target_aligned = align_target_df(
        apply_vol_targeting(prices=prices, targets=target_aligned, cfg_or_block=one_cfg),
        prices.index,
    )

    equity = simulate_execution(
        prices=prices,
        targets=target_aligned.reset_index().rename(columns={"index": "date"}),
        buy_cost=buy_cost,
        sell_cost=sell_cost,
        mode=execution_mode,
    )

    return {
        "combo_idx": int(combo_idx),
        "config": one_cfg,
        "params": params,
        "equity": equity,
        "targets": target_aligned,
    }


def save_meta_folder(
    *,
    result: dict[str, Any],
    prices: pd.DataFrame,
    execution_mode: str,
    buy_cost: float,
    sell_cost: float,
    folder: Path,
) -> None:
    equity = result["equity"]
    full = compute_metrics(equity)
    recent10 = compute_trailing_10y_metrics(equity)

    summary = pd.DataFrame(
        [
            {
                "engine": "meta",
                "combo_idx": result["combo_idx"],
                "execution_mode": execution_mode,
                "buy_cost": buy_cost,
                "sell_cost": sell_cost,
                **full,
                "selection_score": compute_selection_score(full["cagr"], full["mdd"], full["max_recovery_days"]),
                "recent10y_cagr": recent10["cagr"],
                "recent10y_mdd": recent10["mdd"],
                "recent10y_max_recovery_days": recent10["max_recovery_days"],
                "recent10y_seed_multiple": recent10["seed_multiple"],
                "recent10y_selection_score": compute_selection_score(
                    recent10["cagr"], recent10["mdd"], recent10["max_recovery_days"]
                ),
                **{f"param::{k}": serialize_param_value(v) for k, v in result["params"].items()},
            }
        ]
    )

    folder.mkdir(parents=True, exist_ok=True)
    summary.to_csv(folder / "summary.csv", index=False)
    equity.to_csv(folder / "equity_curve.csv", header=True)
    result["raw_equity"].to_csv(folder / "raw_equity_curve.csv", header=True)
    result["targets"].reset_index().rename(columns={"index": "date"}).to_csv(folder / "target_weights.csv", index=False)
    prices.to_csv(folder / "prices.csv")
    result["engine_choice_log"].to_csv(folder / "engine_choice_log.csv", index=False)
    result["picks_df"].to_csv(folder / "picks_top2_weekly.csv", index=False)
    result["holdings_daily"].to_csv(folder / "holdings_daily.csv", index=False)
    result["holdings_weekly"].to_csv(folder / "holdings_weekly.csv", index=False)

    save_json(
        folder / "metrics.json",
        {"full": full, "recent10y": recent10},
    )
    save_json(
        folder / "best_params.json",
        {
            "engine": "meta",
            "combo_idx": result["combo_idx"],
            "execution_mode": execution_mode,
            "buy_cost": buy_cost,
            "sell_cost": sell_cost,
            "params": result["params"],
            "full_config": result["config"],
        },
    )


def save_branch_folder(
    *,
    result: dict[str, Any],
    prices: pd.DataFrame,
    execution_mode: str,
    buy_cost: float,
    sell_cost: float,
    folder: Path,
) -> None:
    equity = result["equity"]
    full = compute_metrics(equity)
    recent10 = compute_trailing_10y_metrics(equity)

    summary = pd.DataFrame(
        [
            {
                "engine": "branch5a",
                "combo_idx": result["combo_idx"],
                "execution_mode": execution_mode,
                "buy_cost": buy_cost,
                "sell_cost": sell_cost,
                **full,
                "selection_score": compute_selection_score(full["cagr"], full["mdd"], full["max_recovery_days"]),
                "recent10y_cagr": recent10["cagr"],
                "recent10y_mdd": recent10["mdd"],
                "recent10y_max_recovery_days": recent10["max_recovery_days"],
                "recent10y_seed_multiple": recent10["seed_multiple"],
                "recent10y_selection_score": compute_selection_score(
                    recent10["cagr"], recent10["mdd"], recent10["max_recovery_days"]
                ),
                **{f"param::{k}": serialize_param_value(v) for k, v in result["params"].items()},
            }
        ]
    )

    folder.mkdir(parents=True, exist_ok=True)
    summary.to_csv(folder / "summary.csv", index=False)
    equity.to_csv(folder / "equity_curve.csv", header=True)
    result["targets"].reset_index().rename(columns={"index": "date"}).to_csv(folder / "target_weights.csv", index=False)
    prices.to_csv(folder / "prices.csv")

    save_json(
        folder / "metrics.json",
        {"full": full, "recent10y": recent10},
    )
    save_json(
        folder / "best_params.json",
        {
            "engine": "branch5a",
            "combo_idx": result["combo_idx"],
            "execution_mode": execution_mode,
            "buy_cost": buy_cost,
            "sell_cost": sell_cost,
            "params": result["params"],
            "full_config": result["config"],
        },
    )


def build_hybrid_targets(
    *,
    idx: pd.DatetimeIndex,
    meta_targets: pd.DataFrame,
    branch_targets: pd.DataFrame,
    meta_state: pd.Series,
    meta_equity: pd.Series,
    branch_equity: pd.Series,
    core_weight: float,
    satellite_weight: float,
    hybrid_rebalance_mode: str,
    hybrid_override_mode: str,
) -> pd.DataFrame:
    hybrid_rebalance_mode = str(hybrid_rebalance_mode).lower().strip()
    hybrid_override_mode = str(hybrid_override_mode).lower().strip()

    meta_targets = align_target_df(meta_targets, idx)
    branch_targets = align_target_df(branch_targets, idx)
    meta_state = meta_state.reindex(idx).ffill().astype(str)
    meta_equity = meta_equity.reindex(idx)
    branch_equity = branch_equity.reindex(idx)

    meta_ret = meta_equity.pct_change().fillna(0.0)
    branch_ret = branch_equity.pct_change().fillna(0.0)

    month_ends = month_end_trading_dates(idx)

    cap_meta = float(core_weight)
    cap_branch = float(satellite_weight)

    rows: list[dict[str, Any]] = []

    for dt in idx:
        state = str(meta_state.loc[dt]).upper()

        if hybrid_override_mode == "meta_bear_crash_full" and state in {"BEAR", "CRASH"}:
            target_w = normalize_trade_weights(meta_targets.loc[dt].to_dict())
            override_hit = True
        else:
            override_hit = False

            if hybrid_rebalance_mode == "always":
                cap_meta_target = float(core_weight)
                cap_branch_target = float(satellite_weight)
            elif hybrid_rebalance_mode == "month_end":
                cap_meta_target = float(cap_meta)
                cap_branch_target = float(cap_branch)
                total = cap_meta_target + cap_branch_target
                if total <= 0:
                    cap_meta_target = float(core_weight)
                    cap_branch_target = float(satellite_weight)
                else:
                    cap_meta_target /= total
                    cap_branch_target /= total
            else:
                raise ValueError(f"unsupported hybrid_rebalance_mode: {hybrid_rebalance_mode}")

            target_w = {
                c: cap_meta_target * float(meta_targets.loc[dt, c]) + cap_branch_target * float(branch_targets.loc[dt, c])
                for c in TRADE_COLS
            }
            target_w = normalize_trade_weights(target_w)

        rows.append(
            {
                "date": dt,
                "override_hit": override_hit,
                "cap_meta_used": float(cap_meta if hybrid_rebalance_mode == "month_end" else core_weight),
                "cap_branch_used": float(cap_branch if hybrid_rebalance_mode == "month_end" else satellite_weight),
                **target_w,
            }
        )

        if hybrid_rebalance_mode == "month_end":
            cap_meta = float(cap_meta) * (1.0 + float(meta_ret.loc[dt]))
            cap_branch = float(cap_branch) * (1.0 + float(branch_ret.loc[dt]))
            total_after = cap_meta + cap_branch
            if total_after > 0:
                cap_meta /= total_after
                cap_branch /= total_after

            if dt in month_ends:
                cap_meta = float(core_weight)
                cap_branch = float(satellite_weight)

        elif hybrid_rebalance_mode == "always":
            cap_meta = float(core_weight)
            cap_branch = float(satellite_weight)

    return pd.DataFrame(rows)


def evaluate_hybrid_combo(
    *,
    prices: pd.DataFrame,
    meta_result: dict[str, Any],
    branch_result: dict[str, Any],
    execution_mode: str,
    buy_cost: float,
    sell_cost: float,
    core_weight: float,
    satellite_weight: float,
    hybrid_rebalance_mode: str,
    hybrid_override_mode: str,
    hybrid_name: str,
    hybrid_vol_block: dict[str, Any],
) -> tuple[pd.Series, pd.DataFrame, dict[str, Any]]:
    idx = prices.index.intersection(meta_result["equity"].index).intersection(branch_result["equity"].index)

    hybrid_targets = build_hybrid_targets(
        idx=idx,
        meta_targets=meta_result["targets"],
        branch_targets=branch_result["targets"],
        meta_state=meta_result["meta_state"],
        meta_equity=meta_result["equity"],
        branch_equity=branch_result["equity"],
        core_weight=core_weight,
        satellite_weight=satellite_weight,
        hybrid_rebalance_mode=hybrid_rebalance_mode,
        hybrid_override_mode=hybrid_override_mode,
    )

    hybrid_targets = align_target_df(hybrid_targets, idx)

    if bool(hybrid_vol_block.get("enabled", False)):
        hybrid_targets = apply_vol_targeting(
            prices=prices.loc[idx],
            targets=hybrid_targets.copy(),
            cfg_or_block=hybrid_vol_block,
        )
        hybrid_targets = align_target_df(hybrid_targets, idx)

    hybrid_equity = simulate_execution(
        prices=prices.loc[idx],
        targets=hybrid_targets.reset_index().rename(columns={"index": "date"}),
        buy_cost=buy_cost,
        sell_cost=sell_cost,
        mode=execution_mode,
    )

    full = compute_metrics(hybrid_equity)
    recent10 = compute_trailing_10y_metrics(hybrid_equity)
    selection_score = compute_selection_score(full["cagr"], full["mdd"], full["max_recovery_days"])
    recent10y_selection_score = compute_selection_score(recent10["cagr"], recent10["mdd"], recent10["max_recovery_days"])

    row = {
        "engine": hybrid_name,
        "meta_combo_idx": meta_result["combo_idx"],
        "branch_combo_idx": branch_result["combo_idx"],
        "execution_mode": execution_mode,
        "buy_cost": buy_cost,
        "sell_cost": sell_cost,
        "core_weight": core_weight,
        "satellite_weight": satellite_weight,
        "hybrid_rebalance_mode": hybrid_rebalance_mode,
        "hybrid_override_mode": hybrid_override_mode,
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
        "hybrid_vol_target_enabled": bool(hybrid_vol_block.get("enabled", False)),
        "hybrid_vol_target_annual_vol": float(hybrid_vol_block.get("target_annual_vol", 0.0)),
        "hybrid_vol_target_lookback_days": int(hybrid_vol_block.get("lookback_days", 0)),
        "hybrid_vol_target_min_scale": float(hybrid_vol_block.get("min_scale", 0.0)),
        "hybrid_vol_target_cash_asset": str(hybrid_vol_block.get("cash_asset", "SGOV_MIX")),
        "hybrid_vol_target_annualization": int(hybrid_vol_block.get("annualization", 252)),
    }

    return hybrid_equity, hybrid_targets, row


def save_hybrid_folder(
    *,
    prices: pd.DataFrame,
    hybrid_equity: pd.Series,
    hybrid_targets: pd.DataFrame,
    row: dict[str, Any],
    meta_result: dict[str, Any],
    branch_result: dict[str, Any],
    folder: Path,
) -> None:
    full = compute_metrics(hybrid_equity)
    recent10 = compute_trailing_10y_metrics(hybrid_equity)

    summary = pd.DataFrame([row])

    folder.mkdir(parents=True, exist_ok=True)
    summary.to_csv(folder / "summary.csv", index=False)
    hybrid_equity.to_csv(folder / "equity_curve.csv", header=True)
    hybrid_targets.to_csv(folder / "target_weights.csv", index=False)
    prices.loc[hybrid_equity.index].to_csv(folder / "prices.csv")

    save_json(
        folder / "metrics.json",
        {"full": full, "recent10y": recent10},
    )
    save_json(
        folder / "best_params.json",
        {
            "engine": row["engine"],
            "execution_mode": row["execution_mode"],
            "buy_cost": row["buy_cost"],
            "sell_cost": row["sell_cost"],
            "core_weight": row["core_weight"],
            "satellite_weight": row["satellite_weight"],
            "hybrid_rebalance_mode": row["hybrid_rebalance_mode"],
            "hybrid_override_mode": row["hybrid_override_mode"],
            "hybrid_vol_targeting": {
                "enabled": bool(row.get("hybrid_vol_target_enabled", False)),
                "target_annual_vol": float(row.get("hybrid_vol_target_annual_vol", 0.0)),
                "lookback_days": int(row.get("hybrid_vol_target_lookback_days", 0)),
                "min_scale": float(row.get("hybrid_vol_target_min_scale", 0.0)),
                "cash_asset": str(row.get("hybrid_vol_target_cash_asset", "SGOV_MIX")),
                "annualization": int(row.get("hybrid_vol_target_annualization", 252)),
            },
            "meta_combo_idx": meta_result["combo_idx"],
            "meta_params": meta_result["params"],
            "branch_combo_idx": branch_result["combo_idx"],
            "branch_params": branch_result["params"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prices-csv", required=True)
    parser.add_argument("--meta-grid-yaml", required=True)
    parser.add_argument("--branch-grid-yaml", required=True)
    parser.add_argument("--execution-mode", required=True)
    parser.add_argument("--buy-cost", type=float, required=True)
    parser.add_argument("--sell-cost", type=float, required=True)
    parser.add_argument("--hybrid-core-weight", type=float, default=0.70)
    parser.add_argument("--hybrid-satellite-weight", type=float, default=0.30)
    parser.add_argument("--hybrid-meta-topn", type=int, default=10)
    parser.add_argument("--hybrid-branch-topn", type=int, default=10)
    parser.add_argument("--hybrid-rebalance-modes", default="always,month_end")
    parser.add_argument("--hybrid-override-modes", default="none,meta_bear_crash_full")

    # NEW: hybrid vol targeting is now independent from meta/branch
    parser.add_argument("--hybrid-vol-enabled", type=str2bool, default=False)
    parser.add_argument("--hybrid-vol-lookback-days", type=int, default=20)
    parser.add_argument("--hybrid-vol-target-annual-vol", type=float, default=0.55)
    parser.add_argument("--hybrid-vol-min-scale", type=float, default=0.35)
    parser.add_argument("--hybrid-vol-cash-asset", default="SGOV_MIX")
    parser.add_argument("--hybrid-vol-annualization", type=int, default=252)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hybrid_name = hybrid_engine_name(
        core_weight=float(args.hybrid_core_weight),
        satellite_weight=float(args.hybrid_satellite_weight),
    )

    hybrid_vol_block = build_hybrid_vol_block_from_args(args)

    prices = pd.read_csv(args.prices_csv, index_col=0, parse_dates=True).sort_index()
    prices.index = pd.to_datetime(prices.index)

    all_df = load_all_results(input_dir)
    if "selection_score" not in all_df.columns:
        all_df["selection_score"] = all_df.apply(
            lambda r: compute_selection_score(r["cagr"], r["mdd"], int(r.get("max_recovery_days", 0))),
            axis=1,
        )
    all_df.to_csv(out_dir / "all_results.csv", index=False)

    meta_rows = all_df[all_df["engine"] == "meta"].copy()
    branch_rows = all_df[all_df["engine"] == "branch5a"].copy()

    if meta_rows.empty or branch_rows.empty:
        raise RuntimeError("meta or branch5a results are empty")

    meta_rows = meta_rows.sort_values(["selection_score", "cagr", "mdd"], ascending=[False, False, False]).reset_index(drop=True)
    branch_rows = branch_rows.sort_values(["selection_score", "cagr", "mdd"], ascending=[False, False, False]).reset_index(drop=True)

    top50_by_score = (
        all_df.sort_values(["engine", "selection_score", "cagr", "mdd"], ascending=[True, False, False, False])
        .groupby("engine", as_index=False)
        .head(50)
    )
    top50_by_cagr = (
        all_df.sort_values(["engine", "cagr", "mdd"], ascending=[True, False, False])
        .groupby("engine", as_index=False)
        .head(50)
    )
    top50_by_mdd = (
        all_df.sort_values(["engine", "mdd", "cagr"], ascending=[True, False, False])
        .groupby("engine", as_index=False)
        .head(50)
    )
    top50_by_score.to_csv(out_dir / "top50_by_score.csv", index=False)
    top50_by_cagr.to_csv(out_dir / "top50_by_cagr.csv", index=False)
    top50_by_mdd.to_csv(out_dir / "top50_by_mdd.csv", index=False)

    meta_best_row = meta_rows.iloc[0]
    branch_best_row = branch_rows.iloc[0]

    meta_best_result = build_meta_result(
        combo_idx=int(meta_best_row["combo_idx"]),
        prices=prices,
        grid_yaml=args.meta_grid_yaml,
        execution_mode=args.execution_mode,
        buy_cost=args.buy_cost,
        sell_cost=args.sell_cost,
    )
    save_meta_folder(
        result=meta_best_result,
        prices=prices,
        execution_mode=args.execution_mode,
        buy_cost=args.buy_cost,
        sell_cost=args.sell_cost,
        folder=out_dir / "meta_best",
    )

    branch_best_result = build_branch_result(
        combo_idx=int(branch_best_row["combo_idx"]),
        prices=prices,
        grid_yaml=args.branch_grid_yaml,
        execution_mode=args.execution_mode,
        buy_cost=args.buy_cost,
        sell_cost=args.sell_cost,
    )
    save_branch_folder(
        result=branch_best_result,
        prices=prices,
        execution_mode=args.execution_mode,
        buy_cost=args.buy_cost,
        sell_cost=args.sell_cost,
        folder=out_dir / "branch5a_best",
    )

    meta_topn = max(1, int(args.hybrid_meta_topn))
    branch_topn = max(1, int(args.hybrid_branch_topn))

    meta_top = meta_rows.head(meta_topn).copy()
    branch_top = branch_rows.head(branch_topn).copy()

    meta_cache: dict[int, dict[str, Any]] = {
        int(meta_best_result["combo_idx"]): meta_best_result
    }
    branch_cache: dict[int, dict[str, Any]] = {
        int(branch_best_result["combo_idx"]): branch_best_result
    }

    rebalance_modes = [x.strip() for x in str(args.hybrid_rebalance_modes).split(",") if x.strip()]
    override_modes = [x.strip() for x in str(args.hybrid_override_modes).split(",") if x.strip()]

    hybrid_rows: list[dict[str, Any]] = []
    best_payload: tuple[pd.Series, pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]] | None = None

    total_combos = len(meta_top) * len(branch_top) * len(rebalance_modes) * len(override_modes)
    done = 0

    for _, mrow in meta_top.iterrows():
        m_idx = int(mrow["combo_idx"])
        if m_idx not in meta_cache:
            meta_cache[m_idx] = build_meta_result(
                combo_idx=m_idx,
                prices=prices,
                grid_yaml=args.meta_grid_yaml,
                execution_mode=args.execution_mode,
                buy_cost=args.buy_cost,
                sell_cost=args.sell_cost,
            )
        meta_result = meta_cache[m_idx]

        for _, brow in branch_top.iterrows():
            b_idx = int(brow["combo_idx"])
            if b_idx not in branch_cache:
                branch_cache[b_idx] = build_branch_result(
                    combo_idx=b_idx,
                    prices=prices,
                    grid_yaml=args.branch_grid_yaml,
                    execution_mode=args.execution_mode,
                    buy_cost=args.buy_cost,
                    sell_cost=args.sell_cost,
                )
            branch_result = branch_cache[b_idx]

            for reb_mode, ov_mode in itertools.product(rebalance_modes, override_modes):
                hybrid_equity, hybrid_targets, row = evaluate_hybrid_combo(
                    prices=prices,
                    meta_result=meta_result,
                    branch_result=branch_result,
                    execution_mode=args.execution_mode,
                    buy_cost=args.buy_cost,
                    sell_cost=args.sell_cost,
                    core_weight=float(args.hybrid_core_weight),
                    satellite_weight=float(args.hybrid_satellite_weight),
                    hybrid_rebalance_mode=reb_mode,
                    hybrid_override_mode=ov_mode,
                    hybrid_name=hybrid_name,
                    hybrid_vol_block=hybrid_vol_block,
                )
                hybrid_rows.append(row)

                if best_payload is None:
                    best_payload = (hybrid_equity, hybrid_targets, row, meta_result, branch_result)
                else:
                    _, _, best_row, _, _ = best_payload
                    if (row["selection_score"], row["cagr"], row["mdd"]) > (
                        best_row["selection_score"],
                        best_row["cagr"],
                        best_row["mdd"],
                    ):
                        best_payload = (hybrid_equity, hybrid_targets, row, meta_result, branch_result)

                done += 1
                print(
                    f"[hybrid-search] {done}/{total_combos} "
                    f"meta={m_idx} branch={b_idx} "
                    f"rebalance={reb_mode} override={ov_mode} "
                    f"score={row['selection_score']:.6f} cagr={row['cagr']:.6f} mdd={row['mdd']:.6f} "
                    f"hybrid_vol_enabled={row['hybrid_vol_target_enabled']}"
                )

    hybrid_candidates = pd.DataFrame(hybrid_rows).sort_values(
        ["selection_score", "cagr", "mdd"], ascending=[False, False, False]
    ).reset_index(drop=True)
    hybrid_candidates.to_csv(out_dir / "hybrid_candidates.csv", index=False)

    if best_payload is None:
        raise RuntimeError("no hybrid candidate evaluated")

    best_hybrid_equity, best_hybrid_targets, best_hybrid_row, best_meta_result, best_branch_result = best_payload
    save_hybrid_folder(
        prices=prices,
        hybrid_equity=best_hybrid_equity,
        hybrid_targets=best_hybrid_targets,
        row=best_hybrid_row,
        meta_result=best_meta_result,
        branch_result=best_branch_result,
        folder=out_dir / hybrid_name,
    )

    summary_rows = [
        {
            "engine": "meta",
            "rows": int(len(meta_rows)),
            "best_score": float(meta_rows.iloc[0]["selection_score"]),
            "best_cagr": float(meta_rows.iloc[0]["cagr"]),
            "best_mdd": float(meta_rows.iloc[0]["mdd"]),
            "worst_mdd": float(meta_rows["mdd"].min()),
        },
        {
            "engine": "branch5a",
            "rows": int(len(branch_rows)),
            "best_score": float(branch_rows.iloc[0]["selection_score"]),
            "best_cagr": float(branch_rows.iloc[0]["cagr"]),
            "best_mdd": float(branch_rows.iloc[0]["mdd"]),
            "worst_mdd": float(branch_rows["mdd"].min()),
        },
        {
            "engine": hybrid_name,
            "rows": int(len(hybrid_candidates)),
            "best_score": float(best_hybrid_row["selection_score"]),
            "best_cagr": float(best_hybrid_row["cagr"]),
            "best_mdd": float(best_hybrid_row["mdd"]),
            "worst_mdd": float(hybrid_candidates["mdd"].min()),
        },
    ]
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)

    print(summary.to_string(index=False))
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()