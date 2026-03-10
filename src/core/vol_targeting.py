from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


TRADE_COLS = ["TQQQ_MIX", "UPRO_MIX", "SOXL_MIX", "SGOV_MIX"]
DEFAULT_RISKY_COLS = ["TQQQ_MIX", "UPRO_MIX", "SOXL_MIX"]
DEFAULT_CASH_ASSET = "SGOV_MIX"


def _extract_vol_targeting_block(cfg_or_block: dict[str, Any] | None) -> dict[str, Any]:
    if not cfg_or_block:
        return {}
    if isinstance(cfg_or_block, dict) and "vol_targeting" in cfg_or_block:
        block = cfg_or_block.get("vol_targeting", {}) or {}
        return dict(block)
    return dict(cfg_or_block)


def disabled_vol_targeting_block(
    *,
    lookback_days: int = 20,
    target_annual_vol: float = 1.0,
    min_scale: float = 1.0,
    max_scale: float = 1.0,
    cash_asset: str = DEFAULT_CASH_ASSET,
    risky_assets: list[str] | None = None,
    annualization: int = 252,
) -> dict[str, Any]:
    return {
        "enabled": False,
        "lookback_days": int(lookback_days),
        "target_annual_vol": float(target_annual_vol),
        "min_scale": float(min_scale),
        "max_scale": float(max_scale),
        "cash_asset": str(cash_asset),
        "risky_assets": list(risky_assets or DEFAULT_RISKY_COLS),
        "annualization": int(annualization),
    }


def merge_vol_targeting_blocks(*cfgs_or_blocks: dict[str, Any] | None) -> dict[str, Any]:
    """
    Legacy utility.

    NOTE:
    - This function no longer needs to be used for hybrid vol targeting.
    - Hybrid vol targeting should be passed as an independent block from
      aggregate_results.py.
    - This function can still be used when the caller explicitly wants to
      merge multiple vol-targeting configs into one conservative block.
    """
    enabled_blocks: list[dict[str, Any]] = []
    fallback_cash = DEFAULT_CASH_ASSET
    fallback_risky = list(DEFAULT_RISKY_COLS)
    fallback_lookback = 20
    fallback_target_annual_vol = 1.0
    fallback_min_scale = 1.0
    fallback_max_scale = 1.0
    fallback_annualization = 252

    for item in cfgs_or_blocks:
        block = _extract_vol_targeting_block(item)
        if not block:
            continue

        if block.get("cash_asset"):
            fallback_cash = str(block.get("cash_asset"))
        if block.get("risky_assets"):
            fallback_risky = [str(x) for x in block.get("risky_assets") if str(x)]

        if "lookback_days" in block:
            fallback_lookback = int(block.get("lookback_days", fallback_lookback))
        if "target_annual_vol" in block:
            fallback_target_annual_vol = float(block.get("target_annual_vol", fallback_target_annual_vol))
        if "min_scale" in block:
            fallback_min_scale = float(block.get("min_scale", fallback_min_scale))
        if "max_scale" in block:
            fallback_max_scale = float(block.get("max_scale", fallback_max_scale))
        if "annualization" in block:
            fallback_annualization = int(block.get("annualization", fallback_annualization))

        if bool(block.get("enabled", False)):
            enabled_blocks.append(block)

    if not enabled_blocks:
        return disabled_vol_targeting_block(
            lookback_days=fallback_lookback,
            target_annual_vol=fallback_target_annual_vol,
            min_scale=fallback_min_scale,
            max_scale=fallback_max_scale,
            cash_asset=fallback_cash,
            risky_assets=fallback_risky,
            annualization=fallback_annualization,
        )

    target_annual_vol = min(float(b.get("target_annual_vol", 1.0)) for b in enabled_blocks)
    lookback_days = max(int(b.get("lookback_days", 20)) for b in enabled_blocks)
    min_scale = min(float(b.get("min_scale", 0.0)) for b in enabled_blocks)
    max_scale = min(float(b.get("max_scale", 1.0)) for b in enabled_blocks)
    annualization = max(int(b.get("annualization", 252)) for b in enabled_blocks)

    return {
        "enabled": True,
        "lookback_days": lookback_days,
        "target_annual_vol": target_annual_vol,
        "min_scale": min_scale,
        "max_scale": max_scale,
        "cash_asset": fallback_cash,
        "risky_assets": fallback_risky,
        "annualization": annualization,
    }


def _prepare_target_df(targets: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    df = targets.copy()
    had_date_col = "date" in df.columns
    if had_date_col:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    for c in TRADE_COLS:
        if c not in df.columns:
            df[c] = 0.0

    return df, had_date_col


def _rolling_var_cov_components(
    rets: pd.DataFrame,
    lookback_days: int,
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.Series]]:
    vars_df = rets.rolling(lookback_days, min_periods=lookback_days).var(ddof=0).shift(1)
    covs: dict[tuple[str, str], pd.Series] = {}
    cols = list(rets.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a = cols[i]
            b = cols[j]
            covs[(a, b)] = rets[a].rolling(lookback_days, min_periods=lookback_days).cov(rets[b]).shift(1)
    return vars_df, covs


def apply_vol_targeting(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    cfg_or_block: dict[str, Any] | None,
) -> pd.DataFrame:
    block = _extract_vol_targeting_block(cfg_or_block)
    enabled = bool(block.get("enabled", False))

    out_df, had_date_col = _prepare_target_df(targets)

    lookback_days = int(block.get("lookback_days", 20))
    target_annual_vol = float(block.get("target_annual_vol", 1.0))
    min_scale = float(block.get("min_scale", 0.0))
    max_scale = float(block.get("max_scale", 1.0))
    annualization = float(block.get("annualization", 252.0))
    cash_asset = str(block.get("cash_asset", DEFAULT_CASH_ASSET))
    risky_assets = [str(x) for x in block.get("risky_assets", DEFAULT_RISKY_COLS) if str(x)]
    risky_assets = [c for c in risky_assets if c in out_df.columns and c in prices.columns]

    out_df["vt_enabled"] = bool(enabled)
    out_df["vt_lookback_days"] = int(lookback_days)
    out_df["vt_target_annual_vol"] = float(target_annual_vol)
    out_df["vt_est_annual_vol"] = np.nan
    out_df["vt_scale"] = 1.0
    out_df["vt_binding"] = False
    out_df["vt_risky_before"] = 0.0
    out_df["vt_risky_after"] = 0.0

    if (not enabled) or (len(risky_assets) == 0):
        if had_date_col:
            return out_df.reset_index().rename(columns={"index": "date"})
        return out_df

    returns = prices[risky_assets].pct_change().fillna(0.0).sort_index()
    vars_df, covs = _rolling_var_cov_components(returns, lookback_days)

    if cash_asset not in out_df.columns:
        out_df[cash_asset] = 0.0

    target_index = out_df.index.intersection(prices.index)
    if len(target_index) == 0:
        if had_date_col:
            return out_df.reset_index().rename(columns={"index": "date"})
        return out_df

    for dt in target_index:
        w = {c: float(out_df.loc[dt, c] or 0.0) for c in TRADE_COLS}
        risky_before = sum(max(0.0, w.get(c, 0.0)) for c in risky_assets)
        out_df.loc[dt, "vt_risky_before"] = float(risky_before)

        if risky_before <= 0.0:
            out_df.loc[dt, "vt_risky_after"] = 0.0
            continue

        port_var = 0.0
        missing_stat = False

        for a in risky_assets:
            var_val = vars_df.loc[dt, a]
            if pd.isna(var_val):
                missing_stat = True
                break
            port_var += (w.get(a, 0.0) ** 2) * float(var_val)

        if not missing_stat:
            for i in range(len(risky_assets)):
                for j in range(i + 1, len(risky_assets)):
                    a = risky_assets[i]
                    b = risky_assets[j]
                    cov_val = covs[(a, b)].loc[dt]
                    if pd.isna(cov_val):
                        missing_stat = True
                        break
                    port_var += 2.0 * float(w.get(a, 0.0)) * float(w.get(b, 0.0)) * float(cov_val)
                if missing_stat:
                    break

        if missing_stat or not math.isfinite(port_var) or port_var <= 0.0:
            est_annual_vol = float("nan")
            scale = 1.0
        else:
            est_annual_vol = math.sqrt(max(port_var, 0.0)) * math.sqrt(max(annualization, 1.0))
            scale_raw = float(target_annual_vol) / float(est_annual_vol) if est_annual_vol > 0 else 1.0
            scale = min(max_scale, scale_raw)
            scale = max(min_scale, scale)
            scale = min(scale, 1.0)

        scale = float(scale)
        risky_after = 0.0
        cash_add = 0.0

        for a in risky_assets:
            prev = max(0.0, w.get(a, 0.0))
            new_val = prev * scale
            out_df.loc[dt, a] = float(new_val)
            risky_after += float(new_val)
            cash_add += float(prev - new_val)

        out_df.loc[dt, cash_asset] = float(max(0.0, float(out_df.loc[dt, cash_asset] or 0.0) + cash_add))
        out_df.loc[dt, "vt_est_annual_vol"] = est_annual_vol
        out_df.loc[dt, "vt_scale"] = scale
        out_df.loc[dt, "vt_binding"] = bool(scale < (1.0 - 1e-12))
        out_df.loc[dt, "vt_risky_after"] = float(risky_after)

    if had_date_col:
        return out_df.reset_index().rename(columns={"index": "date"})
    return out_df