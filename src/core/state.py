from __future__ import annotations

import numpy as np
import pandas as pd


def compute_state_flags(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Priority: CRASH > BULL > BEAR
    Look-ahead prevention: signals use shift(1)

    Includes:
      - crash.main (lb, thr)
      - crash.fast (optional): short lookback crash trigger
      - min_hold_days hysteresis (legacy)
      - bear_fast (optional): short lookback BEAR trigger
        IMPORTANT: bear_fast applied AFTER min_hold so it can force immediate exit
    """
    debug = bool(cfg.get("debug", {}).get("state", False))

    state_cfg = cfg.get("state", {}) or {}

    base = state_cfg.get("base_ticker", state_cfg.get("base"))
    if isinstance(base, list):
        base = base[0] if base else None

    if not base:
        raise ValueError("state.base_ticker is required")

    if base not in prices.columns:
        raise ValueError(f"state.base_ticker '{base}' not in prices columns: {list(prices.columns)}")

    p = prices[base].astype(float)

    ma_days = int(state_cfg["ma_days"])
    min_hold = int(state_cfg.get("min_hold_days", 0))

    ma = p.rolling(ma_days).mean()

    bull_raw = p > ma
    bull = bull_raw.shift(1, fill_value=False).astype(bool)

    # ---- CRASH (main + fast) ----
    crash_cfg = cfg.get("crash", {}) or {}
    crash_enabled = bool(crash_cfg.get("enabled", True))
    crash = pd.Series(False, index=prices.index)

    if crash_enabled:
        lb = int(crash_cfg["lookback_days"])
        thr = float(crash_cfg["threshold"])
        r = p.pct_change(lb).shift(1)
        crash = (r <= thr).fillna(False)

    fast = crash_cfg.get("fast", {}) or {}
    fast_enabled = bool(fast.get("enabled", False))
    if crash_enabled and fast_enabled:
        flb = int(fast.get("lookback_days", 5))
        fthr = float(fast.get("threshold", -0.08))
        fr = p.pct_change(flb).shift(1)
        crash = crash | (fr <= fthr).fillna(False)

    # ---- HOLD FILTER ----
    if min_hold > 0:
        bull = _min_hold_filter(bull, min_hold)

    # ---- BEAR FAST (force BULL off after hold filter) ----
    bear_fast_cfg = cfg.get("bear_fast", {}) or {}
    bear_fast_enabled = bool(bear_fast_cfg.get("enabled", False))
    bear_fast = pd.Series(False, index=prices.index)

    if bear_fast_enabled:
        blb = int(bear_fast_cfg.get("lookback_days", 10))
        bthr = float(bear_fast_cfg.get("threshold", -0.06))
        br = p.pct_change(blb).shift(1)
        bear_fast = (br <= bthr).fillna(False)

        # force BULL off when bear_fast triggers
        bull = bull & (~bear_fast)

    # ---- FINAL STATE ----
    state = pd.Series("BEAR", index=prices.index)
    state.loc[bull] = "BULL"
    state.loc[crash] = "CRASH"

    out = pd.DataFrame(
        {
            "bull_flag": state == "BULL",
            "bear_flag": state == "BEAR",
            "crash_flag": state == "CRASH",
            "state": state,
        },
        index=prices.index,
    )

    if debug:
        print("[STATE-DEBUG] state_counts:", out["state"].value_counts(dropna=False).to_dict())
        if crash_enabled:
            print(f"[STATE-DEBUG] crash_true={int(out['crash_flag'].sum())}")
        if fast_enabled:
            print(
                f"[STATE-DEBUG] crash_fast: "
                f"lb={fast.get('lookback_days')} thr={fast.get('threshold')}"
            )
        if bear_fast_enabled:
            print(
                f"[STATE-DEBUG] bear_fast_true={int(bear_fast.sum())} "
                f"lb={bear_fast_cfg.get('lookback_days')} "
                f"thr={bear_fast_cfg.get('threshold')}"
            )

    return out


def _min_hold_filter(bull_flag: pd.Series, min_hold_days: int) -> pd.Series:
    """
    Run-length enforcement:
      once flag changes, keep previous flag for min_hold_days.
    """
    bull_flag = bull_flag.astype(bool).copy()
    vals = bull_flag.values
    if len(vals) == 0:
        return bull_flag

    out = vals.copy()
    last = out[0]
    hold = 0

    for i in range(1, len(out)):
        if out[i] == last:
            hold = 0
        else:
            hold += 1
            if hold <= min_hold_days:
                out[i] = last
            else:
                last = out[i]
                hold = 0

    return pd.Series(out, index=bull_flag.index)