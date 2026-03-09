from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.state import compute_state_flags


TRADE_COLS = ["TQQQ_MIX", "UPRO_MIX", "SOXL_MIX", "SGOV_MIX"]


def _risk_off_weights(mode: str) -> dict:
    if mode == "SHY_100":
        return {"SHY": 1.0}
    if mode == "BIL_100":
        return {"BIL_MIX": 1.0}
    if mode == "SGOV_100":
        return {"SGOV_MIX": 1.0}

    if mode == "SHY_GLD_50_50":
        return {"SHY": 0.5, "GLD": 0.5}
    if mode == "SHY_70_GLD_30":
        return {"SHY": 0.7, "GLD": 0.3}
    if mode == "GLD_100":
        return {"GLD": 1.0}

    if mode == "SH_100":
        return {"SH_MIX": 1.0}
    if mode == "PSQ_100":
        return {"PSQ_MIX": 1.0}

    return {"SHY": 1.0}


def _trend_trade_col(ticker: str, leverage_mode: str) -> str:
    m = (leverage_mode or "3x").lower().strip()

    if m in ("1x", "spot", "cash", "unlevered"):
        return ticker

    if m in ("2x", "proxy_2x", "mix_2x", "lever_2x"):
        if ticker == "QQQ":
            return "QLD_MIX"
        if ticker == "SPY":
            return "SSO_MIX"
        if ticker == "SOXX":
            return "USD_MIX"
        return ticker

    if ticker == "QQQ":
        return "TQQQ_MIX"
    if ticker == "SPY":
        return "UPRO_MIX"
    if ticker == "SOXX":
        return "SOXL_MIX"
    return ticker


def _meanrev_universe_to_trade_col(ticker: str) -> str:
    if ticker == "QQQ":
        return "QLD_MIX"
    if ticker == "SPY":
        return "SSO_MIX"
    if ticker == "SOXX":
        return "USD_MIX"
    return ticker


def _week_end_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.to_period("W-FRI").to_timestamp("W-FRI")


def _weekly_close(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.resample("W-FRI").last()


def _period_end_trading_dates(idx: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.resample(freq).last().dropna().values)


def _normalize_weights(h: dict) -> dict:
    s = float(sum(h.values()))
    if s <= 0:
        return {}
    return {k: float(v / s) for k, v in h.items()}


def _turnover_cost_frac(w_prev: dict, w_new: dict, buy_cost: float, sell_cost: float) -> float:
    buy_cost = float(buy_cost)
    sell_cost = float(sell_cost)
    if buy_cost <= 0 and sell_cost <= 0:
        return 0.0

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
            sell_turn += (-d)

    cost_frac = buy_turn * buy_cost + sell_turn * sell_cost
    if cost_frac < 0:
        cost_frac = 0.0
    return float(cost_frac)


def _supported_target_weights(h_des: dict[str, float]) -> dict[str, float]:
    out = {"TQQQ_MIX": 0.0, "UPRO_MIX": 0.0, "SOXL_MIX": 0.0, "SGOV_MIX": 0.0}

    defensive_to_sgov = {"SHY", "BIL_MIX", "SGOV_MIX"}

    for k, v in h_des.items():
        w = float(v)
        if w <= 0:
            continue

        if k in out:
            out[k] += w
        elif k in defensive_to_sgov:
            out["SGOV_MIX"] += w
        else:
            raise ValueError(
                f"unsupported target asset in current grid repo: {k}. "
                "현재 실행기는 TQQQ_MIX/UPRO_MIX/SOXL_MIX/SGOV_MIX만 지원함."
            )

    return _normalize_weights(out) if sum(out.values()) > 0 else {"SGOV_MIX": 1.0}


def run_meta_portfolio(
    prices: pd.DataFrame,
    cfg: dict,
    *,
    buy_cost: float | None = None,
    sell_cost: float | None = None,
):
    prices = prices.copy()
    returns = prices.pct_change().fillna(0.0)

    state_df = compute_state_flags(prices, cfg)

    port_cfg = cfg.get("portfolio", {}) or {}
    reb_mode = str(port_cfg.get("rebalance", "weekly")).lower().strip()
    when_mode = str(port_cfg.get("when", "week_end")).lower().strip()

    asset_crash_cfg = cfg.get("asset_crash", {}) or {}

    def _asset_rule(name_upper: str) -> dict:
        sub = asset_crash_cfg.get(name_upper.lower(), {}) or asset_crash_cfg.get(name_upper, {}) or {}
        return {
            "enabled": bool(sub.get("enabled", False)),
            "lookback_days": int(sub.get("lookback_days", 6)),
            "threshold": float(sub.get("threshold", -0.06)),
        }

    asset_crash_rules = {
        "SPY": _asset_rule("SPY"),
        "QQQ": _asset_rule("QQQ"),
        "SOXX": _asset_rule("SOXX"),
    }

    asset_crash_map = {
        "UPRO_MIX": "SPY",
        "SSO_MIX": "SPY",
        "SPY": "SPY",
        "TQQQ_MIX": "QQQ",
        "QLD_MIX": "QQQ",
        "QQQ": "QQQ",
        "SOXL_MIX": "SOXX",
        "USD_MIX": "SOXX",
        "SOXX": "SOXX",
    }

    _under_rets = {}
    for under, rule in asset_crash_rules.items():
        if rule["enabled"] and under in prices.columns:
            lb = int(rule["lookback_days"])
            _under_rets[(under, lb)] = prices[under].pct_change(lb).shift(1)

    rb_cfg = cfg.get("recovery_boost", {}) or {}
    rb_enabled = bool(rb_cfg.get("enabled", False))
    rb_dd_enter = float(rb_cfg.get("dd_enter", -0.20))
    rb_qqq_ma_days = int(rb_cfg.get("qqq_ma_days", 20))
    rb_qqq_mom_days = int(rb_cfg.get("qqq_mom_days", 20))

    rb_from_assets_raw = rb_cfg.get("from_assets", ["SGOV_MIX", "QLD_MIX"])

    def _flatten_str_list(x):
        if isinstance(x, str):
            return [x]
        if isinstance(x, (list, tuple)):
            out = []
            for item in x:
                out.extend(_flatten_str_list(item))
            return out
        return [str(x)]

    rb_from_assets = [str(x) for x in _flatten_str_list(rb_from_assets_raw) if str(x)]
    rb_to_asset = str(rb_cfg.get("to_asset", "TQQQ_MIX"))

    qqq_ma = None
    qqq_mom = None
    if "QQQ" in prices.columns:
        qqq_ma = prices["QQQ"].rolling(rb_qqq_ma_days).mean()
        qqq_mom = prices["QQQ"].pct_change(rb_qqq_mom_days)

    saf_cfg = cfg.get("soxx_admission_filter", {}) or {}
    saf_enabled = bool(saf_cfg.get("enabled", False))
    saf_ret20_positive_required = bool(saf_cfg.get("ret20_positive_required", True))
    saf_ret20_days = int(saf_cfg.get("ret20_days", 20))
    saf_ma20_above_ma50_required = bool(saf_cfg.get("ma20_above_ma50_required", True))
    saf_ma20_days = int(saf_cfg.get("ma20_days", 20))
    saf_ma50_days = int(saf_cfg.get("ma50_days", 50))
    saf_replacement_if_blocked = str(saf_cfg.get("replacement_if_blocked", "USD_MIX"))
    saf_apply_only_rank1 = bool(saf_cfg.get("apply_only_rank1", True))

    soxx_ret20 = None
    soxx_ma20 = None
    soxx_ma50 = None
    if "SOXX" in prices.columns:
        soxx_ret20 = prices["SOXX"].pct_change(saf_ret20_days)
        soxx_ma20 = prices["SOXX"].rolling(saf_ma20_days).mean()
        soxx_ma50 = prices["SOXX"].rolling(saf_ma50_days).mean()

    sea_cfg = cfg.get("sgov_exit_assist", {}) or {}
    sea_enabled = bool(sea_cfg.get("enabled", False))
    sea_apply_only_bull = bool(sea_cfg.get("apply_only_bull", False))
    sea_qqq_ma_days = int(sea_cfg.get("qqq_ma_days", 50))
    sea_spy_ma_days = int(sea_cfg.get("spy_ma_days", 50))
    sea_require_positive_mom = bool(sea_cfg.get("require_positive_mom", False))

    qqq_exit_ma = prices["QQQ"].rolling(sea_qqq_ma_days).mean() if "QQQ" in prices.columns else None
    spy_exit_ma = prices["SPY"].rolling(sea_spy_ma_days).mean() if "SPY" in prices.columns else None

    costs_cfg = cfg.get("costs", {}) or {}
    if buy_cost is None:
        buy_cost = float(costs_cfg.get("buy", 0.0))
    else:
        buy_cost = float(buy_cost)
    if sell_cost is None:
        sell_cost = float(costs_cfg.get("sell", 0.0))
    else:
        sell_cost = float(sell_cost)

    trend_cfg = cfg.get("trend_engine", {}) or {}
    mom_lb = int(trend_cfg.get("mom_lookback_days", 168))
    candidates = trend_cfg.get("candidates", trend_cfg.get("universe", ["QQQ", "SPY", "SOXX"]))
    if len(candidates) == 1 and isinstance(candidates[0], list):
        candidates = candidates[0]

    top_n = int(trend_cfg.get("top_n", 1))
    trend_leverage_mode = str(trend_cfg.get("leverage_mode", "proxy_3x"))

    mom = prices.pct_change(mom_lb)

    mr_cfg = cfg.get("meanrev_engine", {}) or {}
    mr_lb = int(mr_cfg.get("lookback_days", 20))
    mr_drop = float(mr_cfg.get("drop_threshold", -0.12))
    mr_hold = int(mr_cfg.get("hold_days", 5))
    mr_tp = float(mr_cfg.get("take_profit", 0.08))
    mr_sl = float(mr_cfg.get("stop_loss", -0.08))
    mr_candidates = mr_cfg.get("candidates", mr_cfg.get("universe", ["QQQ", "SPY", "SOXX"]))
    if len(mr_candidates) == 1 and isinstance(mr_candidates[0], list):
        mr_candidates = mr_candidates[0]
    mr_base = mr_cfg.get("base", mr_cfg.get("base_ticker", "QQQ"))
    if isinstance(mr_base, list):
        mr_base = mr_base[0] if mr_base else "QQQ"
    if mr_base not in mr_candidates:
        mr_base = "QQQ"

    alloc = cfg["allocator"]
    risk_off_mode = (cfg.get("risk_off", {}) or {}).get("mode", "SHY_100")
    df_w = _risk_off_weights(risk_off_mode)
    df_w = _normalize_weights(df_w) if df_w else {"SHY": 1.0}

    defensive_keys = set(df_w.keys()) | {"SHY", "BIL_MIX", "SGOV_MIX"}

    soxx_gate = cfg.get("soxx_gate", {}) or {}
    soxx_gate_enabled = bool(soxx_gate.get("enabled", False))
    soxx_gate_mode = str(soxx_gate.get("mode", "mom")).lower().strip()
    soxx_gate_mom_lb = int(soxx_gate.get("mom_lookback_days", 20))
    soxx_gate_mom_thr = float(soxx_gate.get("mom_threshold", 0.0))
    soxx_gate_ma_days = int(soxx_gate.get("ma_days", 50))

    def soxx_allowed(dt: pd.Timestamp) -> tuple[bool, float]:
        if "SOXX" not in prices.columns:
            return (False, float("nan"))
        s = prices["SOXX"].astype(float)

        if soxx_gate_mode == "ma":
            ma = s.rolling(soxx_gate_ma_days).mean()
            diff = (s - ma).shift(1).loc[dt]
            if pd.isna(diff):
                return (False, float("nan"))
            return (bool(diff > 0.0), float(diff))

        m = s.pct_change(soxx_gate_mom_lb).shift(1).loc[dt]
        if pd.isna(m):
            return (False, float("nan"))
        return (bool(float(m) > soxx_gate_mom_thr), float(m))

    week_end = _week_end_index(prices.index)
    fridays = prices.index[prices.index.isin(week_end.unique())]

    supported_when_modes = {"week_end"}
    if when_mode not in supported_when_modes:
        raise ValueError(
            f"unsupported portfolio.when: {when_mode}. "
            f"supported={sorted(supported_when_modes)}"
        )

    if reb_mode == "weekly":
        reb_dates = fridays
    elif reb_mode == "biweekly":
        reb_dates = fridays[::2]
    elif reb_mode == "monthly":
        reb_dates = _period_end_trading_dates(prices.index, "ME")
    elif reb_mode == "quarterly":
        reb_dates = _period_end_trading_dates(prices.index, "QE")
    else:
        reb_dates = fridays

    wclose = _weekly_close(prices)

    def get_week_ret(ticker_col: str, wk_end: pd.Timestamp) -> float:
        if ticker_col not in wclose.columns or wk_end not in wclose.index:
            return np.nan
        loc = wclose.index.get_loc(wk_end)
        if loc == 0:
            return np.nan
        prev = wclose.iloc[loc - 1][ticker_col]
        cur = wclose.iloc[loc][ticker_col]
        if pd.isna(prev) or prev == 0 or pd.isna(cur):
            return np.nan
        return float(cur / prev - 1.0)

    current_trend_tradecols: list[str] = []
    current_trend_underlyings: list[str] = []
    picks_rows = []

    mr_active = False
    mr_entry_price = None
    mr_days = 0
    mr_trade_col = None

    h_cur = {"SHY": 1.0}
    equity = 1.0
    curve = []
    engine_choice_log = []
    holdings_daily_rows = []
    target_rows = []

    equity_peak = equity
    idx = prices.index

    for i, dt in enumerate(idx):
        st = state_df.loc[dt, "state"]

        for t, w in h_cur.items():
            holdings_daily_rows.append(
                {"date": str(dt.date()), "ticker": t, "weight": float(w), "state": st}
            )

        daily_ret = 0.0
        for t, w in h_cur.items():
            if t in returns.columns:
                daily_ret += float(returns.loc[dt, t]) * float(w)

        equity *= (1.0 + daily_ret)
        equity_peak = max(equity_peak, equity)
        equity_dd = float(equity / equity_peak - 1.0) if equity_peak > 0 else 0.0

        rebalance_today = bool(dt in reb_dates)

        soxx_gate_applied = False
        soxx_gate_blocked = False
        soxx_gate_value = float("nan")

        asset_crash_hit = False
        asset_crash_under = ""
        asset_crash_value = float("nan")
        asset_crash_lb_used = np.nan
        asset_crash_thr_used = np.nan

        held = list(h_cur.keys())
        for trade_col in held:
            under = asset_crash_map.get(trade_col, "")
            if not under:
                continue

            rule = asset_crash_rules.get(under, {})
            if not rule or (not rule["enabled"]):
                continue

            lb = int(rule["lookback_days"])
            thr = float(rule["threshold"])
            s = _under_rets.get((under, lb))
            if s is None:
                continue

            v = s.loc[dt]
            if pd.notna(v) and float(v) <= thr:
                asset_crash_hit = True
                asset_crash_under = under
                asset_crash_value = float(v)
                asset_crash_lb_used = lb
                asset_crash_thr_used = thr
                break

        if rebalance_today:
            m = mom.loc[dt].reindex(candidates)
            ranked = m.dropna().sort_values(ascending=False)
            top = list(ranked.index[:top_n]) if not ranked.empty else []

            if soxx_gate_enabled and len(top) > 0 and top[0] == "SOXX":
                soxx_gate_applied = True
                ok, val = soxx_allowed(dt)
                soxx_gate_value = val
                if not ok:
                    soxx_gate_blocked = True
                    ranked2 = ranked.drop(index=["SOXX"], errors="ignore")
                    top = list(ranked2.index[:top_n])

            current_trend_underlyings = [t for t in top if t is not None]
            current_trend_tradecols = [_trend_trade_col(t, trend_leverage_mode) for t in current_trend_underlyings]

            wk = _week_end_index(pd.DatetimeIndex([dt]))[0]
            top1 = top[0] if len(top) > 0 else None
            top2 = top[1] if len(top) > 1 else (ranked.index[1] if len(ranked) > 1 else None)

            row = {
                "week_end": str(wk.date()),
                "rank1": top1,
                "rank2": top2,
                "rank1_trade": _trend_trade_col(top1, trend_leverage_mode) if top1 else None,
                "rank2_trade": _trend_trade_col(top2, trend_leverage_mode) if top2 else None,
                "score1_mom": float(ranked.loc[top1]) if (top1 in ranked.index) else np.nan,
                "score2_mom": float(ranked.loc[top2]) if (top2 in ranked.index) else np.nan,
                "rebalance_mode": reb_mode,
                "rebalance_today": True,
                "trend_leverage_mode": trend_leverage_mode,
                "soxx_gate_enabled": bool(soxx_gate_enabled),
                "soxx_gate_applied": bool(soxx_gate_applied),
                "soxx_gate_blocked": bool(soxx_gate_blocked),
                "soxx_gate_mode": soxx_gate_mode,
                "soxx_gate_value": (float(soxx_gate_value) if pd.notna(soxx_gate_value) else np.nan),
                "asset_crash_hit": bool(asset_crash_hit),
                "asset_crash_under": asset_crash_under,
                "asset_crash_value": (float(asset_crash_value) if pd.notna(asset_crash_value) else np.nan),
                "asset_crash_lb_used": asset_crash_lb_used,
                "asset_crash_thr_used": asset_crash_thr_used,
            }
            row["rank1_week_ret"] = get_week_ret(row["rank1_trade"], wk) if row["rank1_trade"] else np.nan
            row["rank2_week_ret"] = get_week_ret(row["rank2_trade"], wk) if row["rank2_trade"] else np.nan
            picks_rows.append(row)

        mr_price_under = prices[mr_base] if mr_base in prices.columns else None
        if mr_price_under is None or mr_price_under.isna().all():
            mr_active = False
            mr_trade_col = None
            mr_entry_price = None
            mr_days = 0
        else:
            r_lb = mr_price_under.pct_change(mr_lb).shift(1).loc[dt]
            if (not mr_active) and pd.notna(r_lb) and (float(r_lb) <= mr_drop):
                mr_active = True
                mr_trade_col = _meanrev_universe_to_trade_col(mr_base)
                mr_entry_price = prices[mr_trade_col].loc[dt] if mr_trade_col in prices.columns else None
                mr_days = 0

            if mr_active:
                mr_days += 1
                cur_px = prices[mr_trade_col].loc[dt] if (mr_trade_col in prices.columns) else np.nan
                if mr_entry_price is None or pd.isna(mr_entry_price) or pd.isna(cur_px):
                    mr_active = False
                    mr_trade_col = None
                    mr_entry_price = None
                    mr_days = 0
                else:
                    pnl = float(cur_px / mr_entry_price - 1.0)
                    if (mr_days >= mr_hold) or (pnl >= mr_tp) or (pnl <= mr_sl):
                        mr_active = False
                        mr_trade_col = None
                        mr_entry_price = None
                        mr_days = 0

        st_key = st.lower()
        w_tr = float(alloc[st_key]["trend"])
        w_mr = float(alloc[st_key]["meanrev"])
        w_df = float(alloc[st_key]["defensive"])

        if asset_crash_hit:
            w_tr = 0.0
            w_mr = 0.0
            w_df = 1.0
            st_key_effective = "asset_crash"
        else:
            st_key_effective = st_key

        h_des: dict[str, float] = {}

        if w_tr > 0 and len(current_trend_tradecols) > 0:
            per = w_tr / len(current_trend_tradecols)
            for tcol in current_trend_tradecols:
                h_des[tcol] = h_des.get(tcol, 0.0) + per

        if w_mr > 0:
            if mr_active and (mr_trade_col in returns.columns):
                h_des[mr_trade_col] = h_des.get(mr_trade_col, 0.0) + w_mr
            else:
                h_des["SHY"] = h_des.get("SHY", 0.0) + w_mr

        if w_df > 0:
            for t, w in df_w.items():
                h_des[t] = h_des.get(t, 0.0) + (w_df * float(w))

        saf_hit = False
        saf_reason = ""
        saf_ret20_prev = float("nan")
        saf_ma20_prev = float("nan")
        saf_ma50_prev = float("nan")

        if saf_enabled and ("SOXL_MIX" in h_des) and ("SOXX" in prices.columns):
            apply_ok = True
            if saf_apply_only_rank1:
                apply_ok = len(current_trend_underlyings) > 0 and current_trend_underlyings[0] == "SOXX"

            block = False
            if apply_ok:
                if saf_ret20_positive_required and soxx_ret20 is not None:
                    ret20_prev = soxx_ret20.shift(1).loc[dt]
                    saf_ret20_prev = float(ret20_prev) if pd.notna(ret20_prev) else np.nan
                    if pd.isna(ret20_prev) or not (float(ret20_prev) > 0.0):
                        block = True
                        saf_reason = "ret20_not_positive"

                if (not block) and saf_ma20_above_ma50_required and (soxx_ma20 is not None) and (soxx_ma50 is not None):
                    ma20_prev = soxx_ma20.shift(1).loc[dt]
                    ma50_prev = soxx_ma50.shift(1).loc[dt]
                    saf_ma20_prev = float(ma20_prev) if pd.notna(ma20_prev) else np.nan
                    saf_ma50_prev = float(ma50_prev) if pd.notna(ma50_prev) else np.nan
                    if pd.isna(ma20_prev) or pd.isna(ma50_prev) or not (float(ma20_prev) > float(ma50_prev)):
                        block = True
                        saf_reason = "ma20_not_above_ma50"

            if block:
                w_soxl = float(h_des.pop("SOXL_MIX", 0.0))
                if w_soxl > 0:
                    h_des[saf_replacement_if_blocked] = h_des.get(saf_replacement_if_blocked, 0.0) + w_soxl
                    saf_hit = True

        rb_hit = False
        rb_price_prev = float("nan")
        rb_ma_prev = float("nan")
        rb_mom_prev = float("nan")
        rb_from_asset_hit = ""

        if rb_enabled and ("QQQ" in prices.columns) and (qqq_ma is not None) and (qqq_mom is not None):
            px_prev = prices["QQQ"].shift(1).loc[dt]
            ma_prev = qqq_ma.shift(1).loc[dt]
            mom_prev = qqq_mom.shift(1).loc[dt]

            qqq_recovery_ok = (
                pd.notna(px_prev)
                and pd.notna(ma_prev)
                and pd.notna(mom_prev)
                and (float(px_prev) > float(ma_prev))
                and (float(mom_prev) > 0.0)
            )

            if equity_dd <= rb_dd_enter and qqq_recovery_ok:
                for from_asset in rb_from_assets:
                    w_from = float(h_des.get(from_asset, 0.0))
                    if w_from > 1e-12:
                        h_des.pop(from_asset, None)
                        h_des[rb_to_asset] = h_des.get(rb_to_asset, 0.0) + w_from
                        rb_hit = True
                        rb_from_asset_hit = from_asset
                        rb_price_prev = float(px_prev)
                        rb_ma_prev = float(ma_prev)
                        rb_mom_prev = float(mom_prev)

        sea_hit = False
        sea_selected_under = ""
        sea_selected_trade = ""
        sea_qqq_px_prev = float("nan")
        sea_qqq_ma_prev = float("nan")
        sea_spy_px_prev = float("nan")
        sea_spy_ma_prev = float("nan")

        if sea_enabled and not asset_crash_hit:
            state_ok = True
            if sea_apply_only_bull:
                state_ok = (st.lower() == "bull")

            if state_ok and len(h_des) > 0 and set(h_des.keys()).issubset(defensive_keys):
                assist_candidates = []

                if "QQQ" in prices.columns and qqq_exit_ma is not None:
                    q_px_prev = prices["QQQ"].shift(1).loc[dt]
                    q_ma_prev = qqq_exit_ma.shift(1).loc[dt]
                    sea_qqq_px_prev = float(q_px_prev) if pd.notna(q_px_prev) else np.nan
                    sea_qqq_ma_prev = float(q_ma_prev) if pd.notna(q_ma_prev) else np.nan
                    if pd.notna(q_px_prev) and pd.notna(q_ma_prev) and (float(q_px_prev) > float(q_ma_prev)):
                        q_score = mom.loc[dt, "QQQ"] if "QQQ" in mom.columns else np.nan
                        if (not sea_require_positive_mom) or (pd.notna(q_score) and float(q_score) > 0.0):
                            assist_candidates.append(("QQQ", float(q_score) if pd.notna(q_score) else -1e18))

                if "SPY" in prices.columns and spy_exit_ma is not None:
                    s_px_prev = prices["SPY"].shift(1).loc[dt]
                    s_ma_prev = spy_exit_ma.shift(1).loc[dt]
                    sea_spy_px_prev = float(s_px_prev) if pd.notna(s_px_prev) else np.nan
                    sea_spy_ma_prev = float(s_ma_prev) if pd.notna(s_ma_prev) else np.nan
                    if pd.notna(s_px_prev) and pd.notna(s_ma_prev) and (float(s_px_prev) > float(s_ma_prev)):
                        s_score = mom.loc[dt, "SPY"] if "SPY" in mom.columns else np.nan
                        if (not sea_require_positive_mom) or (pd.notna(s_score) and float(s_score) > 0.0):
                            assist_candidates.append(("SPY", float(s_score) if pd.notna(s_score) else -1e18))

                if len(assist_candidates) > 0:
                    assist_candidates = sorted(assist_candidates, key=lambda x: x[1], reverse=True)
                    chosen_under = assist_candidates[0][0]
                    chosen_trade = _trend_trade_col(chosen_under, trend_leverage_mode)

                    h_des = {chosen_trade: 1.0}
                    sea_hit = True
                    sea_selected_under = chosen_under
                    sea_selected_trade = chosen_trade

        h_des = _normalize_weights(h_des)
        if not h_des:
            h_des = {"SHY": 1.0}

        target_row = {"date": dt}
        target_row.update(_supported_target_weights(h_des))
        target_rows.append(target_row)

        turnover_sum_abs = 0.0
        cost_frac = 0.0
        traded = False

        if i < len(idx) - 1:
            keys = set(h_cur.keys()) | set(h_des.keys())
            for k in keys:
                turnover_sum_abs += abs(float(h_des.get(k, 0.0)) - float(h_cur.get(k, 0.0)))
            traded = turnover_sum_abs > 1e-12

            cost_frac = _turnover_cost_frac(h_cur, h_des, buy_cost, sell_cost)
            if cost_frac > 0:
                equity *= (1.0 - cost_frac)

            h_cur = h_des

        curve.append(equity)

        engine_choice_log.append(
            {
                "date": str(dt.date()),
                "state": st,
                "state_effective": st_key_effective,
                "equity": float(equity),
                "equity_peak": float(equity_peak),
                "equity_dd": float(equity_dd),
                "w_trend": w_tr,
                "w_meanrev": w_mr,
                "w_defensive": w_df,
                "meanrev_active": bool(mr_active),
                "meanrev_ticker": mr_trade_col if mr_active else "",
                "rebalance_today": bool(rebalance_today),
                "rebalance_mode": reb_mode,
                "trend_leverage_mode": trend_leverage_mode,
                "turnover_sum_abs": float(turnover_sum_abs),
                "turnover_one_way": float(turnover_sum_abs) * 0.5,
                "cost_buy": float(buy_cost),
                "cost_sell": float(sell_cost),
                "cost_frac": float(cost_frac),
                "traded": bool(traded),
                "asset_crash_hit": bool(asset_crash_hit),
                "asset_crash_under": asset_crash_under,
                "asset_crash_value": (float(asset_crash_value) if pd.notna(asset_crash_value) else np.nan),
                "asset_crash_lb_used": asset_crash_lb_used,
                "asset_crash_thr_used": asset_crash_thr_used,
                "recovery_boost_hit": bool(rb_hit),
                "recovery_boost_from_asset": rb_from_asset_hit,
                "recovery_boost_to_asset": rb_to_asset if rb_hit else "",
                "recovery_boost_dd_enter": float(rb_dd_enter),
                "recovery_boost_qqq_price_prev": (float(rb_price_prev) if pd.notna(rb_price_prev) else np.nan),
                "recovery_boost_qqq_ma_prev": (float(rb_ma_prev) if pd.notna(rb_ma_prev) else np.nan),
                "recovery_boost_qqq_mom_prev": (float(rb_mom_prev) if pd.notna(rb_mom_prev) else np.nan),
                "soxx_admission_filter_hit": bool(saf_hit),
                "soxx_admission_filter_reason": saf_reason,
                "soxx_admission_filter_ret20_prev": saf_ret20_prev,
                "soxx_admission_filter_ma20_prev": saf_ma20_prev,
                "soxx_admission_filter_ma50_prev": saf_ma50_prev,
                "soxx_admission_filter_replacement": saf_replacement_if_blocked if saf_hit else "",
                "sgov_exit_assist_hit": bool(sea_hit),
                "sgov_exit_assist_selected_under": sea_selected_under,
                "sgov_exit_assist_selected_trade": sea_selected_trade,
                "sgov_exit_assist_qqq_px_prev": sea_qqq_px_prev,
                "sgov_exit_assist_qqq_ma_prev": sea_qqq_ma_prev,
                "sgov_exit_assist_spy_px_prev": sea_spy_px_prev,
                "sgov_exit_assist_spy_ma_prev": sea_spy_ma_prev,
            }
        )

    equity_series = pd.Series(curve, index=prices.index, name="equity")
    picks_df = pd.DataFrame(picks_rows)
    holdings_daily = pd.DataFrame(holdings_daily_rows)

    if not holdings_daily.empty:
        hd = holdings_daily.copy()
        hd["date"] = pd.to_datetime(hd["date"])
        mat = hd.pivot_table(index="date", columns="ticker", values="weight", aggfunc="sum").fillna(0.0)
        mat["week_end"] = _week_end_index(mat.index)
        wk_mean = mat.groupby("week_end").mean().drop(columns=["week_end"], errors="ignore")
        wk_avg = wk_mean.reset_index().melt(id_vars=["week_end"], var_name="ticker", value_name="avg_weight")
        wk_avg = wk_avg[wk_avg["avg_weight"] > 0]
    else:
        wk_avg = pd.DataFrame(columns=["week_end", "ticker", "avg_weight"])

    def wk_ret_col(ticker: str, wk_end_ts: pd.Timestamp) -> float:
        if ticker not in wclose.columns or wk_end_ts not in wclose.index:
            return np.nan
        loc = wclose.index.get_loc(wk_end_ts)
        if loc == 0:
            return np.nan
        prev = wclose.iloc[loc - 1][ticker]
        cur = wclose.iloc[loc][ticker]
        if pd.isna(prev) or prev == 0 or pd.isna(cur):
            return np.nan
        return float(cur / prev - 1.0)

    if not wk_avg.empty:
        wk_avg["week_ret"] = wk_avg.apply(lambda r: wk_ret_col(r["ticker"], pd.Timestamp(r["week_end"])), axis=1)
        wk_avg["contrib"] = wk_avg["avg_weight"] * wk_avg["week_ret"]
        wk_avg["week_end"] = wk_avg["week_end"].astype(str)

    holdings_weekly = wk_avg
    target_df = pd.DataFrame(target_rows)
    return equity_series, engine_choice_log, picks_df, holdings_daily, holdings_weekly, target_df


def build_meta_targets(
    prices: pd.DataFrame,
    cfg: dict,
    *,
    buy_cost: float | None = None,
    sell_cost: float | None = None,
) -> pd.DataFrame:
    _, _, _, _, _, target_df = run_meta_portfolio(
        prices,
        cfg,
        buy_cost=buy_cost,
        sell_cost=sell_cost,
    )
    return target_df.copy()