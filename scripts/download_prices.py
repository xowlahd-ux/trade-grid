from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import yfinance as yf


AUTO_END_TOKENS = {"", "auto", "latest", "today", "max"}


def resolve_requested_end_date(end_date: str) -> str:
    text = str(end_date or "").strip().lower()
    if text in AUTO_END_TOKENS:
        return pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d")

    try:
        return pd.Timestamp(end_date).strftime("%Y-%m-%d")
    except Exception as e:  # pragma: no cover - defensive error message
        raise ValueError(f"invalid end_date: {end_date!r}") from e


def _pick_price_table(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.levels[0]
        if "Adj Close" in lvl0:
            px = df["Adj Close"].copy()
        elif "Close" in lvl0:
            px = df["Close"].copy()
        else:
            px = df.xs(lvl0[0], level=0, axis=1).copy()
    else:
        if "Adj Close" in df.columns:
            px = df[["Adj Close"]].copy()
        elif "Close" in df.columns:
            px = df[["Close"]].copy()
        else:
            raise ValueError("Neither Adj Close nor Close found.")

    px = px.dropna(how="all").sort_index().ffill()
    return px


def _download_raw_prices(start_date: str, end_date: str) -> pd.DataFrame:
    base_tickers = ["QQQ", "SPY", "SOXX"]
    real_3x = ["TQQQ", "UPRO", "SOXL"]
    real_2x = ["QLD", "SSO", "USD"]
    extra_real = ["BIL", "SGOV", "SH", "PSQ", "SHY"]

    tickers = sorted(set(base_tickers + real_3x + real_2x + extra_real))

    last_err = None
    df = None

    resolved_end_date = resolve_requested_end_date(end_date)
    end_plus = (pd.Timestamp(resolved_end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    for attempt in range(1, 4):
        try:
            df = yf.download(
                tickers=tickers,
                start=start_date,
                end=end_plus,
                auto_adjust=False,
                progress=False,
                group_by="column",
                threads=False,
            )
            if df is not None and not df.empty:
                break
        except Exception as e:
            last_err = e
        time.sleep(2 * attempt)

    if df is None or df.empty:
        raise ValueError(f"Downloaded price data is empty. last_err={last_err!r}")

    return _pick_price_table(df)


def _make_lever_proxy(px: pd.DataFrame, base: str, name: str, k: float) -> None:
    if base not in px.columns:
        return

    base_px = px[base].astype(float)
    first_valid = base_px.first_valid_index()
    if first_valid is None:
        return

    r = base_px.pct_change().fillna(0.0)
    step = (1.0 + k * r).clip(lower=1e-6)
    start_val = float(base_px.loc[first_valid])
    proxy = step.cumprod() * start_val
    px[name] = proxy


def _stitch_real_over_proxy(px: pd.DataFrame, real_col: str, proxy_col: str, out_col: str) -> None:
    if proxy_col not in px.columns:
        return

    mix = px[proxy_col].copy()

    if real_col in px.columns:
        real_start = px[real_col].first_valid_index()
        if real_start is not None:
            if pd.notna(px.at[real_start, real_col]) and pd.notna(mix.at[real_start]):
                denom = float(mix.at[real_start])
                if denom != 0.0:
                    mix = mix * (float(px.at[real_start, real_col]) / denom)
            mix.loc[real_start:] = px.loc[real_start:, real_col]

    px[out_col] = mix


def _backfill_chain(anchor_real: pd.Series, ref_series: pd.Series) -> pd.Series:
    """
    anchor_real의 첫 실제값을 기준으로, 그 이전 구간은 ref_series 수익률로 역방향 백필.
    anchor 이후는 anchor_real 실제값 사용.
    """
    out = ref_series.copy().astype(float)
    out[:] = pd.NA

    real = anchor_real.astype(float)
    ref = ref_series.astype(float)

    real_start = real.first_valid_index()
    if real_start is None:
        return ref.copy()

    idx = ref.index
    if real_start not in idx:
        return ref.copy()

    out.loc[real_start:] = real.loc[real_start:]

    loc = idx.get_loc(real_start)
    if not isinstance(loc, int):
        return out.ffill()

    for i in range(loc - 1, -1, -1):
        d0 = idx[i]
        d1 = idx[i + 1]

        p1 = out.loc[d1]
        r1 = ref.loc[d1] / ref.loc[d0] - 1.0

        if pd.isna(p1) or pd.isna(r1):
            out.loc[d0] = pd.NA
        else:
            denom = 1.0 + float(r1)
            if denom == 0.0:
                out.loc[d0] = pd.NA
            else:
                out.loc[d0] = float(p1) / denom

    return out.ffill()


def _make_cash_chain(px: pd.DataFrame) -> None:
    if "SHY" not in px.columns:
        raise ValueError("SHY is required to build BIL_MIX / SGOV_MIX chain")

    shy = px["SHY"].astype(float)
    px["SHY_PROXY"] = shy.copy()

    if "BIL" in px.columns and px["BIL"].notna().any():
        bil_mix = _backfill_chain(px["BIL"], shy)
    else:
        bil_mix = shy.copy()

    px["BIL_PROXY"] = bil_mix.copy()
    px["BIL_MIX"] = bil_mix.copy()

    if "SGOV" in px.columns and px["SGOV"].notna().any():
        sgov_mix = _backfill_chain(px["SGOV"], px["BIL_MIX"])
    else:
        sgov_mix = px["BIL_MIX"].copy()

    px["SGOV_PROXY"] = sgov_mix.copy()
    px["SGOV_MIX"] = sgov_mix.copy()


def download_prices_and_build_proxies(start_date: str, end_date: str) -> pd.DataFrame:
    px = _download_raw_prices(start_date, end_date)

    _make_lever_proxy(px, "QQQ", "TQQQ_PROXY", 3.0)
    _make_lever_proxy(px, "SPY", "UPRO_PROXY", 3.0)
    _make_lever_proxy(px, "SOXX", "SOXL_PROXY", 3.0)
    _stitch_real_over_proxy(px, "TQQQ", "TQQQ_PROXY", "TQQQ_MIX")
    _stitch_real_over_proxy(px, "UPRO", "UPRO_PROXY", "UPRO_MIX")
    _stitch_real_over_proxy(px, "SOXL", "SOXL_PROXY", "SOXL_MIX")

    _make_lever_proxy(px, "QQQ", "QLD_PROXY", 2.0)
    _make_lever_proxy(px, "SPY", "SSO_PROXY", 2.0)
    _make_lever_proxy(px, "SOXX", "USD_PROXY", 2.0)
    _stitch_real_over_proxy(px, "QLD", "QLD_PROXY", "QLD_MIX")
    _stitch_real_over_proxy(px, "SSO", "SSO_PROXY", "SSO_MIX")
    _stitch_real_over_proxy(px, "USD", "USD_PROXY", "USD_MIX")

    _make_lever_proxy(px, "SPY", "SH_PROXY", -1.0)
    _make_lever_proxy(px, "QQQ", "PSQ_PROXY", -1.0)
    _stitch_real_over_proxy(px, "SH", "SH_PROXY", "SH_MIX")
    _stitch_real_over_proxy(px, "PSQ", "PSQ_PROXY", "PSQ_MIX")

    _make_cash_chain(px)

    preferred_cols = [
        "QQQ",
        "SPY",
        "SOXX",
        "SHY",
        "BIL",
        "SGOV",
        "TQQQ",
        "UPRO",
        "SOXL",
        "QLD",
        "SSO",
        "USD",
        "SH",
        "PSQ",
        "TQQQ_PROXY",
        "UPRO_PROXY",
        "SOXL_PROXY",
        "QLD_PROXY",
        "SSO_PROXY",
        "USD_PROXY",
        "SH_PROXY",
        "PSQ_PROXY",
        "SHY_PROXY",
        "BIL_PROXY",
        "SGOV_PROXY",
        "TQQQ_MIX",
        "UPRO_MIX",
        "SOXL_MIX",
        "QLD_MIX",
        "SSO_MIX",
        "USD_MIX",
        "SH_MIX",
        "PSQ_MIX",
        "BIL_MIX",
        "SGOV_MIX",
    ]
    existing_cols = [c for c in preferred_cols if c in px.columns]
    extra_cols = [c for c in px.columns if c not in existing_cols]
    px = px[existing_cols + extra_cols]

    required = ["QQQ", "SPY", "SOXX", "TQQQ_MIX", "UPRO_MIX", "SOXL_MIX", "SGOV_MIX"]
    missing_required = [c for c in required if c not in px.columns]
    if missing_required:
        raise ValueError(f"Missing required columns after proxy build: {missing_required}")

    first_valid = {c: px[c].first_valid_index() for c in required}
    missing_valid = [c for c, d in first_valid.items() if d is None]
    if missing_valid:
        raise ValueError(f"Required columns still all-NaN: {missing_valid}")

    common_start = max(first_valid.values())
    px = px.loc[common_start:].copy()
    px = px.dropna(subset=required, how="any")

    if px.empty:
        raise ValueError(f"Price data empty after alignment. common_start={common_start}")

    px.index.name = "date"
    return px


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    resolved_end_date = resolve_requested_end_date(args.end_date)
    px = download_prices_and_build_proxies(
        start_date=args.start_date,
        end_date=resolved_end_date,
    )
    px.to_csv(out_csv)

    print(f"saved: {out_csv}")
    print(f"rows={len(px)} cols={len(px.columns)}")
    print(f"requested_end_date={args.end_date} resolved_end_date={resolved_end_date}")
    print(f"start={px.index.min().date()} end={px.index.max().date()}")
    print(f"SGOV_MIX first_valid={px['SGOV_MIX'].first_valid_index()}")
    print(px.head(5).to_string())
    print(px.tail(5).to_string())


if __name__ == "__main__":
    main()
