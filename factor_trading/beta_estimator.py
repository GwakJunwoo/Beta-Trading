"""2팩터 (3Y + 10Y) rolling OLS β 추정 (명세 §2).

회귀식: ``dY_i,t = α_i + β^3Y_i,t · dY_3Y,t + β^10Y_i,t · dY_10Y,t + ε_i,t``
Window = 63영업일 (3M), min_periods = 40.

지표 롤오버 영업일(rollover_flag)은 회귀 인풋에서 제외.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm


WINDOW = 63
MIN_PERIODS = 40


def rolling_beta_2f(
    dy_inst: pd.Series,
    dy_3y:   pd.Series,
    dy_10y:  pd.Series,
    rollover: pd.Series | None = None,
    window: int = WINDOW,
    min_periods: int = MIN_PERIODS,
) -> pd.DataFrame:
    """단일 종목 2-factor rolling β. Returns DataFrame[beta_3y, beta_10y, alpha].

    sample 부족·numerical 이상 등 어떤 에러든 발생 시 NaN frame 반환 — 한 종목 실패가
    전체 universe 회귀를 막지 않도록 광범위 except 로 보호.
    """
    nan_out = pd.DataFrame(np.nan, index=dy_inst.index,
                            columns=["beta_3y", "beta_10y", "alpha"])
    try:
        df = pd.concat([dy_inst.rename("y"),
                        dy_3y.reindex(dy_inst.index).rename("x3"),
                        dy_10y.reindex(dy_inst.index).rename("x10")], axis=1)
        if rollover is not None:
            mask = rollover.reindex(df.index, fill_value=False).astype(bool)
            df.loc[mask, :] = np.nan
        valid = df.dropna()
        # window 보다 sample 적으면 RollingOLS 내부 인덱싱 깨짐 → skip.
        if len(valid) < window:
            return nan_out
        X = sm.add_constant(valid[["x3", "x10"]].values)
        res = RollingOLS(valid["y"].values, X, window=window,
                         min_nobs=min_periods, expanding=False).fit()
        params = res.params
        out = nan_out.copy()
        out.loc[valid.index, "alpha"]    = params[:, 0]
        out.loc[valid.index, "beta_3y"]  = params[:, 1]
        out.loc[valid.index, "beta_10y"] = params[:, 2]
        return out
    except Exception:
        return nan_out


def estimate_all_betas_2f(
    dy_panel: pd.DataFrame,
    dy_3y:   pd.Series,
    dy_10y:  pd.Series,
    rollover: pd.Series | None = None,
    window: int = WINDOW,
    min_periods: int = MIN_PERIODS,
) -> dict[str, pd.DataFrame]:
    """전체 유니버스 2-factor β. Returns {'beta_3y', 'beta_10y', 'alpha'}."""
    idx = dy_panel.index
    b3  = pd.DataFrame(np.nan, index=idx, columns=dy_panel.columns)
    b10 = pd.DataFrame(np.nan, index=idx, columns=dy_panel.columns)
    a0  = pd.DataFrame(np.nan, index=idx, columns=dy_panel.columns)
    for code in dy_panel.columns:
        try:
            out = rolling_beta_2f(dy_panel[code], dy_3y, dy_10y,
                                  rollover=rollover, window=window, min_periods=min_periods)
            b3[code]  = out["beta_3y"]
            b10[code] = out["beta_10y"]
        except Exception:
            # 종목 단위 안전망: 실패 시 NaN 유지
            pass
        a0[code]  = out["alpha"]
    return {"beta_3y": b3, "beta_10y": b10, "alpha": a0}


def sanity_check(
    betas: dict[str, pd.DataFrame],
    bench_3y: str | None,
    bench_10y: str | None,
    meta: pd.DataFrame | None = None,
    remain: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """명세 §2.1 sanity: 3Y 지표 β^3Y≈1, β^10Y≈0 / 10Y 지표 β^10Y≈1, β^3Y≈0 / 5Y 중기물 둘 다 0.3~0.7."""
    b3, b10 = betas["beta_3y"], betas["beta_10y"]
    rows = []
    if bench_3y and bench_3y in b3.columns:
        rows.append({"bond_code": bench_3y, "role": "3Y지표",
                     "β3Y_mean": float(b3[bench_3y].mean(skipna=True)),
                     "β10Y_mean": float(b10[bench_3y].mean(skipna=True))})
    if bench_10y and bench_10y in b3.columns:
        rows.append({"bond_code": bench_10y, "role": "10Y지표",
                     "β3Y_mean": float(b3[bench_10y].mean(skipna=True)),
                     "β10Y_mean": float(b10[bench_10y].mean(skipna=True))})
    # 5Y 근처(3~7년) 평균
    if remain is not None:
        r_avg = remain.mean(axis=0)
        mid = r_avg[(r_avg >= 3) & (r_avg <= 7)].index
        if len(mid):
            rows.append({"bond_code": f"(5Y 중기물 {len(mid)}개 평균)",
                         "role": "5Y중기물",
                         "β3Y_mean":  float(b3[mid].mean().mean()),
                         "β10Y_mean": float(b10[mid].mean().mean())})
    return pd.DataFrame(rows)
