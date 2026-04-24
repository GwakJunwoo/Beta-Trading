"""3팩터 합성 포트폴리오 (명세 §9).

3가지 합성 방법
----------------
1. equal_weight       : 각 팩터 σ로 표준화 후 평균  (단순)
2. risk_parity        : 1/σ 가중. 각 팩터 risk contribution 동일
3. target_vol         : 전체 포트폴리오 연율화 vol을 target에 맞춰 전체 scale

주의: RV는 21d hold 팩터, MOM/CURVE는 1d hold. 합성은 **같은 빈도 PnL series**
사이에서만 의미 있으므로 RV daily PnL을 그대로 쓰되, scale은 표준화로 보정.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# 표준화 / 가중치
# ------------------------------------------------------------------

def zscore_standardize(pnls: dict[str, pd.Series],
                       vol_window: int | None = None) -> pd.DataFrame:
    """각 팩터 PnL을 std로 나눠 unit vol 로 표준화.

    vol_window=None이면 전체기간 단일 σ, 정수면 rolling σ.
    """
    out = {}
    df = pd.DataFrame(pnls)
    for c in df.columns:
        s = df[c]
        if vol_window is None:
            sd = float(s.std(ddof=1))
            out[c] = s / sd if sd > 0 else s
        else:
            sd = s.rolling(vol_window, min_periods=max(30, vol_window // 2)).std(ddof=1)
            out[c] = s / sd.replace(0, np.nan)
    return pd.DataFrame(out)


def equal_weight(pnls: dict[str, pd.Series],
                 standardize: bool = True,
                 vol_window: int | None = None) -> pd.Series:
    """각 팩터 1/N 가중. 선택적 표준화."""
    if standardize:
        df_s = zscore_standardize(pnls, vol_window=vol_window)
    else:
        df_s = pd.DataFrame(pnls)
    return df_s.mean(axis=1, skipna=True).rename("EW")


def risk_parity(pnls: dict[str, pd.Series],
                vol_window: int | None = None) -> pd.Series:
    """1/σ 가중 (rolling σ 또는 전체 σ).

    w_i = (1/σ_i) / Σ(1/σ_j)
    weighted_pnl_t = Σ w_i,t · pnl_i,t
    """
    df = pd.DataFrame(pnls)
    if vol_window is None:
        sds = df.std(ddof=1)
        inv = 1.0 / sds.replace(0, np.nan)
        w = (inv / inv.sum()).values
        return (df.mul(w, axis=1)).sum(axis=1, min_count=1).rename("RP")
    else:
        sds = df.rolling(vol_window, min_periods=max(30, vol_window // 2)).std(ddof=1)
        inv = 1.0 / sds.replace(0, np.nan)
        w = inv.div(inv.sum(axis=1), axis=0)
        return (df * w).sum(axis=1, min_count=1).rename("RP")


def target_vol(pnl: pd.Series, target_ann_vol_bp: float,
               vol_window: int = 63) -> pd.Series:
    """전체 포트폴리오 PnL을 target 연율 vol에 맞게 scale.

    leverage_t = target / (realized_vol_t · √252)
    """
    realized = pnl.rolling(vol_window, min_periods=max(30, vol_window // 2)).std(ddof=1)
    ann = realized * np.sqrt(252.0)
    lev = target_ann_vol_bp / ann.replace(0, np.nan)
    return (pnl * lev.shift(1)).rename("TV")       # lag 1 (현실성)


# ------------------------------------------------------------------
# 종합 요약
# ------------------------------------------------------------------

def combine_summary(pnls: dict[str, pd.Series],
                    vol_window: int | None = None,
                    target_ann_vol_bp: float | None = None) -> dict:
    """EW + RP (+ TV) 합성 결과 일괄 생성. 진단용 요약 포함."""
    out = {
        "EW": equal_weight(pnls, standardize=True, vol_window=vol_window),
        "RP": risk_parity(pnls, vol_window=vol_window),
    }
    if target_ann_vol_bp:
        out["EW_TV"] = target_vol(out["EW"], target_ann_vol_bp)
        out["RP_TV"] = target_vol(out["RP"], target_ann_vol_bp)
    return out


def satellite_overlay(
    main_pnl: pd.Series,
    satellite_pnl: pd.Series,
    sat_weight: float = 0.15,
    standardize_sat: bool = True,
    sat_vol_window: int = 63,
) -> pd.Series:
    """메인 포트폴리오 PnL에 satellite signal 을 w_sat 가중으로 overlay.

    Parameters
    ----------
    main_pnl : 메인 포트폴리오 daily PnL (이미 합성된 상태, 예: 3팩터 DYN)
    satellite_pnl : satellite 팩터 daily PnL (활성일만 non-zero, 대부분 0)
    sat_weight : 0~1. 메인 size 대비 satellite size. 15% = 0.15.
    standardize_sat : satellite 도 메인과 동일한 scale로 맞춤 (vol-normalize).
    sat_vol_window : satellite 의 rolling vol window.

    Returns
    -------
    combined : 메인 + w_sat · satellite_std
    """
    if standardize_sat:
        sat_vol = satellite_pnl.rolling(sat_vol_window,
                                         min_periods=max(30, sat_vol_window // 2)).std(ddof=1)
        sat_vol = sat_vol.shift(1)
        sat_scaled = satellite_pnl / sat_vol.replace(0, np.nan)
    else:
        sat_scaled = satellite_pnl.copy()

    sat_scaled = sat_scaled.fillna(0)
    aligned = main_pnl.add(sat_weight * sat_scaled, fill_value=0)
    return aligned.rename("main+satellite")


def stats_table(pnls: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for k, v in pnls.items():
        s = v.dropna()
        if len(s) < 10:
            rows.append({"factor": k, "n": len(s)})
            continue
        mu, sd = float(s.mean()), float(s.std(ddof=1))
        cum = s.cumsum()
        peak = cum.cummax()
        dd = (cum - peak).min()
        rows.append({
            "factor": k, "n": len(s),
            "mean": mu, "std": sd,
            "sharpe_ann": mu / sd * np.sqrt(252.0) if sd > 0 else np.nan,
            "hit%": float((s > 0).mean()) * 100,
            "max_dd": float(dd),
        })
    return pd.DataFrame(rows).set_index("factor")
