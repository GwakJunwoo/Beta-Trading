"""Duration-neutral 비중 계산 (명세 §6 CURVE 페어용).

BPV(bond) = DV01 ≈ modified_duration · price · 0.0001

간단화: 잔존만기 approximation을 MD로 사용.
  MD ≈ remain_year / (1 + ytm)    (annual comp. 근사)
  BPV_per_100par ≈ MD · 1 · 0.0001   (bp당 가격변화, price=100 가정)

비중 ratio 3Y:10Y = BPV_10Y / BPV_3Y   (10Y 1단위당 3Y N단위)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def modified_duration(remain_year: float, ytm_pct: float) -> float:
    if ytm_pct is None or np.isnan(ytm_pct) or remain_year is None or np.isnan(remain_year):
        return float("nan")
    return float(remain_year) / (1.0 + float(ytm_pct) / 100.0)


def bpv_per_100par(remain_year: float, ytm_pct: float) -> float:
    md = modified_duration(remain_year, ytm_pct)
    return md * 100.0 * 1e-4     # price=100 par, bp당 가격변화


def dv01_weights(
    remain_3y: pd.Series,
    ytm_3y_pct: pd.Series,
    remain_10y: pd.Series,
    ytm_10y_pct: pd.Series,
) -> pd.DataFrame:
    """일자별 (3Y, 10Y)의 BPV와 3Y:10Y 비중 ratio.

    Returns DataFrame with columns:
        bpv_3y, bpv_10y, w_3y_per_10y   (10Y 1단위당 3Y weight)
    """
    idx = remain_3y.index.union(remain_10y.index).sort_values()
    r3 = remain_3y.reindex(idx).astype(float)
    y3 = ytm_3y_pct.reindex(idx).astype(float)
    r10 = remain_10y.reindex(idx).astype(float)
    y10 = ytm_10y_pct.reindex(idx).astype(float)
    bpv3  = r3 / (1 + y3 / 100) * 100.0 * 1e-4
    bpv10 = r10 / (1 + y10 / 100) * 100.0 * 1e-4
    return pd.DataFrame({
        "bpv_3y":  bpv3,
        "bpv_10y": bpv10,
        "w_3y_per_10y": bpv10 / bpv3.replace(0, np.nan),
    }).astype(float)
