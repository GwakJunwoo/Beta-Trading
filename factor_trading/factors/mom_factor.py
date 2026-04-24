"""MOM 팩터 — 21일 누적 dY_3Y의 z-score (명세 §5).

``cum_dY_3Y,t = Σ_{s=t-20..t} dY_3Y,s   = Y_3Y,t - Y_3Y,t-21``
``MOM_t = (cum_dY_3Y,t - rolling_mean) / rolling_std``   (분포는 1년 trailing)

Bang-bang 신호
--------------
``sign(MOM_t)``:
  - MOM > 0 (금리 상승세) → **3Y 현물 SHORT** (금리 오르면 가격 하락)
  - MOM < 0 (금리 하락세) → **3Y 현물 LONG**

2차 튜닝에서 hysteresis (|z|>0.5일 때만 flip) 고려.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


CUM_WINDOW   = 21     # 21d 누적 dY
ZSCORE_WIN   = 252    # 1년 trailing for z-score distribution
ZSCORE_MINP  = 126    # 절반 이상 필요


def compute_mom(
    dy_3y: pd.Series,
    cum_window: int = CUM_WINDOW,
    zscore_win: int = ZSCORE_WIN,
    zscore_minp: int = ZSCORE_MINP,
) -> pd.Series:
    """dY_3Y로부터 MOM z-score 계산.

    Parameters
    ----------
    dy_3y : 3Y 지표 dY(bp) 시계열
    cum_window : 21 (명세)
    zscore_win : 252 (1년 trailing)

    Returns
    -------
    Series — MOM_z_t (unitless)
    """
    cum = dy_3y.rolling(cum_window, min_periods=max(10, cum_window // 2)).sum()
    mu  = cum.rolling(zscore_win, min_periods=zscore_minp).mean()
    sd  = cum.rolling(zscore_win, min_periods=zscore_minp).std(ddof=1)
    z = (cum - mu) / sd.replace(0, np.nan)
    return z.rename("MOM_z")


def mom_raw_cum(dy_3y: pd.Series, cum_window: int = 63) -> pd.Series:
    """Raw cumulative dY_3Y. z-score 없이 signal 만들기용 (Phase4 심층 진단 기반)."""
    minp = max(10, cum_window // 2)
    return dy_3y.rolling(cum_window, min_periods=minp).sum().rename(f"cum_dY_{cum_window}d")


def mom_split_signals(
    cum_dy: pd.Series,
    sigma_window: int = 252,
    sigma_minp: int = 126,
    theta_low: float = 1.0,
    theta_high: float = 2.0,
) -> tuple[pd.Series, pd.Series]:
    """MOM을 pure-momentum(trend) + contrarian 두 sub-signal로 분해.

    구간 분리 (|cum_dy| 를 rolling σ 로 정규화한 z = |cum_dy|/σ 기준):

        z < theta_low                 : 둘 다 0 (신호 약함)
        theta_low ≤ z < theta_high    : trend_sig = −sign(cum_dy),  contra_sig = 0
        z ≥ theta_high                : trend_sig = 0,              contra_sig = +sign(cum_dy)

    ``trend`` + ``contra`` 합은 언제나 -sign(cum)·{−1,0,+1} 의 disjoint 2-signal 구조.

    Parameters
    ----------
    cum_dy : pd.Series
        누적 dY (bp). 부호가 trend 방향을 나타냄.
    sigma_window, sigma_minp : int
        threshold 계산용 rolling σ 윈도우 (default: 252d / 126d, 약 1년).
    theta_low, theta_high : float
        z 기준 임계값. theta_low < theta_high.

    Returns
    -------
    (trend_sig, contra_sig) — 둘 다 bang-bang {−1, 0, +1}.
    """
    if not (0 < theta_low < theta_high):
        raise ValueError(f"0 < theta_low < theta_high 필요: {theta_low}, {theta_high}")
    sigma = cum_dy.rolling(sigma_window, min_periods=sigma_minp).std(ddof=1)
    z = cum_dy / sigma.replace(0, np.nan)
    abs_z = z.abs()
    s_sign = np.sign(cum_dy)

    trend = pd.Series(0.0, index=cum_dy.index, name="MOM_trend_sig")
    contra = pd.Series(0.0, index=cum_dy.index, name="MOM_contra_sig")

    trend_mask  = (abs_z >= theta_low) & (abs_z < theta_high)
    contra_mask = abs_z >= theta_high

    trend.loc[trend_mask]   = -s_sign.loc[trend_mask]     # momentum
    contra.loc[contra_mask] = +s_sign.loc[contra_mask]    # reversal

    # z 가 nan인 날은 둘 다 0 유지
    nan_mask = z.isna()
    trend.loc[nan_mask]  = np.nan
    contra.loc[nan_mask] = np.nan
    return trend, contra


def mom_raw_signal(cum_dy: pd.Series, dead_zone_bp: float = 0.0) -> pd.Series:
    """Raw cumulative dY 기반 bang-bang signal.

    cum_dy > 0 (최근 N일 rate 상승세) → momentum 가정 → SHORT bond → signal = -1
    cum_dy < 0 → LONG bond → signal = +1

    ``dead_zone_bp`` (bp) > 0 이면 |cum_dy| ≤ dead_zone_bp 에서 0 (중립).
    """
    sig = -np.sign(cum_dy)
    if dead_zone_bp > 0:
        sig = sig.where(cum_dy.abs() > dead_zone_bp, 0.0)
    return sig.rename("MOM_raw_signal")


def mom_signal(z: pd.Series, dead_zone: float = 0.0,
               direction: str = "mean_rev") -> pd.Series:
    """Bang-bang signal. +1 = LONG 3Y 현물, -1 = SHORT.

    direction
      - momentum : MOM > 0 (rate 상승세) → 추세 지속 → SHORT  (signal = -sign(z))
      - mean_rev : MOM > 0 → 과열 반전 → LONG   (signal = +sign(z))  ← Stage1 data로 확정

    ``dead_zone`` > 0 이면 |z| ≤ dead_zone에서 0 (중립, hysteresis).
    """
    if direction == "momentum":
        sig = -np.sign(z)
    elif direction == "mean_rev":
        sig = +np.sign(z)
    else:
        raise ValueError(direction)
    if dead_zone > 0:
        sig = sig.where(z.abs() > dead_zone, 0.0)
    return sig.rename(f"MOM_signal_{direction}")
