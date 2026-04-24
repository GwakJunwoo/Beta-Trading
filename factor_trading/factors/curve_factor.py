"""CURVE 팩터 — (dY_10Y − dY_3Y)를 dY_3Y에 회귀한 잔차의 z-score (명세 §6).

2단계
-----
Step 1  slope_t = dY_10Y,t − dY_3Y,t                (slope 변화량)
Step 2  slope_t = a + b · dY_3Y,t + u_t             (OLS on rolling window)
        CURVE_t = u_t                                (MOM과 직교)
Step 3  CURVE_z = z-score(CURVE_t, window=21d)

Bang-bang 포지션
----------------
방향 **미정 — 백테스트로 결정** (명세 §6 미결 항목):
  - 가설 A (mean reversion): CURVE_z > 0 → flatten (10Y LONG, 3Y SHORT)
  - 가설 B (momentum):        CURVE_z > 0 → steepen (반대)

``curve_signal(..., direction='mean_rev')`` 또는 ``'momentum'``으로 전환.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm


RESID_WIN   = 63      # Step 2 OLS 윈도우 (3M)
RESID_MINP  = 40
Z_WIN       = 21      # Step 3 z-score (명세)
Z_MINP      = 15


def _rolling_residual(y: pd.Series, x: pd.Series,
                      window: int = RESID_WIN, min_periods: int = RESID_MINP) -> pd.Series:
    """y = a + b·x + u 의 잔차 u_t (rolling OLS)."""
    df = pd.concat([y.rename("y"), x.reindex(y.index).rename("x")], axis=1)
    v = df.dropna()
    if len(v) < min_periods:
        return pd.Series(np.nan, index=y.index)
    X = sm.add_constant(v["x"].values)
    res = RollingOLS(v["y"].values, X, window=window,
                     min_nobs=min_periods, expanding=False).fit()
    params = res.params                                    # (n, 2): const, x
    # 각 시점의 잔차: y - (a_t + b_t · x_t)
    pred = params[:, 0] + params[:, 1] * v["x"].values
    u = pd.Series(np.nan, index=y.index)
    u.loc[v.index] = v["y"].values - pred
    return u


def compute_curve(
    dy_3y:  pd.Series,
    dy_10y: pd.Series,
    resid_window: int = RESID_WIN,
    resid_minp: int = RESID_MINP,
    z_window: int = Z_WIN,
    z_minp: int = Z_MINP,
) -> pd.Series:
    """CURVE z-score 시계열.

    Returns
    -------
    Series — CURVE_z_t (MOM ⊥)
    """
    slope = (dy_10y.reindex(dy_3y.index) - dy_3y).rename("slope_bp")
    u = _rolling_residual(slope, dy_3y, window=resid_window, min_periods=resid_minp)
    mu = u.rolling(z_window, min_periods=z_minp).mean()
    sd = u.rolling(z_window, min_periods=z_minp).std(ddof=1)
    z = (u - mu) / sd.replace(0, np.nan)
    return z.rename("CURVE_z")


def curve_signal(z: pd.Series, direction: str = "mean_rev",
                 dead_zone: float = 0.0) -> pd.Series:
    """CURVE signal (커브 트레이드 방향).

    반환값의 의미: **+1 = steepen 베팅 (3Y LONG / 10Y SHORT, 듀레이션 중립)**
                  **-1 = flatten 베팅 (3Y SHORT / 10Y LONG)**

    - mean_rev: CURVE_z > 0 (최근 steepen 과열) → flatten → -1
    - momentum: CURVE_z > 0 → steepen 지속 → +1
    """
    if direction == "mean_rev":
        sig = -np.sign(z)
    elif direction == "momentum":
        sig = np.sign(z)
    else:
        raise ValueError(direction)
    if dead_zone > 0:
        sig = sig.where(z.abs() > dead_zone, 0.0)
    return sig.rename(f"CURVE_signal_{direction}")
