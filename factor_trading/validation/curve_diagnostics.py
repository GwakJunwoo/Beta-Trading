"""CURVE 팩터 정교화용 진단 유틸.

Key 질문: mean_rev vs momentum 방향 (명세 §6 미결).
추가: (resid_win × z_win × direction × dead_zone) grid, regime, lag.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm

from .mom_diagnostics import pnl_summary, signal_turnover


def _rolling_residual(y: pd.Series, x: pd.Series,
                      window: int, min_periods: int) -> pd.Series:
    df = pd.concat([y.rename("y"), x.reindex(y.index).rename("x")], axis=1).dropna()
    if len(df) < min_periods:
        return pd.Series(np.nan, index=y.index)
    X = sm.add_constant(df["x"].values)
    res = RollingOLS(df["y"].values, X, window=window,
                     min_nobs=min_periods, expanding=False).fit()
    params = res.params                                    # (n, 2): [const, x]
    pred = params[:, 0] + params[:, 1] * df["x"].values
    u = pd.Series(np.nan, index=y.index)
    u.loc[df.index] = df["y"].values - pred
    return u


def curve_z(
    dy_3y: pd.Series,
    dy_10y: pd.Series,
    resid_win: int = 63,
    resid_minp: int = 40,
    z_win: int = 21,
    z_minp: int | None = None,
) -> pd.Series:
    """CURVE z-score 시계열 (MOM ⊥)."""
    if z_minp is None:
        z_minp = max(z_win // 2 + 1, 10)
    slope = dy_10y.reindex(dy_3y.index) - dy_3y
    u = _rolling_residual(slope, dy_3y, window=resid_win, min_periods=resid_minp)
    mu = u.rolling(z_win, min_periods=z_minp).mean()
    sd = u.rolling(z_win, min_periods=z_minp).std(ddof=1)
    return ((u - mu) / sd.replace(0, np.nan)).rename("CURVE_z")


def curve_signal(z: pd.Series, direction: str = "mean_rev",
                 dead_zone: float = 0.0) -> pd.Series:
    """+1 = steepen 베팅 (3Y LONG / 10Y SHORT, dur-neutral)
       -1 = flatten 베팅.

    mean_rev: CURVE_z > 0 (최근 과도한 steepen) → flatten(-1)
    momentum: CURVE_z > 0 → steepen 지속(+1)
    """
    if direction == "mean_rev":
        sig = -np.sign(z)
    elif direction == "momentum":
        sig = np.sign(z)
    else:
        raise ValueError(direction)
    if dead_zone > 0:
        sig = sig.where(z.abs() > dead_zone, 0.0)
    return sig.rename(f"CURVE_sig_{direction}")


def curve_pnl(signal: pd.Series, dy_3y: pd.Series, dy_10y: pd.Series,
              lag: int = 1) -> pd.Series:
    """``pnl_t = signal_{t-lag} · (dY_10Y,t − dY_3Y,t)`` (bp, yield-space)."""
    s = signal.shift(lag).reindex(dy_3y.index)
    slope_dy = dy_10y.reindex(dy_3y.index) - dy_3y
    return (s * slope_dy).rename("CURVE_pnl_bp")


def curve_parameter_grid(
    dy_3y: pd.Series,
    dy_10y: pd.Series,
    resid_wins: list[int],
    z_wins: list[int],
    directions: list[str],
    dead_zones: list[float],
    start: str | None = None,
    end:   str | None = None,
    lag: int = 1,
) -> pd.DataFrame:
    rows = []
    for rw in resid_wins:
        for zw in z_wins:
            z = curve_z(dy_3y, dy_10y, resid_win=rw, z_win=zw)
            for direction in directions:
                for dz in dead_zones:
                    sig = curve_signal(z, direction=direction, dead_zone=dz)
                    pnl = curve_pnl(sig, dy_3y, dy_10y, lag=lag)
                    if start: pnl = pnl.loc[pnl.index >= pd.Timestamp(start)]
                    if end:   pnl = pnl.loc[pnl.index <= pd.Timestamp(end)]
                    s = pnl_summary(pnl, holding_days=1)
                    t = signal_turnover(sig.loc[pnl.index])
                    rows.append({
                        "resid_win": rw, "z_win": zw,
                        "direction": direction, "dead_zone": dz,
                        "n": s.get("n", 0),
                        "mean_bp":    s.get("mean_bp", np.nan),
                        "std_bp":     s.get("std_bp", np.nan),
                        "sharpe_ann": s.get("sharpe_ann", np.nan),
                        "t_nw":       s.get("t_nw", np.nan),
                        "hit_pct":    s.get("hit_pct", np.nan),
                        "skew":       s.get("skew", np.nan),
                        "flips_per_year": t.get("flips_per_year", np.nan),
                        "avg_hold_days":  t.get("avg_hold_days",  np.nan),
                        "pct_active":     t.get("pct_nonzero",    np.nan),
                    })
    return pd.DataFrame(rows)


def curve_lag_sensitivity(signal: pd.Series, dy_3y: pd.Series, dy_10y: pd.Series,
                          lags: list[int] = [0, 1, 2, 3, 5]) -> pd.DataFrame:
    rows = []
    for L in lags:
        pnl = curve_pnl(signal, dy_3y, dy_10y, lag=L)
        rows.append({"lag": L, **pnl_summary(pnl)})
    return pd.DataFrame(rows).set_index("lag")
