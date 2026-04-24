"""팩터별 단일 P&L series 생성.

규약
----
모든 daily PnL 시계열 단위는 **bp**, **1영업일 보유(hold=1d)** 기준.
- Cross-section (RV, VOL): within-bucket quintile Long Q5 / Short Q1 의 daily LS P&L
- MOM: 3Y 현물 bang-bang → ``pnl_t = signal_{t-1} · (-dY_3Y,t)``
  (전일 신호로 당일 포지션, bond LONG이면 yield 떨어져야 이익 → -dY)
- CURVE: 3Y vs 10Y BPV-중립 페어의 signal-based daily P&L

Cross-section LS P&L 계산
-------------------------
t에서 score로 quintile 매김 → t+1부터 holding.
여기서는 LS_t = mean_{Q5 at t-1}(−dY_i,t) − mean_{Q1 at t-1}(−dY_i,t)
     = mean_Q1(dY) − mean_Q5(dY)   (bp)
하지만 β-hedge 포함한 진짜 P&L을 쓰려면 β-adjusted forward return이 필요.
여기서는 **hold=1d raw** 기준. 장기 hold P&L이 필요하면 forward_return_*()로 별도 계산.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Cross-section: within-bucket quintile
# ------------------------------------------------------------------

def within_bucket_quintile(
    score_panel: pd.DataFrame,
    remain_panel: pd.DataFrame,
    bucket_edges: list[float] = [0, 5, 10, 100],
    n_bins: int = 5,
) -> pd.DataFrame:
    """시점별 종목을 만기 버킷에 배정 후 버킷 내에서 quintile(1..n_bins) 라벨링.

    Parameters
    ----------
    score_panel : DataFrame (date × bond_code)
    remain_panel : 같은 모양. 잔존만기(년).
    bucket_edges : [0, 5, 10, 100]  (≤5, 5~10, >10)
    n_bins : 5 (quintile)

    Returns
    -------
    DataFrame (date × bond_code) with integer quintile label (1..n_bins) 또는 NaN.
    """
    S = score_panel
    R = remain_panel.reindex_like(S)
    out = pd.DataFrame(np.nan, index=S.index, columns=S.columns)
    for i in range(len(bucket_edges) - 1):
        lo, hi = bucket_edges[i], bucket_edges[i + 1]
        in_b = ((R >= lo) & (R < hi))
        sub = S.where(in_b)
        n_valid = sub.notna().sum(axis=1)
        pct = sub.rank(axis=1, pct=True, method="first")
        lab = np.ceil(pct * n_bins).where(sub.notna())
        lab.loc[n_valid < n_bins] = np.nan
        out = out.where(~in_b, lab)
    return out


def xsec_ls_pnl(
    score_panel: pd.DataFrame,
    dy_panel: pd.DataFrame,
    remain_panel: pd.DataFrame,
    bucket_edges: list[float] = [0, 5, 10, 100],
    n_bins: int = 5,
    long_quintile: int = 5,
    short_quintile: int = 1,
    lag: int = 1,
) -> pd.DataFrame:
    """Cross-section LS 일별 P&L (bp, hold=1d).

    score_t → quintile_t → 포지션은 ``lag``일 후부터 적용 (= t+lag일에 open).
    P&L_{t+lag} = mean_{LongQ}(-dY_{t+lag}) - mean_{ShortQ}(-dY_{t+lag})
                = mean_ShortQ(dY) - mean_LongQ(dY)   (bp)

    Returns
    -------
    DataFrame with columns:
        ls_bp, long_bp, short_bp (개별 leg도 진단용으로 반환)
    """
    labels = within_bucket_quintile(score_panel, remain_panel,
                                     bucket_edges=bucket_edges, n_bins=n_bins)
    labels = labels.shift(lag)                                # 전일 신호 → 오늘 포지션
    common = labels.index.intersection(dy_panel.index)
    L = labels.reindex(common)
    D = dy_panel.reindex(index=common, columns=L.columns)
    L_arr, D_arr = L.to_numpy(), D.to_numpy()

    def _mean_where(q):
        mask = (L_arr == q)
        return np.nanmean(np.where(mask, -D_arr, np.nan), axis=1)   # LONG P&L = -dY

    long_pnl  = pd.Series(_mean_where(long_quintile),  index=common, name="long_bp")
    short_pnl = pd.Series(_mean_where(short_quintile), index=common, name="short_bp")
    ls = (long_pnl - short_pnl).rename("ls_bp")
    return pd.concat([ls, long_pnl, short_pnl], axis=1).dropna(how="all")


# ------------------------------------------------------------------
# Time-series: MOM (3Y 현물)
# ------------------------------------------------------------------

def mom_pnl(signal: pd.Series, dy_3y: pd.Series, lag: int = 1) -> pd.Series:
    """MOM 3Y 현물 bang-bang daily P&L (bp).

    ``pnl_t = signal_{t-lag} · (-dY_3Y,t)``
      signal=+1 (bond LONG) → yield 내려야 이익 (-dY>0)
      signal=-1 (bond SHORT) → yield 올라야 이익 (-dY<0)
    """
    s = signal.shift(lag).reindex(dy_3y.index)
    return (s * (-dy_3y)).rename("MOM_pnl_bp")


# ------------------------------------------------------------------
# Time-series: CURVE (3Y/10Y 듀레이션 중립 페어)
# ------------------------------------------------------------------

def curve_pnl(
    signal: pd.Series,
    dy_3y: pd.Series,
    dy_10y: pd.Series,
    lag: int = 1,
) -> pd.Series:
    """CURVE 페어 daily P&L (bp per 10Y leg notional).

    **규약**: LS notional = 10Y 1단위에 대해 3Y는 BPV-중립 ratio 만큼 반대방향.
    단순화: 여기서는 **yield-space basis point P&L**을 잡는다. 가격P&L이 아니라
    "slope change × signal"을 bp 단위로 추적. 절대금액은 portfolio/combiner에서 별도 스케일.

    signal=+1 (steepen 베팅 = 3Y LONG / 10Y SHORT) → slope 오를수록 이익:
        pnl = slope_dY = dY_10Y - dY_3Y   (bp)
        (signal=-1이면 flatten 베팅 = -slope_dY)

    ``pnl_t = signal_{t-lag} · (dY_10Y,t - dY_3Y,t)``

    이 표현은 "dY_bond" 관점에서 잡은 것이라 실제 가격P&L과는 듀레이션 스케일이
    다르지만, 팩터 간 비교(Sharpe, corr) 용으로는 적합.
    """
    s = signal.shift(lag).reindex(dy_3y.index)
    slope_dy = dy_10y.reindex(dy_3y.index) - dy_3y
    # signal=+1 steepen(3Y LONG/10Y SHORT) → 원하는 건 "3Y yield↓ + 10Y yield↑"
    # 이를 slope_dy로 잡으면: steepen일 때 slope_dy>0 → signal·slope_dy>0 ✓
    return (s * slope_dy).rename("CURVE_pnl_bp")
