"""VOL 팩터 정교화 유틸.

핵심 고려사항
-------------
- VOL 정의: ``rolling_std(ε, window)``  (단위 bp)
- Mechanical corr: VOL과 RV가 **같은 잔차 ε**를 사용 →
  vol 측정 구간과 RV 누적 구간이 겹치면 mechanical corr 발생.
  → ``lag`` 도입 (최근 lag 일을 vol 계산에서 제외) 로 완화 가능.

- 방향 (low-vol LONG vs high-vol LONG) 실증 필요 (명세 §4 미결).
- Within-bucket 필수 (장기물 자연히 vol 큼).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rv_diagnostics import quantile_labels_within_bucket


# ------------------------------------------------------------------
# VOL score
# ------------------------------------------------------------------

def compute_vol_lagged(
    residual: pd.DataFrame,
    window: int = 20,
    lag: int = 0,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """ε의 rolling std. lag > 0 이면 최근 lag 일 제외.

        VOL_t = std(ε_{t-lag-window+1 ... t-lag})
    """
    if min_periods is None:
        min_periods = max(int(window * 0.75), 10)
    eps = residual.shift(lag)
    return eps.rolling(window, min_periods=min_periods).std(ddof=1)


# ------------------------------------------------------------------
# VOL LS PnL  (within-bucket, hold=1d)
# ------------------------------------------------------------------

def vol_ls_pnl(
    vol_score: pd.DataFrame,
    dy_panel: pd.DataFrame,
    remain: pd.DataFrame,
    bucket_edges: list[float] = [0, 5, 10, 100],
    n_bins: int = 5,
    direction: str = "low_long",   # low_long | high_long
    exec_lag: int = 1,
) -> pd.Series:
    """Within-bucket VOL quintile → LS daily P&L (bp).

    direction:
      - low_long  : Q1 LONG / Q5 SHORT (low-vol anomaly)
      - high_long : Q5 LONG / Q1 SHORT (high-vol premium)
    """
    if direction == "low_long":
        lq, sq = 1, 5
    elif direction == "high_long":
        lq, sq = 5, 1
    else:
        raise ValueError(direction)

    labels = quantile_labels_within_bucket(vol_score, remain,
                                           bucket_edges=bucket_edges, n_bins=n_bins)
    labels = labels.shift(exec_lag)                  # 실행 지연
    common = labels.index.intersection(dy_panel.index)
    L = labels.reindex(common)
    D = dy_panel.reindex(index=common, columns=L.columns)
    La, Da = L.to_numpy(), D.to_numpy()

    def _m(q):
        mask = (La == q)
        return np.nanmean(np.where(mask, -Da, np.nan), axis=1)    # LONG P&L = -dY

    long_ = pd.Series(_m(lq), index=common)
    short = pd.Series(_m(sq), index=common)
    return (long_ - short).rename("VOL_ls_bp").dropna()


# ------------------------------------------------------------------
# Parameter grid
# ------------------------------------------------------------------

def vol_parameter_grid(
    residual: pd.DataFrame,
    dy_panel: pd.DataFrame,
    remain: pd.DataFrame,
    windows: list[int],
    lags: list[int],
    directions: list[str],
    bucket_edges: list[float] = [0, 5, 10, 100],
    n_bins: int = 5,
    start: str | None = None,
    end:   str | None = None,
    exec_lag: int = 1,
) -> pd.DataFrame:
    from .mom_diagnostics import pnl_summary
    rows = []
    for w in windows:
        for L in lags:
            vs = compute_vol_lagged(residual, window=w, lag=L)
            for direction in directions:
                pnl = vol_ls_pnl(vs, dy_panel, remain,
                                 bucket_edges=bucket_edges, n_bins=n_bins,
                                 direction=direction, exec_lag=exec_lag)
                if start: pnl = pnl.loc[pnl.index >= pd.Timestamp(start)]
                if end:   pnl = pnl.loc[pnl.index <= pd.Timestamp(end)]
                s = pnl_summary(pnl, holding_days=1)
                rows.append({
                    "window": w, "vol_lag": L, "direction": direction,
                    "n": s.get("n", 0),
                    "mean_bp":    s.get("mean_bp", np.nan),
                    "std_bp":     s.get("std_bp", np.nan),
                    "sharpe_ann": s.get("sharpe_ann", np.nan),
                    "t_nw":       s.get("t_nw", np.nan),
                    "hit_pct":    s.get("hit_pct", np.nan),
                    "skew":       s.get("skew", np.nan),
                    "kurt":       s.get("kurt", np.nan),
                })
    return pd.DataFrame(rows)


def rv_vol_mechanical_corr(
    rv_score: pd.DataFrame,
    vol_score: pd.DataFrame,
) -> float:
    """score level에서의 교차 상관 (명세 §7의 (A))."""
    common_idx = rv_score.index.intersection(vol_score.index)
    common_col = rv_score.columns.intersection(vol_score.columns)
    R = rv_score.reindex(index=common_idx, columns=common_col).stack()
    V = vol_score.reindex(index=common_idx, columns=common_col).stack()
    df = pd.concat([R.rename("rv"), V.rename("vol")], axis=1).dropna()
    if len(df) < 2:
        return float("nan")
    return float(df["rv"].corr(df["vol"]))
