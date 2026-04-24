"""MOM 팩터 정교화용 진단 유틸.

Time-series 팩터라 RV와 다른 측면:
- signal turnover (flip 빈도)
- PnL skew/kurt (trend-following 전형)
- hysteresis(dead_zone) 효과
- lag sensitivity
- parameter (cum_window, zscore_win, dead_zone) grid
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rv_diagnostics import newey_west_long_run_var, drawdown_stats, rate_regime_split, tail_event_impact


# ------------------------------------------------------------------
# Core MOM computation (refactor-friendly 버전)
# ------------------------------------------------------------------

def mom_z(
    dy_3y: pd.Series,
    cum_window: int = 21,
    zscore_win: int = 252,
    zscore_minp: int | None = None,
) -> pd.Series:
    if zscore_minp is None:
        zscore_minp = max(zscore_win // 2, 30)
    cum = dy_3y.rolling(cum_window, min_periods=max(10, cum_window // 2)).sum()
    mu  = cum.rolling(zscore_win, min_periods=zscore_minp).mean()
    sd  = cum.rolling(zscore_win, min_periods=zscore_minp).std(ddof=1)
    return ((cum - mu) / sd.replace(0, np.nan)).rename("MOM_z")


def mom_signal(z: pd.Series, dead_zone: float = 0.0,
               direction: str = "mean_rev") -> pd.Series:
    """+1 = LONG bond (3Y yield 하락 기대), -1 = SHORT bond.

    direction
      - momentum : z>0 (최근 rate 상승세) → 추세 지속 가정 → SHORT bond  (signal = -sign(z))
      - mean_rev : z>0 → 과열 반전 가정 → LONG bond  (signal = +sign(z))
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


def mom_pnl(signal: pd.Series, dy_3y: pd.Series, lag: int = 1) -> pd.Series:
    s = signal.shift(lag).reindex(dy_3y.index)
    return (s * (-dy_3y)).rename("MOM_pnl_bp")


# ------------------------------------------------------------------
# PnL 요약 + 분포
# ------------------------------------------------------------------

def pnl_summary(pnl: pd.Series, holding_days: int = 1) -> dict:
    s = pnl.dropna()
    if len(s) < 2:
        return {}
    mu, sd = float(s.mean()), float(s.std(ddof=1))
    sharpe = mu / sd * np.sqrt(252.0 / holding_days) if sd > 0 else float("nan")
    # NW t (MOM은 lag=holding_days-1 굳이 필요 없지만 일관성 위해)
    nw_lag = max(holding_days - 1, 0)
    if nw_lag > 0:
        lrv = newey_west_long_run_var(s.values, lag=nw_lag)
        se = np.sqrt(lrv / len(s)) if lrv > 0 else float("nan")
    else:
        se = sd / np.sqrt(len(s))
    t = mu / se if se and se > 0 else float("nan")
    wins, losses = s[s > 0], s[s < 0]
    return {
        "n": int(len(s)),
        "mean_bp": mu, "std_bp": sd,
        "sharpe_ann": sharpe, "t_nw": t,
        "hit_pct": float((s > 0).mean()) * 100,
        "skew": float(s.skew()), "kurt": float(s.kurt()),
        "avg_win_bp":  float(wins.mean())   if len(wins)   else float("nan"),
        "avg_loss_bp": float(losses.mean()) if len(losses) else float("nan"),
        "win_loss_ratio": (abs(float(wins.mean()) / float(losses.mean()))
                           if len(wins) and len(losses) and float(losses.mean()) != 0
                           else float("nan")),
    }


# ------------------------------------------------------------------
# Signal 진단
# ------------------------------------------------------------------

def signal_turnover(signal: pd.Series) -> dict:
    """신호 flip 빈도 / 평균 보유기간."""
    s = signal.dropna()
    if len(s) < 2:
        return {}
    flips = (s.diff().abs() > 0).sum()
    flip_per_yr = float(flips) / len(s) * 252
    # 평균 같은 방향 유지 일수 (0 제외)
    nz = s[s != 0]
    runs = (nz != nz.shift()).cumsum()
    run_lens = nz.groupby(runs).size()
    return {
        "n": int(len(s)),
        "flips": int(flips),
        "flips_per_year": flip_per_yr,
        "avg_hold_days": float(run_lens.mean()) if len(run_lens) else float("nan"),
        "median_hold_days": float(run_lens.median()) if len(run_lens) else float("nan"),
        "pct_nonzero": float((s != 0).mean()) * 100,
    }


# ------------------------------------------------------------------
# Parameter grid
# ------------------------------------------------------------------

def mom_parameter_grid(
    dy_3y: pd.Series,
    cum_windows: list[int],
    zscore_wins: list[int],
    dead_zones:  list[float],
    directions:  list[str] = ["momentum", "mean_rev"],
    start: str | None = None,
    end:   str | None = None,
    lag: int = 1,
) -> pd.DataFrame:
    """(direction × cum × zwin × dz) 격자. Sharpe + z-bin 단조성 ρ 포함."""
    from .principles import ts_bin_pnl, monotonicity_rho
    rows = []
    for cw in cum_windows:
        for zw in zscore_wins:
            z = mom_z(dy_3y, cum_window=cw, zscore_win=zw)
            # z-bin rho (방향 무관, 부호만 보면 됨)
            mom_raw = -dy_3y
            bin_pnl = ts_bin_pnl(z, mom_raw, lag=lag)
            rho_zbin = monotonicity_rho(bin_pnl)
            for direction in directions:
                for dz in dead_zones:
                    sig = mom_signal(z, dead_zone=dz, direction=direction)
                    pnl = mom_pnl(sig, dy_3y, lag=lag)
                    if start: pnl = pnl.loc[pnl.index >= pd.Timestamp(start)]
                    if end:   pnl = pnl.loc[pnl.index <= pd.Timestamp(end)]
                    stats = pnl_summary(pnl, holding_days=1)
                    turn = signal_turnover(sig.loc[pnl.index])
                    rows.append({
                        "cum_window": cw, "zscore_win": zw,
                        "direction": direction, "dead_zone": dz,
                        "n": stats.get("n", 0),
                        "mean_bp":    stats.get("mean_bp", np.nan),
                        "std_bp":     stats.get("std_bp", np.nan),
                        "sharpe_ann": stats.get("sharpe_ann", np.nan),
                        "t_nw":       stats.get("t_nw", np.nan),
                        "hit_pct":    stats.get("hit_pct", np.nan),
                        "skew":       stats.get("skew", np.nan),
                        "zbin_rho":   rho_zbin,
                        "flips_per_year": turn.get("flips_per_year", np.nan),
                        "avg_hold_days":  turn.get("avg_hold_days", np.nan),
                        "pct_active":     turn.get("pct_nonzero", np.nan),
                    })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Lag sensitivity
# ------------------------------------------------------------------

def lag_sensitivity(
    signal: pd.Series,
    dy_3y: pd.Series,
    lags: list[int] = [0, 1, 2, 3, 5],
) -> pd.DataFrame:
    rows = []
    for L in lags:
        pnl = mom_pnl(signal, dy_3y, lag=L)
        st = pnl_summary(pnl)
        rows.append({"lag": L, **st})
    return pd.DataFrame(rows).set_index("lag")
