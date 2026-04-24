"""3대 원칙 공식 검증 유틸.

원칙
----
1. 이론 타당성 — 사람이 판단 (문서화)
2. 분위수 성과 단조성 — 여기서 수치화
3. 팩터 간 직교성 — 여기서 corr + α-β 분해

포인트
------
- Cross-section 팩터 (RV, VOL): quintile 기준
- Time-series 팩터 (MOM, CURVE): z-score bin 기준
- 직교성: PnL level corr + α-β 회귀 (명세 §7)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ============================================================
# 분위수 단조성 (원칙 2)
# ============================================================

def ts_bin_pnl(
    z_series: pd.Series,
    fwd_pnl_raw: pd.Series,   # 예: -dY_3Y (MOM LONG P&L)
    bins: list[float] = (-np.inf, -2, -1, -0.5, 0.5, 1, 2, np.inf),
    lag: int = 1,
) -> pd.DataFrame:
    """Time-series 팩터 — z 크기 bin별 forward PnL.

    ``PnL_{t+1} = fwd_pnl_raw_{t+1}`` 에 대해
    z_t (lag=1로 shift) 에 따른 bin 평균을 계산.

    fwd_pnl_raw : MOM이면 (-dY_3Y), CURVE steepen이면 (dY_10Y-dY_3Y) 등
                  즉 "+1 포지션의 daily PnL"
    """
    z = z_series.shift(lag)
    s = pd.concat([z.rename("z"), fwd_pnl_raw.rename("pnl")], axis=1).dropna()
    edges = list(bins)
    labels = [f"({edges[i]}, {edges[i+1]}]" for i in range(len(edges)-1)]
    s["bin"] = pd.cut(s["z"], bins=edges, labels=labels, include_lowest=True)
    g = s.groupby("bin", observed=True)["pnl"].agg(
        ["count", "mean", "std", lambda x: float((x > 0).mean()) * 100])
    g.columns = ["n", "mean_bp", "std_bp", "hit%"]
    g["sharpe_ann"] = g["mean_bp"] / g["std_bp"] * np.sqrt(252.0)
    return g


def xs_quintile_fwd_pnl(
    score: pd.DataFrame,
    dy_panel: pd.DataFrame,
    remain: pd.DataFrame,
    bucket_edges: list[float] = [0, 5, 10, 100],
    n_bins: int = 5,
    exec_lag: int = 1,
) -> pd.DataFrame:
    """Cross-section 팩터 — within-bucket quintile별 forward PnL (hold=1d)."""
    from .rv_diagnostics import quantile_labels_within_bucket
    labels = quantile_labels_within_bucket(score, remain,
                                            bucket_edges=bucket_edges, n_bins=n_bins)
    labels = labels.shift(exec_lag)
    common = labels.index.intersection(dy_panel.index)
    L = labels.reindex(common)
    D = dy_panel.reindex(index=common, columns=L.columns)
    La, Da = L.to_numpy(), D.to_numpy()

    rows = []
    for q in range(1, n_bins + 1):
        mask = (La == q)
        pnl_q = np.nanmean(np.where(mask, -Da, np.nan), axis=1)
        s = pd.Series(pnl_q, index=common).dropna()
        mu, sd = float(s.mean()), float(s.std(ddof=1))
        rows.append({
            "Q": q, "n": int(len(s)),
            "mean_bp": mu, "std_bp": sd,
            "sharpe_ann": mu / sd * np.sqrt(252.0) if sd > 0 else np.nan,
            "hit%": float((s > 0).mean()) * 100,
        })
    return pd.DataFrame(rows).set_index("Q")


def monotonicity_rho(df: pd.DataFrame, value_col: str = "mean_bp") -> float:
    """Spearman ρ between bin order and value (±1이면 완벽 단조)."""
    from scipy import stats
    v = df[value_col].dropna().values
    if len(v) < 2: return float("nan")
    rho, _ = stats.spearmanr(np.arange(len(v)), v)
    return float(rho)


# ============================================================
# 직교성 (원칙 3)
# ============================================================

def pnl_corr_matrix(pnls: dict[str, pd.Series]) -> pd.DataFrame:
    df = pd.DataFrame(pnls).dropna(how="all")
    return df.corr()


def alpha_beta_decomp(
    pnls: dict[str, pd.Series],
    target: str,
) -> dict:
    """target PnL을 나머지 팩터 PnL에 회귀 → α (절편) + R² + t-stat.

    alpha t > 1.5 면 target이 나머지로 설명되지 않는 고유 PnL 생성 (명세 §7).
    """
    df = pd.DataFrame(pnls).dropna()
    y = df[target]
    X = df.drop(columns=[target])
    X = sm.add_constant(X)
    res = sm.OLS(y, X).fit()
    alpha = float(res.params.iloc[0])
    alpha_t = float(res.tvalues.iloc[0])
    return {
        "alpha_bp_per_day": alpha,
        "alpha_t": alpha_t,
        "alpha_sharpe_ann": alpha / df[target].std() * np.sqrt(252) if df[target].std() > 0 else np.nan,
        "r_squared": float(res.rsquared),
        "betas": {k: float(v) for k, v in res.params.drop("const").items()},
        "n": int(len(df)),
    }


# ============================================================
# RV × VOL double sort (VOL 이론 재검토)
# ============================================================

def rv_vol_double_sort(
    rv_score: pd.DataFrame,
    vol_score: pd.DataFrame,
    dy_panel: pd.DataFrame,
    remain: pd.DataFrame,
    rv_bins: int = 3,
    vol_bins: int = 3,
    bucket_edges: list[float] = [0, 5, 10, 100],
    exec_lag: int = 1,
) -> pd.DataFrame:
    """같은 시점·같은 만기 버킷 안에서 RV×VOL 2차원 tercile.

    각 (rv_t, vol_t) 셀의 평균 forward PnL (bp/일, LONG side = -dY).
    VOL이 RV controlled 상태에서도 independent information을 주는지 확인.
    """
    from .rv_diagnostics import quantile_labels_within_bucket
    L_rv  = quantile_labels_within_bucket(rv_score,  remain, bucket_edges, n_bins=rv_bins)
    L_vol = quantile_labels_within_bucket(vol_score, remain, bucket_edges, n_bins=vol_bins)
    L_rv  = L_rv.shift(exec_lag)
    L_vol = L_vol.shift(exec_lag)
    common = L_rv.index.intersection(dy_panel.index).intersection(L_vol.index)
    Lr = L_rv.reindex(common).to_numpy()
    Lv = L_vol.reindex(common).to_numpy()
    D  = dy_panel.reindex(index=common, columns=L_rv.columns).to_numpy()

    grid = np.full((rv_bins, vol_bins), np.nan)
    n_grid = np.zeros_like(grid, dtype=int)
    for i in range(rv_bins):
        for j in range(vol_bins):
            mask = (Lr == i + 1) & (Lv == j + 1)
            vals = np.where(mask, -D, np.nan)              # LONG P&L = -dY
            daily = np.nanmean(vals, axis=1)
            ser = pd.Series(daily).dropna()
            grid[i, j]   = float(ser.mean())
            n_grid[i, j] = int(len(ser))

    out = pd.DataFrame(grid,
                       index=[f"RV_Q{i+1}" for i in range(rv_bins)],
                       columns=[f"VOL_Q{j+1}" for j in range(vol_bins)])
    out.index.name = "RV bin"; out.columns.name = "VOL bin"
    return out
