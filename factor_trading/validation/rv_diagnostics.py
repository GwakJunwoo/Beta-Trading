"""RV 팩터 정교화용 진단 유틸 — 1팩터 실험 시 만든 방법론을 2팩터 체계에 이식.

핵심 빌딩블록
--------------
- ``forward_return_2f``              : β-헤지된 N영업일 LONG P&L (bp, 2-factor)
- ``quantile_labels_*``              : simple / within-bucket quintile
- ``quantile_forward_returns``       : 분위수별 평균 forward P&L
- ``run_rv_grid``                    : (horizon × fwd_window) grid — IC, LS Sharpe, 단조성
- ``ls_stats``                       : NW HAC 보정 포함 mean/std/Sharpe/t
- ``drawdown_stats``, ``rate_regime_split``, ``tail_event_impact``
                                     : LS 곡선 변동성 분해용
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# NW long-run variance + LS stats
# ============================================================

def newey_west_long_run_var(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        return float("nan")
    e = x - x.mean()
    g0 = float(e @ e) / n
    s = g0
    for L in range(1, min(lag, n - 1) + 1):
        w = 1.0 - L / (lag + 1.0)
        s += 2.0 * w * float(e[L:] @ e[:-L]) / n
    return max(s, 0.0)


@dataclass
class LSStats:
    n: int
    mean: float
    std: float
    sharpe_ann: float      # IID: mean/std · √(252/N)
    t_nw: float            # NW HAC t: mean / √(LRV/n)
    hit_pct: float
    nw_lag: int


def ls_stats(ls: pd.Series, holding_days: int) -> LSStats:
    s = ls.dropna()
    if len(s) < 2:
        return LSStats(int(len(s)), float("nan"), float("nan"),
                       float("nan"), float("nan"), float("nan"), max(holding_days - 1, 0))
    mu = float(s.mean())
    sd = float(s.std(ddof=1))
    sharpe = mu / sd * np.sqrt(252.0 / holding_days) if sd > 0 else float("nan")
    nw_lag = max(holding_days - 1, 0)
    if nw_lag > 0:
        lrv = newey_west_long_run_var(s.values, lag=nw_lag)
        se = np.sqrt(lrv / len(s)) if lrv > 0 else float("nan")
    else:
        se = sd / np.sqrt(len(s))
    t = mu / se if se and se > 0 else float("nan")
    pos = float((s > 0).mean()) * 100
    return LSStats(int(len(s)), mu, sd, sharpe, t, pos, nw_lag)


def monotonicity(q_means: pd.Series) -> float:
    q = q_means.dropna()
    if len(q) < 2:
        return float("nan")
    rho, _ = stats.spearmanr(np.arange(1, len(q) + 1), q.values)
    return float(rho)


# ============================================================
# Forward return & quintile
# ============================================================

def forward_return_2f(
    ytm_panel: pd.DataFrame,
    dy_3y: pd.Series,
    dy_10y: pd.Series,
    beta_3y: pd.DataFrame,
    beta_10y: pd.DataFrame,
    n_days: int,
) -> pd.DataFrame:
    """2-factor β-hedged N-day LONG P&L (bp).

        P&L = −(ΔY_bond − β^3Y·ΔY_3Y − β^10Y·ΔY_10Y)
    """
    bond_dy = ytm_panel.shift(-n_days) - ytm_panel
    f3_cum  = dy_3y.fillna(0).cumsum()
    f10_cum = dy_10y.fillna(0).cumsum()
    fwd_f3  = (f3_cum.shift(-n_days)  - f3_cum ).reindex(bond_dy.index)
    fwd_f10 = (f10_cum.shift(-n_days) - f10_cum).reindex(bond_dy.index)
    b3  = beta_3y.reindex(bond_dy.index).reindex(columns=bond_dy.columns)
    b10 = beta_10y.reindex(bond_dy.index).reindex(columns=bond_dy.columns)
    return -(bond_dy - b3.mul(fwd_f3, axis=0) - b10.mul(fwd_f10, axis=0))


def quantile_labels_simple(score: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    n_valid = score.notna().sum(axis=1)
    pct = score.rank(axis=1, pct=True, method="first")
    lab = np.ceil(pct * n_bins).where(score.notna())
    drop = n_valid < n_bins
    if drop.any():
        lab.loc[drop] = np.nan
    return lab


def quantile_labels_within_bucket(
    score: pd.DataFrame,
    remain: pd.DataFrame,
    bucket_edges: list[float] = [0, 5, 10, 100],
    n_bins: int = 5,
) -> pd.DataFrame:
    S, R = score, remain.reindex_like(score)
    out = pd.DataFrame(np.nan, index=S.index, columns=S.columns)
    for i in range(len(bucket_edges) - 1):
        lo, hi = bucket_edges[i], bucket_edges[i + 1]
        in_b = (R >= lo) & (R < hi)
        sub = S.where(in_b)
        n_valid = sub.notna().sum(axis=1)
        pct = sub.rank(axis=1, pct=True, method="first")
        lab = np.ceil(pct * n_bins).where(sub.notna())
        lab.loc[n_valid < n_bins] = np.nan
        out = out.where(~in_b, lab)
    return out


def quantile_forward_returns(
    labels: pd.DataFrame, fwd: pd.DataFrame, n_bins: int = 5,
) -> pd.DataFrame:
    common = labels.index.intersection(fwd.index)
    L = labels.reindex(common)
    F = fwd.reindex(index=common, columns=L.columns)
    out = pd.DataFrame(np.nan, index=common, columns=range(1, n_bins + 1))
    L_arr, F_arr = L.to_numpy(), F.to_numpy()
    with np.errstate(invalid="ignore"):
        for q in range(1, n_bins + 1):
            mask = (L_arr == q)
            out.iloc[:, q - 1] = np.nanmean(np.where(mask, F_arr, np.nan), axis=1)
    return out


# ============================================================
# Grid — horizon × forward window
# ============================================================

def run_rv_grid(
    away_dict: dict[str, pd.DataFrame],      # {"1d", "1w", "2w", "1m"} → (date × bond) score
    fwd_dict:  dict[int, pd.DataFrame],      # {1, 5, 10, 21} → β-adj forward P&L
    remain:    pd.DataFrame,
    quintile_mode: str = "within_bucket",    # "simple" | "within_bucket"
    bucket_edges: list[float] = [0, 5, 10, 100],
    n_bins: int = 5,
    start: str | None = None,
    end:   str | None = None,
) -> dict:
    """horizon × fwd grid 실행. mode에 따라 simple / within-bucket quintile."""
    rows = []
    detailed = {}
    for h, F_full in away_dict.items():
        F = F_full
        if start: F = F.loc[F.index >= pd.Timestamp(start)]
        if end:   F = F.loc[F.index <= pd.Timestamp(end)]
        for N, fwd in fwd_dict.items():
            if quintile_mode == "simple":
                lab = quantile_labels_simple(F, n_bins=n_bins)
            else:
                lab = quantile_labels_within_bucket(F, remain,
                                                     bucket_edges=bucket_edges, n_bins=n_bins)
            qr = quantile_forward_returns(lab, fwd, n_bins=n_bins)
            ls = (qr.iloc[:, -1] - qr.iloc[:, 0]).dropna().rename("ls")
            q_means = qr.mean(axis=0)
            st = ls_stats(ls, holding_days=N)
            mono = monotonicity(q_means)

            # IC (Pearson)
            Fc = F.reindex(index=qr.index)
            Fc = Fc.reindex(columns=fwd.columns)
            ic = Fc.corrwith(fwd.reindex(index=qr.index, columns=Fc.columns),
                             axis=1, method="pearson").dropna()
            ic_mean = float(ic.mean()) if len(ic) else float("nan")
            ic_nw_lag = max(N - 1, 0)
            if ic_nw_lag > 0 and len(ic) > 1:
                lrv = newey_west_long_run_var(ic.values, lag=ic_nw_lag)
                ic_t = ic_mean / np.sqrt(lrv / len(ic)) if lrv > 0 else float("nan")
            elif len(ic) > 1:
                ic_t = ic_mean / (ic.std(ddof=1) / np.sqrt(len(ic)))
            else:
                ic_t = float("nan")

            detailed[(h, N)] = {"labels": lab, "qrets": qr, "q_means": q_means,
                                 "ls": ls, "stats": st, "monotonicity": mono,
                                 "ic": ic, "ic_mean": ic_mean, "ic_t_nw": ic_t}
            rows.append({
                "horizon": h, "fwd_N": N, "n_days": st.n,
                "Q1_bp": float(q_means.iloc[0]), "Q5_bp": float(q_means.iloc[-1]),
                "LS_mean_bp": st.mean, "LS_std_bp": st.std,
                "LS_sharpe_ann": st.sharpe_ann, "LS_t_nw": st.t_nw,
                "LS_hit%": st.hit_pct,
                "IC_mean": ic_mean, "IC_t_nw": ic_t,
                "monotonicity": mono,
            })
    return {"stats": pd.DataFrame(rows), "detailed": detailed}


# ============================================================
# LS 곡선 변동성 분해 진단
# ============================================================

def drawdown_stats(pnl: pd.Series) -> dict:
    """Max DD 규모 + peak/trough/recovery 날짜."""
    cum = pnl.fillna(0).cumsum()
    peak = cum.cummax()
    dd = cum - peak
    trough = dd.idxmin()
    max_dd = float(dd.min())
    p = cum.loc[:trough].idxmax()
    recov = cum.loc[trough:]
    r_idx = recov[recov >= cum.loc[p]].index
    recovered_on = r_idx[0] if len(r_idx) else None
    return dict(max_dd=max_dd, peak=p, trough=trough, recovered_on=recovered_on,
                duration_days=(trough - p).days,
                recovery_days=(recovered_on - trough).days if recovered_on is not None else None)


def rate_regime_split(
    pnl: pd.Series,
    dy_3y: pd.Series,
    cum_window: int = 63,
    edges: list[float] = (-np.inf, -25, -5, 5, 25, np.inf),
    labels: list[str] | None = None,
    holding_days: int = 21,
) -> pd.DataFrame:
    """3M 누적 dY_3Y에 따라 regime 분할하고 PnL 통계."""
    if labels is None:
        labels = ["bull강(-25↓)", "bull약(-25~-5)", "flat(±5)",
                  "bear약(5~25)", "bear강(25↑)"]
    f3m = dy_3y.rolling(cum_window, min_periods=30).sum()
    common = f3m.index.intersection(pnl.index)
    df = pd.DataFrame({"pnl": pnl.loc[common], "f3m": f3m.loc[common]}).dropna()
    df["regime"] = pd.cut(df["f3m"], bins=list(edges), labels=labels)
    g = df.groupby("regime", observed=True)["pnl"].agg(
        ["count", "mean", "std", lambda s: float((s > 0).mean()) * 100])
    g.columns = ["n", "mean_bp", "std_bp", "hit%"]
    g["sharpe_ann"] = g["mean_bp"] / g["std_bp"] * np.sqrt(252.0 / holding_days)
    return g


def tail_event_impact(pnl: pd.Series, holding_days: int,
                      ks: list[int] = (0, 5, 10, 20, 30)) -> pd.DataFrame:
    """|pnl| 상위 K개 이벤트 제거 시 Sharpe 변화."""
    s = pnl.dropna()
    order = s.abs().sort_values(ascending=False).index
    rows = []
    for k in ks:
        keep = s.drop(order[:k])
        mu, sd = keep.mean(), keep.std(ddof=1)
        sh = mu / sd * np.sqrt(252.0 / holding_days) if sd > 0 else float("nan")
        rows.append({"K": k, "mean_bp": float(mu), "std_bp": float(sd),
                     "sharpe_ann": sh, "hit%": float((keep > 0).mean()) * 100})
    return pd.DataFrame(rows).set_index("K")
