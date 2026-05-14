"""
03 — 외국인 flow → 미래 yield 상승의 원인 추적.

4가지 분해:
  A) ΔY = β·ΔY_3Y + γ·Δslope + Δε  → flow 와 각 component 의 IC
  B) ε reversion 가설: flow 시점의 ε 부호 vs 미래 ε 변화
  C) Aggregate vs idiosyncratic: 시장 전체 외국인 flow → Y_3Y / Y_10Y 움직임
  D) 만기 bucket × flow 크기 quintile differential
"""
from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

BETA_ROOT = Path(__file__).resolve().parents[3]
FULL_ROOT = Path(r"C:\Users\infomax\Desktop\fullstackjunior")
for p in (BETA_ROOT, FULL_ROOT, FULL_ROOT / "server"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from beta_trading.db import get_connection
from app.routers.beta import _load_label_series, _rolling_two_factor_beta


def load_data(start: str = "2023-01-01"):
    """flow + ktb + bench (3Y/10Y) 로드."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT bond_code, price_date,
                   foreigner_sum_3d, foreigner_sum_5d, foreigner_sum_10d, foreigner_diff_1d
            FROM ktb_trade_flow_features
            WHERE price_date >= %s AND bond_code IS NOT NULL AND bond_code != ''
        """, (start,))
        flow_rows = cur.fetchall()

        cur.execute("""
            SELECT price_date, bond_code, AVG(ytm) AS ytm, AVG(remain_year) AS remain_year
            FROM ktb
            WHERE category='국고채' AND price_date >= %s AND ytm > 0
              AND bond_code IS NOT NULL AND bond_code != ''
            GROUP BY price_date, bond_code
        """, (start,))
        ktb_rows = cur.fetchall()

    flow = pd.DataFrame(flow_rows)
    flow["price_date"] = pd.to_datetime(flow["price_date"])
    for c in flow.columns:
        if c not in ("bond_code", "price_date"):
            flow[c] = pd.to_numeric(flow[c], errors="coerce")

    ktb = pd.DataFrame(ktb_rows)
    ktb["price_date"] = pd.to_datetime(ktb["price_date"])
    ktb["ytm"] = pd.to_numeric(ktb["ytm"], errors="coerce")
    ktb["remain_year"] = pd.to_numeric(ktb["remain_year"], errors="coerce")

    s3 = _load_label_series("3년지표", days=1500)
    s10 = _load_label_series("10년지표", days=1500)
    return flow, ktb, s3, s10


def compute_eps_panel(ktb: pd.DataFrame, s3: pd.Series, s10: pd.Series,
                      window: int = 63, min_periods: int = 20):
    """Level mode ε + β/γ panel (lookahead-safe: β 1일 lag)."""
    ytm_panel = ktb.pivot_table(index="price_date", columns="bond_code",
                                values="ytm", aggfunc="mean").sort_index()
    idx = ytm_panel.index.union(s3.index).union(s10.index).sort_values()
    ytm_panel = ytm_panel.reindex(idx).ffill()
    s3_full = s3.reindex(idx).ffill()
    s10_full = s10.reindex(idx).ffill()
    y_panel = ytm_panel * 100.0
    x1 = s3_full * 100.0
    x2 = (s10_full - s3_full) * 100.0
    beta_lvl, beta_slp, _ = _rolling_two_factor_beta(
        dy_panel=y_panel, dy_level_bp=x1, dy_slope_bp=x2,
        window=window, min_periods=min_periods,
    )
    # 1일 lag (look-ahead 제거)
    beta_lvl_lag = beta_lvl.shift(1)
    beta_slp_lag = beta_slp.shift(1)
    eps = y_panel - beta_lvl_lag.multiply(x1, axis=0) - beta_slp_lag.multiply(x2, axis=0)
    return eps, beta_lvl_lag, beta_slp_lag, s3_full, s10_full, ytm_panel


def ic(x: pd.Series, y: pd.Series) -> dict:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    df = df[(df["x"] != 0) | (df["y"] != 0)]
    if len(df) < 50:
        return {"n": len(df), "ic": np.nan, "pval": np.nan}
    rho, p = spearmanr(df["x"], df["y"])
    return {"n": len(df), "ic": float(rho), "pval": float(p)}


def main():
    print("[load] data ...")
    flow, ktb, s3, s10 = load_data(start="2023-01-01")
    eps, beta_lvl, beta_slp, s3_full, s10_full, ytm_panel = compute_eps_panel(ktb, s3, s10)
    print(f"  eps panel: {eps.shape}, dates {eps.index.min().date()} ~ {eps.index.max().date()}\n")

    # Latest 잔존만기 매핑
    latest_t = ktb["price_date"].max()
    rem_latest = ktb[ktb["price_date"] == latest_t][["bond_code", "remain_year"]].set_index("bond_code")["remain_year"]

    # ── 분해를 위한 panel 빌드 ──
    print("[build] merged panel (flow + eps + Y_3Y + Y_10Y) ...")

    # flow long-form 으로
    flow_use = flow[["bond_code", "price_date", "foreigner_sum_3d", "foreigner_sum_5d",
                     "foreigner_sum_10d", "foreigner_diff_1d"]]

    # forward yield change 분해를 위해 다양한 horizon
    horizons = [5, 10, 21]
    pivot_y = ktb.pivot_table(index="price_date", columns="bond_code", values="ytm",
                              aggfunc="mean").sort_index().ffill(limit=5)
    pivot_eps = eps   # already DataFrame

    # 단일 horizon 21d 로 main analysis
    H = 21
    pivot_y_fwd = pivot_y.shift(-H)
    dy_fwd = (pivot_y_fwd - pivot_y) * 100.0   # bp
    s3_fwd = s3_full.shift(-H)
    dy3_fwd = (s3_fwd - s3_full) * 100.0       # bp
    s10_fwd = s10_full.shift(-H)
    dy10_fwd = (s10_fwd - s10_full) * 100.0
    dslope_fwd = dy10_fwd - dy3_fwd            # bp
    eps_fwd = pivot_eps.shift(-H)
    d_eps_fwd = (eps_fwd - pivot_eps)          # bp (eps 이미 bp 단위)

    # melt to long form
    def melt(df, name):
        return df.stack().reset_index().rename(columns={
            df.index.name or "level_0": "price_date",
            df.columns.name or "level_1": "bond_code",
            0: name,
        })

    pieces = []
    for df, name in [(dy_fwd, f"fwd_dy_{H}d"), (d_eps_fwd, f"fwd_deps_{H}d"),
                     (pivot_eps, "eps_now"), (pivot_y, "y_now")]:
        m = df.stack().reset_index()
        m.columns = ["price_date", "bond_code", name]
        pieces.append(m.set_index(["price_date", "bond_code"]))
    panel = pd.concat(pieces, axis=1).reset_index()

    # add dy3, dslope forward (scalar per date)
    panel["fwd_dy3_21d"] = panel["price_date"].map(dy3_fwd)
    panel["fwd_dslope_21d"] = panel["price_date"].map(dslope_fwd)

    # merge flow
    panel = panel.merge(flow_use, on=["price_date", "bond_code"], how="inner")
    # add remain (latest 만 — approximation)
    panel["remain"] = panel["bond_code"].map(rem_latest)
    panel = panel.dropna(subset=["remain"])
    panel = panel[(panel["remain"] >= 2) & (panel["remain"] <= 13)]

    print(f"  panel: {len(panel):,} rows, {panel['bond_code'].nunique()} bonds\n")

    # ── A) 외국인 flow 가 어느 component 와 상관? ──
    print("=== A) 외국인 sum_3d 가 21일 후 ΔY 의 어느 component 와 상관? ===")
    flow_col = "foreigner_sum_3d"
    print(f"   (flow {flow_col} → forward 21d 변화)")
    print()
    for tgt_col, label in [
        (f"fwd_dy_{H}d",        "ΔY (전체)"),
        ("fwd_dy3_21d",         "ΔY_3Y (시장 평행이동)"),
        ("fwd_dslope_21d",      "Δslope (curve)"),
        (f"fwd_deps_{H}d",      "Δε (idiosyncratic)"),
    ]:
        res = ic(panel[flow_col], panel[tgt_col])
        print(f"  {label:30s}: IC = {res['ic']:+.4f}  (N={res['n']:,}, p={res['pval']:.2e})")

    # ── B) ε reversion 가설 ──
    print("\n=== B) ε reversion 가설 ===")
    print("   flow buy 시점의 ε 부호 + flow buy 후 ε 변화")
    print()
    # flow buy = sum_3d > 0
    p = panel.dropna(subset=[flow_col, "eps_now", f"fwd_deps_{H}d"]).copy()
    p["flow_sign"] = np.sign(p[flow_col])
    grouped = p.groupby("flow_sign").agg(
        N=("eps_now", "size"),
        mean_eps_now=("eps_now", "mean"),
        median_eps_now=("eps_now", "median"),
        pct_eps_neg=("eps_now", lambda x: (x < 0).mean() * 100),
        mean_d_eps=(f"fwd_deps_{H}d", "mean"),
        median_d_eps=(f"fwd_deps_{H}d", "median"),
    ).round(3)
    print(grouped.to_string())
    print()
    print("  해석:")
    print("    flow_sign=+1 (외국인 매수 누적): eps_now < 0 이면 '매수 시점 종목 rich (가격 over-bought)'")
    print("    flow_sign=+1 의 mean_d_eps > 0 이면 'mean revert to 0+'")

    # ── flow 와 eps_now 의 직접 IC ──
    print("\n  → 더 정량적: flow_sum_3d vs 같은 시점의 eps_now IC")
    res = ic(p[flow_col], p["eps_now"])
    print(f"    IC(flow_sum_3d, eps_now) = {res['ic']:+.4f}  (N={res['n']:,})")
    print("    음수면 = 매수 누적 시 그 종목 ε 가 negative (rich) — overbought 신호")

    # ── C) Aggregate vs idiosyncratic ──
    print("\n=== C) Aggregate vs idiosyncratic ===")
    # daily 전체 외국인 net buy 합 (sum_3d 의 일자별 평균)
    daily_agg = flow.groupby("price_date").agg(
        agg_foreigner_sum_3d=("foreigner_sum_3d", "sum"),
        agg_foreigner_diff_1d=("foreigner_diff_1d", "sum"),
    )
    # daily 시장 yield
    daily_agg["y_3y"] = s3_full.reindex(daily_agg.index)
    daily_agg["y_10y"] = s10_full.reindex(daily_agg.index)
    daily_agg["dy_3y_fwd_21"] = (daily_agg["y_3y"].shift(-21) - daily_agg["y_3y"]) * 100
    daily_agg["dy_10y_fwd_21"] = (daily_agg["y_10y"].shift(-21) - daily_agg["y_10y"]) * 100
    daily_agg["dslope_fwd_21"] = daily_agg["dy_10y_fwd_21"] - daily_agg["dy_3y_fwd_21"]

    print("  Aggregate foreigner sum_3d (전체 KTB 합) → 시장 yield 변화 IC:")
    for tgt_col, label in [("dy_3y_fwd_21", "ΔY_3Y"), ("dy_10y_fwd_21", "ΔY_10Y"),
                            ("dslope_fwd_21", "Δslope")]:
        res = ic(daily_agg["agg_foreigner_sum_3d"], daily_agg[tgt_col])
        print(f"    {label:10s}: IC = {res['ic']:+.4f}  (N={res['n']:,})")
    print()
    print("  → agg flow 가 시장 yield (특히 10Y) 와 상관 있다면 macro flow 효과")
    print("  → 종목 specific 효과는 panel 의 idiosyncratic ε 변화 IC 로 확인")

    # ── D) 만기 bucket 별 ε reversion ──
    print("\n=== D) 만기 bucket 별 flow → Δε IC ===")
    panel["rem_bucket"] = pd.cut(panel["remain"], bins=[2, 3, 5, 7, 10, 13],
                                  labels=["2-3Y", "3-5Y", "5-7Y", "7-10Y", "10-13Y"])
    for bucket, sub in panel.groupby("rem_bucket", observed=True):
        res_y = ic(sub[flow_col], sub[f"fwd_dy_{H}d"])
        res_e = ic(sub[flow_col], sub[f"fwd_deps_{H}d"])
        res_eps_now = ic(sub[flow_col], sub["eps_now"])
        print(f"  {bucket}: N={res_y['n']:>5,d}  "
              f"IC(flow,ΔY)={res_y['ic']:+.3f}  "
              f"IC(flow,Δε)={res_e['ic']:+.3f}  "
              f"IC(flow,ε_now)={res_eps_now['ic']:+.3f}")
    print("\n  해석:")
    print("    IC(flow,ε_now) 가 음수 = 매수 시점 종목 rich")
    print("    IC(flow,Δε) 가 양수 = 21일 후 ε 가 positive 방향으로 revert (=가격 down=yield up)")

    # ── E) flow 크기 quintile 별 forward return ──
    print("\n=== E) flow 크기 quintile 별 forward ΔY ===")
    p = panel.dropna(subset=[flow_col, f"fwd_dy_{H}d"]).copy()
    p["flow_q"] = pd.qcut(p[flow_col], q=5, labels=["Q1 (매도)", "Q2", "Q3", "Q4", "Q5 (매수)"],
                          duplicates="drop")
    q = p.groupby("flow_q", observed=True).agg(
        N=(f"fwd_dy_{H}d", "size"),
        mean_dy=(f"fwd_dy_{H}d", "mean"),
        median_dy=(f"fwd_dy_{H}d", "median"),
        mean_eps_now=("eps_now", "mean"),
        mean_d_eps=(f"fwd_deps_{H}d", "mean"),
    ).round(2)
    print(q.to_string())
    print()
    print("  → Q5 (외국인 큰 매수) 의 mean_dy 가 양수 (yield 상승) = flow 후 가격 하락 패턴")
    print("  → Q5 의 mean_eps_now 가 음수 (rich) = overbought 시점")
    print("  → Q5 의 mean_d_eps 가 양수 = 21d 후 mean revert")

    print("\n[done]")


if __name__ == "__main__":
    main()
