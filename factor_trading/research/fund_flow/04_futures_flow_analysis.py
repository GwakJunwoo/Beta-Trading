"""
04 — 외국인 선물 flow vs 현물 flow vs 미래 yield 변화.

가설 (user): "외국인이 선물 중심으로 들어와서 매수 → 시장 강세 정점 → 이후 yield 상승" ?

핵심 검증:
  1. KTB3F / KTB10F 외국인 net buy → 미래 ΔY_3Y / ΔY_10Y / Δslope IC
  2. 현물 aggregate flow vs 선물 flow lead/lag
  3. 선물 누적 (5d/10d) sum 의 cumulative position 효과
  4. "선물 매수 + 현물 매수" 동시 vs 선물만, 현물만 의 forward return
  5. 선물 큰 매수 quintile → 21d forward Y_3Y, Y_10Y 분포

선물 단위: 계약수 (KTB3F: ~1억원/계약, KTB10F: ~1억원/계약 액면 — 추정)
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
from app.routers.beta import _load_label_series

HORIZONS = [1, 3, 5, 10, 21]


def load_futures(start: str = "2020-01-01") -> pd.DataFrame:
    """ktbf_netbuy → long form (foreigner, institution, pension by tenor)."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT price_date, tenor, foreigner, institution, fund, pension, bank, insurance
            FROM ktbf_netbuy
            WHERE price_date >= %s AND tenor IN ('KTB3F','KTB10F')
            ORDER BY price_date, tenor
        """, (start,))
        rows = cur.fetchall()
    df = pd.DataFrame(rows)
    df["price_date"] = pd.to_datetime(df["price_date"])
    for c in ["foreigner", "institution", "fund", "pension", "bank", "insurance"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def load_cash_flow_agg(start: str = "2020-01-01") -> pd.DataFrame:
    """ktb_trade_flow_features → daily aggregate (모든 종목 sum)."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT price_date,
                   SUM(foreigner_diff_1d) AS for_d1,
                   SUM(foreigner_sum_3d)  AS for_s3,
                   SUM(foreigner_sum_5d)  AS for_s5,
                   SUM(foreigner_sum_10d) AS for_s10
            FROM ktb_trade_flow_features
            WHERE price_date >= %s AND bond_code IS NOT NULL AND bond_code != ''
            GROUP BY price_date
            ORDER BY price_date
        """, (start,))
        rows = cur.fetchall()
    df = pd.DataFrame(rows)
    df["price_date"] = pd.to_datetime(df["price_date"])
    for c in df.columns:
        if c != "price_date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def ic(x: pd.Series, y: pd.Series) -> dict:
    s = pd.DataFrame({"x": x, "y": y}).dropna()
    s = s[(s["x"] != 0) | (s["y"] != 0)]
    if len(s) < 30:
        return {"n": len(s), "ic": np.nan, "pval": np.nan}
    rho, p = spearmanr(s["x"], s["y"])
    return {"n": len(s), "ic": float(rho), "pval": float(p)}


def build_daily_panel(start: str = "2020-01-01") -> pd.DataFrame:
    """일자 단위 panel: 선물 (3F,10F foreigner) + 현물 agg + 시장 yield + forward changes."""
    fut = load_futures(start)
    cash = load_cash_flow_agg(start)
    s3 = _load_label_series("3년지표", days=2200)
    s10 = _load_label_series("10년지표", days=2200)

    # 선물 wide 형태: tenor × column
    fut3 = fut[fut["tenor"] == "KTB3F"][["price_date", "foreigner", "institution", "pension", "bank"]]
    fut3.columns = ["price_date", "f3_for", "f3_inst", "f3_pen", "f3_bnk"]
    fut10 = fut[fut["tenor"] == "KTB10F"][["price_date", "foreigner", "institution", "pension", "bank"]]
    fut10.columns = ["price_date", "f10_for", "f10_inst", "f10_pen", "f10_bnk"]

    panel = fut3.merge(fut10, on="price_date", how="outer")
    panel = panel.merge(cash, on="price_date", how="outer")
    panel = panel.sort_values("price_date").reset_index(drop=True)

    # 시장 yield 합치기
    s3.index = pd.to_datetime(s3.index)
    s10.index = pd.to_datetime(s10.index)
    panel["y_3y"] = panel["price_date"].map(s3) * 100.0   # bp 단위
    panel["y_10y"] = panel["price_date"].map(s10) * 100.0
    panel = panel.dropna(subset=["y_3y", "y_10y"]).reset_index(drop=True)

    # forward Δ (모든 horizon)
    for h in HORIZONS:
        panel[f"dy3_fwd_{h}"] = panel["y_3y"].shift(-h) - panel["y_3y"]
        panel[f"dy10_fwd_{h}"] = panel["y_10y"].shift(-h) - panel["y_10y"]
        panel[f"dslope_fwd_{h}"] = panel[f"dy10_fwd_{h}"] - panel[f"dy3_fwd_{h}"]

    # 선물 누적 (sum window)
    for col in ["f3_for", "f10_for"]:
        panel[f"{col}_s3"] = panel[col].rolling(3, min_periods=1).sum()
        panel[f"{col}_s5"] = panel[col].rolling(5, min_periods=1).sum()
        panel[f"{col}_s10"] = panel[col].rolling(10, min_periods=1).sum()

    return panel


def main():
    print("[load] 선물 + 현물 agg + 시장 yield ...")
    panel = build_daily_panel(start="2020-01-01")
    print(f"  panel: {len(panel):,} rows, {panel['price_date'].min().date()} ~ {panel['price_date'].max().date()}\n")

    # ── 0) 선물 vs 현물 외국인 flow 의 일치/괴리 ──
    print("=== 0) 선물 vs 현물 외국인 flow 상관 ===")
    print("    (선물 net buy 와 현물 sum_3d 의 동시점 IC)")
    print()
    for fcol, label in [("f3_for", "KTB3F 외국인 daily"),
                        ("f10_for", "KTB10F 외국인 daily"),
                        ("f3_for_s5", "KTB3F 외국인 5d cum"),
                        ("f10_for_s5", "KTB10F 외국인 5d cum")]:
        res = ic(panel[fcol], panel["for_s5"])
        print(f"  {label:30s} vs 현물(for_s5):  IC={res['ic']:+.3f}  N={res['n']:,}")
    print()

    # ── 1) 선물 외국인 net buy → forward yield IC ──
    print("=== 1) 선물 외국인 net buy → forward ΔY IC ===")
    print("    음수 = buy 후 yield 하락 (정방향 정보)")
    print("    양수 = buy 후 yield 상승 (역방향 / contrarian)")
    print()
    print(f"  {'Signal':28s} {'h=1':>10s} {'h=3':>10s} {'h=5':>10s} {'h=10':>10s} {'h=21':>10s}")
    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for fcol in ["f3_for", "f3_for_s3", "f3_for_s5", "f3_for_s10"]:
        line = f"  {fcol+' → ΔY_3Y':28s}"
        for h in HORIZONS:
            res = ic(panel[fcol], panel[f"dy3_fwd_{h}"])
            line += f" {res['ic']:+.3f}".rjust(11)
        print(line)
    print()
    for fcol in ["f10_for", "f10_for_s3", "f10_for_s5", "f10_for_s10"]:
        line = f"  {fcol+' → ΔY_10Y':28s}"
        for h in HORIZONS:
            res = ic(panel[fcol], panel[f"dy10_fwd_{h}"])
            line += f" {res['ic']:+.3f}".rjust(11)
        print(line)
    print()
    # 슬로프
    print("  슬로프 (Δslope = ΔY_10Y − ΔY_3Y):")
    for fcol in ["f3_for_s5", "f10_for_s5"]:
        line = f"  {fcol+' → Δslope':28s}"
        for h in HORIZONS:
            res = ic(panel[fcol], panel[f"dslope_fwd_{h}"])
            line += f" {res['ic']:+.3f}".rjust(11)
        print(line)
    print()

    # ── 2) 현물 vs 선물 lead-lag ──
    print("=== 2) 현물 vs 선물 lead-lag (외국인) ===")
    print("    선물이 lead 하면 corr(선물(t), 현물(t+k)) 강하게 양수")
    print()
    base = panel[["price_date", "for_s5", "f3_for_s5", "f10_for_s5"]].dropna().reset_index(drop=True)
    for fcol in ["f3_for_s5", "f10_for_s5"]:
        line = f"  {fcol+' (현물 = for_s5)':30s}"
        for lag in [-5, -3, -1, 0, 1, 3, 5]:
            cash_shift = base["for_s5"].shift(-lag)
            res = ic(base[fcol], cash_shift)
            sign = "+" if lag >= 0 else ""
            line += f" lag{sign}{lag}:{res['ic']:+.2f}"
        print(line)
    print("    (lag=+k: 선물(t) vs 현물(t+k). 양수 IC at lag>0 = 선물 leads 현물)")
    print()

    # ── 3) 선물 5d cum quintile → forward yield ──
    print("=== 3) 선물 5d cum quintile → forward Y 평균 변화 (bp) ===")
    print()
    for fcol, ycol, label in [
        ("f3_for_s5", "dy3_fwd_21", "KTB3F 외국인 5d cum → 21d ΔY_3Y"),
        ("f10_for_s5", "dy10_fwd_21", "KTB10F 외국인 5d cum → 21d ΔY_10Y"),
        ("f10_for_s5", "dslope_fwd_21", "KTB10F 외국인 5d cum → 21d Δslope"),
    ]:
        sub = panel[[fcol, ycol]].dropna().copy()
        sub = sub[sub[fcol] != 0]
        sub["q"] = pd.qcut(sub[fcol], q=5, labels=["Q1 매도", "Q2", "Q3", "Q4", "Q5 매수"],
                           duplicates="drop")
        g = sub.groupby("q", observed=True).agg(
            n=(ycol, "size"),
            mean_dy=(ycol, "mean"),
            median_dy=(ycol, "median"),
        ).round(2)
        print(f"  ▶ {label}")
        print(g.to_string())
        print(f"    Q5−Q1 mean diff: {g.loc['Q5 매수','mean_dy'] - g.loc['Q1 매도','mean_dy']:+.2f} bp")
        print()

    # ── 4) "선물 매수 + 현물 매수" 동시 vs 한쪽만 의 forward return ──
    print("=== 4) 선물 × 현물 buy 조합 → 21d forward ΔY_10Y (bp) ===")
    print()
    sub = panel[["f10_for_s5", "for_s5", "dy10_fwd_21", "dy3_fwd_21"]].dropna().copy()
    # 부호로만 분류
    sub["fut_buy"] = sub["f10_for_s5"] > 0
    sub["cash_buy"] = sub["for_s5"] > 0
    g = sub.groupby(["fut_buy", "cash_buy"]).agg(
        n=("dy10_fwd_21", "size"),
        mean_dy10=("dy10_fwd_21", "mean"),
        median_dy10=("dy10_fwd_21", "median"),
        mean_dy3=("dy3_fwd_21", "mean"),
    ).round(2)
    print(g.to_string())
    print()
    print("  해석:")
    print("    (True,True)  = 선물 + 현물 둘다 buy → 시장 최강세 정점 가능성")
    print("    (True,False) = 선물만 buy → 현물 외국인 후행?")
    print("    (False,True) = 현물만 buy → 펀더멘털 매수?")
    print()

    # ── 5) "큰 매수" 극단 케이스 추적 ──
    print("=== 5) 외국인 KTB10F 큰 매수 (top 5%) 의 그 후 yield 추이 ===")
    sub = panel[["price_date", "f10_for_s5", "y_10y", "dy10_fwd_5",
                 "dy10_fwd_10", "dy10_fwd_21"]].dropna().copy()
    thr = sub["f10_for_s5"].quantile(0.95)
    big = sub[sub["f10_for_s5"] >= thr]
    print(f"  threshold (top 5%): f10_for_s5 ≥ {thr:.0f} 계약")
    print(f"  N={len(big):,}")
    print(f"  대형 매수 후 평균 ΔY_10Y:")
    print(f"    5d:  {big['dy10_fwd_5'].mean():+.2f} bp")
    print(f"    10d: {big['dy10_fwd_10'].mean():+.2f} bp")
    print(f"    21d: {big['dy10_fwd_21'].mean():+.2f} bp")
    print()
    # 대형 매도 (bottom 5%)
    thr_lo = sub["f10_for_s5"].quantile(0.05)
    bigs = sub[sub["f10_for_s5"] <= thr_lo]
    print(f"  threshold (bottom 5%): f10_for_s5 ≤ {thr_lo:.0f} 계약")
    print(f"  N={len(bigs):,}")
    print(f"  대형 매도 후 평균 ΔY_10Y:")
    print(f"    5d:  {bigs['dy10_fwd_5'].mean():+.2f} bp")
    print(f"    10d: {bigs['dy10_fwd_10'].mean():+.2f} bp")
    print(f"    21d: {bigs['dy10_fwd_21'].mean():+.2f} bp")
    print()

    # ── 6) 연도별 안정성 ──
    print("=== 6) 연도별 IC (KTB10F 외국인 5d cum → 21d ΔY_10Y) ===")
    panel["year"] = panel["price_date"].dt.year
    for yr, g in panel.groupby("year"):
        res = ic(g["f10_for_s5"], g["dy10_fwd_21"])
        if not np.isnan(res["ic"]):
            print(f"  {yr}: N={res['n']:>4,d}  IC={res['ic']:+.3f}")
    print()

    print("[done]")


if __name__ == "__main__":
    main()
