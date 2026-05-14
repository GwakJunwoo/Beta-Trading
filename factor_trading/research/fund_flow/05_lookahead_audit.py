"""
05 — Look-ahead audit & flow vs momentum 분리.

user 지적: "당일/직전 약세인 날에 외국인이 매도 → 그 후도 약세" 라면,
forward IC 는 단지 momentum continuation 일 뿐, flow 자체의 예측력이 아니다.

검증:
  A) past Δy (t-5→t, t-21→t) 와 flow 의 동시점 상관 — flow 가 trend 의 *반영* 인지
  B) past Δy 와 forward Δy 의 corr — momentum 자체가 얼마나 강한가
  C) flow → forward Δy IC 의 *조건부* 효과: past Δy 통제 후에도 flow IC 가 유지되는가
  D) "(sell + sell) 21d 후 +11bp" 가 단순 momentum 인가 vs flow 추가 효과인가
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


def build_daily_panel(start: str = "2020-01-01") -> pd.DataFrame:
    fut = load_futures(start)
    cash = load_cash_flow_agg(start)
    s3 = _load_label_series("3년지표", days=2200)
    s10 = _load_label_series("10년지표", days=2200)

    fut3 = fut[fut["tenor"] == "KTB3F"][["price_date", "foreigner"]]
    fut3.columns = ["price_date", "f3_for"]
    fut10 = fut[fut["tenor"] == "KTB10F"][["price_date", "foreigner"]]
    fut10.columns = ["price_date", "f10_for"]

    panel = fut3.merge(fut10, on="price_date", how="outer")
    panel = panel.merge(cash, on="price_date", how="outer")
    panel = panel.sort_values("price_date").reset_index(drop=True)

    s3.index = pd.to_datetime(s3.index)
    s10.index = pd.to_datetime(s10.index)
    panel["y_3y"] = panel["price_date"].map(s3) * 100.0
    panel["y_10y"] = panel["price_date"].map(s10) * 100.0
    panel = panel.dropna(subset=["y_3y", "y_10y"]).reset_index(drop=True)

    for h in HORIZONS:
        panel[f"dy3_fwd_{h}"] = panel["y_3y"].shift(-h) - panel["y_3y"]
        panel[f"dy10_fwd_{h}"] = panel["y_10y"].shift(-h) - panel["y_10y"]
        panel[f"dslope_fwd_{h}"] = panel[f"dy10_fwd_{h}"] - panel[f"dy3_fwd_{h}"]

    for col in ["f3_for", "f10_for"]:
        panel[f"{col}_s5"] = panel[col].rolling(5, min_periods=1).sum()
    return panel


def ic(x: pd.Series, y: pd.Series) -> dict:
    s = pd.DataFrame({"x": x, "y": y}).dropna()
    s = s[(s["x"] != 0) | (s["y"] != 0)]
    if len(s) < 30:
        return {"n": len(s), "ic": np.nan, "pval": np.nan}
    rho, p = spearmanr(s["x"], s["y"])
    return {"n": len(s), "ic": float(rho), "pval": float(p)}


def main():
    print("[load] panel ...")
    panel = build_daily_panel(start="2020-01-01")

    # past Δy 추가
    for h in [5, 10, 21]:
        panel[f"dy3_past_{h}"] = panel["y_3y"] - panel["y_3y"].shift(h)
        panel[f"dy10_past_{h}"] = panel["y_10y"] - panel["y_10y"].shift(h)

    # ── A) flow 가 이미 진행중인 trend 의 반영인가? ──
    print("\n=== A) flow vs *past* Δy 상관 (flow 가 trend 반영인지) ===")
    print("    음수 IC: flow 매수 시점에 이미 yield 하락중 (=강세 진행중)")
    print("    양수 IC: flow 매수 시점에 yield 상승중 (=매도세 진행중 → contrarian buy)")
    print()
    print(f"  {'Signal':30s} {'past 5d':>12s} {'past 10d':>12s} {'past 21d':>12s}")
    for fcol, tgt, label in [
        ("f3_for_s5",  "dy3_past",  "KTB3F 5d cum vs past ΔY_3Y"),
        ("f10_for_s5", "dy10_past", "KTB10F 5d cum vs past ΔY_10Y"),
        ("for_s5",     "dy3_past",  "현물 외국인 5d vs past ΔY_3Y"),
        ("for_s5",     "dy10_past", "현물 외국인 5d vs past ΔY_10Y"),
    ]:
        line = f"  {label:30s}"
        for h in [5, 10, 21]:
            res = ic(panel[fcol], panel[f"{tgt}_{h}"])
            line += f" {res['ic']:+.3f}".rjust(12)
        print(line)
    print()

    # ── B) momentum 자체의 강도 ──
    print("=== B) Pure momentum: past Δy → forward Δy IC ===")
    print("    (flow 무관, 단순 시계열 momentum)")
    print()
    print(f"  {'Past→Forward':30s} {'fwd 5d':>10s} {'fwd 10d':>10s} {'fwd 21d':>10s}")
    for past_h in [5, 10, 21]:
        for which, label in [("3", "ΔY_3Y"), ("10", "ΔY_10Y")]:
            line = f"  past_{past_h}d {label:18s}"
            for fwd_h in [5, 10, 21]:
                res = ic(panel[f"dy{which}_past_{past_h}"], panel[f"dy{which}_fwd_{fwd_h}"])
                line += f" {res['ic']:+.3f}".rjust(11)
            print(line)
    print("    음수면 mean revert, 양수면 momentum continuation")
    print()

    # ── C) past Δy 통제 후 flow 의 잔여 효과 ──
    print("=== C) past Δy 통제 후 flow 의 forward IC (residual) ===")
    print("    방법: forward Δy 를 past Δy 로 OLS 회귀 후 residual 과 flow 의 IC")
    print()

    def residual(y_col, x_col):
        df = panel[[y_col, x_col]].dropna()
        if len(df) < 50:
            return pd.Series(dtype=float)
        x = df[x_col].values
        y = df[y_col].values
        b = np.cov(x, y, ddof=0)[0, 1] / np.var(x)
        a = y.mean() - b * x.mean()
        resid = y - (a + b * x)
        return pd.Series(resid, index=df.index)

    print(f"  {'Signal':40s} {'raw IC':>10s} {'past-removed IC':>18s}")
    for fcol, fwd_col, past_col, label in [
        ("f3_for_s5",  "dy3_fwd_21",  "dy3_past_5",  "KTB3F 5d → ΔY_3Y_21d"),
        ("f10_for_s5", "dy10_fwd_21", "dy10_past_5", "KTB10F 5d → ΔY_10Y_21d"),
        ("for_s5",     "dy3_fwd_21",  "dy3_past_5",  "현물 5d → ΔY_3Y_21d"),
        ("for_s5",     "dy10_fwd_21", "dy10_past_5", "현물 5d → ΔY_10Y_21d"),
    ]:
        raw = ic(panel[fcol], panel[fwd_col])
        resid = residual(fwd_col, past_col)
        flow_aligned = panel[fcol].reindex(resid.index)
        clean = ic(flow_aligned, resid)
        print(f"  {label:40s} {raw['ic']:+.3f}".ljust(53) + f"  {clean['ic']:+.3f}".rjust(11))
    print()

    # ── D) (sell + sell) 케이스에 momentum 이 얼마나 작용했나? ──
    print("=== D) (선물 sell + 현물 sell) 케이스의 past vs forward ===")
    sub = panel[["f10_for_s5", "for_s5", "dy10_past_5", "dy10_past_21",
                 "dy10_fwd_5", "dy10_fwd_21"]].dropna().copy()
    sub["fut_buy"] = sub["f10_for_s5"] > 0
    sub["cash_buy"] = sub["for_s5"] > 0
    g = sub.groupby(["fut_buy", "cash_buy"]).agg(
        n=("dy10_fwd_21", "size"),
        past_dy10_5d=("dy10_past_5", "mean"),
        past_dy10_21d=("dy10_past_21", "mean"),
        fwd_dy10_5d=("dy10_fwd_5", "mean"),
        fwd_dy10_21d=("dy10_fwd_21", "mean"),
    ).round(2)
    print(g.to_string())
    print()
    print("  해석:")
    print("    past_dy10 가 양수면 → flow 시점 직전에 이미 yield 상승 중 (약세 진행중)")
    print("    fwd_dy10 가 past_dy10 와 비슷한 부호 = momentum continuation (flow 효과 X)")
    print("    fwd_dy10 가 past 보다 훨씬 크면 → flow 자체의 추가 정보")
    print()

    # ── E) flow strength quintile + past Δy 부호 조합 ──
    print("=== E) flow quintile × past trend 부호 → forward ΔY_10Y (bp) ===")
    sub = panel[["f10_for_s5", "dy10_past_5", "dy10_fwd_21"]].dropna().copy()
    sub = sub[sub["f10_for_s5"] != 0]
    sub["fq"] = pd.qcut(sub["f10_for_s5"], q=5,
                         labels=["Q1 sell", "Q2", "Q3", "Q4", "Q5 buy"], duplicates="drop")
    sub["past_up"] = sub["dy10_past_5"] > 0   # yield 직전 상승 = 약세
    g = sub.groupby(["fq", "past_up"], observed=True).agg(
        n=("dy10_fwd_21", "size"),
        mean_fwd=("dy10_fwd_21", "mean"),
    ).round(2)
    print(g.to_string())
    print()
    print("  핵심: Q1 sell × past_up=True 가 +11bp 수준이면 단순 momentum")
    print("        Q1 sell × past_up=False (강세 진행중인데 sell) 가 양수면 flow 의 진짜 정보")
    print()

    print("[done]")


if __name__ == "__main__":
    main()
