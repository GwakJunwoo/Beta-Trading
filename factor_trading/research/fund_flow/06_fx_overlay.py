"""
06 — FX overlay: 외국인 선물 sell + 현물 buy 의 이유를 환율로 설명할 수 있나?

가설:
  H1 (FX-hedged carry): 외국인 = 현물 long + 선물환 hedge (KRW 약세 우려 시 헷지)
       → USDKRW 상승 (KRW 약세) 시점에 더 많이 일어나야 함
       → 또는 USDKRW 상승 예상 시 (forward) 미리 hedge 진입
  H2 (Information edge): 외국인이 환율 움직임 예상 → KRW 강세 예상 시 KRW 채권 매수
       → 매수 후 KRW 강세 (USDKRW 하락) 나타남
  H3 (Basis trade): 선물 basis 가 cheap 하면 선물 매도 + 현물 매수 (단순 차익)
       → 환율과 무관, basis 자체에 driver

검증:
  A) (fut_buy, cash_buy) 4 조합별 past/forward ΔUSDKRW
  B) USDKRW 변동 IC: 외국인 flow ↔ FX past/forward
  C) FX regime (KRW 약세/강세 진행중) 조건부 외국인 flow → forward yield
  D) (sell+buy) 케이스의 환율 distribution
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

FX_PATH = r"C:\Users\infomax\Desktop\USDKRW_INFOMAX.xlsx"
HORIZONS = [5, 10, 21]


def load_fx() -> pd.Series:
    df = pd.read_excel(FX_PATH, sheet_name="Sheet1", header=None, skiprows=2, usecols=[0, 1])
    df.columns = ["price_date", "usdkrw"]
    df["price_date"] = pd.to_datetime(df["price_date"], errors="coerce")
    df["usdkrw"] = pd.to_numeric(df["usdkrw"], errors="coerce")
    df = df.dropna()
    return df.set_index("price_date")["usdkrw"].sort_index()


def load_futures(start: str = "2020-01-01") -> pd.DataFrame:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT price_date, tenor, foreigner
            FROM ktbf_netbuy
            WHERE price_date >= %s AND tenor IN ('KTB3F','KTB10F')
        """, (start,))
        rows = cur.fetchall()
    df = pd.DataFrame(rows)
    df["price_date"] = pd.to_datetime(df["price_date"])
    df["foreigner"] = pd.to_numeric(df["foreigner"], errors="coerce").fillna(0.0)
    return df


def load_cash_agg(start: str = "2020-01-01") -> pd.DataFrame:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT price_date, SUM(foreigner_sum_5d) AS for_s5
            FROM ktb_trade_flow_features
            WHERE price_date >= %s AND bond_code IS NOT NULL AND bond_code != ''
            GROUP BY price_date
        """, (start,))
        rows = cur.fetchall()
    df = pd.DataFrame(rows)
    df["price_date"] = pd.to_datetime(df["price_date"])
    df["for_s5"] = pd.to_numeric(df["for_s5"], errors="coerce")
    return df


def ic(x, y):
    s = pd.DataFrame({"x": x, "y": y}).dropna()
    s = s[(s["x"] != 0) | (s["y"] != 0)]
    if len(s) < 30:
        return {"n": len(s), "ic": np.nan}
    rho, _ = spearmanr(s["x"], s["y"])
    return {"n": len(s), "ic": float(rho)}


def build_panel(start="2020-01-01"):
    fut = load_futures(start)
    cash = load_cash_agg(start)
    s3 = _load_label_series("3년지표", days=2200)
    s10 = _load_label_series("10년지표", days=2200)
    fx = load_fx()

    fut3 = fut[fut["tenor"] == "KTB3F"][["price_date", "foreigner"]].rename(columns={"foreigner": "f3_for"})
    fut10 = fut[fut["tenor"] == "KTB10F"][["price_date", "foreigner"]].rename(columns={"foreigner": "f10_for"})

    p = fut3.merge(fut10, on="price_date", how="outer").merge(cash, on="price_date", how="outer")
    p = p.sort_values("price_date").reset_index(drop=True)

    s3.index = pd.to_datetime(s3.index)
    s10.index = pd.to_datetime(s10.index)
    p["y_3y"] = p["price_date"].map(s3) * 100.0
    p["y_10y"] = p["price_date"].map(s10) * 100.0
    p["fx"] = p["price_date"].map(fx)
    p = p.dropna(subset=["y_3y", "y_10y", "fx"]).reset_index(drop=True)

    p["f10_for_s5"] = p["f10_for"].rolling(5, min_periods=1).sum()
    for h in HORIZONS:
        p[f"dy3_fwd_{h}"] = p["y_3y"].shift(-h) - p["y_3y"]
        p[f"dy10_fwd_{h}"] = p["y_10y"].shift(-h) - p["y_10y"]
        p[f"dfx_fwd_{h}"] = p["fx"].shift(-h) - p["fx"]              # 원/USD bp 비교용
        p[f"dfx_pct_fwd_{h}"] = (p["fx"].shift(-h) / p["fx"] - 1) * 100  # %
    for h in [5, 10, 21]:
        p[f"dy10_past_{h}"] = p["y_10y"] - p["y_10y"].shift(h)
        p[f"dfx_past_{h}"] = p["fx"] - p["fx"].shift(h)
        p[f"dfx_pct_past_{h}"] = (p["fx"] / p["fx"].shift(h) - 1) * 100

    return p


def main():
    print("[load] panel with FX ...")
    p = build_panel("2020-01-01")
    print(f"  panel: {len(p):,} rows  {p['price_date'].min().date()} ~ {p['price_date'].max().date()}\n")

    # ── A) 4 조합별 past/forward FX 변동 ──
    print("=== A) (선물, 현물) 조합별 past/forward USDKRW 변동 ===")
    print("    (양수 = KRW 약세, 즉 USDKRW 상승)")
    print()
    sub = p[["f10_for_s5", "for_s5", "dfx_past_5", "dfx_past_21",
             "dfx_fwd_5", "dfx_fwd_21", "dy10_fwd_21"]].dropna().copy()
    sub["fut_buy"] = sub["f10_for_s5"] > 0
    sub["cash_buy"] = sub["for_s5"] > 0
    g = sub.groupby(["fut_buy", "cash_buy"]).agg(
        n=("dy10_fwd_21", "size"),
        past5_fx=("dfx_past_5", "mean"),
        past21_fx=("dfx_past_21", "mean"),
        fwd5_fx=("dfx_fwd_5", "mean"),
        fwd21_fx=("dfx_fwd_21", "mean"),
        fwd21_dy10=("dy10_fwd_21", "mean"),
    ).round(2)
    print(g.to_string())
    print()
    print("  해석:")
    print("    past21_fx > 0 = flow 시점 이전 21일간 KRW 약세 진행")
    print("    fwd21_fx  > 0 = flow 후 21일간 KRW 약세 진행")
    print()

    # ── B) FX 변동 vs 외국인 flow IC ──
    print("=== B) 외국인 flow vs USDKRW past/forward IC ===")
    print("    음수 = flow buy 시점에 KRW 강세 (USDKRW 하락)")
    print("    양수 = flow buy 시점에 KRW 약세 (USDKRW 상승)")
    print()
    print(f"  {'Signal':30s} {'past 5d':>10s} {'past 21d':>10s} {'fwd 5d':>10s} {'fwd 21d':>10s}")
    for fcol, label in [("f10_for_s5", "KTB10F 5d cum"),
                         ("for_s5", "현물 외국인 5d")]:
        line = f"  {label:30s}"
        for col in ["dfx_past_5", "dfx_past_21", "dfx_fwd_5", "dfx_fwd_21"]:
            res = ic(p[fcol], p[col])
            line += f" {res['ic']:+.3f}".rjust(11)
        print(line)
    print()

    # ── C) FX regime 조건부 forward yield ──
    print("=== C) FX regime × (선물, 현물) 조합 → 21d forward ΔY_10Y ===")
    print("    fx_past_5 > 0 (KRW 약세 진행중) vs < 0 (KRW 강세 진행중)")
    print()
    sub2 = p[["f10_for_s5", "for_s5", "dfx_past_5", "dy10_fwd_21"]].dropna().copy()
    sub2["fut_buy"] = sub2["f10_for_s5"] > 0
    sub2["cash_buy"] = sub2["for_s5"] > 0
    sub2["fx_weak"] = sub2["dfx_past_5"] > 0   # KRW 약세 진행중
    g = sub2.groupby(["fut_buy", "cash_buy", "fx_weak"]).agg(
        n=("dy10_fwd_21", "size"),
        mean_fwd=("dy10_fwd_21", "mean"),
    ).round(2)
    print(g.to_string())
    print()

    # ── D) (sell + buy) 케이스 환율 distribution ──
    print("=== D) (선물 sell + 현물 buy) 케이스 환율 환경 ===")
    sb = p[(p["f10_for_s5"] < 0) & (p["for_s5"] > 0)].copy()
    print(f"  N = {len(sb)}")
    print(f"  past 5d ΔFX (KRW): mean={sb['dfx_past_5'].mean():+.2f}, median={sb['dfx_past_5'].median():+.2f}")
    print(f"  past 21d ΔFX     : mean={sb['dfx_past_21'].mean():+.2f}, median={sb['dfx_past_21'].median():+.2f}")
    print(f"  fwd 5d ΔFX       : mean={sb['dfx_fwd_5'].mean():+.2f}, median={sb['dfx_fwd_5'].median():+.2f}")
    print(f"  fwd 21d ΔFX      : mean={sb['dfx_fwd_21'].mean():+.2f}, median={sb['dfx_fwd_21'].median():+.2f}")
    print()
    # 전체 평균과 비교
    all_p = p[["dfx_past_5", "dfx_past_21", "dfx_fwd_5", "dfx_fwd_21"]].dropna()
    print(f"  ▶ 전체 baseline (참고):")
    print(f"  past 5d : {all_p['dfx_past_5'].mean():+.2f} | past 21d: {all_p['dfx_past_21'].mean():+.2f}"
          f" | fwd 5d: {all_p['dfx_fwd_5'].mean():+.2f} | fwd 21d: {all_p['dfx_fwd_21'].mean():+.2f}")
    print()

    # ── E) (sell + sell) 와 (sell + buy) 의 fwd ΔFX 비교 ──
    print("=== E) 4 조합의 forward FX% (KRW 변동 % 단위) ===")
    sub3 = p[["f10_for_s5", "for_s5",
              "dfx_pct_fwd_5", "dfx_pct_fwd_10", "dfx_pct_fwd_21"]].dropna().copy()
    sub3["fut_buy"] = sub3["f10_for_s5"] > 0
    sub3["cash_buy"] = sub3["for_s5"] > 0
    g = sub3.groupby(["fut_buy", "cash_buy"]).agg(
        n=("dfx_pct_fwd_5", "size"),
        fwd5_pct=("dfx_pct_fwd_5", "mean"),
        fwd10_pct=("dfx_pct_fwd_10", "mean"),
        fwd21_pct=("dfx_pct_fwd_21", "mean"),
    ).round(3)
    print(g.to_string())
    print()

    # ── F) FX past 통제 후 외국인 flow → forward yield 의 잔여 ──
    print("=== F) past FX + past Δy 둘 다 통제 후 외국인 flow forward IC ===")
    # 2-factor residual: dy10_fwd_21 = a + b1*dy10_past_5 + b2*dfx_past_5 + resid
    cols = ["dy10_fwd_21", "dy10_past_5", "dfx_past_5", "f10_for_s5", "for_s5"]
    rdf = p[cols].dropna().reset_index(drop=True)
    X = rdf[["dy10_past_5", "dfx_past_5"]].values
    y = rdf["dy10_fwd_21"].values
    Xc = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    resid = y - Xc @ beta
    rdf["resid"] = resid
    res_fut = ic(rdf["f10_for_s5"], rdf["resid"])
    res_cash = ic(rdf["for_s5"], rdf["resid"])
    print(f"  control = past Δy_10Y_5d + past ΔFX_5d")
    print(f"  KTB10F 5d cum → ΔY_10Y_21d residual: IC = {res_fut['ic']:+.3f}  N={res_fut['n']:,}")
    print(f"  현물 외국인 5d → ΔY_10Y_21d residual: IC = {res_cash['ic']:+.3f}  N={res_cash['n']:,}")
    print()

    print("[done]")


if __name__ == "__main__":
    main()
