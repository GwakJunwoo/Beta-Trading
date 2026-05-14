"""
12 — 최신 시그널 (V4b) 평가.

매일 close 후 다음 거래일 시그널을 산출:
  - KTB10F 외국인 sum_5d
  - 현물 외국인 aggregate sum_5d
  - USDKRW past 5d 변화 (KRW 强弱)
  → V4b mapping:
       SELL+SELL/KRW强 → -1.5, hold=21d
       SELL+SELL/KRW弱 → -0.7, hold=21d
       SELL+BUY/KRW强  → -1.0, hold=3d
       SELL+BUY/KRW弱  → -0.4, hold=3d
       그 외          → flat
"""
from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

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
DV01_KTB10F = 8.5


def load_fx():
    df = pd.read_excel(FX_PATH, sheet_name="Sheet1", header=None, skiprows=2, usecols=[0, 1])
    df.columns = ["price_date", "usdkrw"]
    df["price_date"] = pd.to_datetime(df["price_date"], errors="coerce")
    df["usdkrw"] = pd.to_numeric(df["usdkrw"], errors="coerce")
    return df.dropna().set_index("price_date")["usdkrw"].sort_index()


def load_panel(start="2025-01-01"):
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""SELECT price_date, foreigner FROM ktbf_netbuy
                       WHERE price_date >= %s AND tenor='KTB10F'""", (start,))
        f10 = pd.DataFrame(cur.fetchall()).rename(columns={"foreigner": "f10_for"})
        cur.execute("""SELECT price_date, SUM(foreigner_sum_5d) AS for_s5,
                              SUM(foreigner_diff_1d) AS for_d1
                       FROM ktb_trade_flow_features
                       WHERE price_date >= %s AND bond_code IS NOT NULL AND bond_code != ''
                       GROUP BY price_date""", (start,))
        cash = pd.DataFrame(cur.fetchall())
    for df in (f10, cash):
        df["price_date"] = pd.to_datetime(df["price_date"])
        for c in df.columns:
            if c != "price_date":
                df[c] = pd.to_numeric(df[c], errors="coerce")

    s10 = _load_label_series("10년지표", days=400)
    s10.index = pd.to_datetime(s10.index)
    fx = load_fx()

    p = f10.merge(cash, on="price_date", how="outer").sort_values("price_date").reset_index(drop=True)
    p["y_10y"] = p["price_date"].map(s10) * 100.0
    p["fx"] = p["price_date"].map(fx)
    p = p.dropna(subset=["y_10y", "fx", "f10_for", "for_s5"]).reset_index(drop=True)
    p["f10_s5"] = p["f10_for"].rolling(5, min_periods=1).sum()
    p["dfx_past_5"] = p["fx"] - p["fx"].shift(5)
    return p


def v4b_evaluate(row):
    fb = row["f10_s5"] > 0
    cb = row["for_s5"] > 0
    krw_strong = row["dfx_past_5"] < 0
    fut = "BUY" if fb else "SELL"
    cash = "BUY" if cb else "SELL"
    fx = "KRW强" if krw_strong else "KRW弱"
    combo = f"{fut}+{cash}/{fx}"
    if not fb and not cb:
        sig, hold = (-1.5 if krw_strong else -0.7), 21
    elif not fb and cb:
        sig, hold = (-1.0 if krw_strong else -0.4), 3
    else:
        sig, hold = 0.0, 0
    return combo, sig, hold


def main():
    print("[load] panel (2025-01- ~ 최신) ...")
    p = load_panel("2025-01-01")
    print(f"  {len(p):,} rows, latest = {p['price_date'].max().date()}\n")

    # 최근 15 영업일
    recent = p.tail(15).copy()
    rows = []
    for _, r in recent.iterrows():
        combo, sig, hold = v4b_evaluate(r)
        rows.append({
            "date": r["price_date"].strftime("%Y-%m-%d"),
            "y_10y_bp": round(r["y_10y"], 1),
            "fx": round(r["fx"], 2),
            "fx_5d_Δ": round(r["dfx_past_5"], 2) if pd.notna(r["dfx_past_5"]) else None,
            "f10_s5": int(r["f10_s5"]),
            "cash_for_s5": int(r["for_s5"]) if pd.notna(r["for_s5"]) else None,
            "combo": combo,
            "sig": sig,
            "hold_d": hold,
        })
    df = pd.DataFrame(rows)
    print("=" * 100)
    print("최근 15 영업일 V4b 시그널")
    print("=" * 100)
    print(df.to_string(index=False))
    print()

    # latest 시그널 디테일
    latest = p.iloc[-1]
    combo, sig, hold = v4b_evaluate(latest)
    print("=" * 100)
    print(f"★ Latest 시그널 — {latest['price_date'].strftime('%Y-%m-%d')}")
    print("=" * 100)
    print(f"\n  10Y yield        : {latest['y_10y']:.1f} bp")
    print(f"  USDKRW           : {latest['fx']:.2f}  (5d Δ: {latest['dfx_past_5']:+.2f}원)")
    print(f"  KTB10F 외국인 5d  : {int(latest['f10_s5']):+,d} 계약")
    print(f"  현물 외국인 5d    : {int(latest['for_s5']):+,d} (억원 추정)")
    print(f"\n  Combo classification : {combo}")
    print(f"  Signal               : {sig}")
    print(f"  Holding              : {hold}d" if hold else "  → FLAT (시그널 없음)")

    if sig != 0:
        action = "SHORT" if sig < 0 else "LONG"
        size_per_unit_per_contract = abs(sig)
        print(f"\n  ▶ Action : KTB10F {action}")
        print(f"     size  : {size_per_unit_per_contract:.2f} unit")
        print(f"     hold  : T+1 ~ T+{hold} (총 {hold} 영업일)")
        # historical 같은 시그널의 평균
        # 이전 시그널 통계는 11_v4_hybrid_hold.py 결과 인용
        hist = {
            (-1.5, 21): {"N": 32, "hit": 78.1, "avg_bp": 22.22},
            (-0.7, 21): {"N": 44, "hit": 68.2, "avg_bp": 5.81},
            (-1.0, 3):  {"N": 282, "hit": 54.3, "avg_bp": 1.44},
            (-0.4, 3):  {"N": 373, "hit": 55.8, "avg_bp": 0.58},
        }
        h = hist.get((sig, hold))
        if h:
            print(f"\n  ▶ Historical 통계 (이 시그널 강도, 2020-2026):")
            print(f"     N          : {h['N']}건")
            print(f"     hit rate   : {h['hit']:.1f}%")
            print(f"     평균 trade  : {h['avg_bp']:+.2f} bp")
            print(f"     1 unit × 1 계약 환산 평균: {h['avg_bp'] * DV01_KTB10F * 10000:,.0f}원")
            # 100 계약 사이즈 추정
            est_100 = sig * (-h['avg_bp'] / sig) * DV01_KTB10F * 100 * 10000  # 음수 처리
            est_won = abs(h['avg_bp']) * DV01_KTB10F * 100 * 10000
            print(f"     100 계약 사이즈 추정 평균 익: ≈ {est_won:,.0f}원 / trade")

    # 최근 같은 시그널 발생일 (last 3 occurrences)
    print()
    print("=" * 100)
    print(f"이 시그널 ({combo}) 최근 발생일 (latest 3건)")
    print("=" * 100)
    same_sig = []
    for _, r in p.iterrows():
        c, s, h = v4b_evaluate(r)
        if c == combo and s != 0:
            same_sig.append((r["price_date"], s, h, r["y_10y"]))
    if len(same_sig) >= 2:
        for d, s, h, y in same_sig[-5:]:
            print(f"  {d.strftime('%Y-%m-%d')}: sig={s}, hold={h}d, entry y10={y:.1f}bp")
    print()

    print("[done]")


if __name__ == "__main__":
    main()
