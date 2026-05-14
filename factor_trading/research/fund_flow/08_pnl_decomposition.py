"""
08 — V2 전략 P&L attribution (trade-level 분해).

목표: 왜 2021/2022 만 익이고 2023/2024 는 손실인가?

분석:
  A) trade-level P&L (entry 일자 기준) 으로 재정의
       trade_pnl_t = sig_t × (-21d forward ΔY_10Y)
       sum_t trade_pnl_t == 누적 daily P&L (∵ overlapping 효과)
  B) 연도별 × 시그널 값별 P&L 분해 (어느 시그널이 익/손실?)
  C) 연도별 × 시그널 부호별 (short vs long) 분해
  D) 시그널 vs forward yield 부호 일치율 (= hit rate by year)
  E) 손실 연도의 worst trades (어떤 macro 일자에 큰 손실?)
  F) regime 변수 (KRW 변동, yield level, US 금리차 ?) 의 연도별 차이
  G) 시그널 frequency (각 시그널의 빈도가 연도별로 달라졌나?)
  H) "in-sample 강한 -2.0 시그널" 의 forward yield 가 2023+ 에는 어떻게 변했나
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
HOLD = 21


def load_fx():
    df = pd.read_excel(FX_PATH, sheet_name="Sheet1", header=None, skiprows=2, usecols=[0, 1])
    df.columns = ["price_date", "usdkrw"]
    df["price_date"] = pd.to_datetime(df["price_date"], errors="coerce")
    df["usdkrw"] = pd.to_numeric(df["usdkrw"], errors="coerce")
    return df.dropna().set_index("price_date")["usdkrw"].sort_index()


def load_panel(start="2020-01-01"):
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""SELECT price_date, foreigner FROM ktbf_netbuy
                       WHERE price_date >= %s AND tenor='KTB10F'""", (start,))
        f10 = pd.DataFrame(cur.fetchall()).rename(columns={"foreigner": "f10_for"})
        cur.execute("""SELECT price_date, SUM(foreigner_sum_5d) AS for_s5
                       FROM ktb_trade_flow_features
                       WHERE price_date >= %s AND bond_code IS NOT NULL AND bond_code != ''
                       GROUP BY price_date""", (start,))
        cash = pd.DataFrame(cur.fetchall())
    for df in (f10, cash):
        df["price_date"] = pd.to_datetime(df["price_date"])
        for c in df.columns:
            if c != "price_date":
                df[c] = pd.to_numeric(df[c], errors="coerce")

    s10 = _load_label_series("10년지표", days=2200)
    s3 = _load_label_series("3년지표", days=2200)
    s10.index = pd.to_datetime(s10.index)
    s3.index = pd.to_datetime(s3.index)
    fx = load_fx()

    p = f10.merge(cash, on="price_date", how="outer").sort_values("price_date").reset_index(drop=True)
    p["y_10y"] = p["price_date"].map(s10) * 100.0
    p["y_3y"] = p["price_date"].map(s3) * 100.0
    p["fx"] = p["price_date"].map(fx)
    p = p.dropna(subset=["y_10y", "y_3y", "fx"]).reset_index(drop=True)

    p["f10_s5"] = p["f10_for"].rolling(5, min_periods=1).sum()
    p["dfx_past_5"] = p["fx"] - p["fx"].shift(5)
    p["dfx_past_21"] = p["fx"] - p["fx"].shift(21)
    p["dy10_past_5"] = p["y_10y"] - p["y_10y"].shift(5)
    p["dy10_past_21"] = p["y_10y"] - p["y_10y"].shift(21)

    p["dy10_fwd_21"] = p["y_10y"].shift(-HOLD) - p["y_10y"]
    p["dy3_fwd_21"] = p["y_3y"].shift(-HOLD) - p["y_3y"]
    p["dfx_fwd_21"] = p["fx"].shift(-HOLD) - p["fx"]
    return p


def signal_v2(row):
    fb = row["f10_s5"] > 0
    cb = row["for_s5"] > 0
    krw_strong = row["dfx_past_5"] < 0
    if not fb and not cb:
        return -1.5 if krw_strong else -0.7
    if not fb and cb:
        return -1.0 if krw_strong else -0.4
    if fb and not cb:
        return -0.3 if krw_strong else 0.0
    return +0.8 if krw_strong else +0.3


def classify_combo(row):
    fb = row["f10_s5"] > 0
    cb = row["for_s5"] > 0
    krw_strong = row["dfx_past_5"] < 0
    fut = "BUY" if fb else "SELL"
    cash = "BUY" if cb else "SELL"
    fx = "KRW强" if krw_strong else "KRW弱"
    return f"{fut}+{cash}/{fx}"


def main():
    print("[load] panel ...")
    p = load_panel("2020-01-01")
    p["sig"] = p.apply(signal_v2, axis=1)
    p["combo"] = p.apply(classify_combo, axis=1)
    p["year"] = p["price_date"].dt.year
    # trade-level P&L: sig × (-21d ΔY_10Y)
    p["trade_pnl"] = p["sig"] * (-p["dy10_fwd_21"])
    trades = p.dropna(subset=["dy10_fwd_21"]).copy()
    print(f"  {len(trades):,} trades  ({trades['price_date'].min().date()} ~ {trades['price_date'].max().date()})\n")

    # 검산: trade sum vs daily-overlap sum
    total_check = trades["trade_pnl"].sum()
    print(f"  Total P&L (trade attribution): {total_check:+.0f} bp\n")

    # ────────────── A) 연도 × 시그널 강도 ──────────────
    print("=" * 88)
    print("A) 연도 × 시그널 강도 분해 (P&L total bp)")
    print("=" * 88)
    pivot_pnl = trades.pivot_table(index="year", columns="sig", values="trade_pnl",
                                    aggfunc="sum").round(0).fillna(0)
    pivot_n = trades.pivot_table(index="year", columns="sig", values="trade_pnl",
                                  aggfunc="size").fillna(0).astype(int)
    print("\nP&L (bp):")
    print(pivot_pnl.to_string())
    print("\nN (trade count):")
    print(pivot_n.to_string())
    print()

    # ────────────── B) 연도 × 시그널 부호 ──────────────
    print("=" * 88)
    print("B) 연도 × 시그널 부호 (short=sig<0, long=sig>0, neutral=0)")
    print("=" * 88)
    trades["side"] = np.where(trades["sig"] > 0, "long",
                       np.where(trades["sig"] < 0, "short", "flat"))
    sb = trades.pivot_table(index="year", columns="side", values="trade_pnl",
                             aggfunc=["sum", "size", "mean"]).round(2)
    print(sb.to_string())
    print()

    # ────────────── C) 시그널 vs forward 부호 일치율 (Hit rate by year) ──────────────
    print("=" * 88)
    print("C) 시그널 부호 vs forward Δy 부호 일치율 (예측 적중률)")
    print("=" * 88)
    # short signal (sig<0): yield 상승 (dy>0) 이면 hit
    # long  signal (sig>0): yield 하락 (dy<0) 이면 hit
    def hit_row(g):
        n = len(g)
        if n == 0:
            return pd.Series({"N": 0, "hit_pct": np.nan, "mean_dy_fwd": np.nan,
                              "mean_pnl": np.nan})
        h = ((g["sig"] < 0) & (g["dy10_fwd_21"] > 0)) | \
            ((g["sig"] > 0) & (g["dy10_fwd_21"] < 0))
        return pd.Series({
            "N": n,
            "hit_pct": h.mean() * 100,
            "mean_dy_fwd": g["dy10_fwd_21"].mean(),
            "mean_pnl": g["trade_pnl"].mean(),
        })

    active = trades[trades["sig"] != 0]
    yh = active.groupby("year").apply(hit_row).round(2)
    print(yh.to_string())
    print()
    # 부호별
    for side, sub in active.groupby("side"):
        print(f"  ▶ {side} only:")
        ys = sub.groupby("year").apply(hit_row).round(2)
        print(ys.to_string())
        print()

    # ────────────── D) 연도 × (fut, cash, fx) 조합 P&L ──────────────
    print("=" * 88)
    print("D) 연도 × 조합 P&L (어느 조합이 어느 해에 익/손?)")
    print("=" * 88)
    pivot_combo_pnl = trades.pivot_table(index="combo", columns="year",
                                          values="trade_pnl", aggfunc="sum").round(0).fillna(0)
    pivot_combo_n = trades.pivot_table(index="combo", columns="year",
                                        values="trade_pnl", aggfunc="size").fillna(0).astype(int)
    print("\nP&L (bp):")
    print(pivot_combo_pnl.to_string())
    print("\nN:")
    print(pivot_combo_n.to_string())
    print()

    # ────────────── E) 손실 연도 (2023, 2024) 의 worst trades ──────────────
    for bad_yr in [2023, 2024]:
        print("=" * 88)
        print(f"E) {bad_yr} worst 10 trades")
        print("=" * 88)
        sub = trades[trades["year"] == bad_yr].copy()
        sub = sub[sub["sig"] != 0]
        worst = sub.nsmallest(10, "trade_pnl")[
            ["price_date", "sig", "combo", "y_10y", "dy10_fwd_21", "dfx_fwd_21", "trade_pnl"]
        ]
        print(worst.to_string(index=False))
        # best 5 for reference
        print(f"\n  best 5 in {bad_yr}:")
        best = sub.nlargest(5, "trade_pnl")[
            ["price_date", "sig", "combo", "y_10y", "dy10_fwd_21", "dfx_fwd_21", "trade_pnl"]
        ]
        print(best.to_string(index=False))
        print()

    # ────────────── F) 연도별 시장 환경 ──────────────
    print("=" * 88)
    print("F) 연도별 시장 환경 (Y_10Y, FX, vol)")
    print("=" * 88)
    env = trades.groupby("year").agg(
        y10_avg=("y_10y", "mean"),
        y10_min=("y_10y", "min"),
        y10_max=("y_10y", "max"),
        y10_change=("y_10y", lambda x: x.iloc[-1] - x.iloc[0]),
        fx_avg=("fx", "mean"),
        fx_change=("fx", lambda x: x.iloc[-1] - x.iloc[0]),
        dy10_fwd_std=("dy10_fwd_21", "std"),
        dy10_fwd_mean=("dy10_fwd_21", "mean"),
    ).round(2)
    print(env.to_string())
    print()
    print("  해석:")
    print("    dy10_fwd_mean > 0 = 그 해 평균적으로 21d 후 yield 가 더 높음 (bond 약세장)")
    print("    dy10_fwd_mean < 0 = 강세장 (rally)")
    print()

    # ────────────── G) 시그널 발생 빈도 변화 ──────────────
    print("=" * 88)
    print("G) 연도별 시그널 발생 빈도 (combo distribution)")
    print("=" * 88)
    freq = trades.pivot_table(index="combo", columns="year",
                               values="trade_pnl", aggfunc="size").fillna(0).astype(int)
    # 비율로
    freq_pct = (freq / freq.sum(axis=0) * 100).round(1)
    print("\n% of year:")
    print(freq_pct.to_string())
    print()

    # ────────────── H) "-1.5 최강 시그널" 의 연도별 변화 ──────────────
    print("=" * 88)
    print("H) sig=-1.5 (SELL+SELL × KRW强) 의 연도별 진단")
    print("=" * 88)
    sg = trades[trades["sig"] == -1.5].copy()
    yh = sg.groupby("year").agg(
        N=("trade_pnl", "size"),
        mean_dy_fwd=("dy10_fwd_21", "mean"),
        median_dy_fwd=("dy10_fwd_21", "median"),
        mean_pnl=("trade_pnl", "mean"),
        total_pnl=("trade_pnl", "sum"),
        hit_pct=("dy10_fwd_21", lambda x: (x > 0).mean() * 100),
    ).round(2)
    print(yh.to_string())
    print("\n  → 이 sig 의 mean_dy_fwd 가 2021-22 양수 크게, 2023+ 0 근처 또는 음수면 = 매커니즘 소멸")
    print()

    # ────────────── I) sig=-1.0 (SELL+BUY × KRW强) 의 연도별 ──────────────
    print("=" * 88)
    print("I) sig=-1.0 (SELL+BUY × KRW强, 정점 의심 시그널) 연도별")
    print("=" * 88)
    sg = trades[trades["sig"] == -1.0].copy()
    yh = sg.groupby("year").agg(
        N=("trade_pnl", "size"),
        mean_dy_fwd=("dy10_fwd_21", "mean"),
        mean_pnl=("trade_pnl", "mean"),
        total_pnl=("trade_pnl", "sum"),
        hit_pct=("dy10_fwd_21", lambda x: (x > 0).mean() * 100),
    ).round(2)
    print(yh.to_string())
    print()

    # ────────────── J) long signal (buy+buy/KRW强 sig=+0.8) 연도별 ──────────────
    print("=" * 88)
    print("J) sig=+0.8 (BUY+BUY × KRW强, carry long) 연도별")
    print("=" * 88)
    sg = trades[trades["sig"] == 0.8].copy()
    yh = sg.groupby("year").agg(
        N=("trade_pnl", "size"),
        mean_dy_fwd=("dy10_fwd_21", "mean"),
        mean_pnl=("trade_pnl", "mean"),
        total_pnl=("trade_pnl", "sum"),
        hit_pct=("dy10_fwd_21", lambda x: (x < 0).mean() * 100),  # long: yield 하락이 hit
    ).round(2)
    print(yh.to_string())
    print("\n  hit_pct = (yield 하락 비율). long 포지션의 hit.")
    print()

    print("[done]")


if __name__ == "__main__":
    main()
