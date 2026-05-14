"""
09 — Holding horizon 변경 비교 (5d, 10d, 21d).

목표:
  2023 10-11월 SELL+BUY/KRW强 시그널 9건 -45~-66bp 손실 → 채권 큰 rally 에 21d 동안 다 깔림
  2024 11월 SELL+SELL/KRW强 -45bp → 강세장 시작에 21d 너무 길어
  → holding 단축으로 reversal 빨리 빠질 수 있는지 검증.

전략 동일 (V2 signal), holding 만 변경.
포지션: KTB10F short/long 1 단위 (시그널 sign 그대로) = bp 단위 P&L.
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
TRADING_DAYS = 252


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
    s10.index = pd.to_datetime(s10.index)
    fx = load_fx()

    p = f10.merge(cash, on="price_date", how="outer").sort_values("price_date").reset_index(drop=True)
    p["y_10y"] = p["price_date"].map(s10) * 100.0
    p["fx"] = p["price_date"].map(fx)
    p = p.dropna(subset=["y_10y", "fx"]).reset_index(drop=True)

    p["f10_s5"] = p["f10_for"].rolling(5, min_periods=1).sum()
    p["dfx_past_5"] = p["fx"] - p["fx"].shift(5)
    p["dy10_1d"] = p["y_10y"].diff()

    for h in [3, 5, 10, 21]:
        p[f"dy10_fwd_{h}"] = p["y_10y"].shift(-h) - p["y_10y"]
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


def daily_pnl_overlap(p: pd.DataFrame, sig: pd.Series, hold: int) -> pd.DataFrame:
    n = len(p)
    pos = np.zeros(n)
    sig_arr = sig.values
    for i in range(n):
        lo = max(0, i - hold)
        pos[i] = sig_arr[lo:i].sum()
    dy = p["dy10_1d"].fillna(0.0).values
    daily_pnl = pos * (-dy)
    out = p[["price_date", "y_10y"]].copy()
    out["year"] = out["price_date"].dt.year
    out["sig"] = sig_arr
    out["pos"] = pos
    out["pnl_bp"] = daily_pnl
    return out


def summarize_yearly(name: str, res: pd.DataFrame):
    r = res.dropna(subset=["pnl_bp"]).copy()
    r = r[r["pos"] != 0]
    yg = r.groupby("year").agg(
        N=("pnl_bp", "size"),
        total_bp=("pnl_bp", "sum"),
        sharpe=("pnl_bp", lambda x: x.mean() / x.std() * np.sqrt(TRADING_DAYS) if x.std() > 0 else np.nan),
    ).round(2)
    total = r["pnl_bp"].sum()
    mu = r["pnl_bp"].mean()
    sd = r["pnl_bp"].std()
    sh = mu / sd * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan
    nyrs = (r["price_date"].max() - r["price_date"].min()).days / 365.25
    per_yr = total / nyrs if nyrs > 0 else 0
    print(f"\n  ▶ {name}: total={total:+.0f}bp, sharpe={sh:+.2f}, per_yr={per_yr:+.0f}bp/y")
    print(yg.to_string())


def main():
    print("[load] panel ...")
    p = load_panel("2020-01-01")
    p["sig"] = p.apply(signal_v2, axis=1)
    p["combo"] = p.apply(classify_combo, axis=1)
    p["year"] = p["price_date"].dt.year
    print(f"  panel: {len(p):,} rows\n")

    HORIZONS = [3, 5, 10, 21]

    # ────────────── A) Holding horizon 별 daily-overlap 백테스트 ──────────────
    print("=" * 88)
    print("A) Holding horizon 별 daily-overlap P&L (V2 시그널, KTB10F duration short/long)")
    print("=" * 88)
    for h in HORIZONS:
        res = daily_pnl_overlap(p, p["sig"], h)
        summarize_yearly(f"hold={h}d", res)
    print()

    # ────────────── B) Trade-attribution: 시그널 → forward h-day Δy 의 부호별 P&L ──────────────
    print("=" * 88)
    print("B) Trade-attribution: 시그널 강도 × horizon 별 평균 trade P&L (bp)")
    print("=" * 88)
    for h in HORIZONS:
        sub = p.dropna(subset=[f"dy10_fwd_{h}"]).copy()
        sub["trade_pnl"] = sub["sig"] * (-sub[f"dy10_fwd_{h}"])
        ag = sub.groupby("sig").agg(
            N=("trade_pnl", "size"),
            mean_pnl=("trade_pnl", "mean"),
            total_pnl=("trade_pnl", "sum"),
            hit=("trade_pnl", lambda x: (x > 0).mean() * 100),
        ).round(2)
        print(f"\n  ▶ hold={h}d")
        print(ag.to_string())
    print()

    # ────────────── C) Side 별 (short / long) 연도별 hold 비교 ──────────────
    print("=" * 88)
    print("C) Side 별 P&L 연도 × horizon")
    print("=" * 88)
    for h in HORIZONS:
        sub = p.dropna(subset=[f"dy10_fwd_{h}"]).copy()
        sub["trade_pnl"] = sub["sig"] * (-sub[f"dy10_fwd_{h}"])
        sub["side"] = np.where(sub["sig"] > 0, "long",
                       np.where(sub["sig"] < 0, "short", "flat"))
        pivot = sub[sub["sig"] != 0].pivot_table(
            index="year", columns="side", values="trade_pnl", aggfunc="sum"
        ).round(0).fillna(0)
        print(f"\n  ▶ hold={h}d (trade P&L sum)")
        print(pivot.to_string())
    print()

    # ────────────── D) 문제 케이스 검증: SELL+BUY/KRW强 의 horizon 별 ──────────────
    print("=" * 88)
    print("D) 문제 시그널의 horizon 별 평균 fwd Δy")
    print("=" * 88)
    targets = ["SELL+BUY/KRW强", "SELL+SELL/KRW强", "BUY+BUY/KRW强"]
    for combo in targets:
        sub = p[p["combo"] == combo].copy()
        print(f"\n  ▶ {combo} (N={len(sub):,})")
        rows = []
        for yr, g in sub.groupby("year"):
            r = {"year": yr, "N": len(g)}
            for h in HORIZONS:
                r[f"mean_dy_{h}d"] = g[f"dy10_fwd_{h}"].mean()
            rows.append(r)
        ydf = pd.DataFrame(rows).round(2)
        print(ydf.to_string(index=False))
    print()

    # ────────────── E) 2023, 2024 specifically ──────────────
    print("=" * 88)
    print("E) 손실 연도 (2023, 2024) horizon 별 short side P&L")
    print("=" * 88)
    for bad_yr in [2023, 2024]:
        print(f"\n  ▶ {bad_yr}")
        for h in HORIZONS:
            sub = p[(p["year"] == bad_yr) & (p["sig"] < 0)].dropna(subset=[f"dy10_fwd_{h}"]).copy()
            sub["pnl"] = sub["sig"] * (-sub[f"dy10_fwd_{h}"])
            total = sub["pnl"].sum()
            hit = (sub["pnl"] > 0).mean() * 100 if len(sub) else np.nan
            print(f"    hold={h:>2}d: N={len(sub):>4d}  total={total:+.0f}bp  hit={hit:.1f}%")
    print()

    # ────────────── F) Hold horizon 별 sharpe summary ──────────────
    print("=" * 88)
    print("F) Summary: hold 별 (전체 + short only) sharpe & total")
    print("=" * 88)
    rows = []
    for h in HORIZONS:
        sub = p.dropna(subset=[f"dy10_fwd_{h}"]).copy()
        sub["trade_pnl"] = sub["sig"] * (-sub[f"dy10_fwd_{h}"])
        for which, mask in [("all", sub["sig"] != 0),
                            ("short_only", sub["sig"] < 0),
                            ("long_only", sub["sig"] > 0)]:
            s = sub[mask]
            mu = s["trade_pnl"].mean()
            sd = s["trade_pnl"].std()
            # trade horizon h, 그러므로 # trades per year ≈ TRADING_DAYS
            # sharpe of trade returns (per trade)
            sh_per_trade = mu / sd if sd > 0 else np.nan
            # annualized: per-trade × sqrt(trades per year)
            # 각 시그널은 매일 발생, 하지만 hold 일간 active, overlapping
            # effective independent trades per year ≈ TRADING_DAYS / h
            sh_ann = sh_per_trade * np.sqrt(TRADING_DAYS / h) if not np.isnan(sh_per_trade) else np.nan
            rows.append({
                "hold": h, "side": which, "N": len(s),
                "mean": round(mu, 2), "std": round(sd, 2),
                "sharpe_ann": round(sh_ann, 2),
                "total": round(s["trade_pnl"].sum(), 0),
            })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    print()
    print("  (sharpe_ann 은 trade-attribution 기준, # indep trades/year ≈ 252/h 가정)")
    print()

    print("[done]")


if __name__ == "__main__":
    main()
