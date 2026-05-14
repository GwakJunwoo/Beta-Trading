"""
13 — V4 의 SELL+BUY hold 를 sweep 해서 best 찾기.

V4 base:
  SELL+SELL → hold=21d (고정)
  SELL+BUY  → hold=?   (sweep: 3, 5, 7, 10, 14, 21)
  BUY+SELL  → 제거
  BUY+BUY   → 제거 (short-only)

각 hold 별로:
  - 7년 sharpe
  - 연도별 P&L
  - SELL+BUY 단독 P&L (이 변경의 marginal impact)
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
    p["year"] = p["price_date"].dt.year
    for h in [3, 5, 7, 10, 14, 21]:
        p[f"dy10_fwd_{h}"] = p["y_10y"].shift(-h) - p["y_10y"]
    return p


def classify_combo(row):
    fb = row["f10_s5"] > 0
    cb = row["for_s5"] > 0
    krw_strong = row["dfx_past_5"] < 0
    fut = "BUY" if fb else "SELL"
    cash = "BUY" if cb else "SELL"
    fx = "KRW强" if krw_strong else "KRW弱"
    return f"{fut}+{cash}/{fx}"


def signal_v4_param(row, h_ss=21, h_sb=3):
    """V4 with parametrized hold. short-only (BUY+BUY, BUY+SELL 제거)."""
    fb = row["f10_s5"] > 0
    cb = row["for_s5"] > 0
    krw_strong = row["dfx_past_5"] < 0
    if not fb and not cb:
        return ((-1.5 if krw_strong else -0.7), h_ss)
    if not fb and cb:
        return ((-1.0 if krw_strong else -0.4), h_sb)
    return (0.0, 0)


def daily_simulation(p, h_ss=21, h_sb=3):
    n = len(p)
    daily_pnl = np.zeros(n)
    daily_pos = np.zeros(n)
    dy1d = p["dy10_1d"].fillna(0.0).values
    rows = p.to_dict("records")
    for i, row in enumerate(rows):
        s, h = signal_v4_param(row, h_ss, h_sb)
        if s == 0 or h == 0:
            continue
        for d in range(i + 1, min(i + h + 1, n)):
            daily_pnl[d] += s * (-dy1d[d])
            daily_pos[d] += s
    out = p[["price_date", "year"]].copy()
    out["pnl_bp"] = daily_pnl
    out["pos"] = daily_pos
    return out


def metrics(dp):
    r = dp[dp["pos"] != 0].copy()
    if len(r) == 0:
        return None
    mu = r["pnl_bp"].mean()
    sd = r["pnl_bp"].std()
    sh = mu / sd * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan
    total = r["pnl_bp"].sum()
    nyrs = (r["price_date"].max() - r["price_date"].min()).days / 365.25
    per_yr = total / nyrs if nyrs > 0 else 0
    cum = r["pnl_bp"].cumsum()
    dd = (cum - cum.cummax()).min()
    return {"N": len(r), "total": total, "per_yr": per_yr, "sharpe": sh,
            "maxDD": dd, "avg_pos": r["pos"].abs().mean()}


def yearly(dp):
    r = dp[dp["pos"] != 0].copy()
    return r.groupby("year").agg(
        total=("pnl_bp", "sum"),
        sharpe=("pnl_bp", lambda x: x.mean() / x.std() * np.sqrt(TRADING_DAYS) if x.std() > 0 else np.nan),
    ).round(1)


def main():
    print("[load] panel ...")
    p = load_panel("2020-01-01")
    p["combo"] = p.apply(classify_combo, axis=1)
    print(f"  {len(p):,} rows\n")

    # ────────────── A) SELL+BUY hold sweep (SELL+SELL=21d 고정) ──────────────
    print("=" * 100)
    print("A) V4b 변형: SELL+SELL=21d 고정, SELL+BUY hold sweep")
    print("=" * 100)
    print(f"\n  {'h_sb':>6s} {'N_active':>10s} {'sharpe':>10s} {'per_yr':>10s} {'total':>10s} {'maxDD':>10s} {'avg_pos':>10s}")
    sb_holds = [3, 5, 7, 10, 14, 21]
    results = {}
    for h_sb in sb_holds:
        dp = daily_simulation(p, h_ss=21, h_sb=h_sb)
        m = metrics(dp)
        results[h_sb] = (dp, m)
        print(f"  {h_sb:>6d} {m['N']:>10,d} {m['sharpe']:>+10.2f} {m['per_yr']:>+10.0f} {m['total']:>+10.0f} {m['maxDD']:>10.0f} {m['avg_pos']:>10.2f}")
    print()

    # ────────────── B) 연도별 sharpe 비교 ──────────────
    print("=" * 100)
    print("B) 연도별 sharpe — SELL+BUY hold 별")
    print("=" * 100)
    yr_sh = {}
    for h_sb in sb_holds:
        yr_sh[h_sb] = yearly(results[h_sb][0])["sharpe"]
    df = pd.DataFrame(yr_sh)
    df.columns = [f"h_sb={h}" for h in sb_holds]
    print("\n" + df.to_string())
    print()

    # ────────────── C) 연도별 total P&L ──────────────
    print("=" * 100)
    print("C) 연도별 total P&L (bp)")
    print("=" * 100)
    yr_pnl = {}
    for h_sb in sb_holds:
        yr_pnl[h_sb] = yearly(results[h_sb][0])["total"]
    df2 = pd.DataFrame(yr_pnl)
    df2.columns = [f"h_sb={h}" for h in sb_holds]
    print("\n" + df2.to_string())
    print()

    # ────────────── D) SELL+BUY 단독 contribution (이 변경의 marginal) ──────────────
    print("=" * 100)
    print("D) SELL+BUY 단독 trade-attribution (horizon 변경시 SELL+BUY 만 어떻게 변하나)")
    print("=" * 100)
    sb_only = p[p["combo"].isin(["SELL+BUY/KRW强", "SELL+BUY/KRW弱"])].copy()
    for h in sb_holds:
        sb_only[f"pnl_{h}"] = sb_only.apply(
            lambda r: ((-1.0 if r["combo"].endswith("KRW强") else -0.4)
                       * (-r[f"dy10_fwd_{h}"])) if pd.notna(r[f"dy10_fwd_{h}"]) else np.nan,
            axis=1)
    rows = []
    for h in sb_holds:
        col = f"pnl_{h}"
        s = sb_only[col].dropna()
        rows.append({
            "h_sb": h,
            "N": len(s),
            "hit_pct": round((s > 0).mean() * 100, 1),
            "mean_bp": round(s.mean(), 3),
            "total_bp": round(s.sum(), 0),
            "max_win": round(s.max(), 1),
            "max_loss": round(s.min(), 1),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print()

    # ────────────── E) SELL+BUY 연도별 P&L by hold ──────────────
    print("=" * 100)
    print("E) SELL+BUY 연도별 trade-level P&L (bp)")
    print("=" * 100)
    yr_rows = []
    for h in sb_holds:
        col = f"pnl_{h}"
        for yr, g in sb_only.groupby("year"):
            yr_rows.append({"h_sb": h, "year": yr, "total": g[col].sum()})
    ydf = pd.DataFrame(yr_rows).pivot_table(index="year", columns="h_sb", values="total").round(0)
    print(ydf.to_string())
    print()

    # ────────────── F) 2026 specifically: 직전 trade 들 ──────────────
    print("=" * 100)
    print("F) 2026년 SELL+BUY 의 horizon 별 평균 (samples small 주의)")
    print("=" * 100)
    sb26 = sb_only[sb_only["year"] == 2026]
    rows = []
    for h in sb_holds:
        col = f"pnl_{h}"
        s = sb26[col].dropna()
        rows.append({
            "h": h, "N": len(s),
            "hit_pct": round((s > 0).mean() * 100, 1) if len(s) else np.nan,
            "mean": round(s.mean(), 2),
            "total": round(s.sum(), 1),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print()

    # ────────────── G) 5/4 sig=-1.0 의 horizon별 expected (직접) ──────────────
    print("=" * 100)
    print("G) 2026-05-04 SELL+BUY/KRW强 (sig=-1.0) 의 horizon 별 이론 P&L")
    print("=" * 100)
    target = p[p["price_date"] == "2026-05-04"]
    if len(target):
        r = target.iloc[0]
        print(f"  entry date: 2026-05-04, entry y10 = {r['y_10y']:.1f}")
        for h in [3, 5, 7, 10, 14, 21]:
            v = r[f"dy10_fwd_{h}"]
            if pd.notna(v):
                pnl = -1.0 * (-v)
                print(f"    hold={h:>2}d: exit Δy = {v:+.1f}bp, trade pnl = {pnl:+.2f} bp")
            else:
                print(f"    hold={h:>2}d: 미래 데이터 없음 (이 영업일 이후 panel 끝)")
    print()

    print("[done]")


if __name__ == "__main__":
    main()
