"""
14 — Regime-adaptive hold for SELL+BUY.

가설:
  강세장 (yield trending down) → SELL+BUY 후 yield reversal 빨라야 함 → hold 짧게
  약세장 (yield trending up)   → SELL+BUY 후 yield 추세 지속 → hold 길게

Regime 정의: past Δy_10Y_N (N = 21d 또는 63d) 의 부호 / 크기

Variants:
  V5a: binary, lookback=21d, threshold=0,  hold_bull=3, hold_bear=10
  V5b: binary, lookback=21d, threshold=0,  hold_bull=3, hold_bear=7
  V5c: binary, lookback=63d, threshold=0,  hold_bull=3, hold_bear=10
  V5d: 3-bucket, lookback=21d, threshold=±5bp,  bull=3, neutral=5, bear=10
  V5e: lookback=21d, threshold=0, bull=5, bear=10  (둘 다 늘림)
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
DV01 = 8.5


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
    for h in [3, 5, 7, 10, 14, 21, 63]:
        p[f"dy10_past_{h}"] = p["y_10y"] - p["y_10y"].shift(h)
        p[f"dy10_fwd_{h}"] = p["y_10y"].shift(-h) - p["y_10y"]
    return p


def signal_v5(row, lookback=21, threshold=0,
              hold_bull=3, hold_neutral=5, hold_bear=10,
              three_bucket=False):
    """V5: regime-adaptive hold for SELL+BUY.
    SELL+SELL stays at hold=21.
    BUY+BUY, BUY+SELL: skipped (short-only).
    """
    fb = row["f10_s5"] > 0
    cb = row["for_s5"] > 0
    krw_strong = row["dfx_past_5"] < 0

    if not fb and not cb:                # SELL+SELL
        return ((-1.5 if krw_strong else -0.7), 21)
    if not fb and cb:                    # SELL+BUY → regime-adaptive
        trend = row[f"dy10_past_{lookback}"]
        if pd.isna(trend):
            h = hold_neutral
        elif three_bucket:
            if trend >= threshold:
                h = hold_bear
            elif trend <= -threshold:
                h = hold_bull
            else:
                h = hold_neutral
        else:
            h = hold_bear if trend >= 0 else hold_bull
        return ((-1.0 if krw_strong else -0.4), h)
    return (0.0, 0)


def daily_sim(p, sig_fn):
    n = len(p)
    daily_pnl = np.zeros(n)
    daily_pos = np.zeros(n)
    dy1d = p["dy10_1d"].fillna(0.0).values
    rows = p.to_dict("records")
    for i, row in enumerate(rows):
        s, h = sig_fn(row)
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
    return {"N": len(r), "total": total, "per_yr": per_yr, "sharpe": sh, "maxDD": dd,
            "avg_pos": r["pos"].abs().mean()}


def yearly(dp):
    r = dp[dp["pos"] != 0].copy()
    return r.groupby("year").agg(
        total=("pnl_bp", "sum"),
        sharpe=("pnl_bp", lambda x: x.mean() / x.std() * np.sqrt(TRADING_DAYS) if x.std() > 0 else np.nan),
    ).round(1)


def main():
    print("[load] panel ...")
    p = load_panel("2020-01-01")
    print(f"  {len(p):,} rows\n")

    # ────────────── A) baseline + variants ──────────────
    variants = {
        "V4b (fixed h=3)":               lambda r: signal_v5(r, lookback=21, hold_bull=3, hold_bear=3),
        "V4c (fixed h=5)":               lambda r: signal_v5(r, lookback=21, hold_bull=5, hold_bear=5),
        "V4d (fixed h=7)":               lambda r: signal_v5(r, lookback=21, hold_bull=7, hold_bear=7),
        "V5a (LB21, bull=3, bear=10)":   lambda r: signal_v5(r, lookback=21, hold_bull=3, hold_bear=10),
        "V5b (LB21, bull=3, bear=7)":    lambda r: signal_v5(r, lookback=21, hold_bull=3, hold_bear=7),
        "V5c (LB63, bull=3, bear=10)":   lambda r: signal_v5(r, lookback=63, hold_bull=3, hold_bear=10),
        "V5d (LB21, bull=5, bear=10)":   lambda r: signal_v5(r, lookback=21, hold_bull=5, hold_bear=10),
        "V5e (LB21, 3-bkt ±5)":          lambda r: signal_v5(r, lookback=21, threshold=5,
                                                              hold_bull=3, hold_neutral=5, hold_bear=10,
                                                              three_bucket=True),
        "V5f (LB21, bull=3, bear=14)":   lambda r: signal_v5(r, lookback=21, hold_bull=3, hold_bear=14),
        "V5g (LB63, bull=5, bear=10)":   lambda r: signal_v5(r, lookback=63, hold_bull=5, hold_bear=10),
    }

    print("=" * 100)
    print("A) Variants summary")
    print("=" * 100)
    print(f"\n  {'variant':35s} {'N':>8s} {'sharpe':>10s} {'per_yr':>10s} {'total':>10s} {'maxDD':>10s} {'avg_pos':>10s}")
    daily_results = {}
    for name, fn in variants.items():
        dp = daily_sim(p, fn)
        m = metrics(dp)
        daily_results[name] = dp
        print(f"  {name:35s} {m['N']:>8,d} {m['sharpe']:>+10.2f} {m['per_yr']:>+10.0f} {m['total']:>+10.0f} {m['maxDD']:>10.0f} {m['avg_pos']:>10.2f}")
    print()

    # ────────────── B) 연도별 sharpe ──────────────
    print("=" * 100)
    print("B) 연도별 sharpe")
    print("=" * 100)
    sh_tbl = {name: yearly(dp)["sharpe"] for name, dp in daily_results.items()}
    df = pd.DataFrame(sh_tbl)
    print("\n" + df.to_string())
    print()

    # ────────────── C) 연도별 total P&L ──────────────
    print("=" * 100)
    print("C) 연도별 total P&L (bp)")
    print("=" * 100)
    pnl_tbl = {name: yearly(dp)["total"] for name, dp in daily_results.items()}
    df2 = pd.DataFrame(pnl_tbl)
    print("\n" + df2.to_string())
    print()

    # ────────────── D) Regime split (best variant 의 trades) ──────────────
    best_name = "V5a (LB21, bull=3, bear=10)"
    print("=" * 100)
    print(f"D) {best_name} 의 SELL+BUY trade-level regime 분해")
    print("=" * 100)

    fn = variants[best_name]
    rows = []
    for _, row in p.iterrows():
        s, h = fn(row)
        if s == 0 or h == 0:
            continue
        # SELL+BUY 만
        combo_fb = row["f10_s5"] > 0
        combo_cb = row["for_s5"] > 0
        if combo_fb or not combo_cb:
            continue
        trend = row["dy10_past_21"]
        fwd = row[f"dy10_fwd_{h}"]
        if pd.isna(fwd):
            continue
        regime = "bear" if (pd.notna(trend) and trend >= 0) else "bull"
        rows.append({
            "year": row["year"],
            "regime": regime,
            "hold": h,
            "trend_21d": trend,
            "fwd_dy": fwd,
            "pnl": s * (-fwd),
        })
    t = pd.DataFrame(rows)
    print(f"\n  Total SELL+BUY trades: {len(t):,}")
    rg = t.groupby("regime").agg(
        N=("pnl", "size"),
        hold=("hold", "mean"),
        hit_pct=("pnl", lambda x: (x > 0).mean() * 100),
        mean_bp=("pnl", "mean"),
        total_bp=("pnl", "sum"),
    ).round(2)
    print("\n  Regime split:")
    print(rg.to_string())
    # 연도 × regime
    yr_rg = t.groupby(["year", "regime"]).agg(
        N=("pnl", "size"),
        hit_pct=("pnl", lambda x: (x > 0).mean() * 100),
        total=("pnl", "sum"),
    ).round(1)
    print("\n  연도 × regime:")
    print(yr_rg.to_string())
    print()

    # ────────────── E) Lookback sensitivity (best variant 만) ──────────────
    print("=" * 100)
    print("E) Lookback (regime detector) sensitivity")
    print("=" * 100)
    print(f"\n  V5 형식 (bull=3, bear=10), threshold=0:")
    print(f"  {'lookback':>10s} {'sharpe':>10s} {'per_yr':>10s} {'total':>10s} {'maxDD':>10s}")
    for lb in [3, 5, 7, 10, 14, 21, 63]:
        dp = daily_sim(p, lambda r: signal_v5(r, lookback=lb, hold_bull=3, hold_bear=10))
        m = metrics(dp)
        print(f"  {lb:>10d} {m['sharpe']:>+10.2f} {m['per_yr']:>+10.0f} {m['total']:>+10.0f} {m['maxDD']:>10.0f}")
    print()

    # ────────────── F) 5/11 기준 V5a 시그널 ──────────────
    print("=" * 100)
    print("F) 5/11 기준 V5a (LB21, bull=3, bear=10) 시그널")
    print("=" * 100)
    recent = p.tail(15)
    for _, r in recent.iterrows():
        s, h = signal_v5(r, lookback=21, hold_bull=3, hold_bear=10)
        trend = r["dy10_past_21"]
        regime = "bull" if (pd.notna(trend) and trend < 0) else "bear" if pd.notna(trend) else "n/a"
        fb = r["f10_s5"] > 0
        cb = r["for_s5"] > 0
        krw = r["dfx_past_5"] < 0
        fut = "BUY" if fb else "SELL"
        cash = "BUY" if cb else "SELL"
        fxr = "KRW强" if krw else "KRW弱"
        combo = f"{fut}+{cash}/{fxr}"
        trend_str = f"{trend:+.1f}" if pd.notna(trend) else "  n/a"
        print(f"  {r['price_date'].strftime('%Y-%m-%d')}: y10={r['y_10y']:.1f}  past21d={trend_str}  regime={regime:5s}  combo={combo:18s}  sig={s:+.2f} hold={h}d")
    print()

    print("[done]")


if __name__ == "__main__":
    main()
