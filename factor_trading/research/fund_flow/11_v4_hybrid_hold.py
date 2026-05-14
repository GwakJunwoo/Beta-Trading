"""
11 — V4: signal-specific holding (hybrid).

Mapping:
  SELL+SELL  → hold=21d (macro regime shift, 길게 작동)
  SELL+BUY   → hold= 3d (hedge action, 단기 효과)
  BUY+SELL   → 제거 (모든 P&L 미미 또는 음수)
  BUY+BUY    → 두 변형:
               V4a: long 유지 (hold=21d)
               V4b: 제거 (short-only)

비교 baseline:
  V2_3d  : V2 시그널, uniform hold=3d (이전 best risk-adjusted)
  V2_21d : V2 시그널, uniform hold=21d (이전 best absolute)
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
DV01_KTB10F = 8.5


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
    for h in [3, 5, 10, 21]:
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


def signal_v2_uniform(row):
    """기존 V2 (참고용)."""
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


def signal_v4(row, keep_long=True):
    """V4: hybrid (sig, hold). BUY+SELL 제거.

    keep_long=True  → V4a (BUY+BUY long 유지)
    keep_long=False → V4b (short-only)
    """
    fb = row["f10_s5"] > 0
    cb = row["for_s5"] > 0
    krw_strong = row["dfx_past_5"] < 0
    if not fb and not cb:                  # SELL+SELL → macro regime, 21d
        return ((-1.5 if krw_strong else -0.7), 21)
    if not fb and cb:                      # SELL+BUY → hedge, 3d
        return ((-1.0 if krw_strong else -0.4), 3)
    if fb and not cb:                      # BUY+SELL → 제거
        return (0.0, 0)
    # BUY+BUY
    if not keep_long:
        return (0.0, 0)
    return ((+0.8 if krw_strong else +0.3), 21)


def trade_attribution(p, sig_hold_fn):
    """Trade-level P&L: 각 entry t 에서 자기 hold 까지 forward yield 변화로 산출."""
    rows = []
    for _, row in p.iterrows():
        s, h = sig_hold_fn(row)
        if s == 0 or h == 0:
            continue
        fwd_col = f"dy10_fwd_{h}"
        v = row[fwd_col]
        if pd.isna(v):
            continue
        rows.append({
            "price_date": row["price_date"],
            "year": row["year"],
            "sig": s, "hold": h,
            "combo": row["combo"],
            "y_10y": row["y_10y"],
            "fwd_dy": v,
            "pnl_bp": s * (-v),
        })
    return pd.DataFrame(rows)


def daily_simulation(p, sig_hold_fn):
    """매 entry 가 자기 hold 일까지 유지되는 daily P&L 시뮬레이션."""
    n = len(p)
    daily_pnl = np.zeros(n)
    daily_pos = np.zeros(n)
    dy1d = p["dy10_1d"].fillna(0.0).values
    rows = p.to_dict("records")
    for i, row in enumerate(rows):
        s, h = sig_hold_fn(row)
        if s == 0 or h == 0:
            continue
        for d in range(i + 1, min(i + h + 1, n)):
            daily_pnl[d] += s * (-dy1d[d])
            daily_pos[d] += s
    out = p[["price_date", "year"]].copy()
    out["pnl_bp"] = daily_pnl
    out["pos"] = daily_pos
    return out


def summarize_daily(name, dp):
    r = dp[dp["pos"] != 0].copy()
    if len(r) == 0:
        print(f"  {name}: no active days")
        return
    mu = r["pnl_bp"].mean()
    sd = r["pnl_bp"].std()
    sh = mu / sd * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan
    total = r["pnl_bp"].sum()
    nyrs = (r["price_date"].max() - r["price_date"].min()).days / 365.25
    per_yr = total / nyrs if nyrs > 0 else 0
    cum = r["pnl_bp"].cumsum()
    dd = (cum - cum.cummax()).min()
    print(f"  {name}: N_active={len(r):,d}  total={total:+.0f}bp  per_yr={per_yr:+.0f}bp  "
          f"sharpe={sh:+.2f}  maxDD={dd:.0f}bp  avg_pos={r['pos'].abs().mean():.2f}")


def yearly_table(name, dp):
    r = dp[dp["pos"] != 0].copy()
    r["year"] = r["price_date"].dt.year
    yg = r.groupby("year").agg(
        N=("pnl_bp", "size"),
        total_bp=("pnl_bp", "sum"),
        sharpe=("pnl_bp", lambda x: x.mean() / x.std() * np.sqrt(TRADING_DAYS) if x.std() > 0 else np.nan),
    ).round(2)
    print(f"\n  ▶ {name} 연도별")
    print(yg.to_string())


def turnover(p, sig_hold_fn):
    n_trades = 0
    sum_size = 0.0
    sum_h = 0
    for _, row in p.iterrows():
        s, h = sig_hold_fn(row)
        if s != 0 and h != 0:
            n_trades += 1
            sum_size += abs(s)
            sum_h += h
    nyrs = (p["price_date"].max() - p["price_date"].min()).days / 365.25
    avg_size = sum_size / n_trades if n_trades else 0
    avg_hold = sum_h / n_trades if n_trades else 0
    return {
        "trades": n_trades,
        "per_year": n_trades / nyrs if nyrs > 0 else 0,
        "avg_size": avg_size,
        "avg_hold": avg_hold,
    }


def main():
    print("[load] panel ...")
    p = load_panel("2020-01-01")
    p["combo"] = p.apply(classify_combo, axis=1)
    print(f"  {len(p):,} rows\n")

    # 시그널 정의
    def sig_v2_3d(r):
        s = signal_v2_uniform(r)
        return (s, 3) if s != 0 else (0.0, 0)

    def sig_v2_21d(r):
        s = signal_v2_uniform(r)
        return (s, 21) if s != 0 else (0.0, 0)

    def sig_v4a(r):
        return signal_v4(r, keep_long=True)

    def sig_v4b(r):
        return signal_v4(r, keep_long=False)

    variants = [
        ("V2 hold=3d  (baseline best risk-adj)", sig_v2_3d),
        ("V2 hold=21d (baseline best abs)",      sig_v2_21d),
        ("V4a hybrid (BUY+SELL drop, BUY+BUY long 유지)", sig_v4a),
        ("V4b hybrid (drop BUY+SELL + BUY+BUY, short only)", sig_v4b),
    ]

    # ────────────── A) Turnover 비교 ──────────────
    print("=" * 88)
    print("A) 회전률 비교")
    print("=" * 88)
    print(f"  {'variant':52s} {'trades/y':>10s} {'avg_size':>10s} {'avg_hold':>10s}")
    for name, fn in variants:
        m = turnover(p, fn)
        print(f"  {name:52s} {m['per_year']:>10.0f} {m['avg_size']:>10.2f} {m['avg_hold']:>10.1f}")
    print()

    # ────────────── B) Daily P&L summary + 연도별 ──────────────
    print("=" * 88)
    print("B) Daily-overlap simulation (active days only)")
    print("=" * 88)
    daily_results = {}
    for name, fn in variants:
        dp = daily_simulation(p, fn)
        daily_results[name] = dp
        summarize_daily(name, dp)
    print()

    print("=" * 88)
    print("C) 연도별 P&L 비교")
    print("=" * 88)
    rows = []
    for name, _ in variants:
        dp = daily_results[name]
        r = dp[dp["pos"] != 0].copy()
        r["year"] = r["price_date"].dt.year
        for yr, g in r.groupby("year"):
            rows.append({"variant": name, "year": yr,
                         "total_bp": g["pnl_bp"].sum(),
                         "sharpe": g["pnl_bp"].mean() / g["pnl_bp"].std() * np.sqrt(TRADING_DAYS)
                                    if g["pnl_bp"].std() > 0 else np.nan})
    yc = pd.DataFrame(rows).pivot_table(index="year", columns="variant", values="total_bp").round(0)
    print("\n  Total P&L (bp) by year:")
    print(yc.to_string())
    ys = pd.DataFrame(rows).pivot_table(index="year", columns="variant", values="sharpe").round(2)
    print("\n  Sharpe by year:")
    print(ys.to_string())
    print()

    # ────────────── D) Trade-level breakdown (V4a, V4b) ──────────────
    print("=" * 88)
    print("D) V4a / V4b: trade-level 통계")
    print("=" * 88)
    for name, fn in [("V4a", sig_v4a), ("V4b", sig_v4b)]:
        t = trade_attribution(p, fn)
        print(f"\n  ▶ {name}: {len(t):,} trades")
        # by combo + hold
        bc = t.groupby(["combo", "hold"]).agg(
            N=("pnl_bp", "size"),
            hit_pct=("pnl_bp", lambda x: (x > 0).mean() * 100),
            total_bp=("pnl_bp", "sum"),
            avg_bp=("pnl_bp", "mean"),
        ).round(2)
        print(bc.to_string())
        # by year
        ty = t.groupby("year").agg(
            N=("pnl_bp", "size"),
            hit_pct=("pnl_bp", lambda x: (x > 0).mean() * 100),
            total_bp=("pnl_bp", "sum"),
        ).round(2)
        print(f"\n   {name} 연도별 trade P&L:")
        print(ty.to_string())
    print()

    # ────────────── E) Sharpe & cumulative chart (text) ──────────────
    print("=" * 88)
    print("E) Cumulative P&L (월말 기준 마지막 영업일 cumulative)")
    print("=" * 88)
    for name, _ in variants:
        dp = daily_results[name]
        r = dp[dp["pos"] != 0].copy()
        r["ym"] = r["price_date"].dt.to_period("M")
        cum = r.set_index("price_date")["pnl_bp"].cumsum()
        mo_last = cum.groupby(cum.index.to_period("M")).last()
        print(f"\n  ▶ {name}")
        # 6개월 단위만 출력
        sample = mo_last.iloc[::6]
        for ym, v in sample.items():
            print(f"    {ym}: {v:+.0f} bp")
        print(f"    final: {mo_last.iloc[-1]:+.0f} bp")
    print()

    # ────────────── F) 회전률 + cost 후 추정 ──────────────
    print("=" * 88)
    print("F) 거래비용 보수적 추정 (round trip 0.12 bp/계약, KTB10F)")
    print("=" * 88)
    print(f"  {'variant':52s} {'gross/y':>10s} {'cost/y':>10s} {'net/y':>10s}")
    for name, fn in variants:
        dp = daily_results[name]
        r = dp[dp["pos"] != 0].copy()
        nyrs = (r["price_date"].max() - r["price_date"].min()).days / 365.25
        gross = r["pnl_bp"].sum() / nyrs
        tm = turnover(p, fn)
        cost = tm["per_year"] * tm["avg_size"] * 0.12   # round trip
        net = gross - cost
        print(f"  {name:52s} {gross:>+10.0f} {cost:>10.0f} {net:>+10.0f}")
    print()

    print("[done]")


if __name__ == "__main__":
    main()
