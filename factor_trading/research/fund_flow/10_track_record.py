"""
10 — Track record + turnover for V2 시그널 × hold=3d.

목표:
  - 매일 진입 / 3d 청산 의 실제 trade 회전률 (turnover, # trades/year)
  - 연도별 trade summary (N, win/loss, mean, max DD)
  - Top winners / losers (실제 entry date, sig, combo)
  - short-only 변형 비교
  - effective position size 분포
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
HOLD = 3
TRADING_DAYS = 252
DV01_KTB10F = 8.5  # 만원/bp/계약 (대략)


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
    p[f"dy10_fwd_{HOLD}"] = p["y_10y"].shift(-HOLD) - p["y_10y"]
    p["y10_exit"] = p["y_10y"].shift(-HOLD)
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


def build_trades(p: pd.DataFrame, sig_filter="all") -> pd.DataFrame:
    """매 (sig != 0) 영업일 = 1 trade. hold=3d. KTB10F 단독."""
    q = p.copy()
    if sig_filter == "short_only":
        q = q[q["sig"] < 0]
    elif sig_filter == "long_only":
        q = q[q["sig"] > 0]
    q = q[(q["sig"] != 0) & q[f"dy10_fwd_{HOLD}"].notna()].copy()
    q["pnl_bp"] = q["sig"] * (-q[f"dy10_fwd_{HOLD}"])
    q["side"] = np.where(q["sig"] > 0, "LONG", "SHORT")
    q["abs_sig"] = q["sig"].abs()
    return q.reset_index(drop=True)


def turnover_metrics(p: pd.DataFrame, sig_filter="all"):
    """
    회전률:
      avg position (gross) = mean over time of |Σ active sigs|
      gross daily traded = avg(|new entries| + |exits|) per day
      annual_turnover = annual_trades × avg_size  vs  avg position
    """
    q = p.copy()
    if sig_filter == "short_only":
        q["sig_use"] = q["sig"].where(q["sig"] < 0, 0.0)
    elif sig_filter == "long_only":
        q["sig_use"] = q["sig"].where(q["sig"] > 0, 0.0)
    else:
        q["sig_use"] = q["sig"]

    n = len(q)
    sig = q["sig_use"].values
    pos = np.zeros(n)
    for i in range(n):
        lo = max(0, i - HOLD)
        pos[i] = sig[lo:i].sum()
    gross_pos = np.abs(pos)
    active_days = (gross_pos > 0).sum()
    avg_pos = gross_pos[gross_pos > 0].mean() if active_days > 0 else 0
    new_trades = (sig != 0).sum()
    avg_size = np.abs(sig[sig != 0]).mean() if new_trades else 0
    yrs = (q["price_date"].max() - q["price_date"].min()).days / 365.25

    return {
        "n_total_days": n,
        "n_trade_days": int(new_trades),
        "trades_per_year": new_trades / yrs,
        "avg_trade_size_unit": avg_size,
        "avg_gross_position_unit": avg_pos,
        "active_pct": active_days / n * 100,
        "avg_hold_days": HOLD,
    }


def main():
    print("[load] panel ...")
    p = load_panel("2020-01-01")
    p["sig"] = p.apply(signal_v2, axis=1)
    p["combo"] = p.apply(classify_combo, axis=1)
    p["year"] = p["price_date"].dt.year
    print(f"  {len(p):,} rows\n")

    # ────────────── A) Turnover metrics ──────────────
    print("=" * 88)
    print(f"A) 회전률 (hold={HOLD}d)")
    print("=" * 88)
    for f in ["all", "short_only", "long_only"]:
        m = turnover_metrics(p, sig_filter=f)
        print(f"\n  ▶ {f}")
        for k, v in m.items():
            if isinstance(v, float):
                print(f"    {k:30s}: {v:>10.2f}")
            else:
                print(f"    {k:30s}: {v:>10}")
    print()
    print("  주: 'unit' = 시그널 강도 (0.3 ~ 1.5). 1 unit ≈ KTB10F 1계약 매핑 가능")
    print(f"  KTB10F DV01 ≈ {DV01_KTB10F}만원/bp/계약")
    print()

    # ────────────── B) Trade list 전체 통계 ──────────────
    print("=" * 88)
    print(f"B) Trade 전체 통계 (V2 × hold={HOLD}d, 매일 신규 진입)")
    print("=" * 88)
    for f in ["all", "short_only"]:
        t = build_trades(p, sig_filter=f)
        n = len(t)
        wins = (t["pnl_bp"] > 0).sum()
        losses = (t["pnl_bp"] < 0).sum()
        flat = (t["pnl_bp"] == 0).sum()
        total = t["pnl_bp"].sum()
        avg = t["pnl_bp"].mean()
        avg_win = t.loc[t["pnl_bp"] > 0, "pnl_bp"].mean()
        avg_loss = t.loc[t["pnl_bp"] < 0, "pnl_bp"].mean()
        wl = avg_win / (-avg_loss) if avg_loss < 0 else np.inf
        yrs = (t["price_date"].max() - t["price_date"].min()).days / 365.25
        print(f"\n  ▶ {f}")
        print(f"    Trades        : {n:,d}  ({n/yrs:.0f}/year, avg every {yrs*252/n:.1f} trading days)")
        print(f"    Win / Loss    : {wins:,d} / {losses:,d}  (hit={wins/(wins+losses)*100:.1f}%)")
        print(f"    Avg win  bp   : {avg_win:+.2f}")
        print(f"    Avg loss bp   : {avg_loss:+.2f}")
        print(f"    Win/Loss ratio: {wl:.2f}")
        print(f"    Total bp      : {total:+.0f}")
        print(f"    Avg trade bp  : {avg:+.3f}")
    print()

    # ────────────── C) Trade list — 연도별 ──────────────
    print("=" * 88)
    print(f"C) 연도별 trade 통계 (V2 × hold={HOLD}d, short_only)")
    print("=" * 88)
    t = build_trades(p, sig_filter="short_only")
    yr = t.groupby("year").agg(
        N=("pnl_bp", "size"),
        wins=("pnl_bp", lambda x: (x > 0).sum()),
        losses=("pnl_bp", lambda x: (x < 0).sum()),
        hit_pct=("pnl_bp", lambda x: (x > 0).mean() * 100),
        total_bp=("pnl_bp", "sum"),
        avg_bp=("pnl_bp", "mean"),
        max_win=("pnl_bp", "max"),
        max_loss=("pnl_bp", "min"),
    ).round(2)
    print(yr.to_string())
    print()

    # ────────────── D) Trade list — combo 별 ──────────────
    print("=" * 88)
    print(f"D) 시그널 조합 별 trade 통계 (short_only)")
    print("=" * 88)
    cb = t.groupby("combo").agg(
        N=("pnl_bp", "size"),
        hit_pct=("pnl_bp", lambda x: (x > 0).mean() * 100),
        total_bp=("pnl_bp", "sum"),
        avg_bp=("pnl_bp", "mean"),
    ).round(2).sort_values("total_bp", ascending=False)
    print(cb.to_string())
    print()

    # ────────────── E) Top winners ──────────────
    print("=" * 88)
    print(f"E) Top 15 winning trades (short_only)")
    print("=" * 88)
    top = t.nlargest(15, "pnl_bp")[["price_date", "sig", "combo", "y_10y", "y10_exit",
                                     f"dy10_fwd_{HOLD}", "pnl_bp"]].copy()
    top.columns = ["entry", "sig", "combo", "entry_y10", "exit_y10", "Δy_bp", "pnl_bp"]
    print(top.to_string(index=False))
    print()

    # ────────────── F) Top losers ──────────────
    print("=" * 88)
    print(f"F) Top 15 losing trades (short_only)")
    print("=" * 88)
    bot = t.nsmallest(15, "pnl_bp")[["price_date", "sig", "combo", "y_10y", "y10_exit",
                                      f"dy10_fwd_{HOLD}", "pnl_bp"]].copy()
    bot.columns = ["entry", "sig", "combo", "entry_y10", "exit_y10", "Δy_bp", "pnl_bp"]
    print(bot.to_string(index=False))
    print()

    # ────────────── G) Monthly P&L (cumulative) ──────────────
    print("=" * 88)
    print(f"G) 월별 trade P&L (short_only)")
    print("=" * 88)
    t["ym"] = t["price_date"].dt.to_period("M")
    m = t.groupby("ym").agg(
        N=("pnl_bp", "size"),
        total_bp=("pnl_bp", "sum"),
        hit_pct=("pnl_bp", lambda x: (x > 0).mean() * 100),
    ).round(1)
    print(m.to_string())
    print()

    # ────────────── H) 직접 KTB10F 만원 환산 ──────────────
    print("=" * 88)
    print(f"H) KTB10F 1계약 = 1 unit 으로 환산 (DV01 {DV01_KTB10F}만원/bp 가정)")
    print("=" * 88)
    yr_won = t.groupby("year").agg(
        N=("pnl_bp", "size"),
        total_bp=("pnl_bp", "sum"),
    )
    yr_won["total_원_per_unit"] = (yr_won["total_bp"] * DV01_KTB10F * 10000).astype(int)
    yr_won["per_year_원"] = yr_won["total_원_per_unit"]
    print(yr_won.to_string())
    print()
    total_won = yr_won["total_원_per_unit"].sum()
    yrs = (t["price_date"].max() - t["price_date"].min()).days / 365.25
    print(f"  Total: {total_won:,d}원  ({total_won/yrs:,.0f}원/year per 1 시그널 unit)")
    print(f"  → 시그널 평균 size {t['abs_sig'].mean():.2f} unit, max 1.5 unit")
    print()

    # ────────────── I) 시그널 강도별 trade record ──────────────
    print("=" * 88)
    print(f"I) 시그널 강도 × 연도별 trade 결과 (short_only, hold={HOLD}d)")
    print("=" * 88)
    pivot_pnl = t.pivot_table(index="sig", columns="year", values="pnl_bp",
                               aggfunc="sum").round(0).fillna(0)
    pivot_n = t.pivot_table(index="sig", columns="year", values="pnl_bp",
                             aggfunc="size").fillna(0).astype(int)
    print("\n  P&L bp:")
    print(pivot_pnl.to_string())
    print("\n  N:")
    print(pivot_n.to_string())
    print()

    print("[done]")


if __name__ == "__main__":
    main()
