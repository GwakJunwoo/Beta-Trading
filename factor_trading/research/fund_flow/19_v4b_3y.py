"""
19 — V4b 스킴을 KTB3F (3년 선물) 에 적용.

V4b 와 동일한 분류:
  - 외국인 KTB3F 5d cum  (sign 만 사용)
  - 외국인 현물 5d aggregate  (sign 만 사용)
  - USDKRW 5d 변화  (KRW 强弱)

Target: ΔY_3Y (3년 yield 변동, bp)

매핑 (V4b 그대로):
  SELL+SELL/KRW强 → -1.5, hold=21d
  SELL+SELL/KRW弱 → -0.7, hold=21d
  SELL+BUY/KRW强  → -1.0, hold=3d
  SELL+BUY/KRW弱  → -0.4, hold=3d
  BUY+SELL, BUY+BUY → 제거 (short-only)

분석:
  A) 4 조합 × FX 의 fwd ΔY_3Y matrix (V4b 의 stage 06 동등)
  B) hold sweep
  C) 최종 백테스트 + 메트릭 + 연도별 + 차트
  D) 5/11 기준 시그널
  E) KTB10F (V4b) vs KTB3F 비교
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

BETA_ROOT = Path(__file__).resolve().parents[3]
FULL_ROOT = Path(r"C:\Users\infomax\Desktop\fullstackjunior")
for p in (BETA_ROOT, FULL_ROOT, FULL_ROOT / "server"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from beta_trading.db import get_connection
from app.routers.beta import _load_label_series

FX_PATH = r"C:\Users\infomax\Desktop\USDKRW_INFOMAX.xlsx"
CHART_DIR = Path(__file__).parent / "charts"
DV01_KTB3F = 2.8       # 만원/bp/계약 (대략, duration ~2.8)
DV01_KTB10F = 8.5
TRADING_DAYS = 252

for fname in ["Malgun Gothic", "NanumGothic", "AppleGothic"]:
    try:
        plt.rcParams["font.family"] = fname
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 140


def load_fx():
    df = pd.read_excel(FX_PATH, sheet_name="Sheet1", header=None, skiprows=2, usecols=[0, 1])
    df.columns = ["price_date", "usdkrw"]
    df["price_date"] = pd.to_datetime(df["price_date"], errors="coerce")
    df["usdkrw"] = pd.to_numeric(df["usdkrw"], errors="coerce")
    return df.dropna().set_index("price_date")["usdkrw"].sort_index()


def load_panel(start="2020-01-01"):
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""SELECT price_date, foreigner FROM ktbf_netbuy
                       WHERE price_date >= %s AND tenor='KTB3F'""", (start,))
        f3 = pd.DataFrame(cur.fetchall()).rename(columns={"foreigner": "f3_for"})
        cur.execute("""SELECT price_date, SUM(foreigner_sum_5d) AS for_s5
                       FROM ktb_trade_flow_features
                       WHERE price_date >= %s AND bond_code IS NOT NULL AND bond_code != ''
                       GROUP BY price_date""", (start,))
        cash = pd.DataFrame(cur.fetchall())
    for df in (f3, cash):
        df["price_date"] = pd.to_datetime(df["price_date"])
        for c in df.columns:
            if c != "price_date":
                df[c] = pd.to_numeric(df[c], errors="coerce")
    s3 = _load_label_series("3년지표", days=2200)
    s3.index = pd.to_datetime(s3.index)
    fx = load_fx()
    p = f3.merge(cash, on="price_date", how="outer").sort_values("price_date").reset_index(drop=True)
    p["y_3y"] = p["price_date"].map(s3) * 100.0
    p["fx"] = p["price_date"].map(fx)
    p = p.dropna(subset=["y_3y", "fx"]).reset_index(drop=True)
    p["f3_s5"] = p["f3_for"].rolling(5, min_periods=1).sum()
    p["dfx_past_5"] = p["fx"] - p["fx"].shift(5)
    p["dy3_1d"] = p["y_3y"].diff()
    p["year"] = p["price_date"].dt.year
    for h in [3, 5, 7, 10, 14, 21]:
        p[f"dy3_past_{h}"] = p["y_3y"] - p["y_3y"].shift(h)
        p[f"dy3_fwd_{h}"] = p["y_3y"].shift(-h) - p["y_3y"]
    return p


def classify_combo(row):
    fb = row["f3_s5"] > 0
    cb = row["for_s5"] > 0
    krw_strong = row["dfx_past_5"] < 0
    fut = "BUY" if fb else "SELL"
    cash = "BUY" if cb else "SELL"
    fx = "KRW强" if krw_strong else "KRW弱"
    return f"{fut}+{cash}/{fx}"


def signal_v4b_3y(row, h_ss=21, h_sb=3):
    fb = row["f3_s5"] > 0
    cb = row["for_s5"] > 0
    krw_strong = row["dfx_past_5"] < 0
    if not fb and not cb:
        return ((-1.5 if krw_strong else -0.7), h_ss)
    if not fb and cb:
        return ((-1.0 if krw_strong else -0.4), h_sb)
    return (0.0, 0)


def ic(x, y):
    s = pd.DataFrame({"x": x, "y": y}).dropna()
    s = s[(s["x"] != 0) | (s["y"] != 0)]
    if len(s) < 30:
        return {"n": len(s), "ic": np.nan}
    rho, _ = spearmanr(s["x"], s["y"])
    return {"n": len(s), "ic": float(rho)}


def daily_sim(p, h_ss=21, h_sb=3):
    n = len(p)
    daily_pnl = np.zeros(n)
    daily_pos = np.zeros(n)
    dy1d = p["dy3_1d"].fillna(0.0).values
    rows = p.to_dict("records")
    for i, row in enumerate(rows):
        s, h = signal_v4b_3y(row, h_ss, h_sb)
        if s == 0 or h == 0:
            continue
        for d in range(i + 1, min(i + h + 1, n)):
            daily_pnl[d] += s * (-dy1d[d])
            daily_pos[d] += s
    out = p[["price_date", "year"]].copy()
    out["pnl_bp"] = daily_pnl
    out["pos"] = daily_pos
    return out


def trade_attr(p, h_ss=21, h_sb=3):
    rows = []
    for _, row in p.iterrows():
        s, h = signal_v4b_3y(row, h_ss, h_sb)
        if s == 0 or h == 0:
            continue
        fwd = row[f"dy3_fwd_{h}"]
        if pd.isna(fwd):
            continue
        rows.append({
            "price_date": row["price_date"], "year": row["year"],
            "sig": s, "hold": h, "combo": classify_combo(row),
            "y_entry": row["y_3y"], "fwd_dy": fwd,
            "pnl_bp": s * (-fwd),
        })
    return pd.DataFrame(rows)


def metrics(dp):
    r = dp[dp["pos"] != 0].copy()
    if len(r) == 0:
        return None
    mu = r["pnl_bp"].mean()
    sd = r["pnl_bp"].std()
    sh = mu / sd * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan
    cum = r["pnl_bp"].cumsum()
    mdd = (cum - cum.cummax()).min()
    total = r["pnl_bp"].sum()
    nyrs = (r["price_date"].max() - r["price_date"].min()).days / 365.25
    return {"N": len(r), "total": total, "per_yr": total / nyrs if nyrs > 0 else 0,
            "sharpe": sh, "maxDD": mdd, "avg_pos": r["pos"].abs().mean()}


def main():
    print("[load] panel ...")
    p = load_panel("2020-01-01")
    p["combo"] = p.apply(classify_combo, axis=1)
    print(f"  {len(p):,} rows  {p['price_date'].min().date()} ~ {p['price_date'].max().date()}\n")

    # ── A) 4 조합 × FX 의 forward 21d ΔY_3Y matrix ──
    print("=" * 90)
    print("A) 4 조합별 fwd 21d ΔY_3Y (KTB3F flow + 현물 + KRW)")
    print("=" * 90)
    sub = p[["f3_s5", "for_s5", "dfx_past_5", "dy3_fwd_21"]].dropna().copy()
    sub["fut_buy"] = sub["f3_s5"] > 0
    sub["cash_buy"] = sub["for_s5"] > 0
    sub["krw_strong"] = sub["dfx_past_5"] < 0
    g = sub.groupby(["fut_buy", "cash_buy", "krw_strong"]).agg(
        n=("dy3_fwd_21", "size"),
        mean_dy3=("dy3_fwd_21", "mean"),
        median=("dy3_fwd_21", "median"),
    ).round(2)
    print(g.to_string())
    print()

    # ── B) IC by combo (스킴 검증) ──
    print("=" * 90)
    print("B) KTB3F flow ↔ forward ΔY_3Y IC (horizon별)")
    print("=" * 90)
    print(f"\n  {'Signal':30s} {'h=3':>8s} {'h=5':>8s} {'h=10':>8s} {'h=21':>8s}")
    for fcol, label in [("f3_s5", "KTB3F 외국인 5d cum"),
                         ("for_s5", "현물 외국인 5d")]:
        line = f"  {label:30s}"
        for h in [3, 5, 10, 21]:
            r = ic(p[fcol], p[f"dy3_fwd_{h}"])
            line += f" {r['ic']:+.3f}".rjust(9)
        print(line)
    print()

    # ── C) Hold sweep (SELL+BUY 만 변화, SELL+SELL=21d 고정) ──
    print("=" * 90)
    print("C) V4b-3Y hold sweep (SELL+SELL=21d, SELL+BUY hold 변화)")
    print("=" * 90)
    print(f"\n  {'h_sb':>6s} {'N':>8s} {'sharpe':>9s} {'per_yr':>10s} {'total':>10s} {'maxDD':>10s}")
    sb_holds = [3, 5, 7, 10, 14, 21]
    results = {}
    for h in sb_holds:
        dp = daily_sim(p, h_ss=21, h_sb=h)
        m = metrics(dp)
        results[h] = dp
        print(f"  {h:>6d} {m['N']:>8,d} {m['sharpe']:>+9.2f} {m['per_yr']:>+10.0f} "
              f"{m['total']:>+10.0f} {m['maxDD']:>10.0f}")
    print()

    # ── D) 연도별 (best variant) ──
    print("=" * 90)
    print("D) 연도별 P&L (h_sb 별)")
    print("=" * 90)
    yr_tbl = {}
    for h in sb_holds:
        r = results[h][results[h]["pos"] != 0]
        yr_tbl[f"h_sb={h}"] = r.groupby("year")["pnl_bp"].sum().round(0)
    df = pd.DataFrame(yr_tbl)
    print("\n  Total P&L (bp) by year:")
    print(df.to_string())
    print()

    # ── E) Best variant 의 trade-level 분해 ──
    # 우선 V4b 와 동일 hold 매핑 (h_ss=21, h_sb=3) 으로 진행
    best_h_sb = 3
    print("=" * 90)
    print(f"E) Trade-level breakdown (h_ss=21, h_sb={best_h_sb})")
    print("=" * 90)
    t = trade_attr(p, h_ss=21, h_sb=best_h_sb)
    cb = t.groupby(["combo", "hold"]).agg(
        N=("pnl_bp", "size"),
        hit_pct=("pnl_bp", lambda x: (x > 0).mean() * 100),
        total_bp=("pnl_bp", "sum"),
        avg_bp=("pnl_bp", "mean"),
    ).round(2)
    print("\n  Combo × hold:")
    print(cb.to_string())

    yr = t.groupby("year").agg(
        N=("pnl_bp", "size"),
        hit_pct=("pnl_bp", lambda x: (x > 0).mean() * 100),
        total_bp=("pnl_bp", "sum"),
        avg_bp=("pnl_bp", "mean"),
    ).round(2)
    print("\n  연도별:")
    print(yr.to_string())
    print()

    # ── F) 5/11 기준 시그널 ──
    print("=" * 90)
    print(f"F) 5/11 기준 V4b-3Y 시그널 (직전 10일)")
    print("=" * 90)
    recent = p.tail(15)
    for _, r in recent.iterrows():
        s, h = signal_v4b_3y(r)
        fb = r["f3_s5"] > 0
        cb = r["for_s5"] > 0
        krw = r["dfx_past_5"] < 0
        fut = "BUY" if fb else "SELL"
        cash = "BUY" if cb else "SELL"
        fxr = "KRW强" if krw else "KRW弱"
        combo = f"{fut}+{cash}/{fxr}"
        print(f"  {r['price_date'].strftime('%Y-%m-%d')}: y3={r['y_3y']:.1f}  "
              f"f3_s5={int(r['f3_s5']):+,}  cash_s5={int(r['for_s5']):+,}  "
              f"fx5d={r['dfx_past_5']:+.1f}  combo={combo:18s}  sig={s:+.2f} hold={h}d")
    print()

    # ── G) KTB10F (V4b) 대비 ──
    print("=" * 90)
    print("G) 비교: V4b (KTB10F → ΔY_10Y) vs V4b-3Y (KTB3F → ΔY_3Y)")
    print("=" * 90)
    print(f"\n  metric            V4b (10F)    V4b-3Y (3F)")
    print(f"  -----            ----------    -----------")
    # V4b 결과 (15_report_charts 출력 그대로): per_yr 264, sharpe 1.48, MDD -180, 731 trades
    m_3y = metrics(results[3])
    print(f"  Per_yr (bp)        +264          {m_3y['per_yr']:+.0f}")
    print(f"  Sharpe (ann)       +1.48         {m_3y['sharpe']:+.2f}")
    print(f"  MaxDD (bp)         -180          {m_3y['maxDD']:.0f}")
    print(f"  N active days      1,059         {m_3y['N']:,d}")
    print(f"  DV01 (만/bp/계약)  8.5          ~2.8")
    print(f"  100계약 per_yr(만) +22,500       ~ {m_3y['per_yr'] * DV01_KTB3F * 100:,.0f}")
    print()

    # ── 차트 ──
    print("=" * 90)
    print("차트 생성 ...")
    print("=" * 90)
    CHART_DIR.mkdir(exist_ok=True)
    dp = results[3]
    r = dp.copy()
    r["cum_pnl"] = r["pnl_bp"].cumsum()
    r["peak"] = r["cum_pnl"].cummax()
    r["dd"] = r["cum_pnl"] - r["peak"]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                              gridspec_kw={"height_ratios": [2.5, 1]})
    ax = axes[0]
    ax.fill_between(r["price_date"], 0, r["cum_pnl"],
                     where=r["cum_pnl"] >= 0, alpha=0.25, color="#2a9d8f")
    ax.fill_between(r["price_date"], 0, r["cum_pnl"],
                     where=r["cum_pnl"] < 0, alpha=0.25, color="#e76f51")
    ax.plot(r["price_date"], r["cum_pnl"], color="#264653", lw=2)
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    final = r["cum_pnl"].iloc[-1]
    sharpe = metrics(dp)["sharpe"]
    ax.set_title(f"V4b-3Y 누적 P&L (KTB3F short, target ΔY_3Y)  Final {final:+.0f}bp  Sharpe {sharpe:+.2f}",
                  fontsize=13, weight="bold")
    ax.set_ylabel("Cumulative P&L (bp)")
    ax.grid(alpha=0.3)

    # drawdown
    ax = axes[1]
    ax.fill_between(r["price_date"], 0, r["dd"], color="#e76f51", alpha=0.35)
    ax.plot(r["price_date"], r["dd"], color="#a8331b", lw=1.2)
    mdd = r["dd"].min()
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.set_title(f"Drawdown (MaxDD {mdd:.0f}bp)", fontsize=11)
    ax.set_ylabel("DD (bp)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "20_v4b_3y_cumulative.png", bbox_inches="tight")
    plt.close(fig)
    print("  OK 20_v4b_3y_cumulative.png")

    # 연도별 + combo
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    yr_pnl = r.groupby("year")["pnl_bp"].sum()
    colors = ["#2a9d8f" if v >= 0 else "#e76f51" for v in yr_pnl]
    axes[0].bar(yr_pnl.index.astype(str), yr_pnl.values, color=colors, edgecolor="#264653")
    for i, v in enumerate(yr_pnl.values):
        axes[0].text(i, v + (3 if v >= 0 else -7), f"{v:+.0f}", ha="center", fontsize=10, weight="bold")
    axes[0].axhline(0, color="gray", lw=0.7, ls="--")
    axes[0].set_title("V4b-3Y 연도별 P&L (bp)", fontsize=12, weight="bold")
    axes[0].set_ylabel("P&L (bp)"); axes[0].grid(alpha=0.3, axis="y")

    cb_pnl = t.groupby("combo")["pnl_bp"].sum().sort_values()
    colors2 = ["#2a9d8f" if v >= 0 else "#e76f51" for v in cb_pnl]
    axes[1].barh(cb_pnl.index, cb_pnl.values, color=colors2, edgecolor="#264653")
    for i, v in enumerate(cb_pnl.values):
        axes[1].text(v + (3 if v >= 0 else -10), i, f"{v:+.0f}",
                      va="center", fontsize=9, weight="bold")
    axes[1].axvline(0, color="gray", lw=0.7, ls="--")
    axes[1].set_title("시그널 조합별 P&L (trade-level)", fontsize=12, weight="bold")
    axes[1].set_xlabel("P&L (bp)"); axes[1].grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "21_v4b_3y_breakdown.png", bbox_inches="tight")
    plt.close(fig)
    print("  OK 21_v4b_3y_breakdown.png")

    print("\n[done]")


if __name__ == "__main__":
    main()
