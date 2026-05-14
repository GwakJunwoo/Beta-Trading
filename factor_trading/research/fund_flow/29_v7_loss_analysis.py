"""
29 — V7-clean 손실/저익 연도 분해 분석.

Target periods:
  2020 +520만  (small)
  2023 -510만 ★ 마이너스
  2025 +740만  (small)
  비교: 2022 +1,756 / 2024 +2,512 (good years)

분해 방향:
  A) 연도 × cell 별 P&L 분해 — 어느 cell 이 손실 만들었나
  B) Worst trades top 20 of bad years
  C) 시장 환경: yield level, slope level, FX, daily vol
  D) 시그널 발생 빈도 변화 (cell 별 N)
  E) Expected (cell mean) vs Actual (fwd_dslope) 격차 — over-confident?
  F) Drawdown timeline 의 시작/끝
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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
DV01_KTB10F = 8.5
DV01_KTB3F = 2.8
RATIO = DV01_KTB10F / DV01_KTB3F
TRADING_DAYS = 252
HOLD = 21
TC_10F_BP = 0.12
TC_3F_BP = 0.05

for fname in ["Malgun Gothic", "NanumGothic", "AppleGothic"]:
    try:
        plt.rcParams["font.family"] = fname
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

RULE_V7_CLEAN = {
    "1001": +2.0, "1100": +1.0, "1101": +1.0, "1000": +0.5,
    "0111": -0.5,
}


def load_fx():
    df = pd.read_excel(FX_PATH, sheet_name="Sheet1", header=None, skiprows=2, usecols=[0, 1])
    df.columns = ["price_date", "usdkrw"]
    df["price_date"] = pd.to_datetime(df["price_date"], errors="coerce")
    df["usdkrw"] = pd.to_numeric(df["usdkrw"], errors="coerce")
    return df.dropna().set_index("price_date")["usdkrw"].sort_index()


def load_panel(start="2020-01-01"):
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT
              f.price_date,
              CASE
                WHEN k.avg_remain IS NULL THEN 'unknown'
                WHEN k.avg_remain <= 4 THEN 'b3F'
                WHEN k.avg_remain <= 7 THEN 'b5F'
                WHEN k.avg_remain <= 13 THEN 'b10F'
                ELSE 'b30F'
              END AS bucket,
              SUM(f.foreigner_sum_5d) AS for_s5
            FROM ktb_trade_flow_features f
            LEFT JOIN (
              SELECT bond_code, AVG(remain_year) AS avg_remain
              FROM ktb WHERE category='국고채' AND remain_year IS NOT NULL
              GROUP BY bond_code
            ) k ON f.bond_code = k.bond_code
            WHERE f.bond_code IS NOT NULL AND f.bond_code != ''
              AND f.price_date >= %s
            GROUP BY f.price_date, bucket
        """, (start,))
        cash_rows = cur.fetchall()
        cur.execute("""SELECT price_date, tenor, foreigner FROM ktbf_netbuy
                       WHERE price_date >= %s AND tenor IN ('KTB3F','KTB10F')""", (start,))
        fut_rows = cur.fetchall()

    cash = pd.DataFrame(cash_rows)
    cash["price_date"] = pd.to_datetime(cash["price_date"])
    cash["for_s5"] = pd.to_numeric(cash["for_s5"], errors="coerce")
    cash = cash.pivot_table(index="price_date", columns="bucket",
                              values="for_s5", aggfunc="sum").reset_index()
    fut = pd.DataFrame(fut_rows)
    fut["price_date"] = pd.to_datetime(fut["price_date"])
    fut["foreigner"] = pd.to_numeric(fut["foreigner"], errors="coerce").fillna(0)
    fut = fut.pivot_table(index="price_date", columns="tenor",
                            values="foreigner").reset_index()

    s3 = _load_label_series("3년지표", days=2200)
    s10 = _load_label_series("10년지표", days=2200)
    s3.index = pd.to_datetime(s3.index)
    s10.index = pd.to_datetime(s10.index)
    fx = load_fx()

    p = cash.merge(fut, on="price_date", how="outer").sort_values("price_date").reset_index(drop=True)
    p["y_3y"] = p["price_date"].map(s3) * 100.0
    p["y_10y"] = p["price_date"].map(s10) * 100.0
    p["fx"] = p["price_date"].map(fx)
    p = p.dropna(subset=["y_3y", "y_10y", "fx"]).reset_index(drop=True)

    for col in ["b3F", "b5F", "b10F", "b30F"]:
        if col not in p.columns:
            p[col] = 0.0
        p[col] = p[col].fillna(0)
    p["f3"]  = p["KTB3F"].rolling(5, min_periods=1).sum()
    p["f10"] = p["KTB10F"].rolling(5, min_periods=1).sum()
    p["slope"] = p["y_10y"] - p["y_3y"]
    p["dy3_1d"] = p["y_3y"].diff()
    p["dy10_1d"] = p["y_10y"].diff()
    p["dslope_1d"] = p["dy10_1d"] - p["dy3_1d"]
    p["year"] = p["price_date"].dt.year
    p["dy3_fwd_21"] = p["y_3y"].shift(-HOLD) - p["y_3y"]
    p["dy10_fwd_21"] = p["y_10y"].shift(-HOLD) - p["y_10y"]
    p["dslope_fwd_21"] = p["dy10_fwd_21"] - p["dy3_fwd_21"]
    p["s_f10"] = (p["f10"] > 0).astype(int)
    p["s_f3"] = (p["f3"] > 0).astype(int)
    p["s_b10F"] = (p["b10F"] > 0).astype(int)
    p["s_b3F"] = (p["b3F"] > 0).astype(int)
    p["cell"] = (p["s_f10"].astype(str) + p["s_f3"].astype(str)
                  + p["s_b10F"].astype(str) + p["s_b3F"].astype(str))
    return p


def make_trades(p, rule):
    rows = []
    n = len(p)
    for i in range(n):
        c = p["cell"].iloc[i]
        if c not in rule:
            continue
        if i + HOLD >= n:
            continue
        size = rule[c]
        pos_10 = -size
        pos_3 = +size * RATIO
        fwd_dy10 = p["dy10_fwd_21"].iloc[i]
        fwd_dy3 = p["dy3_fwd_21"].iloc[i]
        fwd_dslope = p["dslope_fwd_21"].iloc[i]
        if pd.isna(fwd_dy10) or pd.isna(fwd_dy3):
            continue
        gross = pos_10 * (-fwd_dy10) * DV01_KTB10F + pos_3 * (-fwd_dy3) * DV01_KTB3F
        cost = abs(pos_10) * TC_10F_BP * DV01_KTB10F + abs(pos_3) * TC_3F_BP * DV01_KTB3F
        net = gross - cost
        rows.append({
            "entry_date": p["price_date"].iloc[i],
            "exit_date": p["price_date"].iloc[i + HOLD],
            "year": p["year"].iloc[i],
            "cell": c,
            "direction": "STEEPENER" if size > 0 else "FLATTENER",
            "size_unit": size,
            "pos_10F": pos_10, "pos_3F": pos_3,
            "y10_entry": p["y_10y"].iloc[i],
            "y3_entry": p["y_3y"].iloc[i],
            "slope_entry": p["slope"].iloc[i],
            "fwd_dy10": fwd_dy10, "fwd_dy3": fwd_dy3, "fwd_dslope": fwd_dslope,
            "expected_dslope_sign": "+" if size > 0 else "-",
            "actual_dslope_sign": "+" if fwd_dslope > 0 else "-",
            "directionally_right": (size > 0 and fwd_dslope > 0) or (size < 0 and fwd_dslope < 0),
            "gross_pnl": gross, "cost": cost, "net_pnl": net,
        })
    return pd.DataFrame(rows)


def main():
    print("[load] panel ...")
    p = load_panel("2020-01-01")
    trades = make_trades(p, RULE_V7_CLEAN)
    print(f"  {len(p):,} days, {len(trades)} trades\n")

    # ── A) 연도 × cell P&L 분해 ──
    print("=" * 100)
    print("A) 연도 × cell 별 net P&L (만)")
    print("=" * 100)
    pivot = trades.pivot_table(index="cell", columns="year",
                                 values="net_pnl", aggfunc="sum").round(0).fillna(0)
    pivot_n = trades.pivot_table(index="cell", columns="year",
                                   values="net_pnl", aggfunc="size").fillna(0).astype(int)
    pivot_hit = trades.pivot_table(index="cell", columns="year",
                                     values="directionally_right",
                                     aggfunc=lambda x: (x.mean()*100)).round(1).fillna(0)
    print("\n  Net P&L (만):")
    print(pivot.to_string())
    print("\n  N trades:")
    print(pivot_n.to_string())
    print("\n  Hit% (directional):")
    print(pivot_hit.to_string())
    print()

    # ── B) 시장 환경 ──
    print("=" * 100)
    print("B) 연도별 시장 환경")
    print("=" * 100)
    env = p.groupby("year").agg(
        y10_avg=("y_10y", "mean"),
        y10_change=("y_10y", lambda x: x.iloc[-1] - x.iloc[0]),
        y10_vol=("dy10_1d", lambda x: x.std()),
        slope_avg=("slope", "mean"),
        slope_change=("slope", lambda x: x.iloc[-1] - x.iloc[0]),
        slope_vol=("dslope_1d", lambda x: x.std()),
        fx_change=("fx", lambda x: x.iloc[-1] - x.iloc[0]),
        dslope_fwd21_mean=("dslope_fwd_21", "mean"),
    ).round(2)
    print(f"\n  y10_avg : 그 해 평균 10Y yield (bp)")
    print(f"  slope_change: 그 해 slope (10Y-3Y) 시작→끝 변화 (bp)")
    print(f"  dslope_fwd21_mean: 모든 일자의 forward 21d Δslope 평균 (시장 전체 slope 추세)")
    print()
    print(env.to_string())
    print()
    print("  → slope_change 음수 = curve flattener regime (10Y가 3Y보다 더 강세)")
    print("  → slope_change 양수 = curve steepener regime")
    print()

    # ── C) Worst trades by bad year ──
    print("=" * 100)
    print("C) Worst 10 trades (bad years: 2020, 2023, 2025)")
    print("=" * 100)
    for yr in [2020, 2023, 2025]:
        sub = trades[trades["year"] == yr].copy()
        if sub.empty:
            print(f"\n  ▶ {yr}: no trades")
            continue
        worst = sub.nsmallest(min(8, len(sub)), "net_pnl")[
            ["entry_date", "cell", "direction", "size_unit",
             "y10_entry", "slope_entry", "fwd_dy10", "fwd_dy3", "fwd_dslope", "net_pnl"]
        ].copy()
        worst["entry_date"] = worst["entry_date"].dt.strftime("%Y-%m-%d")
        for c in ["y10_entry", "slope_entry", "fwd_dy10", "fwd_dy3", "fwd_dslope", "net_pnl", "size_unit"]:
            worst[c] = worst[c].round(2)
        print(f"\n  ▶ {yr} (total {sub['net_pnl'].sum():+,.0f}만, {len(sub)} trades, hit {sub['directionally_right'].mean()*100:.0f}%)")
        print(worst.to_string(index=False))

        # best 3
        best = sub.nlargest(3, "net_pnl")[
            ["entry_date", "cell", "direction", "size_unit", "fwd_dslope", "net_pnl"]
        ].copy()
        best["entry_date"] = best["entry_date"].dt.strftime("%Y-%m-%d")
        for c in ["size_unit", "fwd_dslope", "net_pnl"]:
            best[c] = best[c].round(2)
        print(f"\n   {yr} best 3:")
        print(best.to_string(index=False))
    print()

    # ── D) Expected vs Actual by year & cell ──
    print("=" * 100)
    print("D) Cell sign 의 진실 vs 예측: 연도별 actual Δslope 평균")
    print("=" * 100)
    expected_signs = {c: ("+" if sz > 0 else "-") for c, sz in RULE_V7_CLEAN.items()}
    print(f"\n  cell  expected  ", end="")
    years = sorted(trades["year"].unique())
    for y in years:
        print(f"{y:>8d} ", end="")
    print()
    print("  ----  --------  " + " ".join(["--------"] * len(years)))
    for c, sz in RULE_V7_CLEAN.items():
        line = f"  {c}  {expected_signs[c]:>8s}  "
        for y in years:
            sub = trades[(trades["cell"] == c) & (trades["year"] == y)]
            if len(sub):
                actual_mean = sub["fwd_dslope"].mean()
                line += f"{actual_mean:>+7.2f}  "
            else:
                line += f"{'  -':>8s} "
        print(line)
    print("\n  → 예측 부호와 actual 평균 부호 mismatch 한 연도/cell 파악")
    print()

    # ── E) Drawdown timeline ──
    print("=" * 100)
    print("E) 손실 시점 timeline (daily cumulative net P&L drawdown)")
    print("=" * 100)
    # daily reconstruction
    n = len(p)
    daily_pnl = np.zeros(n)
    daily_cost = np.zeros(n)
    dy10_1d = p["dy10_1d"].fillna(0.0).values
    dy3_1d = p["dy3_1d"].fillna(0.0).values
    cells = p["cell"].values
    for i in range(n):
        c = cells[i]
        if c not in RULE_V7_CLEAN:
            continue
        size = RULE_V7_CLEAN[c]
        pos_10 = -size
        pos_3 = +size * RATIO
        cost = abs(pos_10) * TC_10F_BP * DV01_KTB10F + abs(pos_3) * TC_3F_BP * DV01_KTB3F
        ent = min(i + 1, n - 1)
        ext = min(i + HOLD, n - 1)
        daily_cost[ent] += cost / 2
        daily_cost[ext] += cost / 2
        for d in range(i + 1, min(i + HOLD + 1, n)):
            daily_pnl[d] += pos_10 * (-dy10_1d[d]) * DV01_KTB10F + pos_3 * (-dy3_1d[d]) * DV01_KTB3F
    daily_net = daily_pnl - daily_cost
    cum = np.cumsum(daily_net)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak

    # Top 5 drawdowns periods
    print("\n  Top 5 drawdown 최저점:")
    bottom_idx = np.argsort(dd)[:5]
    for idx in bottom_idx:
        d = p["price_date"].iloc[idx]
        # 찾기: 이전 peak 시점
        peak_v = peak[idx]
        peak_idx = np.argmax(cum[:idx+1])
        peak_d = p["price_date"].iloc[peak_idx]
        days_dd = (d - peak_d).days
        print(f"    {d.strftime('%Y-%m-%d')}: DD={dd[idx]:+,.0f}만 "
              f"(peak {peak_d.strftime('%Y-%m-%d')} @ {peak_v:+,.0f}만, {days_dd}일 후)")
    print()

    # ── F) 시그널 발생 빈도 변화 ──
    print("=" * 100)
    print("F) Cell 시그널 발생 빈도 (연도별)")
    print("=" * 100)
    print("\n  Cell 별 trade 수:")
    print(pivot_n.to_string())
    print()
    total_per_year = pivot_n.sum(axis=0)
    print(f"\n  연도별 총 trade 수: {dict(total_per_year)}")
    print()

    # ── G) 결론 ──
    print("=" * 100)
    print("G) 진단 요약")
    print("=" * 100)
    print(f"\n  연도별 V7-clean net P&L:")
    yr_pnl = trades.groupby("year")["net_pnl"].sum().round(0)
    yr_hit = trades.groupby("year")["directionally_right"].mean().mul(100).round(1)
    yr_n = trades.groupby("year").size()
    for yr in sorted(yr_pnl.index):
        slope_chg = env.loc[yr, "slope_change"]
        actual_dslope = env.loc[yr, "dslope_fwd21_mean"]
        print(f"    {yr}: P&L={yr_pnl[yr]:>+8,.0f}만 (N={yr_n[yr]:>3d}, hit={yr_hit[yr]:5.1f}%) | "
              f"slope_change={slope_chg:>+7.2f}bp | mean_fwd_dslope={actual_dslope:>+5.2f}")
    print()

    # ── Excel ──
    CHART_DIR.mkdir(exist_ok=True)
    xlsx = CHART_DIR / "V7clean_loss_analysis.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xl:
        pivot.reset_index().to_excel(xl, sheet_name="Year_x_Cell_pnl", index=False)
        pivot_n.reset_index().to_excel(xl, sheet_name="Year_x_Cell_N", index=False)
        pivot_hit.reset_index().to_excel(xl, sheet_name="Year_x_Cell_hit", index=False)
        env.reset_index().to_excel(xl, sheet_name="Market_env", index=False)
        # All trades
        t = trades.copy()
        t["entry_date"] = t["entry_date"].dt.strftime("%Y-%m-%d")
        t["exit_date"] = t["exit_date"].dt.strftime("%Y-%m-%d")
        for c in t.select_dtypes(include=["float64"]).columns:
            t[c] = t[c].round(2)
        t.to_excel(xl, sheet_name="All_trades", index=False)
        # Bad years detail
        for yr in [2020, 2023, 2025]:
            sub = t[t["year"] == yr]
            if not sub.empty:
                sub.to_excel(xl, sheet_name=f"Trades_{yr}", index=False)
    print(f"[save] {xlsx}\n")

    # ── 차트 ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # 1. Yearly P&L by cell (stacked bar)
    pivot.T.plot(kind="bar", stacked=True, ax=axes[0, 0], width=0.7, edgecolor="white",
                   colormap="Set2")
    axes[0, 0].axhline(0, color="gray", lw=0.7, ls="--")
    axes[0, 0].set_title("V7-clean 연도 × cell P&L (stacked)")
    axes[0, 0].set_ylabel("Net P&L (만)")
    axes[0, 0].legend(title="cell", loc="upper left", fontsize=8)
    axes[0, 0].grid(alpha=0.3, axis="y")

    # 2. Slope change vs P&L
    axes[0, 1].scatter(env["slope_change"], yr_pnl.reindex(env.index).values,
                         s=80, color="#264653", alpha=0.8)
    for yr in env.index:
        axes[0, 1].annotate(str(yr), (env.loc[yr, "slope_change"], yr_pnl.get(yr, 0)),
                              fontsize=10, xytext=(5, 5), textcoords="offset points")
    axes[0, 1].axhline(0, color="gray", lw=0.5)
    axes[0, 1].axvline(0, color="gray", lw=0.5)
    axes[0, 1].set_xlabel("Slope change (bp, 연간)")
    axes[0, 1].set_ylabel("V7-clean P&L (만)")
    axes[0, 1].set_title("연도별 P&L vs 시장 slope 변화")
    axes[0, 1].grid(alpha=0.3)

    # 3. Drawdown
    axes[1, 0].fill_between(p["price_date"], 0, dd, color="#e76f51", alpha=0.35)
    axes[1, 0].plot(p["price_date"], dd, color="#a8331b", lw=1.2)
    axes[1, 0].set_title("V7-clean Drawdown timeline")
    axes[1, 0].set_ylabel("DD (만)")
    axes[1, 0].xaxis.set_major_locator(mdates.YearLocator())
    axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1, 0].grid(alpha=0.3)

    # 4. Year x Hit %
    pivot_hit.T.plot(kind="bar", ax=axes[1, 1], width=0.7, edgecolor="white",
                       colormap="Set2")
    axes[1, 1].axhline(50, color="red", lw=0.7, ls="--", label="50% breakeven")
    axes[1, 1].set_title("연도 × cell hit rate (directional %)")
    axes[1, 1].set_ylabel("Hit %")
    axes[1, 1].legend(loc="lower left", fontsize=8, ncol=2)
    axes[1, 1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(CHART_DIR / "31_v7_loss_analysis.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[chart] OK 31_v7_loss_analysis.png\n[done]")


if __name__ == "__main__":
    main()
