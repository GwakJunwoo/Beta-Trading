"""
34 — TP+7/SL-7 룰의 TIMEOUT 분석 + 2022/2023 손실 deep dive.

분석:
  A) TIMEOUT trades 상세 (P&L 분포, cum_pnl_bp 분포)
     → TIMEOUT 가 양수 / 음수 어느 쪽이 많은가?
     → 만약 TIMEOUT 가 양수가 많으면 TP 너무 멀어서 capture 못 함
  B) 2022, 2023 trade-by-trade
     → 어느 cell 이 fail
     → 시장 슬로프 어떻게 움직였나
  C) TP 단축 sensitivity (4, 5, 6 vs 7)
     → 더 짧은 TP 가 같은 슬립 가정에서 어떤가?
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
SIZE_10F_PER_UNIT = 20
SIZE_3F_PER_UNIT = round(SIZE_10F_PER_UNIT * DV01_KTB10F / DV01_KTB3F)
TC_10F_BP = 0.12
TC_3F_BP = 0.05
TRADING_DAYS = 252
MAX_HOLD = 21
MAX_10F_NOTIONAL = 100
SLIP_BP = 0.5

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
    p["year"] = p["price_date"].dt.year
    p["s_f10"] = (p["f10"] > 0).astype(int)
    p["s_f3"] = (p["f3"] > 0).astype(int)
    p["s_b10F"] = (p["b10F"] > 0).astype(int)
    p["s_b3F"] = (p["b3F"] > 0).astype(int)
    p["cell"] = (p["s_f10"].astype(str) + p["s_f3"].astype(str)
                  + p["s_b10F"].astype(str) + p["s_b3F"].astype(str))
    return p


def backtest_slippage(p, rule, tp_bp, sl_bp, slip=SLIP_BP):
    n = len(p)
    y10 = p["y_10y"].values
    y3 = p["y_3y"].values
    cells = p["cell"].values
    dates = p["price_date"].values
    active = []
    daily_pnl = np.zeros(n)
    daily_cost = np.zeros(n)
    closed = []
    for i in range(n):
        still_active = []
        for tr in active:
            held = i - tr["entry_idx"]
            if i > tr["entry_idx"]:
                dy10 = y10[i] - y10[i - 1]
                dy3 = y3[i] - y3[i - 1]
                trade_dpnl = tr["pos_10"] * (-dy10) * DV01_KTB10F + tr["pos_3"] * (-dy3) * DV01_KTB3F
                tr["cum_pnl"] += trade_dpnl
                daily_pnl[i] += trade_dpnl
            avg_dv01 = (abs(tr["pos_10"]) * DV01_KTB10F + abs(tr["pos_3"]) * DV01_KTB3F) / 2.0
            pnl_bp = tr["cum_pnl"] / avg_dv01 if avg_dv01 > 0 else 0.0
            exit_reason = None
            realized_bp = pnl_bp
            if pnl_bp >= tp_bp:
                exit_reason = "TP"
                realized_bp = tp_bp - slip
            elif pnl_bp <= sl_bp:
                exit_reason = "SL"
                realized_bp = sl_bp - slip
            elif held >= MAX_HOLD:
                exit_reason = "TIMEOUT"
            if exit_reason:
                if exit_reason in ("TP", "SL"):
                    target = realized_bp * avg_dv01
                    daily_pnl[i] += target - tr["cum_pnl"]
                    tr["cum_pnl"] = target
                    pnl_bp = realized_bp
                cost = (abs(tr["pos_10"]) * TC_10F_BP * DV01_KTB10F
                         + abs(tr["pos_3"]) * TC_3F_BP * DV01_KTB3F)
                daily_cost[i] += cost
                tr["exit_idx"] = i
                tr["exit_date"] = pd.Timestamp(dates[i])
                tr["held_days"] = held
                tr["exit_reason"] = exit_reason
                tr["pnl_bp_final"] = pnl_bp
                tr["net_pnl"] = tr["cum_pnl"] - cost - tr["entry_cost"]
                tr["exit_y10"] = float(y10[i])
                tr["exit_y3"] = float(y3[i])
                closed.append(tr)
            else:
                still_active.append(tr)
        active = still_active
        cur_10F = sum(tr["pos_10"] for tr in active)
        c = cells[i]
        if c in rule:
            size_units = rule[c]
            new_pos_10 = -size_units * SIZE_10F_PER_UNIT
            new_pos_3 = +size_units * SIZE_3F_PER_UNIT
            if abs(cur_10F + new_pos_10) <= MAX_10F_NOTIONAL:
                entry_cost = (abs(new_pos_10) * TC_10F_BP * DV01_KTB10F
                                + abs(new_pos_3) * TC_3F_BP * DV01_KTB3F)
                daily_cost[i] += entry_cost
                active.append({
                    "entry_idx": i, "entry_date": pd.Timestamp(dates[i]),
                    "cell": c, "size_units": size_units,
                    "pos_10": new_pos_10, "pos_3": new_pos_3,
                    "entry_y10": float(y10[i]), "entry_y3": float(y3[i]),
                    "cum_pnl": 0.0, "entry_cost": entry_cost,
                })
    for tr in active:
        avg_dv01 = (abs(tr["pos_10"]) * DV01_KTB10F + abs(tr["pos_3"]) * DV01_KTB3F) / 2.0
        pnl_bp = tr["cum_pnl"] / avg_dv01 if avg_dv01 > 0 else 0.0
        cost = (abs(tr["pos_10"]) * TC_10F_BP * DV01_KTB10F
                 + abs(tr["pos_3"]) * TC_3F_BP * DV01_KTB3F)
        daily_cost[n - 1] += cost
        tr["exit_idx"] = n - 1
        tr["exit_date"] = pd.Timestamp(dates[n - 1])
        tr["held_days"] = n - 1 - tr["entry_idx"]
        tr["exit_reason"] = "END"
        tr["pnl_bp_final"] = pnl_bp
        tr["net_pnl"] = tr["cum_pnl"] - cost - tr["entry_cost"]
        tr["exit_y10"] = float(y10[n - 1])
        tr["exit_y3"] = float(y3[n - 1])
        closed.append(tr)
    daily_net = daily_pnl - daily_cost
    daily = p[["price_date", "year", "y_10y", "y_3y", "slope"]].copy()
    daily["daily_pnl_net"] = daily_net
    daily["cum_pnl_net"] = daily_net.cumsum()
    daily["drawdown"] = daily["cum_pnl_net"] - daily["cum_pnl_net"].cummax()
    return daily, pd.DataFrame(closed)


def main():
    print("[load] panel ...")
    p = load_panel("2020-01-01")
    print(f"  {len(p):,} rows\n")

    print("Running backtest (TP+7/SL-7, slip 0.5/0.5) ...")
    daily, trades = backtest_slippage(p, RULE_V7_CLEAN, 7.0, -7.0, slip=SLIP_BP)
    trades["year"] = pd.to_datetime(trades["entry_date"]).dt.year
    print(f"  {len(trades)} trades\n")

    # ── A) TIMEOUT 분석 ──
    print("=" * 100)
    print("A) TIMEOUT trades 분석")
    print("=" * 100)
    timeout = trades[trades["exit_reason"] == "TIMEOUT"].copy()
    others = trades[trades["exit_reason"] != "TIMEOUT"].copy()
    print(f"\n  TIMEOUT count: {len(timeout)} / {len(trades)} ({len(timeout)/len(trades)*100:.1f}%)")
    print(f"  TIMEOUT net P&L sum: {timeout['net_pnl'].sum():+,.0f}만")
    print(f"  TIMEOUT mean: {timeout['net_pnl'].mean():+,.0f}만")
    print(f"  TIMEOUT median: {timeout['net_pnl'].median():+,.0f}만")
    print(f"  TIMEOUT 양수 trades: {(timeout['net_pnl'] > 0).sum()} ({(timeout['net_pnl'] > 0).mean()*100:.1f}%)")
    print(f"  TIMEOUT 음수 trades: {(timeout['net_pnl'] < 0).sum()} ({(timeout['net_pnl'] < 0).mean()*100:.1f}%)")
    print()
    print(f"  TIMEOUT pnl_bp_final 분포:")
    print(f"    mean: {timeout['pnl_bp_final'].mean():+.2f} bp")
    print(f"    median: {timeout['pnl_bp_final'].median():+.2f} bp")
    print(f"    p25: {timeout['pnl_bp_final'].quantile(0.25):+.2f} bp")
    print(f"    p75: {timeout['pnl_bp_final'].quantile(0.75):+.2f} bp")
    print(f"    min: {timeout['pnl_bp_final'].min():+.2f} bp")
    print(f"    max: {timeout['pnl_bp_final'].max():+.2f} bp")
    print()
    print(f"  → TIMEOUT 가 양수 평균이면 TP+7 너무 멀음 (실제 alpha 는 작은 bp 에서 fade)")
    print()

    # ── B) Cell 별 TIMEOUT 분포 ──
    print("=" * 100)
    print("B) Cell 별 exit_reason 분포")
    print("=" * 100)
    cell_exit = trades.groupby(["cell", "exit_reason"]).size().unstack(fill_value=0)
    cell_pnl = trades.groupby(["cell", "exit_reason"])["net_pnl"].mean().unstack(fill_value=0).round(0)
    print("\n  Trade count by cell × exit_reason:")
    print(cell_exit.to_string())
    print("\n  Avg net P&L by cell × exit_reason (만):")
    print(cell_pnl.to_string())
    print()

    # ── C) 2022, 2023 분석 ──
    for yr in [2022, 2023]:
        print("=" * 100)
        print(f"C) {yr} trade 분석")
        print("=" * 100)
        sub = trades[trades["year"] == yr].copy()
        if len(sub) == 0:
            print(f"  {yr}: 없음")
            continue
        print(f"\n  Total: {len(sub)} trades")
        print(f"  Net: {sub['net_pnl'].sum():+,.0f}만")
        # Exit reason 분포
        er_cnt = sub["exit_reason"].value_counts()
        er_pnl = sub.groupby("exit_reason")["net_pnl"].agg(["sum", "mean", "count"])
        print(f"\n  Exit reason 분포:")
        print(er_pnl.round(0).to_string())
        # Cell 별
        cell_yr = sub.groupby("cell")["net_pnl"].agg(["count", "sum", "mean"]).round(0)
        print(f"\n  Cell 별:")
        print(cell_yr.to_string())
        # Worst trades
        print(f"\n  Worst 8 trades:")
        worst = sub.nsmallest(8, "net_pnl")[
            ["entry_date", "exit_date", "cell", "size_units", "exit_reason",
             "entry_y10", "exit_y10", "entry_y3", "exit_y3",
             "pnl_bp_final", "net_pnl", "held_days"]
        ].copy()
        worst["entry_date"] = pd.to_datetime(worst["entry_date"]).dt.strftime("%Y-%m-%d")
        worst["exit_date"] = pd.to_datetime(worst["exit_date"]).dt.strftime("%Y-%m-%d")
        for c in worst.select_dtypes(include=["float64"]).columns:
            worst[c] = worst[c].round(2)
        print(worst.to_string(index=False))
        print()

        # 시장 환경 (slope)
        sub_p = p[p["year"] == yr]
        print(f"  시장 슬로프 환경 (slope = y10 - y3):")
        print(f"    mean: {sub_p['slope'].mean():.1f} bp")
        print(f"    start -> end: {sub_p['slope'].iloc[0]:.1f} -> {sub_p['slope'].iloc[-1]:.1f} ({sub_p['slope'].iloc[-1]-sub_p['slope'].iloc[0]:+.1f} bp)")
        print(f"    min: {sub_p['slope'].min():.1f} bp (날짜: {sub_p.loc[sub_p['slope'].idxmin(), 'price_date'].date()})")
        print(f"    max: {sub_p['slope'].max():.1f} bp (날짜: {sub_p.loc[sub_p['slope'].idxmax(), 'price_date'].date()})")
        print()

    # ── D) TP 단축 sensitivity ──
    print("=" * 100)
    print("D) TP 단축 sensitivity (SL 같은 값으로 R/R 1.0 유지, slip 0.5/0.5)")
    print("=" * 100)
    print(f"\n  {'TP':>4s} {'SL':>5s} {'Trades':>7s} {'Net':>10s} {'Per_yr':>9s} {'Sharpe':>7s} "
          f"{'MaxDD':>10s} {'Hit%':>5s} {'W/L':>5s} {'Avg hold':>9s} "
          f"{'TP%':>5s} {'SL%':>5s} {'TO%':>5s} {'TO_PnL_avg':>11s}")
    print("  " + "-" * 110)
    for tp in [3, 4, 5, 6, 7]:
        sl = -tp
        d, t = backtest_slippage(p, RULE_V7_CLEAN, float(tp), float(sl), slip=SLIP_BP)
        net = d["daily_pnl_net"].sum()
        nyrs = len(d) / TRADING_DAYS
        s_nz = d["daily_pnl_net"][d["daily_pnl_net"] != 0]
        sh = s_nz.mean()/s_nz.std()*np.sqrt(TRADING_DAYS) if len(s_nz) > 1 and s_nz.std() > 0 else 0
        mdd = d["drawdown"].min()
        wins = t[t["net_pnl"] > 0]
        losses = t[t["net_pnl"] < 0]
        hit = len(wins)/(len(wins)+len(losses))*100 if (len(wins)+len(losses)) else 0
        wl = wins["net_pnl"].mean()/-losses["net_pnl"].mean() if len(losses) and losses["net_pnl"].mean() < 0 else None
        er = t["exit_reason"].value_counts(normalize=True)*100
        to_trades = t[t["exit_reason"] == "TIMEOUT"]
        to_pnl_avg = to_trades["net_pnl"].mean() if len(to_trades) else 0
        wl_str = f"{wl:.2f}" if wl else "N/A"
        print(f"  +{tp:>3d} {sl:>5d} {len(t):>7d} {net:>+10,.0f} {net/nyrs:>+9,.0f} {sh:>+7.2f} "
              f"{mdd:>+10,.0f} {hit:>5.1f} {wl_str:>5s} {t['held_days'].mean():>9.1f} "
              f"{er.get('TP',0):>5.1f} {er.get('SL',0):>5.1f} {er.get('TIMEOUT',0):>5.1f} "
              f"{to_pnl_avg:>+11,.0f}")
    print()

    # ── Excel ──
    CHART_DIR.mkdir(exist_ok=True)
    xlsx = CHART_DIR / "V7_timeout_2022_2023.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xl:
        # Timeout summary
        timeout_summary = pd.DataFrame([{
            "Count": len(timeout),
            "of total": f"{len(timeout)/len(trades)*100:.1f}%",
            "Net P&L sum (만)": round(timeout["net_pnl"].sum(), 0),
            "Mean (만)": round(timeout["net_pnl"].mean(), 0),
            "Median (만)": round(timeout["net_pnl"].median(), 0),
            "Positive count": (timeout["net_pnl"] > 0).sum(),
            "Negative count": (timeout["net_pnl"] < 0).sum(),
            "Mean pnl_bp": round(timeout["pnl_bp_final"].mean(), 2),
            "Median pnl_bp": round(timeout["pnl_bp_final"].median(), 2),
        }])
        timeout_summary.to_excel(xl, sheet_name="Timeout_summary", index=False)
        # Cell x Exit
        cell_exit.reset_index().to_excel(xl, sheet_name="Cell_x_Exit_N", index=False)
        cell_pnl.reset_index().to_excel(xl, sheet_name="Cell_x_Exit_PnL", index=False)
        # 2022, 2023 trades
        for yr in [2022, 2023]:
            sub = trades[trades["year"] == yr].copy()
            if len(sub):
                for c in sub.select_dtypes(include=["object"]).columns:
                    if "date" in c.lower():
                        sub[c] = pd.to_datetime(sub[c]).dt.strftime("%Y-%m-%d")
                for c in sub.select_dtypes(include=["float64"]).columns:
                    sub[c] = sub[c].round(2)
                sub.to_excel(xl, sheet_name=f"Trades_{yr}", index=False)
        # All timeout trades
        timeout_all = timeout.copy()
        for c in timeout_all.select_dtypes(include=["object"]).columns:
            if "date" in c.lower():
                timeout_all[c] = pd.to_datetime(timeout_all[c]).dt.strftime("%Y-%m-%d")
        for c in timeout_all.select_dtypes(include=["float64"]).columns:
            timeout_all[c] = timeout_all[c].round(2)
        timeout_all.to_excel(xl, sheet_name="All_Timeout_trades", index=False)
    print(f"[save] {xlsx}\n")

    # ── Charts ──
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    # 1. TIMEOUT pnl_bp histogram
    axes[0, 0].hist(timeout["pnl_bp_final"], bins=30, color="#264653", edgecolor="white", alpha=0.8)
    axes[0, 0].axvline(0, color="red", lw=1.5, ls="--", label="0")
    axes[0, 0].axvline(timeout["pnl_bp_final"].mean(), color="#2a9d8f", lw=2,
                         label=f"mean {timeout['pnl_bp_final'].mean():+.2f}bp")
    axes[0, 0].set_title(f"TIMEOUT trades pnl_bp_final 분포 (N={len(timeout)})")
    axes[0, 0].set_xlabel("pnl_bp at timeout"); axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3, axis="y")

    # 2. 2022 + 2023 P&L by month
    sub_2223 = trades[trades["year"].isin([2022, 2023])].copy()
    sub_2223["ym"] = pd.to_datetime(sub_2223["entry_date"]).dt.to_period("M").astype(str)
    monthly = sub_2223.groupby("ym")["net_pnl"].sum()
    colors_m = ["#2a9d8f" if v > 0 else "#e76f51" for v in monthly.values]
    axes[0, 1].bar(monthly.index, monthly.values, color=colors_m, edgecolor="white")
    axes[0, 1].axhline(0, color="gray", lw=0.7, ls="--")
    axes[0, 1].set_title("2022-2023 월별 net P&L (entry month 기준)")
    axes[0, 1].set_ylabel("Net P&L (만)")
    axes[0, 1].tick_params(axis="x", rotation=90, labelsize=7)
    axes[0, 1].grid(alpha=0.3, axis="y")

    # 3. Slope timeline 2022-2023
    p_2223 = p[p["year"].isin([2022, 2023])]
    axes[1, 0].plot(p_2223["price_date"], p_2223["slope"], color="#264653", lw=1.2)
    axes[1, 0].axhline(0, color="gray", lw=0.5)
    axes[1, 0].set_title("2022-2023 slope (10Y-3Y) timeline")
    axes[1, 0].set_ylabel("Slope (bp)")
    axes[1, 0].grid(alpha=0.3)

    # 4. TP sensitivity bar (re-run for chart)
    tp_results = []
    for tp in [3, 4, 5, 6, 7]:
        d, t = backtest_slippage(p, RULE_V7_CLEAN, float(tp), -float(tp), slip=SLIP_BP)
        net = d["daily_pnl_net"].sum()
        s_nz = d["daily_pnl_net"][d["daily_pnl_net"] != 0]
        sh = s_nz.mean()/s_nz.std()*np.sqrt(TRADING_DAYS) if len(s_nz) > 1 and s_nz.std() > 0 else 0
        wins = t[t["net_pnl"] > 0]
        losses = t[t["net_pnl"] < 0]
        wl = wins["net_pnl"].mean()/-losses["net_pnl"].mean() if len(losses) and losses["net_pnl"].mean() < 0 else 0
        tp_results.append({"TP": tp, "Net (만)": net, "Sharpe": sh, "W/L": wl,
                           "Hit": len(wins)/max((len(wins)+len(losses)), 1)*100})
    tp_df = pd.DataFrame(tp_results)
    ax2 = axes[1, 1]
    ax2.bar(tp_df["TP"].astype(str), tp_df["Net (만)"],
             color=["#2a9d8f" if v > 0 else "#e76f51" for v in tp_df["Net (만)"]],
             edgecolor="white")
    for i, row in tp_df.iterrows():
        ax2.text(i, row["Net (만)"], f"sh{row['Sharpe']:.2f}\nW/L{row['W/L']:.2f}",
                  ha="center", fontsize=8)
    ax2.set_title("R/R 1.0 대칭 TP/SL 별 Net P&L")
    ax2.set_xlabel("TP=|SL| (bp)")
    ax2.set_ylabel("Net (만)")
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(CHART_DIR / "36_v7_timeout_analysis.png", bbox_inches="tight")
    plt.close(fig)
    print("[chart] OK 36_v7_timeout_analysis.png")
    print("[done]")


if __name__ == "__main__":
    main()
