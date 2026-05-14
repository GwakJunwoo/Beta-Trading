"""
33 — V7-clean with realistic slippage TP/SL.

가정:
  TP trigger (daily close pnl_bp >= TP):
    실제 청산 = (TP - slip_tp) bp 로 capped
    예: TP+3, slip_tp=0.5 → realized = +2.5 (intraday 익절)
  SL trigger (daily close pnl_bp <= SL):
    실제 청산 = (SL - slip_sl) bp (더 깊은 손실)
    예: SL-3, slip_sl=1.0 → realized = -4 (intraday SL 늦음 가정)

기본 slippage:
  slip_tp = 0.5 bp
  slip_sl = 1.0 bp

Grid:
  TP: 2, 3, 4, 5, 7
  SL: -2, -3, -4, -5, -7
  R/R 제약: 비대칭 허용 (W/L > 1.0 위해 SL 더 짧게 가능)

조건:
  W/L ratio >= 1.0 (사용자 핵심 기준)
  Sharpe >= 0.8
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
SIZE_10F_PER_UNIT = 20
SIZE_3F_PER_UNIT = round(SIZE_10F_PER_UNIT * DV01_KTB10F / DV01_KTB3F)
TC_10F_BP = 0.12
TC_3F_BP = 0.05
TRADING_DAYS = 252
MAX_HOLD = 21
MAX_10F_NOTIONAL = 100

SLIP_TP_BP = 0.5
SLIP_SL_BP = 0.5

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
    p["year"] = p["price_date"].dt.year
    p["s_f10"] = (p["f10"] > 0).astype(int)
    p["s_f3"] = (p["f3"] > 0).astype(int)
    p["s_b10F"] = (p["b10F"] > 0).astype(int)
    p["s_b3F"] = (p["b3F"] > 0).astype(int)
    p["cell"] = (p["s_f10"].astype(str) + p["s_f3"].astype(str)
                  + p["s_b10F"].astype(str) + p["s_b3F"].astype(str))
    return p


def backtest_slippage(p, rule, tp_bp, sl_bp, slip_tp=SLIP_TP_BP, slip_sl=SLIP_SL_BP):
    """TP/SL trigger 시 realistic slippage 적용.

    pnl_bp >= TP  → realized = TP - slip_tp  (intraday 익절, 살짝 일찍 끊음)
    pnl_bp <= SL  → realized = SL - slip_sl  (intraday SL, 살짝 늦음)
    """
    n = len(p)
    y10 = p["y_10y"].values
    y3 = p["y_3y"].values
    cells = p["cell"].values
    dates = p["price_date"].values
    active = []
    daily_pnl = np.zeros(n)
    daily_cost = np.zeros(n)
    daily_pos10 = np.zeros(n)
    daily_pos3 = np.zeros(n)
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
                realized_bp = tp_bp - slip_tp   # +2.5 if TP+3, slip 0.5
            elif pnl_bp <= sl_bp:
                exit_reason = "SL"
                realized_bp = sl_bp - slip_sl   # -4 if SL-3, slip 1
            elif held >= MAX_HOLD:
                exit_reason = "TIMEOUT"
            if exit_reason:
                # adjust cum_pnl + daily_pnl with slippage cap
                if exit_reason in ("TP", "SL"):
                    target_cum_pnl = realized_bp * avg_dv01
                    adjustment = target_cum_pnl - tr["cum_pnl"]
                    daily_pnl[i] += adjustment
                    tr["cum_pnl"] = target_cum_pnl
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
            if abs(cur_10F + new_pos_10) > MAX_10F_NOTIONAL:
                pass
            else:
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
        daily_pos10[i] = sum(tr["pos_10"] for tr in active)
        daily_pos3[i] = sum(tr["pos_3"] for tr in active)
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
        closed.append(tr)
    daily_net = daily_pnl - daily_cost
    daily = p[["price_date", "year"]].copy()
    daily["pos_10F"] = daily_pos10
    daily["pos_3F"] = daily_pos3
    daily["daily_pnl_gross"] = daily_pnl
    daily["daily_cost"] = daily_cost
    daily["daily_pnl_net"] = daily_net
    daily["cum_pnl_net"] = daily_net.cumsum()
    daily["peak"] = daily["cum_pnl_net"].cummax()
    daily["drawdown_man"] = daily["cum_pnl_net"] - daily["peak"]
    return daily, pd.DataFrame(closed)


def metrics(daily, trades):
    s_n = daily["daily_pnl_net"][daily["daily_pnl_net"] != 0]
    sh = s_n.mean() / s_n.std() * np.sqrt(TRADING_DAYS) if len(s_n) > 1 and s_n.std() > 0 else 0
    downside = s_n[s_n < 0]
    sortino = s_n.mean() / downside.std() * np.sqrt(TRADING_DAYS) if len(downside) > 1 and downside.std() > 0 else 0
    net = daily["daily_pnl_net"].sum()
    nyrs = len(daily) / TRADING_DAYS
    mdd = daily["drawdown_man"].min()
    if len(trades):
        wins = trades[trades["net_pnl"] > 0]
        losses = trades[trades["net_pnl"] < 0]
        hit = len(wins) / (len(wins) + len(losses)) * 100 if (len(wins) + len(losses)) else 0
        avg_w = wins["net_pnl"].mean() if len(wins) else 0
        avg_l = losses["net_pnl"].mean() if len(losses) else 0
        wl = avg_w / -avg_l if avg_l < 0 else None
        avg_hold = trades["held_days"].mean()
        er = trades["exit_reason"].value_counts(normalize=True) * 100
        worst = trades["net_pnl"].min()
        best = trades["net_pnl"].max()
    else:
        hit = wl = avg_w = avg_l = avg_hold = worst = best = 0
        er = pd.Series()
    return {
        "Trades": len(trades),
        "Net (만)": round(net, 0),
        "Per_yr (만)": round(net / nyrs, 0) if nyrs > 0 else 0,
        "Sharpe": round(sh, 2),
        "Sortino": round(sortino, 2),
        "MaxDD (만)": round(mdd, 0),
        "Calmar": round((net / nyrs) / abs(mdd), 2) if mdd != 0 else None,
        "Hit (%)": round(hit, 1),
        "Avg win (만)": round(avg_w, 1),
        "Avg loss (만)": round(avg_l, 1),
        "W/L ratio": round(wl, 2) if wl else None,
        "Worst trade (만)": round(worst, 0),
        "Best trade (만)": round(best, 0),
        "Avg hold (d)": round(avg_hold, 1),
        "TP %": round(er.get("TP", 0), 1),
        "SL %": round(er.get("SL", 0), 1),
        "TIMEOUT %": round(er.get("TIMEOUT", 0), 1),
    }


def main():
    print("[load] panel ...")
    p = load_panel("2020-01-01")
    print(f"  {len(p):,} rows  {p['price_date'].min().date()} ~ {p['price_date'].max().date()}\n")
    print(f"[slippage] TP slip: -{SLIP_TP_BP}bp (intraday 익절)")
    print(f"           SL slip: -{SLIP_SL_BP}bp (intraday SL 늦음, 더 손실)\n")

    # ── Grid search ──
    tp_grid = [2.0, 3.0, 4.0, 5.0, 7.0]
    sl_grid = [-2.0, -3.0, -4.0, -5.0, -7.0]

    rows = []
    print("=" * 110)
    print("Grid search (with realistic slippage)")
    print("=" * 110)
    print(f"\n  {'TP':>4s} {'SL':>5s} {'Trades':>7s} {'Net':>10s} {'Per_yr':>9s} {'Sharpe':>7s} "
          f"{'Sortino':>7s} {'MDD':>10s} {'Calmar':>7s} {'Hit%':>5s} {'W/L':>5s} "
          f"{'AvgW':>7s} {'AvgL':>7s} {'Worst':>8s}")
    print("  " + "-" * 110)

    for tp in tp_grid:
        for sl in sl_grid:
            d, t = backtest_slippage(p, RULE_V7_CLEAN, tp, sl)
            m = metrics(d, t)
            m["TP"] = tp
            m["SL"] = sl
            rows.append(m)
            wl_str = f"{m['W/L ratio']:.2f}" if m['W/L ratio'] else "N/A"
            print(f"  {tp:>4.1f} {sl:>5.1f} {m['Trades']:>7d} {m['Net (만)']:>+10,.0f} "
                  f"{m['Per_yr (만)']:>+9,.0f} {m['Sharpe']:>+7.2f} {m['Sortino']:>+7.2f} "
                  f"{m['MaxDD (만)']:>+10,.0f} {str(m['Calmar']):>7s} {m['Hit (%)']:>5.1f} "
                  f"{wl_str:>5s} {m['Avg win (만)']:>+7.0f} {m['Avg loss (만)']:>+7.0f} "
                  f"{m['Worst trade (만)']:>+8,.0f}")

    df = pd.DataFrame(rows)
    # Filter: W/L >= 1.0
    print()
    print("=" * 110)
    print("Filter: W/L >= 1.0 (사용자 핵심 기준)")
    print("=" * 110)
    df_wl = df[df["W/L ratio"].notna() & (df["W/L ratio"] >= 1.0)].sort_values("Sharpe", ascending=False)
    if len(df_wl):
        print()
        cols_show = ["TP", "SL", "Trades", "Net (만)", "Per_yr (만)", "Sharpe", "Sortino",
                     "MaxDD (만)", "Calmar", "Hit (%)", "W/L ratio", "Worst trade (만)"]
        print(df_wl[cols_show].to_string(index=False))
    else:
        print("\n  W/L >= 1.0 만족하는 조합 없음")
    print()

    # Best by sharpe + W/L >= 1
    if len(df_wl):
        best = df_wl.iloc[0]
    else:
        best = df.sort_values("Sharpe", ascending=False).iloc[0]
    print("=" * 110)
    print(f"Best: TP+{best['TP']}, SL{best['SL']} (slip: TP {SLIP_TP_BP}, SL {SLIP_SL_BP})")
    print("=" * 110)
    daily_b, trades_b = backtest_slippage(p, RULE_V7_CLEAN, best["TP"], best["SL"])
    m_b = metrics(daily_b, trades_b)
    print()
    for k, v in m_b.items():
        print(f"  {k:>22s}: {v}")
    print()

    # 연도별
    yr = daily_b.groupby("year")["daily_pnl_net"].sum().round(0)
    print("연도별 (best):")
    print(yr.to_string())
    print()

    # 비교: 이전 V7-clean TP+3/SL-3 (slippage 없는 버전)
    print("=" * 110)
    print("이전 (no slippage) vs 본 (with slippage) 비교")
    print("=" * 110)
    # No slippage (slip=0,0)
    d_old, t_old = backtest_slippage(p, RULE_V7_CLEAN, 3.0, -3.0, slip_tp=0, slip_sl=0)
    m_old = metrics(d_old, t_old)
    # With slippage default
    d_def, t_def = backtest_slippage(p, RULE_V7_CLEAN, 3.0, -3.0,
                                       slip_tp=SLIP_TP_BP, slip_sl=SLIP_SL_BP)
    m_def = metrics(d_def, t_def)
    comp_df = pd.DataFrame([
        {"Variant": "TP+3/SL-3 no slip", **m_old},
        {"Variant": "TP+3/SL-3 slip 0.5/1.0", **m_def},
        {"Variant": f"BEST TP+{best['TP']}/SL{best['SL']} slip 0.5/1.0", **m_b},
    ])
    print("\n" + comp_df.to_string(index=False))
    print()

    # 5/11 시그널
    fmt = lambda b: "BUY" if int(b) else "SELL"
    latest = p.iloc[-1]
    cell = latest["cell"]
    print("=" * 110)
    print("5/11 시그널 (Best 룰)")
    print("=" * 110)
    print(f"  cell: {cell}  (f10={fmt(cell[0])}, f3={fmt(cell[1])}, b10F={fmt(cell[2])}, b3F={fmt(cell[3])})")
    if cell in RULE_V7_CLEAN:
        sz = RULE_V7_CLEAN[cell]
        pos_10 = -sz * SIZE_10F_PER_UNIT
        pos_3 = +sz * SIZE_3F_PER_UNIT
        print(f"  --> {'STEEPENER' if sz > 0 else 'FLATTENER'} size {abs(sz):.1f}")
        print(f"      KTB10F {'SHORT' if pos_10 < 0 else 'LONG '} {abs(pos_10):.0f}계약 ({abs(pos_10)}억)")
        print(f"      KTB3F  {'LONG ' if pos_3 > 0 else 'SHORT'} {abs(pos_3):.0f}계약 ({abs(pos_3)}억)")
        print(f"      TP +{best['TP']}bp / SL {best['SL']}bp (intraday slippage 가정)")
        print(f"      Max 21d hold")
    print()

    # ── Excel ──
    CHART_DIR.mkdir(exist_ok=True)
    xlsx = CHART_DIR / "V7clean_slippage_grid.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xl:
        df.sort_values("Sharpe", ascending=False).to_excel(xl, sheet_name="Grid_all", index=False)
        if len(df_wl):
            df_wl.to_excel(xl, sheet_name="WL_ge_1.0", index=False)
        comp_df.to_excel(xl, sheet_name="Comparison", index=False)
        yr.reset_index().to_excel(xl, sheet_name="Yearly_best", index=False)
        t = trades_b.copy()
        for c in t.select_dtypes(include=["object"]).columns:
            if "date" in c.lower():
                t[c] = pd.to_datetime(t[c]).dt.strftime("%Y-%m-%d")
        for c in t.select_dtypes(include=["float64"]).columns:
            t[c] = t[c].round(2)
        t.to_excel(xl, sheet_name="Trades_best", index=False)
        d = daily_b.copy()
        d["price_date"] = d["price_date"].dt.strftime("%Y-%m-%d")
        for c in d.select_dtypes(include=["float64"]).columns:
            d[c] = d[c].round(2)
        d.to_excel(xl, sheet_name="Daily_best", index=False)
    print(f"[save] {xlsx}")

    # 차트
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                              gridspec_kw={"height_ratios": [2.5, 1]})
    axes[0].plot(d_def["price_date"], d_def["cum_pnl_net"], color="#e76f51", lw=1.6, alpha=0.7,
                  label=f"TP+3/SL-3 slip0.5/1.0: {m_def['Net (만)']:+,.0f}만 sh{m_def['Sharpe']:+.2f} W/L{m_def['W/L ratio']}")
    axes[0].plot(daily_b["price_date"], daily_b["cum_pnl_net"], color="#264653", lw=2.2,
                  label=f"BEST TP+{best['TP']}/SL{best['SL']}: {m_b['Net (만)']:+,.0f}만 sh{m_b['Sharpe']:+.2f} W/L{m_b['W/L ratio']}")
    axes[0].axhline(0, color="gray", lw=0.7, ls="--")
    axes[0].set_title("V7-clean with realistic slippage (TP-0.5, SL-1.0)",
                       fontsize=13, weight="bold")
    axes[0].set_ylabel("Cum Net P&L (만)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=9)

    axes[1].fill_between(daily_b["price_date"], 0, daily_b["drawdown_man"],
                          color="#264653", alpha=0.35)
    axes[1].plot(d_def["price_date"], d_def["drawdown_man"], color="#e76f51", lw=1.0, alpha=0.7)
    axes[1].plot(daily_b["price_date"], daily_b["drawdown_man"], color="#264653", lw=1.4)
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("DD (만)")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "35_v7_slippage.png", bbox_inches="tight")
    plt.close(fig)
    print("[chart] OK 35_v7_slippage.png")
    print("[done]")


if __name__ == "__main__":
    main()
