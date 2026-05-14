"""
31 — V7 / V7-v2 with TP/SL exit + position cap.

전략 변경:
  Entry:  시그널 발동 → KTB10F 20억 (=20계약) + KTB3F delta-매칭 사이즈 (1 unit base)
          Cell size_unit 그대로 (1001 = 2 units = 40억 10선 + 120억 3선 등)
          신규 진입 전 max 100억 cap 확인 (KTB10F 총 노출 ≤ 100계약)
  Exit:   TP/SL 달성 또는 max 21d hold
  Size:   1 unit = 10선 20계약 (20억) + 3선 ~60계약 (delta 매칭)
          DV01: 10선 20계약 × 8.5만 = 170만/bp
                3선  60계약 × 2.8만 = 168만/bp (거의 매칭)
  TP/SL:  Trade-level P&L bp (avg DV01 기준) 으로 grid search
          TP: +1, +2, +3, +5, +7, +10 bp
          SL: -1, -2, -3, -5, -7 bp
          R/R 제약: |SL| ≤ 1.5 × TP

비교: V7-clean (5 cells) vs V7-clean-v2 (4 STEEPENER) 각자 best TP/SL 찾기
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
SIZE_10F_PER_UNIT = 20            # 1 unit = 10선 20계약 (=20억)
SIZE_3F_PER_UNIT = round(SIZE_10F_PER_UNIT * DV01_KTB10F / DV01_KTB3F)   # ~60.7 → 61 contracts
TC_10F_BP = 0.12
TC_3F_BP = 0.05
TRADING_DAYS = 252
MAX_HOLD = 21
MAX_10F_NOTIONAL = 100            # 10선 max 100계약 (100억)

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
RULE_V7_CLEAN_V2 = {
    "1001": +2.0, "1100": +1.0, "1101": +1.0, "1000": +0.5,
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


def backtest_tpsl(p, rule, tp_bp, sl_bp):
    """TP/SL exit + max 100억 cap.

    Trade P&L bp = cum_pnl / avg_dv01_won
       avg_dv01_won = (|pos_10F| × DV01_10F + |pos_3F| × DV01_3F) / 2  (만원/bp)
    """
    n = len(p)
    y10 = p["y_10y"].values
    y3 = p["y_3y"].values
    cells = p["cell"].values
    dates = p["price_date"].values

    active = []   # list of dicts
    daily_pnl = np.zeros(n)
    daily_cost = np.zeros(n)
    daily_pos10 = np.zeros(n)
    daily_pos3 = np.zeros(n)
    closed = []

    for i in range(n):
        # 1) Update existing positions (daily mark-to-market + TP/SL check)
        still_active = []
        for tr in active:
            held = i - tr["entry_idx"]
            # daily P&L (i 일 close 가격 기준)
            if i > tr["entry_idx"]:
                dy10 = y10[i] - y10[i - 1]
                dy3 = y3[i] - y3[i - 1]
                trade_dpnl = tr["pos_10"] * (-dy10) * DV01_KTB10F + tr["pos_3"] * (-dy3) * DV01_KTB3F
                tr["cum_pnl"] += trade_dpnl
                daily_pnl[i] += trade_dpnl

            # Trade-level P&L bp (avg DV01)
            avg_dv01 = (abs(tr["pos_10"]) * DV01_KTB10F + abs(tr["pos_3"]) * DV01_KTB3F) / 2.0
            pnl_bp = tr["cum_pnl"] / avg_dv01 if avg_dv01 > 0 else 0.0

            # Exit conditions
            exit_reason = None
            if pnl_bp >= tp_bp:
                exit_reason = "TP"
            elif pnl_bp <= sl_bp:
                exit_reason = "SL"
            elif held >= MAX_HOLD:
                exit_reason = "TIMEOUT"

            if exit_reason:
                # exit cost
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

        # 2) 현재 active position 의 KTB10F 누적 노출 계산
        cur_10F = sum(tr["pos_10"] for tr in active)

        # 3) 새 시그널 진입 체크
        c = cells[i]
        if c in rule:
            size_units = rule[c]
            new_pos_10 = -size_units * SIZE_10F_PER_UNIT     # steepener: short 10F
            new_pos_3 = +size_units * SIZE_3F_PER_UNIT       # long 3F
            # max 100억 cap (절대값 기준)
            if abs(cur_10F + new_pos_10) > MAX_10F_NOTIONAL:
                pass   # skip (capacity 초과)
            else:
                entry_cost = (abs(new_pos_10) * TC_10F_BP * DV01_KTB10F
                                + abs(new_pos_3) * TC_3F_BP * DV01_KTB3F)
                daily_cost[i] += entry_cost
                tr = {
                    "entry_idx": i,
                    "entry_date": pd.Timestamp(dates[i]),
                    "cell": c,
                    "size_units": size_units,
                    "pos_10": new_pos_10, "pos_3": new_pos_3,
                    "entry_y10": float(y10[i]), "entry_y3": float(y3[i]),
                    "cum_pnl": 0.0,
                    "entry_cost": entry_cost,
                }
                active.append(tr)

        # daily aggregate positions
        daily_pos10[i] = sum(tr["pos_10"] for tr in active)
        daily_pos3[i] = sum(tr["pos_3"] for tr in active)

    # Close out 끝까지 남은 position (max hold 21d 도달 못 한 경우)
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
    trades_df = pd.DataFrame(closed)
    return daily, trades_df


def metrics(daily, trades, name):
    s_n = daily["daily_pnl_net"][daily["daily_pnl_net"] != 0]
    sh_n = s_n.mean() / s_n.std() * np.sqrt(TRADING_DAYS) if len(s_n) > 1 and s_n.std() > 0 else 0
    gross = daily["daily_pnl_gross"].sum()
    net = daily["daily_pnl_net"].sum()
    cost = daily["daily_cost"].sum()
    nyrs = len(daily) / TRADING_DAYS
    mdd = daily["drawdown_man"].min()
    if len(trades):
        hit = (trades["net_pnl"] > 0).mean() * 100
        avg_win = trades.loc[trades["net_pnl"] > 0, "net_pnl"].mean() if (trades["net_pnl"] > 0).any() else 0
        avg_loss = trades.loc[trades["net_pnl"] < 0, "net_pnl"].mean() if (trades["net_pnl"] < 0).any() else 0
        avg_hold = trades["held_days"].mean()
        exit_reasons = trades["exit_reason"].value_counts(normalize=True) * 100
    else:
        hit = avg_win = avg_loss = avg_hold = 0
        exit_reasons = pd.Series()
    return {
        "name": name, "Trades": len(trades),
        "Net (만)": round(net, 0), "Cost (만)": round(cost, 0),
        "Per_yr (만)": round(net / nyrs, 0) if nyrs > 0 else 0,
        "Sharpe net": round(sh_n, 2),
        "MaxDD (만)": round(mdd, 0), "Hit (%)": round(hit, 1),
        "Avg win (만)": round(avg_win, 1), "Avg loss (만)": round(avg_loss, 1),
        "W/L ratio": round(avg_win / -avg_loss, 2) if avg_loss < 0 else None,
        "Avg hold (d)": round(avg_hold, 1),
        "Calmar": round((net/nyrs) / abs(mdd), 2) if mdd != 0 else None,
        "TP%": round(exit_reasons.get("TP", 0), 1),
        "SL%": round(exit_reasons.get("SL", 0), 1),
        "TIMEOUT%": round(exit_reasons.get("TIMEOUT", 0), 1),
    }


def grid_search(p, rule, name):
    tp_grid = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    sl_grid = [-1.0, -2.0, -3.0, -5.0, -7.0, -10.0]
    rows = []
    for tp in tp_grid:
        for sl in sl_grid:
            if abs(sl) > 2.0 * tp:
                continue   # R/R 제약 (SL 너무 깊은 케이스 제외)
            daily, trades = backtest_tpsl(p, rule, tp, sl)
            m = metrics(daily, trades, f"{name} TP{tp}/SL{sl}")
            m["TP"] = tp
            m["SL"] = sl
            rows.append(m)
    return pd.DataFrame(rows)


def main():
    print("[load] panel ...")
    p = load_panel("2020-01-01")
    print(f"  {len(p):,} rows  {p['price_date'].min().date()} ~ {p['price_date'].max().date()}\n")
    print(f"[size] 1 unit = KTB10F {SIZE_10F_PER_UNIT}계약 + KTB3F {SIZE_3F_PER_UNIT}계약 (delta-매칭)")
    print(f"       1 unit DV01: 10선 {SIZE_10F_PER_UNIT * DV01_KTB10F:.0f}만/bp, "
          f"3선 {SIZE_3F_PER_UNIT * DV01_KTB3F:.0f}만/bp")
    print(f"       Max 10선 노출: {MAX_10F_NOTIONAL}계약\n")

    # ── Grid search ──
    print("=" * 90)
    print("Grid search: V7-clean")
    print("=" * 90)
    g1 = grid_search(p, RULE_V7_CLEAN, "V7-clean")
    g1_top = g1.sort_values("Sharpe net", ascending=False).head(10)
    print("\nTop 10 by sharpe net:")
    cols = ["TP", "SL", "Trades", "Net (만)", "Per_yr (만)", "Sharpe net",
            "MaxDD (만)", "Hit (%)", "Avg hold (d)", "TP%", "SL%", "TIMEOUT%"]
    print(g1_top[cols].to_string(index=False))
    print()

    print("=" * 90)
    print("Grid search: V7-clean-v2 (no 0111)")
    print("=" * 90)
    g2 = grid_search(p, RULE_V7_CLEAN_V2, "V7-clean-v2")
    g2_top = g2.sort_values("Sharpe net", ascending=False).head(10)
    print("\nTop 10 by sharpe net:")
    print(g2_top[cols].to_string(index=False))
    print()

    # Best 별 비교
    best1 = g1.loc[g1["Sharpe net"].idxmax()]
    best2 = g2.loc[g2["Sharpe net"].idxmax()]
    print("=" * 90)
    print("Best 비교")
    print("=" * 90)
    print(f"\n  V7-clean best:    TP=+{best1['TP']}, SL={best1['SL']}, sharpe={best1['Sharpe net']:+.2f}, "
          f"per_yr={best1['Per_yr (만)']:+,.0f}만, MDD={best1['MaxDD (만)']:+,.0f}만")
    print(f"  V7-clean-v2 best: TP=+{best2['TP']}, SL={best2['SL']}, sharpe={best2['Sharpe net']:+.2f}, "
          f"per_yr={best2['Per_yr (만)']:+,.0f}만, MDD={best2['MaxDD (만)']:+,.0f}만")
    print()

    # Best 적용한 백테스트 detail
    print("=" * 90)
    print(f"V7-clean detail (TP={best1['TP']}, SL={best1['SL']})")
    print("=" * 90)
    daily1, trades1 = backtest_tpsl(p, RULE_V7_CLEAN, best1["TP"], best1["SL"])
    m1 = metrics(daily1, trades1, "V7-clean best")
    for k, v in m1.items():
        if k == "name":
            continue
        print(f"    {k:>22s}: {v}")

    print()
    print("=" * 90)
    print(f"V7-clean-v2 detail (TP={best2['TP']}, SL={best2['SL']})")
    print("=" * 90)
    daily2, trades2 = backtest_tpsl(p, RULE_V7_CLEAN_V2, best2["TP"], best2["SL"])
    m2 = metrics(daily2, trades2, "V7-clean-v2 best")
    for k, v in m2.items():
        if k == "name":
            continue
        print(f"    {k:>22s}: {v}")
    print()

    # 연도별
    print("=" * 90)
    print("연도별 P&L (net, 만):")
    print("=" * 90)
    yr1 = daily1.groupby("year")["daily_pnl_net"].sum().round(0)
    yr2 = daily2.groupby("year")["daily_pnl_net"].sum().round(0)
    yr_df = pd.DataFrame({"V7-clean (TP/SL)": yr1, "V7-clean-v2 (TP/SL)": yr2})
    print(yr_df.to_string())
    print()

    # 5/11 시그널
    fmt = lambda b: "BUY" if int(b) else "SELL"
    latest = p.iloc[-1]
    cell = latest["cell"]
    print("=" * 90)
    print("5/11 시그널")
    print("=" * 90)
    print(f"  cell: {cell}  (f10={fmt(cell[0])}, f3={fmt(cell[1])}, b10F={fmt(cell[2])}, b3F={fmt(cell[3])})")
    for rule_name, rule, best in [("V7-clean", RULE_V7_CLEAN, best1),
                                    ("V7-clean-v2", RULE_V7_CLEAN_V2, best2)]:
        if cell in rule:
            sz = rule[cell]
            pos_10 = -sz * SIZE_10F_PER_UNIT
            pos_3 = +sz * SIZE_3F_PER_UNIT
            print(f"\n  {rule_name} (TP=+{best['TP']}bp, SL={best['SL']}bp, max 21d):")
            print(f"    KTB10F {'SHORT' if pos_10 < 0 else 'LONG '} {abs(pos_10):.0f}계약 ({abs(pos_10)}억)")
            print(f"    KTB3F  {'LONG ' if pos_3 > 0 else 'SHORT'} {abs(pos_3):.0f}계약 ({abs(pos_3)}억)")
        else:
            print(f"\n  {rule_name}: FLAT")
    print()

    # ── Excel ──
    CHART_DIR.mkdir(exist_ok=True)
    xlsx = CHART_DIR / "V7_TPSL_grid.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xl:
        g1.sort_values("Sharpe net", ascending=False).to_excel(xl, sheet_name="Grid_V7clean", index=False)
        g2.sort_values("Sharpe net", ascending=False).to_excel(xl, sheet_name="Grid_V7v2", index=False)
        pd.DataFrame([m1, m2]).to_excel(xl, sheet_name="Best_summary", index=False)
        yr_df.reset_index().to_excel(xl, sheet_name="Yearly", index=False)
        for tdf, sheet in [(trades1, "Trades_V7clean"), (trades2, "Trades_V7v2")]:
            t = tdf.copy()
            for c in t.select_dtypes(include=["object"]).columns:
                if "date" in c.lower():
                    t[c] = pd.to_datetime(t[c]).dt.strftime("%Y-%m-%d")
            for c in t.select_dtypes(include=["float64"]).columns:
                t[c] = t[c].round(2)
            t.to_excel(xl, sheet_name=sheet, index=False)
        for ddf, sheet in [(daily1, "Daily_V7clean"), (daily2, "Daily_V7v2")]:
            d = ddf.copy()
            d["price_date"] = d["price_date"].dt.strftime("%Y-%m-%d")
            for c in d.select_dtypes(include=["float64"]).columns:
                d[c] = d[c].round(2)
            d.to_excel(xl, sheet_name=sheet, index=False)
    print(f"[save] {xlsx}\n")

    # Chart
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                              gridspec_kw={"height_ratios": [2.5, 1]})
    axes[0].plot(daily1["price_date"], daily1["cum_pnl_net"], color="#e76f51", lw=1.8,
                  label=f"V7-clean TP+{best1['TP']}/SL{best1['SL']}: {m1['Net (만)']:+,.0f}만 sharpe {m1['Sharpe net']:+.2f}")
    axes[0].plot(daily2["price_date"], daily2["cum_pnl_net"], color="#264653", lw=2.2,
                  label=f"V7-clean-v2 TP+{best2['TP']}/SL{best2['SL']}: {m2['Net (만)']:+,.0f}만 sharpe {m2['Sharpe net']:+.2f}")
    axes[0].axhline(0, color="gray", lw=0.7, ls="--")
    axes[0].set_title("V7 / V7-v2 with TP/SL (best grid)", fontsize=13, weight="bold")
    axes[0].set_ylabel("Cumulative Net P&L (만)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper left")

    axes[1].fill_between(daily2["price_date"], 0, daily2["drawdown_man"], color="#264653", alpha=0.3)
    axes[1].plot(daily1["price_date"], daily1["drawdown_man"], color="#e76f51", lw=1.0, alpha=0.7,
                  label=f"v1 MDD {m1['MaxDD (만)']:,.0f}")
    axes[1].plot(daily2["price_date"], daily2["drawdown_man"], color="#264653", lw=1.4,
                  label=f"v2 MDD {m2['MaxDD (만)']:,.0f}")
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("DD (만)")
    axes[1].legend(loc="lower right")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "33_v7_tpsl.png", bbox_inches="tight")
    plt.close(fig)
    print("[chart] OK 33_v7_tpsl.png")
    print("[done]")


if __name__ == "__main__":
    main()
