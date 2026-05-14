"""
30 — V7-clean-v2: 0111 (유일 FLATTENER) 제거 후 STEEPENER only.

V7-clean (28번) 의 0111 cell 이 worst trades 의 주범:
  - 2020: 8 worst 중 7개가 0111 (steepener regime 에서 FLATTENER fail)
  - 2023: 8 worst 중 6개가 0111
  - 0111 expected = -, actual mean 부호 자주 양수로 뒤집힘
  - 시장이 strong flattener regime 일 때만 작동

V7-clean-v2: 0111 제거 → 4 STEEPENER cells only
  1001 +2.0  (mean +7.71, hit 91%, N=23)
  1100 +1.0  (mean +3.45, hit 70%, N=23)
  1101 +1.0  (mean +2.90, hit 77%, N=68)
  1000 +0.5  (mean +2.36, hit 75%, N=8)

비교: V7-clean (5 cells) vs V7-clean-v2 (4 STEEPENER)
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
    p["dy3_1d"] = p["y_3y"].diff()
    p["dy10_1d"] = p["y_10y"].diff()
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


def backtest(p, rule):
    n = len(p)
    daily_pnl = np.zeros(n)
    daily_cost = np.zeros(n)
    daily_pos10 = np.zeros(n)
    daily_pos3 = np.zeros(n)
    dy10_1d = p["dy10_1d"].fillna(0.0).values
    dy3_1d = p["dy3_1d"].fillna(0.0).values
    cells = p["cell"].values
    dates = p["price_date"].values
    trades = []
    for i in range(n):
        c = cells[i]
        if c not in rule:
            continue
        size = rule[c]
        pos_10 = -size
        pos_3 = +size * RATIO
        i_exit = min(i + HOLD, n - 1)
        fwd_dy10 = p["dy10_fwd_21"].iloc[i] if i + HOLD < n else np.nan
        fwd_dy3 = p["dy3_fwd_21"].iloc[i] if i + HOLD < n else np.nan
        fwd_dslope = p["dslope_fwd_21"].iloc[i] if i + HOLD < n else np.nan
        t_gross = (pos_10 * (-fwd_dy10) * DV01_KTB10F
                    + pos_3 * (-fwd_dy3) * DV01_KTB3F) if pd.notna(fwd_dy10) else np.nan
        cost = (abs(pos_10) * TC_10F_BP * DV01_KTB10F
                + abs(pos_3) * TC_3F_BP * DV01_KTB3F)
        t_net = t_gross - cost if pd.notna(t_gross) else np.nan
        trades.append({
            "entry_date": pd.Timestamp(dates[i]),
            "exit_date": pd.Timestamp(dates[i_exit]),
            "year": int(p["year"].iloc[i]),
            "cell": c,
            "direction": "STEEPENER" if size > 0 else "FLATTENER",
            "size_unit": size,
            "pos_10F": round(pos_10, 3), "pos_3F": round(pos_3, 3),
            "fwd_dy10_bp": float(fwd_dy10) if pd.notna(fwd_dy10) else np.nan,
            "fwd_dy3_bp": float(fwd_dy3) if pd.notna(fwd_dy3) else np.nan,
            "fwd_dslope_bp": float(fwd_dslope) if pd.notna(fwd_dslope) else np.nan,
            "gross_pnl": float(t_gross) if pd.notna(t_gross) else np.nan,
            "cost": float(cost),
            "net_pnl": float(t_net) if pd.notna(t_net) else np.nan,
        })
        entry_d = min(i + 1, n - 1)
        exit_d = i_exit
        daily_cost[entry_d] += cost / 2
        daily_cost[exit_d] += cost / 2
        for d in range(i + 1, min(i + HOLD + 1, n)):
            daily_pos10[d] += pos_10
            daily_pos3[d] += pos_3
            daily_pnl[d] += pos_10 * (-dy10_1d[d]) * DV01_KTB10F \
                            + pos_3 * (-dy3_1d[d]) * DV01_KTB3F
    daily_net = daily_pnl - daily_cost
    daily = p[["price_date", "year"]].copy()
    daily["pos_10F"] = daily_pos10
    daily["pos_3F"] = daily_pos3
    daily["net_DV01_man"] = daily_pos10 * DV01_KTB10F + daily_pos3 * DV01_KTB3F
    daily["daily_pnl_gross"] = daily_pnl
    daily["daily_cost"] = daily_cost
    daily["daily_pnl_net"] = daily_net
    daily["cum_pnl_net"] = daily_net.cumsum()
    daily["peak"] = daily["cum_pnl_net"].cummax()
    daily["drawdown_man"] = daily["cum_pnl_net"] - daily["peak"]
    return daily, pd.DataFrame(trades)


def metrics(daily, trades, name):
    s_g = daily["daily_pnl_gross"][daily["daily_pnl_gross"] != 0]
    s_n = daily["daily_pnl_net"][daily["daily_pnl_net"] != 0]
    sh_g = s_g.mean() / s_g.std() * np.sqrt(TRADING_DAYS) if len(s_g) > 1 and s_g.std() > 0 else 0
    sh_n = s_n.mean() / s_n.std() * np.sqrt(TRADING_DAYS) if len(s_n) > 1 and s_n.std() > 0 else 0
    gross = daily["daily_pnl_gross"].sum()
    net = daily["daily_pnl_net"].sum()
    cost = daily["daily_cost"].sum()
    nyrs = len(daily) / TRADING_DAYS
    mdd = daily["drawdown_man"].min()
    hit = (trades["net_pnl"] > 0).mean() * 100 if len(trades) else 0
    avg_win = trades.loc[trades["net_pnl"] > 0, "net_pnl"].mean() if (trades["net_pnl"] > 0).any() else 0
    avg_loss = trades.loc[trades["net_pnl"] < 0, "net_pnl"].mean() if (trades["net_pnl"] < 0).any() else 0
    return {
        "name": name, "Trades": len(trades),
        "Gross (만)": round(gross, 0), "Cost (만)": round(cost, 0),
        "Net (만)": round(net, 0), "Per_yr (만)": round(net / nyrs, 0) if nyrs > 0 else 0,
        "Sharpe gross": round(sh_g, 2), "Sharpe net": round(sh_n, 2),
        "MaxDD (만)": round(mdd, 0), "Hit (%)": round(hit, 1),
        "Avg win (만)": round(avg_win, 1), "Avg loss (만)": round(avg_loss, 1),
        "W/L ratio": round(avg_win / -avg_loss, 2) if avg_loss < 0 else None,
        "Avg concurrent pos": round(len(trades) * HOLD / len(daily), 1),
        "Active days": len(s_n),
        "Calmar (per_yr/|MDD|)": round((net/nyrs) / abs(mdd), 2) if mdd != 0 else None,
    }


def walk_forward(p, rule_template):
    n = len(p)
    daily_pnl = np.zeros(n)
    daily_cost = np.zeros(n)
    dy10_1d = p["dy10_1d"].fillna(0.0).values
    dy3_1d = p["dy3_1d"].fillna(0.0).values
    cells_seq = p["cell"].values
    WARM_UP = 252
    REFIT_FREQ = 63
    MIN_N = 3
    cur_rule = {}
    last_refit = -REFIT_FREQ - 1
    n_trades = 0
    for i in range(n):
        if i - last_refit >= REFIT_FREQ and i >= WARM_UP:
            train_idx = i - HOLD - 1
            if train_idx > 0:
                train_p = p.iloc[:train_idx + 1].dropna(subset=["dslope_fwd_21"])
                tbl = train_p.groupby("cell").agg(
                    N=("dslope_fwd_21", "size"),
                    mean=("dslope_fwd_21", "mean"),
                )
                cur_rule = {}
                for c, original_size in rule_template.items():
                    if c not in tbl.index:
                        continue
                    if tbl.loc[c, "N"] < MIN_N:
                        continue
                    actual_sign = np.sign(tbl.loc[c, "mean"])
                    template_sign = np.sign(original_size)
                    if actual_sign != template_sign:
                        continue
                    cur_rule[c] = original_size
                last_refit = i
        if i < WARM_UP or not cur_rule:
            continue
        c = cells_seq[i]
        if c not in cur_rule:
            continue
        size = cur_rule[c]
        pos_10 = -size
        pos_3 = +size * RATIO
        cost = abs(pos_10) * TC_10F_BP * DV01_KTB10F + abs(pos_3) * TC_3F_BP * DV01_KTB3F
        n_trades += 1
        ent = min(i + 1, n - 1)
        ext = min(i + HOLD, n - 1)
        daily_cost[ent] += cost / 2
        daily_cost[ext] += cost / 2
        for d in range(i + 1, min(i + HOLD + 1, n)):
            daily_pnl[d] += pos_10 * (-dy10_1d[d]) * DV01_KTB10F \
                            + pos_3 * (-dy3_1d[d]) * DV01_KTB3F
    daily_net = daily_pnl - daily_cost
    return daily_net, n_trades


def main():
    print("[load] panel ...")
    p = load_panel("2020-01-01")
    print(f"  {len(p):,} rows  {p['price_date'].min().date()} ~ {p['price_date'].max().date()}\n")

    # ── Backtest 둘 다 ──
    daily_v1, trades_v1 = backtest(p, RULE_V7_CLEAN)
    daily_v2, trades_v2 = backtest(p, RULE_V7_CLEAN_V2)
    m1 = metrics(daily_v1, trades_v1, "V7-clean (5 cells, with 0111)")
    m2 = metrics(daily_v2, trades_v2, "V7-clean-v2 (4 STEEPENER only)")

    print("=" * 90)
    print("Backtest 비교 (cost-adjusted, in-sample priori)")
    print("=" * 90)
    for m in [m1, m2]:
        print(f"\n  {m['name']}")
        for k, v in m.items():
            if k == "name":
                continue
            if isinstance(v, float):
                print(f"    {k:>26s}: {v:>+12,.2f}")
            elif isinstance(v, int):
                print(f"    {k:>26s}: {v:>+12,d}")
            else:
                print(f"    {k:>26s}: {v}")

    # ── 연도별 ──
    print("\n연도별 P&L (net, 만):")
    yr_v1 = daily_v1.groupby("year")["daily_pnl_net"].sum().round(0)
    yr_v2 = daily_v2.groupby("year")["daily_pnl_net"].sum().round(0)
    yr_df = pd.DataFrame({"V7-clean": yr_v1, "V7-clean-v2": yr_v2,
                            "diff (v2-v1)": (yr_v2 - yr_v1).round(0)})
    print(yr_df.to_string())
    print()

    # ── Hit rate 비교 ──
    print("Hit rate 비교 (net P&L > 0 비율):")
    yr_hit1 = trades_v1.groupby("year").apply(lambda g: (g["net_pnl"] > 0).mean() * 100).round(1)
    yr_hit2 = trades_v2.groupby("year").apply(lambda g: (g["net_pnl"] > 0).mean() * 100).round(1)
    hit_df = pd.DataFrame({"V7-clean": yr_hit1, "V7-clean-v2": yr_hit2})
    print(hit_df.to_string())
    print()

    # ── Walk-forward 비교 ──
    print("=" * 90)
    print("Walk-forward 비교")
    print("=" * 90)
    wf1_pnl, wf1_n = walk_forward(p, RULE_V7_CLEAN)
    wf2_pnl, wf2_n = walk_forward(p, RULE_V7_CLEAN_V2)
    nyrs = len(p) / TRADING_DAYS
    for pnl_arr, n_tr, name in [(wf1_pnl, wf1_n, "WF V7-clean"),
                                  (wf2_pnl, wf2_n, "WF V7-clean-v2")]:
        active = pnl_arr[pnl_arr != 0]
        sh = active.mean() / active.std() * np.sqrt(TRADING_DAYS) if len(active) > 1 and active.std() > 0 else 0
        total = pnl_arr.sum()
        cum = np.cumsum(pnl_arr)
        mdd = (cum - np.maximum.accumulate(cum)).min()
        print(f"\n  {name}")
        print(f"    Trades: {n_tr}")
        print(f"    Total: {total:+,.0f} 만")
        print(f"    Per_yr: {total/nyrs:+,.0f} 만/y")
        print(f"    Sharpe: {sh:+.2f}")
        print(f"    MaxDD: {mdd:+,.0f} 만")
    print()

    # ── 5/11 시그널 ──
    fmt = lambda b: "BUY" if int(b) else "SELL"
    latest = p.iloc[-1]
    cell = latest["cell"]
    print("=" * 90)
    print("5/11 V7-clean-v2 시그널")
    print("=" * 90)
    print(f"  cell: {cell}  (f10={fmt(cell[0])}, f3={fmt(cell[1])}, b10F={fmt(cell[2])}, b3F={fmt(cell[3])})")
    if cell in RULE_V7_CLEAN_V2:
        sz = RULE_V7_CLEAN_V2[cell]
        print(f"  --> STEEPENER size {abs(sz):.1f}")
        print(f"      KTB10F SHORT {abs(-sz):.2f} 계약")
        print(f"      KTB3F  LONG  {abs(sz*RATIO):.2f} 계약")
    else:
        print(f"  --> FLAT")
    print()

    # ── Excel ──
    CHART_DIR.mkdir(exist_ok=True)
    xlsx = CHART_DIR / "V7clean_v2_track_record.xlsx"
    rule_df = pd.DataFrame([
        {"cell": c, "f10": fmt(c[0]), "f3": fmt(c[1]), "b10F": fmt(c[2]), "b3F": fmt(c[3]),
         "direction": "STEEPENER" if sz > 0 else "FLATTENER",
         "size_unit": sz,
         "pos_10F_ctr": -sz, "pos_3F_ctr": round(sz * RATIO, 3)}
        for c, sz in RULE_V7_CLEAN_V2.items()
    ])
    summary_df = pd.DataFrame([m1, m2])
    wf_summary = pd.DataFrame({
        "Variant": ["V7-clean (WF)", "V7-clean-v2 (WF)"],
        "Trades": [wf1_n, wf2_n],
        "Total (만)": [round(wf1_pnl.sum(), 0), round(wf2_pnl.sum(), 0)],
        "Per_yr (만)": [round(wf1_pnl.sum()/nyrs, 0), round(wf2_pnl.sum()/nyrs, 0)],
        "Sharpe": [
            round(wf1_pnl[wf1_pnl != 0].mean() / wf1_pnl[wf1_pnl != 0].std() * np.sqrt(TRADING_DAYS), 2)
            if (wf1_pnl[wf1_pnl != 0]).std() > 0 else 0,
            round(wf2_pnl[wf2_pnl != 0].mean() / wf2_pnl[wf2_pnl != 0].std() * np.sqrt(TRADING_DAYS), 2)
            if (wf2_pnl[wf2_pnl != 0]).std() > 0 else 0,
        ],
    })
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xl:
        summary_df.to_excel(xl, sheet_name="Summary_compare", index=False)
        rule_df.to_excel(xl, sheet_name="V7cleanV2_rule", index=False)
        yr_df.reset_index().to_excel(xl, sheet_name="Yearly_compare", index=False)
        hit_df.reset_index().to_excel(xl, sheet_name="Hit_compare", index=False)
        wf_summary.to_excel(xl, sheet_name="WalkForward_compare", index=False)
        t = trades_v2.copy()
        t["entry_date"] = t["entry_date"].dt.strftime("%Y-%m-%d")
        t["exit_date"] = t["exit_date"].dt.strftime("%Y-%m-%d")
        for c in t.select_dtypes(include=["float64"]).columns:
            t[c] = t[c].round(2)
        t.to_excel(xl, sheet_name="Trades_v2", index=False)
        d = daily_v2.copy()
        d["price_date"] = d["price_date"].dt.strftime("%Y-%m-%d")
        for c in d.select_dtypes(include=["float64"]).columns:
            d[c] = d[c].round(2)
        d.to_excel(xl, sheet_name="Daily_v2", index=False)
    print(f"[save] {xlsx}")

    # 차트
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                              gridspec_kw={"height_ratios": [2.5, 1]})
    axes[0].plot(daily_v1["price_date"], daily_v1["cum_pnl_net"],
                  color="#e76f51", lw=1.7, alpha=0.7,
                  label=f"V7-clean (5 cells)  {m1['Net (만)']:+,.0f}만  Sharpe {m1['Sharpe net']:+.2f}")
    axes[0].plot(daily_v2["price_date"], daily_v2["cum_pnl_net"],
                  color="#264653", lw=2.2,
                  label=f"V7-clean-v2 (no 0111)  {m2['Net (만)']:+,.0f}만  Sharpe {m2['Sharpe net']:+.2f}")
    axes[0].axhline(0, color="gray", lw=0.7, ls="--")
    axes[0].set_title("V7-clean vs V7-clean-v2 (0111 제거)", fontsize=13, weight="bold")
    axes[0].set_ylabel("Cumulative Net P&L (만)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper left")

    dd1 = daily_v1["drawdown_man"]
    dd2 = daily_v2["drawdown_man"]
    axes[1].fill_between(daily_v2["price_date"], 0, dd2, color="#264653", alpha=0.3,
                          label=f"v2 MDD {dd2.min():,.0f}")
    axes[1].plot(daily_v1["price_date"], dd1, color="#e76f51", lw=1.0, alpha=0.7,
                  label=f"v1 MDD {dd1.min():,.0f}")
    axes[1].plot(daily_v2["price_date"], dd2, color="#264653", lw=1.4)
    axes[1].axhline(0, color="gray", lw=0.7, ls="--")
    axes[1].set_title("Drawdown comparison")
    axes[1].set_ylabel("DD (만)")
    axes[1].legend(loc="lower right")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "32_v7_clean_v2.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[chart] OK 32_v7_clean_v2.png")
    print("[done]")


if __name__ == "__main__":
    main()
