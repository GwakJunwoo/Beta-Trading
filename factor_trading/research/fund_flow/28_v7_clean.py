"""
28 — V7-clean: sign-stable cells 만 활성화한 delta-neutral slope pair.

27번 cell stability test 결과:
  STABLE (3 sub-period 부호 일관) — V7-clean 에 사용:
    1001 STEEPENER  P1+7.59 P2+6.68 P3+8.33   (N=23 total, hit 91%)
    1100 STEEPENER  P1+6.35 P2+0.26 P3+4.09   (N=23 total)
    1101 STEEPENER  P1+2.07 P2+3.27 P3+3.10   (N=68 total, hit 77%)
    1000 STEEPENER  P1 N=0  P2+2.12 P3+2.77   (N=8 total — small but stable)
    0111 FLATTENER  P1-2.44 P2-0.21 P3-1.12   (N=179 total)

  FLIPPED (regime dependent) — V7-clean 에서 제거:
    0011  P1-5.68 P2-2.97 P3+0.37 ❌ (강세장에 뒤집힘, V7 최대 driver였음)
    1011  P1-3.58 P2+0.82 P3-1.24 ❌
    0101  P1+3.70 P2-4.93 P3+1.51 ❌

V7-clean rule:
  1001 +2.0   STEEPENER (강도 최강)
  1100 +1.0   STEEPENER
  1101 +1.0   STEEPENER
  1000 +0.5   STEEPENER (small N)
  0111 -0.5   FLATTENER (only flattener)

Backtest:
  Entry T+1, Exit T+21 close, Overlapping, delta-neutral pair (KTB10F + KTB3F)
  Cost: KTB10F 0.12bp, KTB3F 0.05bp round trip
  비교: V7 (8 cell, 원래) vs V7-clean (5 stable) vs Walk-forward V7-clean
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
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 140

RULE_V7_ORIG = {
    "1001": +2.0, "1100": +1.0, "1101": +1.0, "1000": +0.5,
    "0011": -1.0, "0111": -0.5, "1011": -0.5, "0101": -0.3,
}
RULE_V7_CLEAN = {
    "1001": +2.0,   # stable STEEPENER (strongest)
    "1100": +1.0,
    "1101": +1.0,
    "1000": +0.5,
    "0111": -0.5,   # only stable FLATTENER
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


def backtest(p, rule, apply_cost=True):
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
            "cell": c,
            "direction": "STEEPENER" if size > 0 else "FLATTENER",
            "size_unit": size,
            "pos_10F_ctr": round(pos_10, 3),
            "pos_3F_ctr": round(pos_3, 3),
            "fwd_dy10_bp": float(fwd_dy10) if pd.notna(fwd_dy10) else np.nan,
            "fwd_dy3_bp": float(fwd_dy3) if pd.notna(fwd_dy3) else np.nan,
            "fwd_dslope_bp": float(fwd_dslope) if pd.notna(fwd_dslope) else np.nan,
            "trade_gross_man": float(t_gross) if pd.notna(t_gross) else np.nan,
            "trade_cost_man": float(cost),
            "trade_net_man": float(t_net) if pd.notna(t_net) else np.nan,
        })
        if apply_cost:
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
    hit = (trades["trade_net_man"] > 0).mean() * 100 if len(trades) else 0
    return {
        "name": name, "Trades": len(trades),
        "Gross (만)": round(gross, 0), "Cost (만)": round(cost, 0),
        "Net (만)": round(net, 0), "Per_yr (만)": round(net / nyrs, 0) if nyrs > 0 else 0,
        "Sharpe gross": round(sh_g, 2), "Sharpe net": round(sh_n, 2),
        "MaxDD (만)": round(mdd, 0), "Hit (%)": round(hit, 1),
        "Active days": len(s_n),
        "Avg concurrent pos": round(len(trades) * HOLD / len(daily), 1),
    }


def walk_forward_clean(p, rule_template):
    """V7-clean rule 의 cell sign 만 사용 (5 stable cells).
    매 시점 [start, t-22] 데이터로 cell mean 의 부호만 확인, 변화 없으면 그대로 사용.
    """
    n = len(p)
    daily_pnl = np.zeros(n)
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
                        continue   # 학습 데이터에서 부호 다르면 skip
                    cur_rule[c] = original_size   # 사이즈는 priori 그대로
                last_refit = i
        if i < WARM_UP or not cur_rule:
            continue
        c = cells_seq[i]
        if c not in cur_rule:
            continue
        size = cur_rule[c]
        pos_10 = -size
        pos_3 = +size * RATIO
        n_trades += 1
        for d in range(i + 1, min(i + HOLD + 1, n)):
            daily_pnl[d] += pos_10 * (-dy10_1d[d]) * DV01_KTB10F \
                            + pos_3 * (-dy3_1d[d]) * DV01_KTB3F
    return daily_pnl, n_trades


def main():
    print("[load] panel ...")
    p = load_panel("2020-01-01")
    print(f"  {len(p):,} rows  {p['price_date'].min().date()} ~ {p['price_date'].max().date()}\n")

    fmt = lambda b: "BUY" if int(b) else "SELL"
    print("V7-clean Rule (5 stable cells):")
    for c, sz in sorted(RULE_V7_CLEAN.items(), key=lambda x: -x[1]):
        direction = "STEEPENER" if sz > 0 else "FLATTENER"
        pos_10 = -sz
        pos_3 = sz * RATIO
        action10 = f"{'SHORT' if pos_10 < 0 else 'LONG '} {abs(pos_10):.2f}"
        action3 = f"{'LONG ' if pos_3 > 0 else 'SHORT'} {abs(pos_3):.2f}"
        print(f"  {c} (f10={fmt(c[0])}, f3={fmt(c[1])}, b10F={fmt(c[2])}, b3F={fmt(c[3])}): "
              f"{direction:10s} size {abs(sz):.1f}  10F:{action10}  3F:{action3}")
    print()
    print("Removed (sign-flipped, regime-dependent):")
    for c in sorted(set(RULE_V7_ORIG) - set(RULE_V7_CLEAN)):
        print(f"  {c} (sign flip across sub-periods)")
    print()

    # ── 백테스트 ──
    daily_orig, trades_orig = backtest(p, RULE_V7_ORIG, apply_cost=True)
    daily_clean, trades_clean = backtest(p, RULE_V7_CLEAN, apply_cost=True)
    m_orig = metrics(daily_orig, trades_orig, "V7 orig (8 cells)")
    m_clean = metrics(daily_clean, trades_clean, "V7-clean (5 stable)")

    print("=" * 90)
    print("Backtest 비교 (in-sample, cost-adjusted)")
    print("=" * 90)
    for m in [m_orig, m_clean]:
        print(f"\n  {m['name']}")
        for k, v in m.items():
            if k == "name":
                continue
            if isinstance(v, float):
                print(f"    {k:>22s}: {v:>+12,.2f}")
            else:
                print(f"    {k:>22s}: {v:>+12,d}" if isinstance(v, int) else f"    {k:>22s}: {v}")

    # ── 연도별 ──
    print("\n연도별 P&L (net, 만):")
    yr_orig = daily_orig.groupby("year")["daily_pnl_net"].sum().round(0)
    yr_clean = daily_clean.groupby("year")["daily_pnl_net"].sum().round(0)
    yr_df = pd.DataFrame({"V7 orig": yr_orig, "V7-clean": yr_clean})
    print(yr_df.to_string())
    print()

    # ── Walk-forward V7-clean ──
    print("=" * 90)
    print("Walk-forward V7-clean (cell sign 도 매 분기 재확인, 부호 다르면 skip)")
    print("=" * 90)
    wf_pnl, wf_n_trades = walk_forward_clean(p, RULE_V7_CLEAN)
    wf_total = wf_pnl.sum()
    wf_active = wf_pnl[wf_pnl != 0]
    wf_sh = wf_active.mean() / wf_active.std() * np.sqrt(TRADING_DAYS) if len(wf_active) > 1 and wf_active.std() > 0 else 0
    wf_cum = wf_pnl.cumsum()
    wf_mdd = (wf_cum - np.maximum.accumulate(wf_cum)).min()
    nyrs = len(p) / TRADING_DAYS
    print(f"\n  Trades: {wf_n_trades}")
    print(f"  Total: {wf_total:+,.0f} 만")
    print(f"  Per_yr: {wf_total/nyrs:+,.0f} 만/y")
    print(f"  Sharpe: {wf_sh:+.2f}")
    print(f"  MaxDD: {wf_mdd:+,.0f} 만")
    wf_yr = (pd.DataFrame({"date": p["price_date"], "pnl": wf_pnl})
              .set_index("date").resample("YE").sum())
    wf_yr["year"] = wf_yr.index.year
    print("\n  연도별:")
    print(wf_yr[["year", "pnl"]].to_string(index=False))
    print()

    # ── 5/11 시그널 ──
    latest = p.iloc[-1]
    cell = latest["cell"]
    print("=" * 90)
    print("5/11 V7-clean 시그널")
    print("=" * 90)
    print(f"  cell: {cell}  (f10={fmt(cell[0])}, f3={fmt(cell[1])}, b10F={fmt(cell[2])}, b3F={fmt(cell[3])})")
    if cell in RULE_V7_CLEAN:
        sz = RULE_V7_CLEAN[cell]
        direction = "STEEPENER" if sz > 0 else "FLATTENER"
        print(f"  --> {direction} size {abs(sz):.1f}")
        print(f"      KTB10F {'SHORT' if -sz < 0 else 'LONG '} {abs(-sz):.2f} 계약")
        print(f"      KTB3F  {'LONG ' if sz*RATIO > 0 else 'SHORT'} {abs(sz*RATIO):.2f} 계약")
    else:
        print(f"  --> FLAT (cell not in V7-clean rule)")
    print()

    # ── Excel + Chart ──
    CHART_DIR.mkdir(exist_ok=True)
    xlsx = CHART_DIR / "V7clean_track_record.xlsx"
    rule_df = pd.DataFrame([
        {"cell": c, "f10": fmt(c[0]), "f3": fmt(c[1]), "b10F": fmt(c[2]), "b3F": fmt(c[3]),
         "direction": "STEEPENER" if sz > 0 else "FLATTENER",
         "size_unit": sz,
         "pos_10F_ctr": -sz, "pos_3F_ctr": round(sz * RATIO, 3)}
        for c, sz in RULE_V7_CLEAN.items()
    ])
    summary_df = pd.DataFrame([m_orig, m_clean])
    removed_df = pd.DataFrame([
        {"cell": "0011", "removed_because": "P3 sign flip (-5.68 -> +0.37)"},
        {"cell": "1011", "removed_because": "P2 sign flip (-3.58 -> +0.82 -> -1.24)"},
        {"cell": "0101", "removed_because": "P1, P3 sign flip (+3.70 -> -4.93 -> +1.51)"},
    ])
    wf_summary = pd.DataFrame({
        "Metric": ["Trades", "Total (만)", "Per_yr", "Sharpe", "MaxDD"],
        "Value": [wf_n_trades, round(wf_total, 0), round(wf_total/nyrs, 0),
                   round(wf_sh, 2), round(wf_mdd, 0)],
    })
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xl:
        summary_df.to_excel(xl, sheet_name="Summary", index=False)
        rule_df.to_excel(xl, sheet_name="V7clean_rule", index=False)
        removed_df.to_excel(xl, sheet_name="Removed_cells", index=False)
        yr_df.reset_index().to_excel(xl, sheet_name="Yearly", index=False)
        wf_summary.to_excel(xl, sheet_name="WalkForward_V7clean", index=False)
        wf_yr[["year", "pnl"]].to_excel(xl, sheet_name="WF_yearly", index=False)
        t = trades_clean.copy()
        for c in t.select_dtypes(include=["float64"]).columns:
            t[c] = t[c].round(2)
        t.to_excel(xl, sheet_name="Trades_clean", index=False)
        d = daily_clean.copy()
        d["price_date"] = d["price_date"].dt.strftime("%Y-%m-%d")
        for c in d.select_dtypes(include=["float64"]).columns:
            d[c] = d[c].round(2)
        d.to_excel(xl, sheet_name="Daily_clean", index=False)
    print(f"[save] {xlsx}")

    # 차트
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                              gridspec_kw={"height_ratios": [2.5, 1]})
    axes[0].plot(daily_orig["price_date"], daily_orig["cum_pnl_net"],
                  color="#e76f51", lw=1.6, alpha=0.7,
                  label=f"V7 orig (8 cells)  {m_orig['Net (만)']:+,.0f}만  Sharpe {m_orig['Sharpe net']:+.2f}")
    axes[0].plot(daily_clean["price_date"], daily_clean["cum_pnl_net"],
                  color="#264653", lw=2.2,
                  label=f"V7-clean (5 stable)  {m_clean['Net (만)']:+,.0f}만  Sharpe {m_clean['Sharpe net']:+.2f}")
    axes[0].plot(p["price_date"], wf_cum, color="#2a9d8f", lw=1.4, alpha=0.7, ls="--",
                  label=f"V7-clean Walk-fwd  {wf_total:+,.0f}만  Sharpe {wf_sh:+.2f}")
    axes[0].axhline(0, color="gray", lw=0.7, ls="--")
    axes[0].set_title("V7 vs V7-clean (sign-stable only) vs Walk-forward",
                       fontsize=13, weight="bold")
    axes[0].set_ylabel("Cumulative Net P&L (만)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper left")

    dd = daily_clean["drawdown_man"]
    axes[1].fill_between(daily_clean["price_date"], 0, dd, color="#e76f51", alpha=0.35)
    axes[1].plot(daily_clean["price_date"], dd, color="#a8331b", lw=1.2)
    axes[1].set_title(f"V7-clean Drawdown (MaxDD {m_clean['MaxDD (만)']:,.0f}만)")
    axes[1].set_ylabel("DD (만)")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "30_v7_clean.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[chart] OK 30_v7_clean.png\n")
    print("[done]")


if __name__ == "__main__":
    main()
