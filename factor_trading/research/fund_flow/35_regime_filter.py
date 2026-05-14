"""
35 - Regime filter 리서치.

설정: V7-clean (5 cells incl. 0111), TP+6/SL-6, slip 0.5/0.5.

목표: 각 cell 의 trade 를 시장 regime 변수로 분해 → best filter 찾기.

Regime 변수 후보:
  - slope_level (절대 수치)
  - slope_past_5d / 10d / 21d / 63d (단~장기 추세)
  - y10_level
  - y10_past_21d
  - slope vs MA20/60
  - dy10_past_5d, dy3_past_5d
  - fx_past_5d / 21d

분석:
  A) 각 trade entry 시점의 regime variables 추출
  B) Cell × regime quintile 별 net P&L
  C) 가장 informative regime variable per cell 식별
  D) Best filter rule 도출 (각 cell 별 activation condition)
  E) Filter 적용 후 백테스트 vs baseline
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

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
TP_BP = 6.0
SL_BP = -6.0
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

    # Regime variables
    for n in [5, 10, 21, 63]:
        p[f"slope_past_{n}"] = p["slope"] - p["slope"].shift(n)
        p[f"y10_past_{n}"] = p["y_10y"] - p["y_10y"].shift(n)
        p[f"y3_past_{n}"] = p["y_3y"] - p["y_3y"].shift(n)
        p[f"fx_past_{n}"] = p["fx"] - p["fx"].shift(n)
    p["slope_ma20"] = p["slope"].rolling(20).mean()
    p["slope_ma60"] = p["slope"].rolling(60).mean()
    p["slope_vs_ma20"] = p["slope"] - p["slope_ma20"]
    p["slope_vs_ma60"] = p["slope"] - p["slope_ma60"]
    p["slope_zscore_60"] = (p["slope"] - p["slope_ma60"]) / p["slope"].rolling(60).std()

    p["s_f10"] = (p["f10"] > 0).astype(int)
    p["s_f3"] = (p["f3"] > 0).astype(int)
    p["s_b10F"] = (p["b10F"] > 0).astype(int)
    p["s_b3F"] = (p["b3F"] > 0).astype(int)
    p["cell"] = (p["s_f10"].astype(str) + p["s_f3"].astype(str)
                  + p["s_b10F"].astype(str) + p["s_b3F"].astype(str))
    return p


def backtest_slippage(p, rule, tp_bp, sl_bp, slip=SLIP_BP, filter_fn=None):
    """filter_fn(cell, regime_dict) -> True/False (활성 여부)"""
    n = len(p)
    y10 = p["y_10y"].values
    y3 = p["y_3y"].values
    cells = p["cell"].values
    dates = p["price_date"].values
    active = []
    daily_pnl = np.zeros(n)
    daily_cost = np.zeros(n)
    closed = []
    # 정렬된 regime columns
    regime_cols = [
        "slope", "slope_past_5", "slope_past_21", "slope_past_63",
        "y10_past_5", "y10_past_21", "y3_past_5", "y3_past_21",
        "fx_past_5", "fx_past_21",
        "slope_vs_ma20", "slope_vs_ma60", "slope_zscore_60",
    ]
    p_dict = {c: p[c].values for c in regime_cols if c in p.columns}
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
                exit_reason = "TP"; realized_bp = tp_bp - slip
            elif pnl_bp <= sl_bp:
                exit_reason = "SL"; realized_bp = sl_bp - slip
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
                closed.append(tr)
            else:
                still_active.append(tr)
        active = still_active
        cur_10F = sum(tr["pos_10"] for tr in active)
        c = cells[i]
        if c in rule:
            # filter check
            if filter_fn is not None:
                regime_dict = {col: p_dict[col][i] for col in p_dict if i < len(p_dict[col])}
                if not filter_fn(c, regime_dict):
                    pass
                else:
                    # 진입 허용
                    size_units = rule[c]
                    new_pos_10 = -size_units * SIZE_10F_PER_UNIT
                    new_pos_3 = +size_units * SIZE_3F_PER_UNIT
                    if abs(cur_10F + new_pos_10) <= MAX_10F_NOTIONAL:
                        entry_cost = (abs(new_pos_10) * TC_10F_BP * DV01_KTB10F
                                       + abs(new_pos_3) * TC_3F_BP * DV01_KTB3F)
                        daily_cost[i] += entry_cost
                        # Save entry regime for analysis
                        tr_new = {
                            "entry_idx": i, "entry_date": pd.Timestamp(dates[i]),
                            "cell": c, "size_units": size_units,
                            "pos_10": new_pos_10, "pos_3": new_pos_3,
                            "entry_y10": float(y10[i]), "entry_y3": float(y3[i]),
                            "cum_pnl": 0.0, "entry_cost": entry_cost,
                        }
                        for col in regime_cols:
                            if col in p.columns:
                                tr_new[f"entry_{col}"] = float(p[col].iloc[i]) if pd.notna(p[col].iloc[i]) else np.nan
                        active.append(tr_new)
            else:
                size_units = rule[c]
                new_pos_10 = -size_units * SIZE_10F_PER_UNIT
                new_pos_3 = +size_units * SIZE_3F_PER_UNIT
                if abs(cur_10F + new_pos_10) <= MAX_10F_NOTIONAL:
                    entry_cost = (abs(new_pos_10) * TC_10F_BP * DV01_KTB10F
                                   + abs(new_pos_3) * TC_3F_BP * DV01_KTB3F)
                    daily_cost[i] += entry_cost
                    tr_new = {
                        "entry_idx": i, "entry_date": pd.Timestamp(dates[i]),
                        "cell": c, "size_units": size_units,
                        "pos_10": new_pos_10, "pos_3": new_pos_3,
                        "entry_y10": float(y10[i]), "entry_y3": float(y3[i]),
                        "cum_pnl": 0.0, "entry_cost": entry_cost,
                    }
                    for col in regime_cols:
                        if col in p.columns:
                            tr_new[f"entry_{col}"] = float(p[col].iloc[i]) if pd.notna(p[col].iloc[i]) else np.nan
                    active.append(tr_new)
    for tr in active:
        cost = (abs(tr["pos_10"]) * TC_10F_BP * DV01_KTB10F
                 + abs(tr["pos_3"]) * TC_3F_BP * DV01_KTB3F)
        daily_cost[n - 1] += cost
        tr["exit_idx"] = n - 1
        tr["exit_date"] = pd.Timestamp(dates[n - 1])
        tr["held_days"] = n - 1 - tr["entry_idx"]
        tr["exit_reason"] = "END"
        avg_dv01 = (abs(tr["pos_10"]) * DV01_KTB10F + abs(tr["pos_3"]) * DV01_KTB3F) / 2.0
        tr["pnl_bp_final"] = tr["cum_pnl"] / avg_dv01 if avg_dv01 > 0 else 0
        tr["net_pnl"] = tr["cum_pnl"] - cost - tr["entry_cost"]
        closed.append(tr)
    daily_net = daily_pnl - daily_cost
    daily = p[["price_date", "year"]].copy()
    daily["daily_pnl_net"] = daily_net
    daily["cum_pnl_net"] = daily_net.cumsum()
    daily["drawdown"] = daily["cum_pnl_net"] - daily["cum_pnl_net"].cummax()
    return daily, pd.DataFrame(closed)


def perf(daily, trades, name="x"):
    s_n = daily["daily_pnl_net"][daily["daily_pnl_net"] != 0]
    sh = s_n.mean()/s_n.std()*np.sqrt(TRADING_DAYS) if len(s_n) > 1 and s_n.std() > 0 else 0
    net = daily["daily_pnl_net"].sum()
    nyrs = len(daily) / TRADING_DAYS
    mdd = daily["drawdown"].min()
    wins = trades[trades["net_pnl"] > 0]
    losses = trades[trades["net_pnl"] < 0]
    hit = len(wins)/(len(wins)+len(losses))*100 if (len(wins)+len(losses)) else 0
    wl = wins["net_pnl"].mean()/-losses["net_pnl"].mean() if len(losses) and losses["net_pnl"].mean() < 0 else None
    return {
        "name": name, "Trades": len(trades),
        "Net (만)": round(net, 0), "Per_yr (만)": round(net/nyrs, 0),
        "Sharpe": round(sh, 2),
        "MaxDD (만)": round(mdd, 0),
        "Hit (%)": round(hit, 1),
        "W/L ratio": round(wl, 2) if wl else None,
    }


def main():
    print("[load] panel ...")
    p = load_panel("2020-01-01")
    print(f"  {len(p):,} rows\n")

    # ── Baseline ──
    print("=" * 100)
    print("Baseline: V7-clean (5 cells incl. 0111), TP+6/SL-6, slip 0.5")
    print("=" * 100)
    daily_b, trades_b = backtest_slippage(p, RULE_V7_CLEAN, TP_BP, SL_BP, slip=SLIP_BP)
    trades_b["year"] = pd.to_datetime(trades_b["entry_date"]).dt.year
    m_b = perf(daily_b, trades_b, "Baseline")
    print()
    for k, v in m_b.items():
        if k != "name":
            print(f"  {k:>14s}: {v}")
    print()

    # ── A) Regime variables × Cell quintile 분석 ──
    print("=" * 100)
    print("A) 각 cell × regime variable quintile 별 net P&L")
    print("=" * 100)
    regime_vars = [
        "entry_slope", "entry_slope_past_5", "entry_slope_past_21", "entry_slope_past_63",
        "entry_y10_past_5", "entry_y10_past_21",
        "entry_fx_past_5", "entry_fx_past_21",
        "entry_slope_vs_ma20", "entry_slope_vs_ma60", "entry_slope_zscore_60",
    ]

    cells_in_rule = sorted(RULE_V7_CLEAN.keys())
    for cell in cells_in_rule:
        sub = trades_b[trades_b["cell"] == cell].copy()
        if len(sub) < 20:
            print(f"\n  -> {cell} (N={len(sub)}, sum {sub['net_pnl'].sum():+,.0f}만): too few trades, skip")
            continue
        direction = "STEEPENER" if RULE_V7_CLEAN[cell] > 0 else "FLATTENER"
        print(f"\n  -> {cell} ({direction}, N={len(sub)}, total {sub['net_pnl'].sum():+,.0f}만)")
        for rv in regime_vars:
            if rv not in sub.columns:
                continue
            ss = sub.dropna(subset=[rv])
            if len(ss) < 10:
                continue
            try:
                ss["q"] = pd.qcut(ss[rv], q=4, labels=["Q1 low", "Q2", "Q3", "Q4 high"],
                                    duplicates="drop")
            except Exception:
                continue
            g = ss.groupby("q", observed=True).agg(
                N=("net_pnl", "size"),
                mean=("net_pnl", "mean"),
                sum=("net_pnl", "sum"),
                hit=("net_pnl", lambda x: (x > 0).mean() * 100),
            ).round(0)
            # 가장 좋은 quintile 과 가장 나쁜 quintile P&L 차
            if "Q4 high" in g.index and "Q1 low" in g.index:
                diff = g.loc["Q4 high", "sum"] - g.loc["Q1 low", "sum"]
            else:
                diff = 0
            corr = ss[rv].corr(ss["net_pnl"], method="spearman")
            print(f"    {rv:<25s} corr={corr:+.3f}  diff(Q4-Q1)={diff:+,.0f}만")
        print()

    # ── B) Top regime variable per cell (correlation 기준) ──
    print("=" * 100)
    print("B) Cell 별 가장 informative regime variable (|spearman corr| 기준)")
    print("=" * 100)
    top_per_cell = {}
    print()
    for cell in cells_in_rule:
        sub = trades_b[trades_b["cell"] == cell].copy()
        if len(sub) < 20:
            continue
        best_rv = None
        best_corr = 0
        for rv in regime_vars:
            if rv not in sub.columns:
                continue
            ss = sub.dropna(subset=[rv])
            if len(ss) < 10:
                continue
            corr = ss[rv].corr(ss["net_pnl"], method="spearman")
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_rv = rv
        top_per_cell[cell] = (best_rv, best_corr)
        direction = "STEEPENER" if RULE_V7_CLEAN[cell] > 0 else "FLATTENER"
        print(f"  {cell} {direction}: best regime var = {best_rv}, corr = {best_corr:+.3f}")
    print()

    # ── C) Best filter 만들기 ──
    print("=" * 100)
    print("C) Filter rule 후보 - 각 cell 활성 조건 (data-driven)")
    print("=" * 100)
    print()
    # 각 cell 마다 best regime var 사용 + threshold 는 median 또는 quintile 기반
    cell_filter_rule = {}   # cell -> (regime_var, condition lambda, threshold)
    for cell, (rv, corr) in top_per_cell.items():
        if rv is None or abs(corr) < 0.1:
            cell_filter_rule[cell] = (None, None, None)
            print(f"  {cell}: no strong filter (best corr={corr:+.3f}) → 항상 활성")
            continue
        sub = trades_b[trades_b["cell"] == cell].dropna(subset=[rv]).copy()
        # 상위/하위 50% 중 어느 쪽이 더 좋은가
        median_rv = sub[rv].median()
        upper = sub[sub[rv] > median_rv]["net_pnl"]
        lower = sub[sub[rv] <= median_rv]["net_pnl"]
        if upper.mean() > lower.mean():
            condition = "above_median"
            threshold = median_rv
            avg_upper = upper.mean()
            avg_lower = lower.mean()
            print(f"  {cell}: {rv} > {threshold:+.2f} 활성 "
                  f"(upper avg {avg_upper:+,.0f} vs lower avg {avg_lower:+,.0f})")
        else:
            condition = "below_median"
            threshold = median_rv
            avg_upper = upper.mean()
            avg_lower = lower.mean()
            print(f"  {cell}: {rv} <= {threshold:+.2f} 활성 "
                  f"(lower avg {avg_lower:+,.0f} vs upper avg {avg_upper:+,.0f})")
        cell_filter_rule[cell] = (rv, condition, threshold)
    print()

    # ── D) Filter 적용 백테스트 ──
    print("=" * 100)
    print("D) Filter 적용 백테스트 vs Baseline")
    print("=" * 100)

    def filter_fn(cell, regime_dict):
        rule = cell_filter_rule.get(cell)
        if not rule or rule[0] is None:
            return True
        rv = rule[0]
        rv_key = rv.replace("entry_", "")   # strip prefix
        if rv_key not in regime_dict:
            return True
        val = regime_dict[rv_key]
        if pd.isna(val):
            return True
        if rule[1] == "above_median":
            return val > rule[2]
        elif rule[1] == "below_median":
            return val <= rule[2]
        return True

    daily_f, trades_f = backtest_slippage(p, RULE_V7_CLEAN, TP_BP, SL_BP, slip=SLIP_BP, filter_fn=filter_fn)
    trades_f["year"] = pd.to_datetime(trades_f["entry_date"]).dt.year
    m_f = perf(daily_f, trades_f, "Filtered")

    print(f"\n  {'Metric':>14s}  {'Baseline':>12s}  {'Filtered':>12s}  {'Diff':>10s}")
    for k in ["Trades", "Net (만)", "Per_yr (만)", "Sharpe", "MaxDD (만)", "Hit (%)", "W/L ratio"]:
        bv = m_b[k]
        fv = m_f[k]
        if isinstance(bv, (int, float)) and isinstance(fv, (int, float)):
            diff = fv - bv
            print(f"  {k:>14s}  {bv:>12}  {fv:>12}  {diff:>+10}")
        else:
            print(f"  {k:>14s}  {bv:>12}  {fv:>12}  -")
    print()

    # 연도별 비교
    yr_b = daily_b.groupby("year")["daily_pnl_net"].sum().round(0)
    yr_f = daily_f.groupby("year")["daily_pnl_net"].sum().round(0)
    yr_df = pd.DataFrame({"Baseline": yr_b, "Filtered": yr_f, "Diff": (yr_f - yr_b).round(0)})
    print("연도별 비교:")
    print(yr_df.to_string())
    print()

    # ── Excel ──
    CHART_DIR.mkdir(exist_ok=True)
    xlsx = CHART_DIR / "V7clean_regime_filter.xlsx"
    summary_df = pd.DataFrame([m_b, m_f])
    rule_df = pd.DataFrame([
        {"cell": c,
         "best_regime_var": rule[0] if rule else None,
         "condition": rule[1] if rule else None,
         "threshold": rule[2] if rule and rule[2] is not None else None}
        for c, rule in cell_filter_rule.items()
    ])
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xl:
        summary_df.to_excel(xl, sheet_name="Baseline_vs_Filtered", index=False)
        rule_df.to_excel(xl, sheet_name="Filter_rule", index=False)
        yr_df.reset_index().to_excel(xl, sheet_name="Yearly", index=False)
        for tdf, sheet in [(trades_b, "Trades_baseline"), (trades_f, "Trades_filtered")]:
            t = tdf.copy()
            for c in t.select_dtypes(include=["object"]).columns:
                if "date" in c.lower():
                    t[c] = pd.to_datetime(t[c]).dt.strftime("%Y-%m-%d")
            for c in t.select_dtypes(include=["float64"]).columns:
                t[c] = t[c].round(2)
            t.to_excel(xl, sheet_name=sheet, index=False)
    print(f"[save] {xlsx}\n")

    # 차트
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                              gridspec_kw={"height_ratios": [2.5, 1]})
    axes[0].plot(daily_b["price_date"], daily_b["cum_pnl_net"], color="#e76f51", lw=1.6, alpha=0.7,
                  label=f"Baseline: {m_b['Net (만)']:+,.0f}만 sh{m_b['Sharpe']:+.2f} W/L{m_b['W/L ratio']}")
    axes[0].plot(daily_f["price_date"], daily_f["cum_pnl_net"], color="#264653", lw=2.2,
                  label=f"Filtered: {m_f['Net (만)']:+,.0f}만 sh{m_f['Sharpe']:+.2f} W/L{m_f['W/L ratio']}")
    axes[0].axhline(0, color="gray", lw=0.7, ls="--")
    axes[0].set_title("V7-clean TP+6/SL-6: Baseline vs Regime-filter", fontsize=13, weight="bold")
    axes[0].set_ylabel("Cum Net P&L (만)")
    axes[0].grid(alpha=0.3); axes[0].legend(loc="upper left")

    axes[1].plot(daily_b["price_date"], daily_b["drawdown"], color="#e76f51", lw=1.0, alpha=0.7,
                  label=f"Baseline MDD {m_b['MaxDD (만)']:,.0f}")
    axes[1].plot(daily_f["price_date"], daily_f["drawdown"], color="#264653", lw=1.4,
                  label=f"Filtered MDD {m_f['MaxDD (만)']:,.0f}")
    axes[1].axhline(0, color="gray", lw=0.7, ls="--")
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("DD (만)"); axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "37_regime_filter.png", bbox_inches="tight")
    plt.close(fig)
    print("[chart] OK 37_regime_filter.png")
    print("[done]")


if __name__ == "__main__":
    main()
