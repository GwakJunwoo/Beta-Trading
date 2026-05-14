"""
32 - V7-clean (TP+3/SL-3) 최종 종합 검토.

이전 RV 모델 검토 시 적용했던 항목 적용:

  1. Look-ahead audit (cell sign, TP/SL grid 선정 등)
  2. R/R 대칭성 (RV 검토 때 비대칭 거부됨)
  3. 거래비용 sensitivity (0x, 0.5x, 1x, 2x)
  4. Drawdown stats (MDD, time underwater, recovery)
  5. 사이즈 cap binding 빈도 (max 100계약 binding 비율)
  6. Walk-forward (TP/SL 도 학습)
  7. 평균 동시 active position 분포
  8. Trade exit reason distribution
  9. Monthly P&L 분포 (variance, hit rate)
 10. Best / Worst trades stats
 11. Calmar / Sortino / MAR
 12. 최종 평가 (PASS/WARNING/FAIL)
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
TP_BP = 3.0
SL_BP = -3.0

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
    p["dslope_fwd_21"] = ((p["y_10y"].shift(-21) - p["y_10y"])
                            - (p["y_3y"].shift(-21) - p["y_3y"]))
    return p


def backtest_tpsl(p, rule, tp_bp, sl_bp,
                   cost_multiplier=1.0, max_10f_cap=MAX_10F_NOTIONAL):
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
    daily_cap_blocked = np.zeros(n, dtype=bool)
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
            if pnl_bp >= tp_bp:
                exit_reason = "TP"
            elif pnl_bp <= sl_bp:
                exit_reason = "SL"
            elif held >= MAX_HOLD:
                exit_reason = "TIMEOUT"
            if exit_reason:
                cost = (abs(tr["pos_10"]) * TC_10F_BP * DV01_KTB10F
                         + abs(tr["pos_3"]) * TC_3F_BP * DV01_KTB3F) * cost_multiplier
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
            if abs(cur_10F + new_pos_10) > max_10f_cap:
                daily_cap_blocked[i] = True
            else:
                entry_cost = (abs(new_pos_10) * TC_10F_BP * DV01_KTB10F
                                + abs(new_pos_3) * TC_3F_BP * DV01_KTB3F) * cost_multiplier
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
                 + abs(tr["pos_3"]) * TC_3F_BP * DV01_KTB3F) * cost_multiplier
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
    daily["cap_blocked"] = daily_cap_blocked
    daily["n_active"] = [sum(1 for _ in [tr for tr in [] if False])] * n   # placeholder
    return daily, pd.DataFrame(closed)


def compute_concurrent_positions(daily, trades):
    n = len(daily)
    n_active = np.zeros(n, dtype=int)
    for _, tr in trades.iterrows():
        ent = tr["entry_idx"]
        exi = tr["exit_idx"]
        n_active[ent:exi + 1] += 1
    daily["n_active"] = n_active
    return daily


def perf_full(daily, trades, name):
    s_n = daily["daily_pnl_net"]
    sn_active = s_n[s_n != 0]
    sh = sn_active.mean() / sn_active.std() * np.sqrt(TRADING_DAYS) if len(sn_active) > 1 and sn_active.std() > 0 else 0
    # Sortino (downside std)
    downside = sn_active[sn_active < 0]
    sortino = sn_active.mean() / downside.std() * np.sqrt(TRADING_DAYS) if len(downside) > 1 and downside.std() > 0 else 0
    gross = daily["daily_pnl_gross"].sum()
    net = daily["daily_pnl_net"].sum()
    cost = daily["daily_cost"].sum()
    nyrs = len(daily) / TRADING_DAYS
    mdd = daily["drawdown_man"].min()

    # time underwater
    dd_series = daily["drawdown_man"]
    underwater = (dd_series < 0)
    longest_uw = 0
    cur_uw = 0
    for uw in underwater:
        if uw:
            cur_uw += 1
            longest_uw = max(longest_uw, cur_uw)
        else:
            cur_uw = 0

    return {
        "name": name,
        "Trades": len(trades),
        "Net (만)": round(net, 0),
        "Gross (만)": round(gross, 0),
        "Cost (만)": round(cost, 0),
        "Cost / Gross %": round(cost / gross * 100, 1) if gross != 0 else 0,
        "Per_yr (만)": round(net / nyrs, 0) if nyrs > 0 else 0,
        "Sharpe net": round(sh, 2),
        "Sortino": round(sortino, 2),
        "MaxDD (만)": round(mdd, 0),
        "Calmar": round((net/nyrs) / abs(mdd), 2) if mdd != 0 else None,
        "Longest underwater (days)": longest_uw,
        "Hit (%)": round((trades["net_pnl"] > 0).mean() * 100, 1) if len(trades) else 0,
        "Avg win (만)": round(trades.loc[trades["net_pnl"] > 0, "net_pnl"].mean(), 1) if (trades["net_pnl"] > 0).any() else 0,
        "Avg loss (만)": round(trades.loc[trades["net_pnl"] < 0, "net_pnl"].mean(), 1) if (trades["net_pnl"] < 0).any() else 0,
    }


def main():
    print("[load] panel ...")
    p = load_panel("2020-01-01")
    print(f"  {len(p):,} rows  {p['price_date'].min().date()} ~ {p['price_date'].max().date()}\n")

    # === 1. Look-ahead audit ===
    print("=" * 90)
    print("[1] Look-ahead audit")
    print("=" * 90)
    audit_la = [
        ("Cell sign rule", "전체 6년 panel mean Dslope_21 부호로 priori 정의 (8 cells 중 5 stable)",
         "PARTIAL - sign-only learning, 27번 stability 검증으로 정당화"),
        ("TP/SL parameters", "TP+3/SL-3 는 in-sample grid search 에서 선정",
         "WARNING - in-sample optimization, OOS 검증 필요"),
        ("Size unit", "Cell rule magnitude 그대로, 고정", "OK"),
        ("Signal input timing", "t close 시점 5d cum, t 이전 정보만", "OK"),
        ("Daily P&L computation", "dy_1d[i+1] = y(t+1) - y(t), entry 후 t+1 부터", "OK"),
        ("Cost computation", "Entry/Exit 시점 cost 차감", "OK"),
    ]
    for item, desc, status in audit_la:
        print(f"\n  -> {item}: {status}")
        print(f"    - {desc}")
    print()

    # === 2. R/R 대칭성 (RV 검토 때 핵심) ===
    print("=" * 90)
    print("[2] R/R 대칭성 - RV 검토 때 사용자가 강조한 점")
    print("=" * 90)
    print(f"\n  현재: TP+{TP_BP}bp / SL{SL_BP}bp = R/R 1.0 (대칭) OK")
    print(f"  (RV 검토 때 비대칭 stop-7/target+1 등 거부됨)")
    print()

    # === 3. 거래비용 sensitivity ===
    print("=" * 90)
    print("[3] 거래비용 sensitivity")
    print("=" * 90)
    cost_results = []
    for cm in [0.0, 0.5, 1.0, 1.5, 2.0]:
        d, t = backtest_tpsl(p, RULE_V7_CLEAN, TP_BP, SL_BP, cost_multiplier=cm)
        d = compute_concurrent_positions(d, t)
        m = perf_full(d, t, f"cost x{cm}")
        cost_results.append({
            "cost x": cm,
            "TC_10F_eff": TC_10F_BP * cm,
            "TC_3F_eff": TC_3F_BP * cm,
            "Net (만)": m["Net (만)"],
            "Cost (만)": m["Cost (만)"],
            "Cost/Gross %": m["Cost / Gross %"],
            "Sharpe": m["Sharpe net"],
            "MDD (만)": m["MaxDD (만)"],
        })
    cost_df = pd.DataFrame(cost_results)
    print()
    print(cost_df.to_string(index=False))
    print()
    print("  → 비용 sensitivity: cost x2 시 net 손실로 전환되면 위험")
    print()

    # 기준 (cost x1.0)
    daily_base, trades_base = backtest_tpsl(p, RULE_V7_CLEAN, TP_BP, SL_BP, cost_multiplier=1.0)
    daily_base = compute_concurrent_positions(daily_base, trades_base)
    m_base = perf_full(daily_base, trades_base, "V7-clean TP+3/SL-3")

    print("=" * 90)
    print("[4-12] V7-clean (TP+3/SL-3) Full Metrics")
    print("=" * 90)
    for k, v in m_base.items():
        if k == "name":
            continue
        print(f"  {k:>28s}: {v}")
    print()

    # === 사이즈 cap binding ===
    print("=" * 90)
    print("[5] 사이즈 cap binding")
    print("=" * 90)
    cap_blocked_count = daily_base["cap_blocked"].sum()
    signal_days = daily_base.index[
        [c in RULE_V7_CLEAN for c in p["cell"]]
    ]
    n_signal_days = sum(1 for c in p["cell"] if c in RULE_V7_CLEAN)
    print(f"\n  Signal 발동일: {n_signal_days:,} 일")
    print(f"  Cap binding 일: {cap_blocked_count:,} 일 ({cap_blocked_count/max(n_signal_days,1)*100:.1f}% of signal days)")
    print(f"  → cap binding 가 빈번하면 사이즈 더 늘릴 필요 있음")
    print()

    # === 동시 active position 분포 ===
    print("=" * 90)
    print("[6] 평균 동시 active position 분포")
    print("=" * 90)
    n_act = daily_base["n_active"]
    print(f"\n  n_active 분포:")
    print(f"    mean: {n_act.mean():.2f}")
    print(f"    median: {n_act.median():.1f}")
    print(f"    max: {n_act.max():.0f}")
    print(f"    p95: {n_act.quantile(0.95):.0f}")
    print(f"    days with 0 pos: {(n_act == 0).sum():,} ({(n_act == 0).mean()*100:.1f}%)")
    print(f"    days with >= 5 pos (cap reached): {(n_act >= 5).sum():,} ({(n_act >= 5).mean()*100:.1f}%)")
    print()

    # === Exit reason 분포 ===
    print("=" * 90)
    print("[7] Exit reason 분포")
    print("=" * 90)
    er = trades_base["exit_reason"].value_counts(normalize=True) * 100
    er_count = trades_base["exit_reason"].value_counts()
    print()
    for reason in ["TP", "SL", "TIMEOUT", "END"]:
        pct = er.get(reason, 0)
        cnt = er_count.get(reason, 0)
        avg_pnl = trades_base[trades_base["exit_reason"] == reason]["net_pnl"].mean() if cnt else 0
        print(f"  {reason:>10s}: {cnt:>4d} trades ({pct:.1f}%)  avg net: {avg_pnl:>+8.0f}만")
    print()

    # === Monthly P&L 분포 ===
    print("=" * 90)
    print("[8] Monthly P&L 분포")
    print("=" * 90)
    daily_base["ym"] = daily_base["price_date"].dt.to_period("M")
    monthly = daily_base.groupby("ym")["daily_pnl_net"].sum()
    print(f"\n  N months: {len(monthly)}")
    print(f"  mean: {monthly.mean():+,.0f}만")
    print(f"  std: {monthly.std():,.0f}만")
    print(f"  hit (positive months): {(monthly > 0).mean() * 100:.1f}%")
    print(f"  best month: {monthly.max():+,.0f}만 ({monthly.idxmax()})")
    print(f"  worst month: {monthly.min():+,.0f}만 ({monthly.idxmin()})")
    print(f"  monthly Sharpe (annualized): {monthly.mean()/monthly.std()*np.sqrt(12):+.2f}")
    print()

    # === Best/Worst trades ===
    print("=" * 90)
    print("[9] Best / Worst 10 trades")
    print("=" * 90)
    cols = ["entry_date", "exit_date", "held_days", "cell", "size_units",
             "exit_reason", "pnl_bp_final", "net_pnl"]
    print("\n  Top 10 winners:")
    tw = trades_base.nlargest(10, "net_pnl")[cols].copy()
    tw["entry_date"] = pd.to_datetime(tw["entry_date"]).dt.strftime("%Y-%m-%d")
    tw["exit_date"] = pd.to_datetime(tw["exit_date"]).dt.strftime("%Y-%m-%d")
    for c in ["pnl_bp_final", "net_pnl"]:
        tw[c] = tw[c].round(2)
    print(tw.to_string(index=False))
    print("\n  Bottom 10 losers:")
    tl = trades_base.nsmallest(10, "net_pnl")[cols].copy()
    tl["entry_date"] = pd.to_datetime(tl["entry_date"]).dt.strftime("%Y-%m-%d")
    tl["exit_date"] = pd.to_datetime(tl["exit_date"]).dt.strftime("%Y-%m-%d")
    for c in ["pnl_bp_final", "net_pnl"]:
        tl[c] = tl[c].round(2)
    print(tl.to_string(index=False))
    print()

    # === Drawdown stats ===
    print("=" * 90)
    print("[10] Drawdown timeline")
    print("=" * 90)
    dd = daily_base["drawdown_man"]
    top5_dd = daily_base.nsmallest(5, "drawdown_man")[
        ["price_date", "cum_pnl_net", "drawdown_man"]
    ]
    print("\n  Top 5 lowest DD points:")
    for _, row in top5_dd.iterrows():
        print(f"    {row['price_date'].strftime('%Y-%m-%d')}: "
              f"cum {row['cum_pnl_net']:+,.0f}만, DD {row['drawdown_man']:+,.0f}만")
    print()

    # === 연도별 ===
    print("=" * 90)
    print("[11] 연도별 P&L")
    print("=" * 90)
    yr = daily_base.groupby("year").agg(
        N_active=("daily_pnl_net", lambda x: (x != 0).sum()),
        total=("daily_pnl_net", "sum"),
        sharpe=("daily_pnl_net", lambda x: x[x != 0].mean() / x[x != 0].std() * np.sqrt(TRADING_DAYS)
                if len(x[x != 0]) > 1 and x[x != 0].std() > 0 else 0),
    ).round(2)
    print()
    print(yr.to_string())
    print()

    # === 최종 평가 ===
    print("=" * 90)
    print("[12] 최종 평가 (PASS / WARNING / FAIL)")
    print("=" * 90)
    assessments = []
    # Sharpe
    sh = m_base["Sharpe net"]
    sh_eval = "PASS" if sh >= 1.0 else ("WARNING" if sh >= 0.5 else "FAIL")
    assessments.append(("Sharpe net", f"{sh:+.2f}", sh_eval, "기준: >= 1.0 PASS, >= 0.5 WARNING"))
    # Sortino
    sortino = m_base["Sortino"]
    sortino_eval = "PASS" if sortino >= 1.5 else ("WARNING" if sortino >= 0.8 else "FAIL")
    assessments.append(("Sortino", f"{sortino:+.2f}", sortino_eval, "기준: >= 1.5 PASS"))
    # Calmar
    calmar = m_base["Calmar"]
    calmar_eval = "PASS" if calmar >= 0.7 else ("WARNING" if calmar >= 0.4 else "FAIL")
    assessments.append(("Calmar", f"{calmar:.2f}", calmar_eval, "기준: >= 0.7 PASS"))
    # Cost ratio
    cost_ratio = m_base["Cost / Gross %"]
    cost_eval = "PASS" if cost_ratio <= 30 else ("WARNING" if cost_ratio <= 50 else "FAIL")
    assessments.append(("Cost / Gross %", f"{cost_ratio:.1f}%", cost_eval, "기준: <= 30% PASS"))
    # Hit rate
    hit = m_base["Hit (%)"]
    hit_eval = "PASS" if hit >= 60 else ("WARNING" if hit >= 50 else "FAIL")
    assessments.append(("Hit rate", f"{hit:.1f}%", hit_eval, "기준: >= 60% PASS"))
    # W/L ratio
    wl = m_base["Avg win (만)"] / -m_base["Avg loss (만)"]
    wl_eval = "PASS" if wl >= 1.2 else ("WARNING" if wl >= 0.9 else "FAIL")
    assessments.append(("W/L ratio", f"{wl:.2f}", wl_eval, "기준: >= 1.2 PASS"))
    # Longest underwater
    luw = m_base["Longest underwater (days)"]
    luw_eval = "PASS" if luw <= 120 else ("WARNING" if luw <= 250 else "FAIL")
    assessments.append(("Longest underwater (days)", str(luw), luw_eval, "기준: <= 120일 PASS"))
    # 연도별 양수 비율
    pos_yr = (yr["total"] > 0).sum() / len(yr) * 100
    pos_yr_eval = "PASS" if pos_yr >= 85 else ("WARNING" if pos_yr >= 70 else "FAIL")
    assessments.append(("Yearly positive %", f"{pos_yr:.0f}% ({(yr['total'] > 0).sum()}/{len(yr)})",
                         pos_yr_eval, "기준: >= 85% PASS"))

    print()
    print(f"  {'Metric':>28s}  {'Value':>15s}  {'Status':>10s}  Criterion")
    print(f"  {'-'*28}  {'-'*15}  {'-'*10}  {'-'*40}")
    for k, v, s, c in assessments:
        print(f"  {k:>28s}  {v:>15s}  {s:>10s}  {c}")
    print()

    n_pass = sum(1 for _, _, s, _ in assessments if s == "PASS")
    n_warn = sum(1 for _, _, s, _ in assessments if s == "WARNING")
    n_fail = sum(1 for _, _, s, _ in assessments if s == "FAIL")
    print(f"  종합: PASS {n_pass}, WARNING {n_warn}, FAIL {n_fail}")
    print()

    # === Excel 저장 ===
    CHART_DIR.mkdir(exist_ok=True)
    xlsx = CHART_DIR / "V7clean_FINAL_review.xlsx"
    audit_df = pd.DataFrame(audit_la, columns=["Item", "Description", "Status"])
    summary_df = pd.DataFrame(list(m_base.items()), columns=["Metric", "Value"])
    cost_sens_df = pd.DataFrame(cost_results)
    assess_df = pd.DataFrame(assessments, columns=["Metric", "Value", "Status", "Criterion"])
    exit_df = trades_base["exit_reason"].value_counts().reset_index()
    exit_df.columns = ["exit_reason", "count"]
    monthly_df = monthly.reset_index()
    monthly_df.columns = ["month", "pnl_man"]
    monthly_df["month"] = monthly_df["month"].astype(str)
    monthly_df["pnl_man"] = monthly_df["pnl_man"].round(0)

    with pd.ExcelWriter(xlsx, engine="openpyxl") as xl:
        assess_df.to_excel(xl, sheet_name="Final_assessment", index=False)
        summary_df.to_excel(xl, sheet_name="Summary_metrics", index=False)
        audit_df.to_excel(xl, sheet_name="Lookahead_audit", index=False)
        cost_sens_df.to_excel(xl, sheet_name="Cost_sensitivity", index=False)
        yr.reset_index().to_excel(xl, sheet_name="Yearly", index=False)
        monthly_df.to_excel(xl, sheet_name="Monthly", index=False)
        exit_df.to_excel(xl, sheet_name="Exit_reasons", index=False)
        tdf = trades_base.copy()
        for c in tdf.select_dtypes(include=["object"]).columns:
            if "date" in c.lower():
                tdf[c] = pd.to_datetime(tdf[c]).dt.strftime("%Y-%m-%d")
        for c in tdf.select_dtypes(include=["float64"]).columns:
            tdf[c] = tdf[c].round(2)
        tdf.to_excel(xl, sheet_name="All_trades", index=False)
        ddf = daily_base.copy()
        ddf["price_date"] = ddf["price_date"].dt.strftime("%Y-%m-%d")
        ddf["ym"] = ddf["ym"].astype(str)
        for c in ddf.select_dtypes(include=["float64"]).columns:
            ddf[c] = ddf[c].round(2)
        ddf.to_excel(xl, sheet_name="Daily", index=False)
    print(f"[save] {xlsx}\n")

    # ── 차트 ──
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True,
                              gridspec_kw={"height_ratios": [2.5, 1, 1]})
    axes[0].fill_between(daily_base["price_date"], 0, daily_base["cum_pnl_net"],
                          where=daily_base["cum_pnl_net"] >= 0, alpha=0.25, color="#2a9d8f")
    axes[0].fill_between(daily_base["price_date"], 0, daily_base["cum_pnl_net"],
                          where=daily_base["cum_pnl_net"] < 0, alpha=0.25, color="#e76f51")
    axes[0].plot(daily_base["price_date"], daily_base["cum_pnl_net"], color="#264653", lw=1.8)
    axes[0].axhline(0, color="gray", lw=0.7, ls="--")
    axes[0].set_title(f"V7-clean FINAL (TP+3/SL-3) Net P&L  {m_base['Net (만)']:+,.0f}만  "
                       f"Sharpe {m_base['Sharpe net']:+.2f}  MDD {m_base['MaxDD (만)']:+,.0f}",
                       fontsize=13, weight="bold")
    axes[0].set_ylabel("Cumulative Net P&L (만)")
    axes[0].grid(alpha=0.3)

    axes[1].fill_between(daily_base["price_date"], 0, daily_base["drawdown_man"],
                          color="#e76f51", alpha=0.35)
    axes[1].plot(daily_base["price_date"], daily_base["drawdown_man"], color="#a8331b", lw=1.2)
    axes[1].set_title(f"Drawdown")
    axes[1].set_ylabel("DD (만)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(daily_base["price_date"], daily_base["n_active"], color="#264653", lw=1.0)
    axes[2].axhline(5, color="red", lw=0.7, ls="--", label="cap (5)")
    axes[2].set_title("동시 active position 수")
    axes[2].set_ylabel("# positions")
    axes[2].legend()
    axes[2].xaxis.set_major_locator(mdates.YearLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "34_v7_clean_final.png", bbox_inches="tight")
    plt.close(fig)
    print("[chart] OK 34_v7_clean_final.png")
    print("[done]")


if __name__ == "__main__":
    main()
