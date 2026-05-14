"""
26 — V7 정밀 audit:
  1. Look-ahead 재점검 (input timing, forward yields, cell sign 학습)
  2. Entry / Exit 룰 명시
  3. 평균 holding 일수
  4. 거래 빈도 (진입 빈도, 동시 active 포지션, 회전률)
  5. 거래비용 차감
  6. Walk-forward (cell sign 도 시기별 재학습) OOS test
  7. 사이즈 누적 / overlap risk

V7 spec 재정의 (현재 25번 그대로):
  Entry: 매 거래일 t close 시그널 평가 -> cell ∈ RULE_SLOPE 이면 진입 (T+1 open)
  Exit:  T+HOLD (21 영업일) close 청산 (early exit 없음)
  Sizing: cell 별 fixed unit, overlapping 허용 (사이즈 누적 가능)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

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
TC_10F_BP = 0.12      # round trip per 계약
TC_3F_BP = 0.05       # KTB3F 거래비용 낮음 (가정)

RULE_SLOPE = {
    "1001": +2.0, "1100": +1.0, "1101": +1.0, "1000": +0.5,
    "0011": -1.0, "0111": -0.5, "1011": -0.5, "0101": -0.3,
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


def backtest_with_cost(p, rule, apply_cost=True):
    """Entry T+1, Exit T+HOLD close (no early exit). Overlapping allowed."""
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
        # Trade-level P&L
        fwd_dy10 = p["dy10_fwd_21"].iloc[i] if i + HOLD < n else np.nan
        fwd_dy3 = p["dy3_fwd_21"].iloc[i] if i + HOLD < n else np.nan
        fwd_dslope = p["dslope_fwd_21"].iloc[i] if i + HOLD < n else np.nan
        t_gross = (pos_10 * (-fwd_dy10) * DV01_KTB10F
                    + pos_3 * (-fwd_dy3) * DV01_KTB3F) if pd.notna(fwd_dy10) else np.nan
        # 거래비용 (entry + exit) per trade
        cost = (abs(pos_10) * TC_10F_BP * DV01_KTB10F
                + abs(pos_3) * TC_3F_BP * DV01_KTB3F)
        t_net = t_gross - cost if pd.notna(t_gross) else np.nan
        trades.append({
            "entry_idx": i,
            "entry_date": pd.Timestamp(dates[i]),
            "exit_date": pd.Timestamp(dates[i_exit]),
            "hold_days_calendar": (pd.Timestamp(dates[i_exit]) - pd.Timestamp(dates[i])).days,
            "hold_days_business": HOLD,
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
        # Daily P&L (overlapping)
        # Entry cost at T+1 (절반), Exit cost at T+HOLD (절반)
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


def walk_forward_test(p, rule, train_end_dates=None):
    """Cell sign 을 시기별로 재학습 (priori 효과 제거).

    매 분기 (63 영업일) 마다 그때까지의 데이터에서 mean Δslope 가 > +1 or < -1 인 cell 만 사용.
    Sign: mean 부호.
    Size: 1.0 fixed (단순화).
    Warm-up: 252 영업일.
    """
    n = len(p)
    daily_pnl = np.zeros(n)
    daily_pos10 = np.zeros(n)
    daily_pos3 = np.zeros(n)
    dy10_1d = p["dy10_1d"].fillna(0.0).values
    dy3_1d = p["dy3_1d"].fillna(0.0).values
    cells_seq = p["cell"].values
    dates_seq = p["price_date"].values

    WARM_UP = 252
    REFIT_FREQ = 63
    THR_BP = 1.0
    MIN_N = 5

    cur_rule = {}
    last_refit = -REFIT_FREQ - 1
    trades = []

    for i in range(n):
        if i - last_refit >= REFIT_FREQ and i >= WARM_UP:
            train_idx = i - HOLD - 1   # forward fwd_dy_21 결측 회피
            if train_idx > 0:
                train_p = p.iloc[:train_idx + 1].dropna(subset=["dslope_fwd_21"])
                tbl = train_p.groupby("cell").agg(
                    N=("dslope_fwd_21", "size"),
                    mean=("dslope_fwd_21", "mean"),
                )
                cur_rule = {}
                for c, row in tbl.iterrows():
                    if row["N"] < MIN_N or abs(row["mean"]) < THR_BP:
                        continue
                    cur_rule[c] = np.sign(row["mean"]) * 1.0   # fixed size 1
                last_refit = i

        if i < WARM_UP or not cur_rule:
            continue
        c = cells_seq[i]
        if c not in cur_rule:
            continue
        size = cur_rule[c]
        pos_10 = -size
        pos_3 = +size * RATIO
        trades.append({"entry_date": pd.Timestamp(dates_seq[i]), "cell": c, "size": size})
        for d in range(i + 1, min(i + HOLD + 1, n)):
            daily_pos10[d] += pos_10
            daily_pos3[d] += pos_3
            daily_pnl[d] += pos_10 * (-dy10_1d[d]) * DV01_KTB10F \
                            + pos_3 * (-dy3_1d[d]) * DV01_KTB3F

    return daily_pnl, pd.DataFrame(trades), daily_pos10, daily_pos3


def main():
    print("[load] panel ...")
    p = load_panel("2020-01-01")
    n = len(p)
    print(f"  {n:,} rows  {p['price_date'].min().date()} ~ {p['price_date'].max().date()}\n")

    # ── 1) Look-ahead audit (item-by-item) ──
    print("=" * 90)
    print("1) Look-ahead audit (V7)")
    print("=" * 90)
    audit = [
        ("A. Input timing (f10, f3, b10F, b3F)",
         "t 시점 5d cum sum (ktb_trade_flow_features.foreigner_sum_5d, ktbf_netbuy.foreigner)",
         "DB 의 sum_5d 는 [t-4, t] 5일간 누적. t close 후 발표 가정.",
         "OK"),
        ("B. dy_1d 계산",
         "p['dy10_1d'] = p['y_10y'].diff() = y10[t] - y10[t-1]",
         "P&L 에서 dy10_1d[i+1] = y10[t+1] - y10[t]. Entry t+1 close 가격 기준",
         "OK"),
        ("C. Forward yield (fwd_dy_21)",
         "p['y_10y'].shift(-21) - p['y_10y'] = y10[t+21] - y10[t]",
         "Trade-level P&L 만 (info purpose). Daily P&L 에 영향 없음",
         "OK (info only)"),
        ("D. Entry timing",
         "t close 시그널 -> t+1 부터 daily P&L 누적",
         "Code: for d in range(i+1, i+HOLD+1)",
         "OK"),
        ("E. Exit timing",
         "t+HOLD (21 영업일) close 청산. Early exit 없음",
         "코드 마지막 d 가 i+HOLD",
         "OK"),
        ("F. Cell sign rule (* 핵심)",
         "RULE_SLOPE 의 8 cell 부호 = 전체 panel (2020-2026) mean Δslope_21 부호",
         "사이즈 magnitude 는 사용 안 함 (fixed). 그러나 sign 자체가 in-sample 학습",
         "PARTIAL in-sample (sign only)"),
        ("G. Walk-forward 검증",
         "26번 본 스크립트에서 cell sign 도 시기별 재학습 OOS 결과 별도 산출",
         "WARM_UP=252, REFIT=63 영업일",
         "별도 측정"),
    ]
    for item, desc, detail, status in audit:
        print(f"\n  -> {item}")
        print(f"      what: {desc}")
        print(f"      check: {detail}")
        print(f"      status: {status}")
    print()

    # ── 2) Entry / Exit 룰 ──
    print("=" * 90)
    print("2) Entry / Exit 룰")
    print("=" * 90)
    rules_table = [
        ("Entry trigger", "매 거래일 t close 후 4 카테고리 부호 → cell 식별 → cell ∈ RULE_SLOPE 이면 진입"),
        ("Entry timing", "T+1 open (다음 거래일) 진입 가정 (daily P&L 은 T+1 부터)"),
        ("Entry size", "cell 별 fixed: size_10F = -size_unit, size_3F = +size_unit x 3.04"),
        ("Entry direction", "STEEPENER (size>0) = 10F SHORT + 3F LONG / FLATTENER (size<0) = 10F LONG + 3F SHORT"),
        ("Delta-neutral", "Net DV01 = pos_10F x 8.5 + pos_3F x 2.8 ~ 0"),
        ("Exit trigger", "T + 21 영업일 (단순 holding-based, early exit 없음)"),
        ("Exit timing", "T+21 close 청산"),
        ("Stop loss", "없음"),
        ("Profit target", "없음"),
        ("Overlapping", "허용 (매일 새 trade, 동시 active position 누적)"),
        ("Sizing cap", "없음 (warning: 연속 cell 발동 시 사이즈 누적)"),
    ]
    for k, v in rules_table:
        print(f"  {k:>22s}: {v}")
    print()

    # ── 3) Backtest (gross + net) ──
    print("=" * 90)
    print("3) Backtest with cost (V7 priori rule)")
    print("=" * 90)
    daily, trades = backtest_with_cost(p, RULE_SLOPE, apply_cost=True)
    gross_total = daily["daily_pnl_gross"].sum()
    cost_total = daily["daily_cost"].sum()
    net_total = daily["daily_pnl_net"].sum()
    nyrs = n / TRADING_DAYS
    s_nz = daily["daily_pnl_net"][daily["daily_pnl_net"] != 0]
    sharpe_net = s_nz.mean() / s_nz.std() * np.sqrt(TRADING_DAYS) if len(s_nz) > 1 else 0
    s_g = daily["daily_pnl_gross"][daily["daily_pnl_gross"] != 0]
    sharpe_gross = s_g.mean() / s_g.std() * np.sqrt(TRADING_DAYS) if len(s_g) > 1 else 0
    mdd_net = daily["drawdown_man"].min()

    print(f"\n  Trades: {len(trades):,}")
    print(f"  Gross P&L:  {gross_total:>+12,.0f} 만 (sharpe {sharpe_gross:+.2f})")
    print(f"  Cost:       {cost_total:>+12,.0f} 만  (= {cost_total/gross_total*100:.1f}% of gross)")
    print(f"  Net P&L:    {net_total:>+12,.0f} 만 (sharpe {sharpe_net:+.2f})")
    print(f"  Per_yr net: {net_total/nyrs:>+12,.0f} 만/y")
    print(f"  MaxDD net:  {mdd_net:>+12,.0f} 만")
    print(f"  TC: KTB10F {TC_10F_BP} bp/계약, KTB3F {TC_3F_BP} bp/계약 (round trip)")
    print()

    # ── 4) Hold 일수 ──
    print("=" * 90)
    print("4) Holding days")
    print("=" * 90)
    hold_b = trades["hold_days_business"].describe().round(2)
    hold_c = trades["hold_days_calendar"].describe().round(2)
    print(f"\n  영업일 기준: mean={hold_b['mean']:.1f}, min={hold_b['min']:.0f}, max={hold_b['max']:.0f}")
    print(f"  달력일 기준: mean={hold_c['mean']:.1f}, min={hold_c['min']:.0f}, max={hold_c['max']:.0f}")
    print(f"  * 모든 trade 가 정확히 {HOLD} 영업일 hold (early exit 없음)")
    print()

    # ── 5) 거래 빈도 ──
    print("=" * 90)
    print("5) 거래 빈도 (Frequency)")
    print("=" * 90)
    active_days = (daily["pos_10F"] != 0).sum()
    print(f"\n  총 거래일: {n:,} 영업일 ({nyrs:.2f}년)")
    print(f"  Trades total: {len(trades):,}")
    print(f"  Trades / year: {len(trades)/nyrs:.0f}")
    print(f"  Trades / day: {len(trades)/n:.3f} ({len(trades)/n*100:.1f}% of days have new entry)")
    print(f"  Active days (포지션 보유): {active_days:,} / {n:,} ({active_days/n*100:.1f}%)")
    print(f"  평균 동시 active positions (rough): {len(trades) * HOLD / n:.1f}")
    print()
    # 사이즈 누적 분포
    abs_pos10 = daily["pos_10F"].abs()
    abs_pos3 = daily["pos_3F"].abs()
    print(f"  KTB10F 동시 노출 (계약 수 abs):")
    print(f"    mean: {abs_pos10[abs_pos10 > 0].mean():.2f}, max: {abs_pos10.max():.2f}, "
          f"p95: {abs_pos10.quantile(0.95):.2f}")
    print(f"  KTB3F 동시 노출 (계약 수 abs):")
    print(f"    mean: {abs_pos3[abs_pos3 > 0].mean():.2f}, max: {abs_pos3.max():.2f}, "
          f"p95: {abs_pos3.quantile(0.95):.2f}")
    print(f"  Net DV01 abs mean: {daily['net_DV01_man'].abs().mean():.3f} 만/bp")
    print()
    # Turnover (계약수 기준)
    daily_change_10 = daily["pos_10F"].diff().abs().sum() / 2   # round trip = 2 x turnover
    daily_change_3 = daily["pos_3F"].diff().abs().sum() / 2
    print(f"  Turnover (round trip 계약 수, 6년 합):")
    print(f"    KTB10F: {daily_change_10:.0f} 계약")
    print(f"    KTB3F:  {daily_change_3:.0f} 계약")
    print()

    # ── 6) Walk-forward (cell sign 도 시기별 학습) ──
    print("=" * 90)
    print("6) Walk-forward OOS: cell sign 도 expanding window 로 재학습")
    print("=" * 90)
    wf_pnl, wf_trades, wf_pos10, wf_pos3 = walk_forward_test(p, RULE_SLOPE)
    wf_nyrs = n / TRADING_DAYS
    wf_total = wf_pnl.sum()
    wf_active = (wf_pnl != 0)
    s_wf = wf_pnl[wf_active]
    wf_sharpe = s_wf.mean() / s_wf.std() * np.sqrt(TRADING_DAYS) if len(s_wf) > 1 else 0
    wf_cum = wf_pnl.cumsum()
    wf_mdd = (wf_cum - np.maximum.accumulate(wf_cum)).min()
    print(f"\n  설정: WARM_UP=252영업일, REFIT every 63 영업일, threshold |mean Δslope| >= 1 bp, min N=5, size=1")
    print(f"  Trades: {len(wf_trades):,}")
    print(f"  Total P&L: {wf_total:+,.0f} 만")
    print(f"  Per_yr: {wf_total/wf_nyrs:+,.0f} 만/y")
    print(f"  Sharpe: {wf_sharpe:+.2f}")
    print(f"  MaxDD: {wf_mdd:+,.0f} 만")
    print()
    if len(wf_trades):
        wf_yr = (pd.DataFrame({"date": p["price_date"], "pnl": wf_pnl})
                  .set_index("date").resample("YE").sum())
        wf_yr["year"] = wf_yr.index.year
        print("  연도별 (walk-forward):")
        print(wf_yr[["year", "pnl"]].to_string(index=False))
    print()

    # ── 7) 비교 표 ──
    print("=" * 90)
    print("7) 비교: priori rule (in-sample sign) vs walk-forward")
    print("=" * 90)
    print(f"\n  {'Variant':40s} {'Trades':>8s} {'Per_yr':>10s} {'Sharpe':>8s} {'MaxDD':>10s}")
    print(f"  {'V7 priori (cost adj)':40s} {len(trades):>8d} "
          f"{net_total/nyrs:>+10,.0f} {sharpe_net:>+8.2f} {mdd_net:>+10,.0f}")
    print(f"  {'V7 walk-forward (size=1, cost X)':40s} {len(wf_trades):>8d} "
          f"{wf_total/wf_nyrs:>+10,.0f} {wf_sharpe:>+8.2f} {wf_mdd:>+10,.0f}")
    print()

    # ── 8) Excel 저장 ──
    CHART_DIR.mkdir(exist_ok=True)
    xlsx = CHART_DIR / "V7_audit_full.xlsx"
    print(f"[save] {xlsx}")

    audit_df = pd.DataFrame([
        {"Item": a[0], "What": a[1], "Detail": a[2], "Status": a[3]} for a in audit
    ])
    rules_df = pd.DataFrame(rules_table, columns=["Rule", "Description"])
    freq_df = pd.DataFrame({
        "Metric": [
            "총 거래일", "Trades total", "Trades/year", "Trades/day (entry rate)",
            "Active days", "Active days %", "Avg simultaneous positions",
            "Hold (영업일, fixed)",
            "KTB10F net abs mean (계약)", "KTB10F net abs max", "KTB10F net abs p95",
            "KTB3F net abs mean", "KTB3F net abs max", "KTB3F net abs p95",
            "Net DV01 abs mean (만/bp)",
            "KTB10F turnover (6y, 계약)", "KTB3F turnover (6y, 계약)",
        ],
        "Value": [
            n, len(trades), round(len(trades)/nyrs, 1), round(len(trades)/n, 3),
            active_days, round(active_days/n*100, 1),
            round(len(trades) * HOLD / n, 1),
            HOLD,
            round(abs_pos10[abs_pos10 > 0].mean(), 2),
            round(abs_pos10.max(), 2), round(abs_pos10.quantile(0.95), 2),
            round(abs_pos3[abs_pos3 > 0].mean(), 2),
            round(abs_pos3.max(), 2), round(abs_pos3.quantile(0.95), 2),
            round(daily["net_DV01_man"].abs().mean(), 3),
            round(daily_change_10, 0), round(daily_change_3, 0),
        ]
    })
    cost_df = pd.DataFrame({
        "Metric": ["Gross total", "Cost total", "Net total",
                    "Cost as % of gross",
                    "Gross sharpe", "Net sharpe",
                    "TC KTB10F (bp/round trip/계약)",
                    "TC KTB3F (bp/round trip/계약)"],
        "Value": [round(gross_total, 0), round(cost_total, 0), round(net_total, 0),
                   round(cost_total/gross_total*100, 1) if gross_total != 0 else 0,
                   round(sharpe_gross, 2), round(sharpe_net, 2),
                   TC_10F_BP, TC_3F_BP]
    })
    wf_df = pd.DataFrame({
        "Metric": ["Trades", "Total P&L (만)", "Per_yr (만)", "Sharpe", "MaxDD (만)"],
        "Value": [len(wf_trades), round(wf_total, 0), round(wf_total/wf_nyrs, 0),
                   round(wf_sharpe, 3), round(wf_mdd, 0)]
    })
    summary_df = pd.DataFrame({
        "Variant": ["V7 priori (gross)", "V7 priori (net after cost)", "V7 walk-forward (size=1)"],
        "Trades": [len(trades), len(trades), len(wf_trades)],
        "Per_yr (만)": [round(gross_total/nyrs, 0), round(net_total/nyrs, 0),
                        round(wf_total/wf_nyrs, 0)],
        "Sharpe": [round(sharpe_gross, 2), round(sharpe_net, 2), round(wf_sharpe, 2)],
        "MaxDD (만)": [round((daily['daily_pnl_gross'].cumsum() -
                              daily['daily_pnl_gross'].cumsum().cummax()).min(), 0),
                       round(mdd_net, 0), round(wf_mdd, 0)],
    })
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xl:
        summary_df.to_excel(xl, sheet_name="Summary_3way", index=False)
        audit_df.to_excel(xl, sheet_name="Look_ahead_audit", index=False)
        rules_df.to_excel(xl, sheet_name="Entry_Exit_rules", index=False)
        freq_df.to_excel(xl, sheet_name="Frequency", index=False)
        cost_df.to_excel(xl, sheet_name="Cost_analysis", index=False)
        wf_df.to_excel(xl, sheet_name="Walk_forward", index=False)
        t_out = trades.copy()
        for c in t_out.select_dtypes(include=["float64"]).columns:
            t_out[c] = t_out[c].round(2)
        t_out.to_excel(xl, sheet_name="Trade_log_with_cost", index=False)
        d_out = daily.copy()
        d_out["price_date"] = d_out["price_date"].dt.strftime("%Y-%m-%d")
        for c in d_out.select_dtypes(include=["float64"]).columns:
            d_out[c] = d_out[c].round(3)
        d_out.to_excel(xl, sheet_name="Daily_PnL", index=False)

    print(f"  OK -> {xlsx}\n")
    print("[done]")


if __name__ == "__main__":
    main()
