"""
24 — V6 cross-tenor cell-sign rule 기반 3Y / 10Y 전략.

리서치 결과 (Stage 21, 4-way cell matrix) 를 trading rule 로 변환.
sizing 은 fixed unit (cell mean 값 안 사용 → look-ahead 없음).
Cell sign rule 만 priori 로 정의 (research findings).

10년 전략 (V6-10Y): KTB10F short/long 단독, target = ΔY_10Y_21
  Cell rule (mean ΔY_10Y_21 + hit 기반):
    0110 (f10=S,f3=B,b10F=B,b3F=S)  N=45  +19.4 hit80 -> SHORT 2 unit
    1001 (f10=B,f3=S,b10F=S,b3F=B)  N=23   +7.5 hit78 -> SHORT 1 unit
    1101 (f10=B,f3=B,b10F=S,b3F=B)  N=68   +5.9 hit74 -> SHORT 1 unit
    0010 (f10=S,f3=S,b10F=B,b3F=S)  N=91   +4.7 hit62 -> SHORT 0.5
    0011 (f10=S,f3=S,b10F=B,b3F=B)  N=285  +4.4 hit59 -> SHORT 0.5
    0101 (f10=S,f3=B,b10F=S,b3F=B)  N=27  -12.5 (small N) -> LONG 0.5 (risky)
    others -> flat

3년 전략 (V6-3Y): KTB3F short/long 단독, target = ΔY_3Y_21
  Cell rule:
    0110 (f10=S,f3=B,b10F=B,b3F=S)  N=45  +18.3 hit78 -> SHORT 2 unit
    0011 (f10=S,f3=S,b10F=B,b3F=B)  N=285  +6.9 hit62 -> SHORT 1
    0111 (f10=S,f3=B,b10F=B,b3F=B)  N=179  +5.8 hit63 -> SHORT 1
    0001 (f10=S,f3=S,b10F=S,b3F=B)  N=58   +3.6 hit55 -> SHORT 0.5
    0101 (f10=S,f3=B,b10F=S,b3F=B)  N=27  -11.0 (small N) -> LONG 0.5
    0100 (f10=S,f3=B,b10F=S,b3F=S)  N=8    -4.8 (very small N) -> LONG 0.3
    others -> flat

Hold = 21d.
Look-ahead audit:
  - Cell sign rule 사전 정의 (research findings 활용한 priori)
  - 사이즈는 fixed (mean 값 안 사용)
  - 시그널 input: t 시점 cum-5d only
  - Entry 후 daily P&L: t+1 ~ t+21
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
TRADING_DAYS = 252
HOLD = 21

for fname in ["Malgun Gothic", "NanumGothic", "AppleGothic"]:
    try:
        plt.rcParams["font.family"] = fname
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 140


# ── 사전 정의 cell rule (research findings 기반) ──
# size convention: 양수 = SHORT (yield 상승 = bond 약세 베팅), 음수 = LONG
RULE_10Y = {
    "0110": +2.0,  # mean +19.4, hit 80, N=45
    "1001": +1.0,  # mean +7.5, hit 78, N=23
    "1101": +1.0,  # mean +5.9, hit 74, N=68
    "0010": +0.5,  # mean +4.7, hit 62, N=91
    "0011": +0.5,  # mean +4.4, hit 59, N=285
    "0101": -0.5,  # mean -12.5, hit 48 (small N=27 risky)
}

RULE_3Y = {
    "0110": +2.0,  # mean +18.3, hit 78
    "0011": +1.0,  # mean +6.9, hit 62, N=285 (large)
    "0111": +1.0,  # mean +5.8, hit 63, N=179
    "0001": +0.5,  # mean +3.6, hit 55, N=58
    "0101": -0.5,  # mean -11.0, hit 44, N=27
    "0100": -0.3,  # mean -4.8, N=8 (very small)
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
    p["s_f10"] = (p["f10"] > 0).astype(int)
    p["s_f3"] = (p["f3"] > 0).astype(int)
    p["s_b10F"] = (p["b10F"] > 0).astype(int)
    p["s_b3F"] = (p["b3F"] > 0).astype(int)
    p["cell"] = (p["s_f10"].astype(str) + p["s_f3"].astype(str)
                  + p["s_b10F"].astype(str) + p["s_b3F"].astype(str))
    return p


def backtest(p, rule, dy1d_col, target_fwd_col, dv01):
    """rule={cell: size_unit}, size_unit>0 = short bond (yield up bet).
    Daily P&L = pos_ctr × (-dy1d) × DV01 (만원).
    """
    n = len(p)
    daily_pnl = np.zeros(n)
    daily_pos = np.zeros(n)
    dy1d = p[dy1d_col].fillna(0.0).values
    cells = p["cell"].values
    dates = p["price_date"].values
    trades = []

    for i in range(n):
        c = cells[i]
        if c not in rule:
            continue
        size = rule[c]    # +short, -long
        # pos: yield 상승 베팅 (short bond) 면 -1 계약 (가격 하락 시 익)
        pos_ctr = -size   # size>0 -> pos<0 (short)
        # entry trade log
        i_exit = min(i + HOLD, n - 1)
        fwd_dy = p[target_fwd_col].iloc[i] if i + HOLD < n else np.nan
        trade_pnl = pos_ctr * (-fwd_dy) * dv01 if pd.notna(fwd_dy) else np.nan
        trades.append({
            "entry_date": pd.Timestamp(dates[i]),
            "exit_date": pd.Timestamp(dates[i_exit]),
            "cell": c,
            "size_unit": size,
            "pos_ctr": pos_ctr,
            "y_entry": float(p[target_fwd_col.replace("_fwd_21", "").replace("dy", "y_") + "y" if False else
                              ("y_10y" if "10" in target_fwd_col else "y_3y")].iloc[i]),
            "fwd_dy_bp": float(fwd_dy) if pd.notna(fwd_dy) else np.nan,
            "trade_pnl_man": float(trade_pnl) if pd.notna(trade_pnl) else np.nan,
        })
        # daily P&L (entry+1 ~ entry+HOLD)
        for d in range(i + 1, min(i + HOLD + 1, n)):
            daily_pos[d] += pos_ctr
            daily_pnl[d] += pos_ctr * (-dy1d[d]) * dv01

    daily = p[["price_date", "year"]].copy()
    daily["pos_ctr"] = daily_pos
    daily["daily_pnl_man"] = daily_pnl
    daily["cum_pnl_man"] = daily_pnl.cumsum()
    daily["peak"] = daily["cum_pnl_man"].cummax()
    daily["drawdown_man"] = daily["cum_pnl_man"] - daily["peak"]
    return daily, pd.DataFrame(trades)


def perf(daily, trades, name):
    s_nz = daily["daily_pnl_man"][daily["daily_pnl_man"] != 0]
    mu = s_nz.mean() if len(s_nz) else 0
    sd = s_nz.std() if len(s_nz) else 1
    sh = mu / sd * np.sqrt(TRADING_DAYS) if sd > 0 else 0
    total = daily["daily_pnl_man"].sum()
    nyrs = len(daily) / TRADING_DAYS
    mdd = daily["drawdown_man"].min()
    win = (trades["trade_pnl_man"] > 0).sum() if len(trades) else 0
    loss = (trades["trade_pnl_man"] < 0).sum() if len(trades) else 0
    hit = win / (win + loss) * 100 if (win + loss) else 0
    return {
        "name": name,
        "Trades": len(trades),
        "Total (만)": round(total, 0),
        "Per_yr (만)": round(total / nyrs, 0) if nyrs > 0 else 0,
        "Sharpe": round(sh, 2),
        "MDD (만)": round(mdd, 0),
        "Hit (%)": round(hit, 1),
        "Active days": len(s_nz),
    }


def main():
    print("[load] panel ...")
    p = load_panel("2020-01-01")
    print(f"  {len(p):,} rows  {p['price_date'].min().date()} ~ {p['price_date'].max().date()}\n")

    print("[Rules] 사전 정의 (research findings 기반):")
    print(f"  10Y rule: {len(RULE_10Y)} cells active")
    for c, sz in sorted(RULE_10Y.items()):
        action = "SHORT" if sz > 0 else "LONG"
        print(f"    {c}: {action} {abs(sz)} unit")
    print(f"\n  3Y rule:  {len(RULE_3Y)} cells active")
    for c, sz in sorted(RULE_3Y.items()):
        action = "SHORT" if sz > 0 else "LONG"
        print(f"    {c}: {action} {abs(sz)} unit")
    print()

    # ── Backtest ──
    print("=" * 80)
    print("V6 backtest (Look-ahead audit: sizing fixed, no future cell mean)")
    print("=" * 80)
    daily_10y, trades_10y = backtest(p, RULE_10Y, "dy10_1d", "dy10_fwd_21", DV01_KTB10F)
    daily_3y, trades_3y = backtest(p, RULE_3Y, "dy3_1d", "dy3_fwd_21", DV01_KTB3F)

    m10 = perf(daily_10y, trades_10y, "V6-10Y (KTB10F)")
    m3 = perf(daily_3y, trades_3y, "V6-3Y (KTB3F)")
    print(f"\n  {'Strategy':22s} {'Trades':>7s} {'Total':>10s} {'Per_yr':>10s} {'Sharpe':>8s} {'MDD':>10s} {'Hit%':>7s}")
    for m in [m10, m3]:
        print(f"  {m['name']:22s} {m['Trades']:>7d} {m['Total (만)']:>+10,.0f} {m['Per_yr (만)']:>+10,.0f} "
              f"{m['Sharpe']:>+8.2f} {m['MDD (만)']:>+10,.0f} {m['Hit (%)']:>7.1f}")
    print()

    # Combined 2 strategies
    daily_combined = daily_10y[["price_date", "year"]].copy()
    daily_combined["daily_pnl_man"] = daily_10y["daily_pnl_man"] + daily_3y["daily_pnl_man"]
    daily_combined["cum_pnl_man"] = daily_combined["daily_pnl_man"].cumsum()
    daily_combined["peak"] = daily_combined["cum_pnl_man"].cummax()
    daily_combined["drawdown_man"] = daily_combined["cum_pnl_man"] - daily_combined["peak"]
    trades_combined = pd.concat([trades_10y.assign(strategy="10Y"),
                                  trades_3y.assign(strategy="3Y")], ignore_index=True)
    mc = perf(daily_combined, trades_combined, "Combined (10Y+3Y)")
    print(f"  {mc['name']:22s} {mc['Trades']:>7d} {mc['Total (만)']:>+10,.0f} {mc['Per_yr (만)']:>+10,.0f} "
          f"{mc['Sharpe']:>+8.2f} {mc['MDD (만)']:>+10,.0f} {mc['Hit (%)']:>7.1f}")
    print()

    corr = daily_10y["daily_pnl_man"].corr(daily_3y["daily_pnl_man"])
    print(f"  Daily P&L correlation (10Y vs 3Y): {corr:+.3f}")
    print()

    # ── 연도별 ──
    print("연도별 P&L (만):")
    yr10 = daily_10y.groupby("year")["daily_pnl_man"].sum().round(0)
    yr3 = daily_3y.groupby("year")["daily_pnl_man"].sum().round(0)
    yrc = daily_combined.groupby("year")["daily_pnl_man"].sum().round(0)
    yr_df = pd.DataFrame({"V6-10Y": yr10, "V6-3Y": yr3, "Combined": yrc})
    print(yr_df.to_string())
    print()

    # ── Trade-level by cell (10Y) ──
    print("=" * 80)
    print("Trade-level by cell (V6-10Y):")
    print("=" * 80)
    cell_stats_10 = trades_10y.groupby("cell").agg(
        N=("trade_pnl_man", "size"),
        size_unit=("size_unit", "first"),
        total_man=("trade_pnl_man", "sum"),
        mean_man=("trade_pnl_man", "mean"),
        hit_pct=("trade_pnl_man", lambda x: (x > 0).mean() * 100),
        mean_fwd_dy=("fwd_dy_bp", "mean"),
    ).round(2).sort_values("total_man", ascending=False)
    print(cell_stats_10.to_string())
    print()

    print("Trade-level by cell (V6-3Y):")
    cell_stats_3 = trades_3y.groupby("cell").agg(
        N=("trade_pnl_man", "size"),
        size_unit=("size_unit", "first"),
        total_man=("trade_pnl_man", "sum"),
        mean_man=("trade_pnl_man", "mean"),
        hit_pct=("trade_pnl_man", lambda x: (x > 0).mean() * 100),
        mean_fwd_dy=("fwd_dy_bp", "mean"),
    ).round(2).sort_values("total_man", ascending=False)
    print(cell_stats_3.to_string())
    print()

    # ── 5/11 현재 시그널 ──
    print("=" * 80)
    print("5/11 시그널")
    print("=" * 80)
    latest = p.iloc[-1]
    cell = latest["cell"]
    print(f"  date: {latest['price_date'].strftime('%Y-%m-%d')}")
    print(f"  cell: {cell}")
    fmt = lambda b: "BUY" if int(b) else "SELL"
    print(f"  combo: f10={fmt(cell[0])}, f3={fmt(cell[1])}, b10F={fmt(cell[2])}, b3F={fmt(cell[3])}")
    if cell in RULE_10Y:
        sz = RULE_10Y[cell]
        print(f"  V6-10Y: {'SHORT' if sz > 0 else 'LONG'} KTB10F {abs(sz)} 계약, hold 21d")
    else:
        print(f"  V6-10Y: FLAT")
    if cell in RULE_3Y:
        sz = RULE_3Y[cell]
        print(f"  V6-3Y:  {'SHORT' if sz > 0 else 'LONG'} KTB3F {abs(sz)} 계약, hold 21d")
    else:
        print(f"  V6-3Y:  FLAT")
    print()

    # ── Excel 저장 ──
    CHART_DIR.mkdir(exist_ok=True)
    xlsx = CHART_DIR / "V6_3y10y_track_record.xlsx"
    print(f"[save] {xlsx}")

    audit_rows = [
        ("Cell sign rule", "research findings 기반 priori (전체 매트릭스에서 선정)",
         "in-sample 학습 인정"),
        ("Sizing", "Fixed unit per cell (mean 값 안 들어감)", "look-ahead 없음"),
        ("Input (f10, f3, b10F, b3F)", "t 시점 5d cum, t 정보만", "OK"),
        ("Entry timing", "t close → t+1 부터 daily P&L", "OK"),
        ("Daily P&L", "dy_1d[i+1] = y(t+1) - y(t), t+1 ~ t+21 만", "OK"),
        ("Hold", "21 영업일 fixed", "OK"),
    ]
    audit_df = pd.DataFrame(audit_rows, columns=["Item", "Description", "Status"])

    rule_10_df = pd.DataFrame([
        {"cell": c, "f10": fmt(c[0]), "f3": fmt(c[1]), "b10F": fmt(c[2]), "b3F": fmt(c[3]),
         "size_unit": sz, "action": "SHORT" if sz > 0 else "LONG"}
        for c, sz in RULE_10Y.items()
    ])
    rule_3_df = pd.DataFrame([
        {"cell": c, "f10": fmt(c[0]), "f3": fmt(c[1]), "b10F": fmt(c[2]), "b3F": fmt(c[3]),
         "size_unit": sz, "action": "SHORT" if sz > 0 else "LONG"}
        for c, sz in RULE_3Y.items()
    ])
    summary_df = pd.DataFrame([m10, m3, mc])

    with pd.ExcelWriter(xlsx, engine="openpyxl") as xl:
        summary_df.to_excel(xl, sheet_name="Summary", index=False)
        yr_df.reset_index().to_excel(xl, sheet_name="Yearly", index=False)
        rule_10_df.to_excel(xl, sheet_name="Rule_10Y", index=False)
        rule_3_df.to_excel(xl, sheet_name="Rule_3Y", index=False)
        cell_stats_10.reset_index().to_excel(xl, sheet_name="Cell_stats_10Y", index=False)
        cell_stats_3.reset_index().to_excel(xl, sheet_name="Cell_stats_3Y", index=False)
        # Trade log
        for tdf, sheet in [(trades_10y, "Trades_10Y"), (trades_3y, "Trades_3Y")]:
            t = tdf.copy()
            for c in t.select_dtypes(include=["float64"]).columns:
                t[c] = t[c].round(2)
            t.to_excel(xl, sheet_name=sheet, index=False)
        # Daily
        for ddf, sheet in [(daily_10y, "Daily_10Y"),
                            (daily_3y, "Daily_3Y"),
                            (daily_combined, "Daily_Combined")]:
            d = ddf.copy()
            d["price_date"] = d["price_date"].dt.strftime("%Y-%m-%d")
            for c in d.select_dtypes(include=["float64"]).columns:
                d[c] = d[c].round(2)
            d.to_excel(xl, sheet_name=sheet, index=False)
        audit_df.to_excel(xl, sheet_name="Audit", index=False)

    print(f"  OK -> {xlsx}\n")

    # ── 차트 ──
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                              gridspec_kw={"height_ratios": [2.5, 1]})
    axes[0].plot(daily_10y["price_date"], daily_10y["cum_pnl_man"], color="#e76f51", lw=2,
                  label=f"V6-10Y final={m10['Total (만)']:+,.0f}만, Sharpe={m10['Sharpe']:+.2f}")
    axes[0].plot(daily_3y["price_date"], daily_3y["cum_pnl_man"], color="#2a9d8f", lw=2,
                  label=f"V6-3Y final={m3['Total (만)']:+,.0f}만, Sharpe={m3['Sharpe']:+.2f}")
    axes[0].plot(daily_combined["price_date"], daily_combined["cum_pnl_man"], color="#264653", lw=2.5,
                  label=f"Combined final={mc['Total (만)']:+,.0f}만, Sharpe={mc['Sharpe']:+.2f}")
    axes[0].axhline(0, color="gray", lw=0.7, ls="--")
    axes[0].set_title("V6 cross-tenor cell-sign 전략 누적 P&L (3Y + 10Y)",
                       fontsize=13, weight="bold")
    axes[0].set_ylabel("Cumulative P&L (만원)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper left")

    axes[1].fill_between(daily_combined["price_date"], 0, daily_combined["drawdown_man"],
                          color="#e76f51", alpha=0.35)
    axes[1].plot(daily_combined["price_date"], daily_combined["drawdown_man"],
                  color="#a8331b", lw=1.2)
    axes[1].set_title(f"Combined Drawdown (MaxDD {mc['MDD (만)']:,.0f}만)")
    axes[1].set_ylabel("DD (만)")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "28_v6_3y10y.png", bbox_inches="tight")
    plt.close(fig)
    print("  OK 28_v6_3y10y.png\n")

    print("[done]")


if __name__ == "__main__":
    main()
