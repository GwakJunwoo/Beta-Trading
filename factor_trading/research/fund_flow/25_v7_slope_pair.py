"""
25 — V7 Delta-neutral curve pair (3Y + 10Y 항상 페어).

Strategy:
  매 시그널 발동 시 KTB10F + KTB3F DV01 균형 페어 (듀레이션 0).
  - STEEPENER (mean Δslope > 0):  KTB10F SHORT + KTB3F LONG (DV01 매칭)
  - FLATTENER (mean Δslope < 0):  KTB10F LONG  + KTB3F SHORT (DV01 매칭)
  사이즈: cell 별 priori fixed unit (research findings 기반).

Cell rule (Stage 21 cross-tenor matrix 의 mean Δslope_21 기반):
  STEEPENERS (양수 cell):
    1001 (f10=B, f3=S, b10F=S, b3F=B)  +7.71, hit 91, N=23  -> 2.0
    1100 (f10=B, f3=B, b10F=S, b3F=S)  +3.45, hit 70, N=23  -> 1.0
    1101 (f10=B, f3=B, b10F=S, b3F=B)  +2.90, hit 77, N=68  -> 1.0
    1000 (f10=B, f3=S, b10F=S, b3F=S)  +2.36, hit 75, N=8   -> 0.5

  FLATTENERS (음수 cell):
    0011 (f10=S, f3=S, b10F=B, b3F=B)  -2.53, hit 41, N=285 -> 1.0
    0111 (f10=S, f3=B, b10F=B, b3F=B)  -1.28, hit 44, N=179 -> 0.5
    1011 (f10=B, f3=S, b10F=B, b3F=B)  -1.37, hit 49, N=164 -> 0.5
    0101 (f10=S, f3=B, b10F=S, b3F=B)  -1.50, hit 44, N=27  -> 0.3

Sizing rule:
  size_unit > 0 = STEEPENER bet
  KTB10F position = -size_unit × scale  (short 10F if steepener)
  KTB3F  position = +size_unit × scale × (DV01_10F / DV01_3F)  (≈ 3 계약 per 1 unit)
  --> DV01 balanced, delta-neutral, slope only

Hold = 21 영업일.
Look-ahead audit:
  - Cell sign rule: priori (research 매트릭스, 사전 정의)
  - Sizing: fixed, no mean magnitude lookup
  - 시그널 input: t 시점 5d cum only
  - Daily P&L: t+1 ~ t+21
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
RATIO = DV01_KTB10F / DV01_KTB3F   # ~3.04
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


# ── Cell rule (size 양수=STEEPENER, 음수=FLATTENER) ──
RULE_SLOPE = {
    # STEEPENERS
    "1001": +2.0,   # mean +7.71, hit 91, N=23
    "1100": +1.0,   # mean +3.45, hit 70, N=23
    "1101": +1.0,   # mean +2.90, hit 77, N=68
    "1000": +0.5,   # mean +2.36, hit 75, N=8 (small)
    # FLATTENERS
    "0011": -1.0,   # mean -2.53, hit 41, N=285 (large)
    "0111": -0.5,   # mean -1.28, hit 44, N=179
    "1011": -0.5,   # mean -1.37, hit 49, N=164
    "0101": -0.3,   # mean -1.50, hit 44, N=27
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


def backtest_slope_pair(p, rule):
    """매 cell 시그널 발동 → KTB10F + KTB3F DV01 균형 페어 진입.

    Convention:
      size > 0 = STEEPENER bet (10Y 약세, slope ↑)
        pos_10F = -size (short)
        pos_3F  = +size × RATIO (long ~3 계약)
      size < 0 = FLATTENER bet (10Y 강세, slope ↓)
        pos_10F = -size (즉 +|size|, long)
        pos_3F  = +size × RATIO (즉 -|size| × RATIO, short)

    Daily P&L (만원):
      pos_10F × (-dy10_1d) × DV01_KTB10F + pos_3F × (-dy3_1d) × DV01_KTB3F
    """
    n = len(p)
    daily_pnl = np.zeros(n)
    daily_pos10 = np.zeros(n)
    daily_pos3 = np.zeros(n)
    daily_dur = np.zeros(n)    # net DV01 (만/bp), 0에 가까워야 함
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
        # Position (계약 수)
        pos_10 = -size              # steepener -> short 10F
        pos_3 = +size * RATIO       # steepener -> long ~3계약 3F

        # Trade-level P&L (21d hold approx)
        i_exit = min(i + HOLD, n - 1)
        fwd_dy10 = p["dy10_fwd_21"].iloc[i] if i + HOLD < n else np.nan
        fwd_dy3 = p["dy3_fwd_21"].iloc[i] if i + HOLD < n else np.nan
        fwd_dslope = p["dslope_fwd_21"].iloc[i] if i + HOLD < n else np.nan
        if pd.notna(fwd_dy10) and pd.notna(fwd_dy3):
            t_pnl = pos_10 * (-fwd_dy10) * DV01_KTB10F + pos_3 * (-fwd_dy3) * DV01_KTB3F
        else:
            t_pnl = np.nan
        # net DV01 (실제 check)
        net_dv01 = pos_10 * DV01_KTB10F + pos_3 * DV01_KTB3F   # 만/bp, theoretically 0

        direction = "STEEPENER" if size > 0 else "FLATTENER"
        trades.append({
            "entry_date": pd.Timestamp(dates[i]),
            "exit_date": pd.Timestamp(dates[i_exit]),
            "cell": c,
            "direction": direction,
            "size_unit": size,
            "pos_10F_ctr": round(pos_10, 3),
            "pos_3F_ctr": round(pos_3, 3),
            "net_DV01_man": round(net_dv01, 4),   # 듀레이션 중립 확인
            "y10_entry": float(p["y_10y"].iloc[i]),
            "y3_entry": float(p["y_3y"].iloc[i]),
            "fwd_dy10_bp": float(fwd_dy10) if pd.notna(fwd_dy10) else np.nan,
            "fwd_dy3_bp": float(fwd_dy3) if pd.notna(fwd_dy3) else np.nan,
            "fwd_dslope_bp": float(fwd_dslope) if pd.notna(fwd_dslope) else np.nan,
            "trade_pnl_man": float(t_pnl) if pd.notna(t_pnl) else np.nan,
        })

        # Daily P&L (overlap)
        for d in range(i + 1, min(i + HOLD + 1, n)):
            daily_pos10[d] += pos_10
            daily_pos3[d] += pos_3
            daily_pnl[d] += pos_10 * (-dy10_1d[d]) * DV01_KTB10F \
                            + pos_3 * (-dy3_1d[d]) * DV01_KTB3F

    # Net DV01 시계열 (모든 active position 의 net DV01 합)
    daily_dur = daily_pos10 * DV01_KTB10F + daily_pos3 * DV01_KTB3F

    daily = p[["price_date", "year", "y_10y", "y_3y"]].copy()
    daily["dy10_1d"] = p["dy10_1d"]
    daily["dy3_1d"] = p["dy3_1d"]
    daily["pos_10F"] = daily_pos10
    daily["pos_3F"] = daily_pos3
    daily["net_DV01_man"] = daily_dur
    daily["daily_pnl_man"] = daily_pnl
    daily["cum_pnl_man"] = daily_pnl.cumsum()
    daily["peak"] = daily["cum_pnl_man"].cummax()
    daily["drawdown_man"] = daily["cum_pnl_man"] - daily["peak"]
    return daily, pd.DataFrame(trades)


def perf(daily, trades):
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
    avg_win = trades.loc[trades["trade_pnl_man"] > 0, "trade_pnl_man"].mean() if win else 0
    avg_loss = trades.loc[trades["trade_pnl_man"] < 0, "trade_pnl_man"].mean() if loss else 0
    return {
        "Period": f"{daily['price_date'].min().date()} ~ {daily['price_date'].max().date()}",
        "Total days": len(daily),
        "Trades": len(trades),
        "Total P&L (만)": round(total, 0),
        "Per_yr (만)": round(total / nyrs, 0) if nyrs > 0 else 0,
        "Sharpe": round(sh, 3),
        "MaxDD (만)": round(mdd, 0),
        "Hit (%)": round(hit, 1),
        "Avg win (만)": round(avg_win, 1),
        "Avg loss (만)": round(avg_loss, 1),
        "W/L ratio": round(avg_win / -avg_loss, 2) if loss and avg_loss < 0 else None,
        "Active days": len(s_nz),
        "Net DV01 mean (만/bp)": round(daily["net_DV01_man"].abs().mean(), 3),
    }


def main():
    print("[load] panel ...")
    p = load_panel("2020-01-01")
    print(f"  {len(p):,} rows  {p['price_date'].min().date()} ~ {p['price_date'].max().date()}\n")

    print("[Rule] V7 cell slope direction (priori):")
    print(f"  RATIO (KTB3F per 1 KTB10F unit, DV01 균형) = {RATIO:.3f}\n")
    fmt = lambda b: "BUY" if int(b) else "SELL"
    for c, sz in sorted(RULE_SLOPE.items(), key=lambda x: -x[1]):
        direction = "STEEPENER" if sz > 0 else "FLATTENER"
        pos_10 = -sz
        pos_3 = sz * RATIO
        action_10 = f"{'SHORT' if pos_10 < 0 else 'LONG '} {abs(pos_10):.2f}"
        action_3 = f"{'LONG ' if pos_3 > 0 else 'SHORT'} {abs(pos_3):.2f}"
        print(f"  {c} (f10={fmt(c[0])}, f3={fmt(c[1])}, b10F={fmt(c[2])}, b3F={fmt(c[3])}): "
              f"{direction:10s} size={abs(sz):.1f}  "
              f"10F: {action_10}  3F: {action_3}")
    print()

    daily, trades = backtest_slope_pair(p, RULE_SLOPE)

    print("=" * 80)
    print("V7 Backtest Summary")
    print("=" * 80)
    s = perf(daily, trades)
    for k, v in s.items():
        if isinstance(v, float):
            print(f"  {k:>26s}: {v:>+12,.2f}")
        else:
            print(f"  {k:>26s}: {v}")
    print()

    # 연도별
    yr = daily.groupby("year").agg(
        N_days=("daily_pnl_man", lambda x: (x != 0).sum()),
        total_man=("daily_pnl_man", "sum"),
        sharpe=("daily_pnl_man", lambda x: x[x != 0].mean() / x[x != 0].std() * np.sqrt(TRADING_DAYS)
                if len(x[x != 0]) > 1 and x[x != 0].std() > 0 else 0),
    ).round(2)
    print("연도별:")
    print(yr.to_string())
    print()

    # Direction 별 통계
    print("=" * 80)
    print("Direction 별 (STEEPENER vs FLATTENER):")
    print("=" * 80)
    dr = trades.groupby("direction").agg(
        N=("trade_pnl_man", "size"),
        total_man=("trade_pnl_man", "sum"),
        mean_man=("trade_pnl_man", "mean"),
        hit_pct=("trade_pnl_man", lambda x: (x > 0).mean() * 100),
        mean_fwd_dslope=("fwd_dslope_bp", "mean"),
    ).round(2)
    print(dr.to_string())
    print()

    # Cell-level
    print("Cell-level trade stats:")
    cs = trades.groupby("cell").agg(
        N=("trade_pnl_man", "size"),
        direction=("direction", "first"),
        size_unit=("size_unit", "first"),
        total_man=("trade_pnl_man", "sum"),
        mean_man=("trade_pnl_man", "mean"),
        hit_pct=("trade_pnl_man", lambda x: (x > 0).mean() * 100),
        mean_fwd_dslope=("fwd_dslope_bp", "mean"),
    ).round(2).sort_values("total_man", ascending=False)
    print(cs.to_string())
    print()

    # 5/11 시그널
    print("=" * 80)
    print("5/11 시그널")
    print("=" * 80)
    latest = p.iloc[-1]
    cell = latest["cell"]
    print(f"  date: {latest['price_date'].strftime('%Y-%m-%d')}")
    print(f"  cell: {cell}")
    print(f"  combo: f10={fmt(cell[0])}, f3={fmt(cell[1])}, b10F={fmt(cell[2])}, b3F={fmt(cell[3])}")
    if cell in RULE_SLOPE:
        sz = RULE_SLOPE[cell]
        direction = "STEEPENER" if sz > 0 else "FLATTENER"
        pos_10 = -sz
        pos_3 = sz * RATIO
        print(f"  --> {direction} size {abs(sz):.1f}")
        print(f"      KTB10F {'SHORT' if pos_10 < 0 else 'LONG'} {abs(pos_10):.2f} 계약")
        print(f"      KTB3F  {'SHORT' if pos_3 < 0 else 'LONG'} {abs(pos_3):.2f} 계약")
        print(f"      Hold 21 영업일")
        print(f"      Net DV01 = 0 (delta-neutral, slope 노출만)")
    else:
        print(f"  --> FLAT (rule 없음)")
    print()

    # ── Excel 저장 ──
    CHART_DIR.mkdir(exist_ok=True)
    xlsx = CHART_DIR / "V7_slope_pair_track_record.xlsx"
    print(f"[save] {xlsx}")

    audit_rows = [
        ("Cell sign rule", "Stage 21 매트릭스에서 사전 정의 (Δslope mean 부호 기반)",
         "in-sample structure"),
        ("Sizing", "Fixed unit per cell (mean magnitude 안 사용)", "OK"),
        ("Pair entry", "항상 KTB10F + KTB3F DV01 균형 (RATIO≈3.04)", "delta-neutral"),
        ("Net DV01", "각 trade 의 net DV01 거의 0 (반올림 오차만)", "확인됨"),
        ("Input timing", "t 시점 5d cum, t 정보만", "OK"),
        ("Entry", "t close → t+1 부터 daily P&L", "OK"),
        ("Daily P&L", "dy_1d[i+1] = y(t+1) - y(t)", "OK"),
        ("Target", "slope 변동 (Δ10Y - Δ3Y), 듀레이션 무관", "curve only"),
    ]
    audit_df = pd.DataFrame(audit_rows, columns=["Item", "Description", "Status"])

    rule_df = pd.DataFrame([
        {"cell": c, "f10": fmt(c[0]), "f3": fmt(c[1]), "b10F": fmt(c[2]), "b3F": fmt(c[3]),
         "direction": "STEEPENER" if sz > 0 else "FLATTENER",
         "size_unit": sz,
         "pos_10F_ctr": -sz, "pos_3F_ctr": round(sz * RATIO, 3)}
        for c, sz in RULE_SLOPE.items()
    ])

    summary_df = pd.DataFrame(list(s.items()), columns=["Metric", "Value"])

    with pd.ExcelWriter(xlsx, engine="openpyxl") as xl:
        summary_df.to_excel(xl, sheet_name="Summary", index=False)
        yr.reset_index().to_excel(xl, sheet_name="Yearly", index=False)
        rule_df.to_excel(xl, sheet_name="Rule", index=False)
        dr.reset_index().to_excel(xl, sheet_name="Direction_stats", index=False)
        cs.reset_index().to_excel(xl, sheet_name="Cell_stats", index=False)
        t_out = trades.copy()
        for c in t_out.select_dtypes(include=["float64"]).columns:
            t_out[c] = t_out[c].round(3)
        t_out.to_excel(xl, sheet_name="Trade_log", index=False)
        d_out = daily.copy()
        d_out["price_date"] = d_out["price_date"].dt.strftime("%Y-%m-%d")
        for c in d_out.select_dtypes(include=["float64"]).columns:
            d_out[c] = d_out[c].round(3)
        d_out.to_excel(xl, sheet_name="Daily_PnL", index=False)
        audit_df.to_excel(xl, sheet_name="Audit", index=False)

    print(f"  OK -> {xlsx}\n")

    # ── 차트 ──
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True,
                              gridspec_kw={"height_ratios": [2.2, 1, 1]})
    axes[0].fill_between(daily["price_date"], 0, daily["cum_pnl_man"],
                          where=daily["cum_pnl_man"] >= 0, alpha=0.25, color="#2a9d8f")
    axes[0].fill_between(daily["price_date"], 0, daily["cum_pnl_man"],
                          where=daily["cum_pnl_man"] < 0, alpha=0.25, color="#e76f51")
    axes[0].plot(daily["price_date"], daily["cum_pnl_man"], color="#264653", lw=1.8)
    axes[0].axhline(0, color="gray", lw=0.7, ls="--")
    axes[0].set_title(f"V7 Slope-only Curve Pair (Delta-neutral)  Final {s['Total P&L (만)']:+,.0f}만  "
                       f"Sharpe {s['Sharpe']:+.2f}",
                       fontsize=13, weight="bold")
    axes[0].set_ylabel("Cumulative P&L (만원)")
    axes[0].grid(alpha=0.3)

    axes[1].fill_between(daily["price_date"], 0, daily["drawdown_man"],
                          color="#e76f51", alpha=0.35)
    axes[1].plot(daily["price_date"], daily["drawdown_man"], color="#a8331b", lw=1.2)
    axes[1].set_title(f"Drawdown (MaxDD {s['MaxDD (만)']:,.0f}만)")
    axes[1].set_ylabel("DD (만)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(daily["price_date"], daily["net_DV01_man"], color="#444", lw=0.8)
    axes[2].axhline(0, color="red", lw=0.5, ls="--", label="0 (delta-neutral)")
    axes[2].set_title(f"Net DV01 (만/bp) — 듀레이션 노출 (이상적 0, 실제 mean abs {s['Net DV01 mean (만/bp)']:.3f})")
    axes[2].set_ylabel("Net DV01 (만/bp)")
    axes[2].legend()
    axes[2].xaxis.set_major_locator(mdates.YearLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "29_v7_slope_pair.png", bbox_inches="tight")
    plt.close(fig)
    print("  OK 29_v7_slope_pair.png")

    print("\n[done]")


if __name__ == "__main__":
    main()
