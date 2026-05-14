"""
23 — V5-B audit + 트랙 레코드 엑셀 저장.

Look-ahead audit:
  (A) 시그널 input: t 시점 5d cum, t 정보만 -> OK
  (B) Cell mean (sizing): 22번 in-sample 은 전체 panel 학습 = look-ahead
      -> 본 스크립트는 expanding window 로 매 63 영업일마다 t 이전 데이터만으로 재학습
  (C) Daily P&L: dy_1d 는 t+1 - t close, t 진입 후 t+1 ~ t+21 변동만 사용 -> OK
  (D) Train minimum: warm-up 252 영업일 (1년) 미만은 신호 안 냄

출력:
  Excel: research/fund_flow/charts/V5B_track_record.xlsx
    sheets:
      - Summary       : 메트릭 요약
      - Trade_log     : trade-by-trade (entry date, cell, signal, pos, fwd, pnl)
      - Daily_PnL     : 일별 P&L + cumulative + drawdown + positions
      - Cell_table_final : final expanding window cell table (참고)
      - Audit         : look-ahead audit checklist + 결과

Walk-forward expanding window:
  매 63 영업일마다 cell_tbl 재학습.
  warm-up: 첫 252 영업일은 시그널 안 냄 (cell 별 N 부족).
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
TRADING_DAYS = 252
HOLD = 21
WARM_UP = 252
REFIT_FREQ = 63        # 분기마다 cell_tbl 재학습
SCALE_NORM_DSLOPE = 3.0
SIZE_CAP = 3.0
THRESHOLD_BP = 1.0
MIN_CELL_N = 5         # cell 당 최소 N (학습 데이터)


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
    for h in [21]:
        p[f"dy3_fwd_{h}"] = p["y_3y"].shift(-h) - p["y_3y"]
        p[f"dy10_fwd_{h}"] = p["y_10y"].shift(-h) - p["y_10y"]
        p[f"dslope_fwd_{h}"] = p[f"dy10_fwd_{h}"] - p[f"dy3_fwd_{h}"]
    p["s_f10"] = (p["f10"] > 0).astype(int)
    p["s_f3"] = (p["f3"] > 0).astype(int)
    p["s_b10F"] = (p["b10F"] > 0).astype(int)
    p["s_b3F"] = (p["b3F"] > 0).astype(int)
    p["cell"] = (p["s_f10"].astype(str) + p["s_f3"].astype(str)
                  + p["s_b10F"].astype(str) + p["s_b3F"].astype(str))
    return p


def cell_table_at(train_p):
    """train_p (forward dy 결측 제외) 기준 cell mean. fwd_dy 가 있는 row 만 사용."""
    g = train_p.dropna(subset=["dy10_fwd_21", "dslope_fwd_21"]).copy()
    if len(g) == 0:
        return pd.DataFrame()
    tbl = g.groupby("cell").agg(
        N=("dy10_fwd_21", "size"),
        mean_dy10=("dy10_fwd_21", "mean"),
        mean_dslope=("dslope_fwd_21", "mean"),
        hit_dslope=("dslope_fwd_21", lambda x: (x > 0).mean() * 100),
    )
    return tbl


def backtest_v5b_walkforward(p):
    """매 REFIT_FREQ 영업일마다 cell_tbl 재학습.
       매 entry 시점 t 에서 [start, t-HOLD-1] 까지만 cell_tbl 학습 (forward fwd_dy_21 결측 회피).
    """
    n = len(p)
    daily_pnl = np.zeros(n)
    daily_pos10 = np.zeros(n)
    daily_pos3 = np.zeros(n)
    dy10_1d = p["dy10_1d"].fillna(0.0).values
    dy3_1d = p["dy3_1d"].fillna(0.0).values
    cells_seq = p["cell"].values
    dates_seq = p["price_date"].values

    cur_tbl = pd.DataFrame()
    last_refit = -REFIT_FREQ - 1
    trades = []

    for i in range(n):
        # 1) cell_tbl 재학습 시점
        if i - last_refit >= REFIT_FREQ and i >= WARM_UP:
            # train_p: 인덱스 [0, i - HOLD - 1] 까지 (forward fwd_dy_21 결측 회피)
            train_idx = i - HOLD - 1
            if train_idx > 0:
                cur_tbl = cell_table_at(p.iloc[:train_idx + 1])
                last_refit = i

        if i < WARM_UP or cur_tbl.empty:
            continue

        # 2) 현재 cell 시그널
        c = cells_seq[i]
        if c not in cur_tbl.index:
            continue
        row = cur_tbl.loc[c]
        if int(row["N"]) < MIN_CELL_N:
            continue
        m = row["mean_dslope"]
        if abs(m) < THRESHOLD_BP:
            continue

        sz = min(abs(m) / SCALE_NORM_DSLOPE, SIZE_CAP) * np.sign(m)
        # steepener (m > 0) -> 10F short + 3F long
        # flattener (m < 0) -> 10F long + 3F short
        pos_10 = -sz
        pos_3 = +sz * (DV01_KTB10F / DV01_KTB3F)

        # 3) trade 기록
        i_exit = min(i + HOLD, n - 1)
        fwd_dy10 = p["dy10_fwd_21"].iloc[i] if i + HOLD < n else np.nan
        fwd_dslope = p["dslope_fwd_21"].iloc[i] if i + HOLD < n else np.nan
        # trade P&L (full hold approx): pos_10 × (-fwd_dy10) × DV01_10 + pos_3 × (-fwd_dy3) × DV01_3
        fwd_dy3 = p["dy3_fwd_21"].iloc[i] if i + HOLD < n else np.nan
        trade_pnl = (pos_10 * (-fwd_dy10) * DV01_KTB10F
                      + pos_3 * (-fwd_dy3) * DV01_KTB3F)
        trades.append({
            "entry_date": pd.Timestamp(dates_seq[i]),
            "exit_date": pd.Timestamp(dates_seq[i_exit]),
            "cell": c,
            "f10": "BUY" if int(c[0]) else "SELL",
            "f3": "BUY" if int(c[1]) else "SELL",
            "b10F": "BUY" if int(c[2]) else "SELL",
            "b3F": "BUY" if int(c[3]) else "SELL",
            "cell_N_train": int(row["N"]),
            "cell_mean_dslope_train": float(row["mean_dslope"]),
            "cell_hit_dslope_train": float(row["hit_dslope"]),
            "signal_dslope": float(m),
            "pos_10F_ctr": float(pos_10),
            "pos_3F_ctr": float(pos_3),
            "y_10y_entry": float(p["y_10y"].iloc[i]),
            "y_3y_entry": float(p["y_3y"].iloc[i]),
            "fwd_dy10_bp": float(fwd_dy10) if pd.notna(fwd_dy10) else np.nan,
            "fwd_dy3_bp": float(fwd_dy3) if pd.notna(fwd_dy3) else np.nan,
            "fwd_dslope_bp": float(fwd_dslope) if pd.notna(fwd_dslope) else np.nan,
            "trade_pnl_man": float(trade_pnl) if pd.notna(trade_pnl) else np.nan,
        })

        # 4) Daily P&L 누적 (overlap)
        for d in range(i + 1, min(i + HOLD + 1, n)):
            daily_pos10[d] += pos_10
            daily_pos3[d] += pos_3
            daily_pnl[d] += (pos_10 * (-dy10_1d[d]) * DV01_KTB10F
                              + pos_3 * (-dy3_1d[d]) * DV01_KTB3F)

    out = p[["price_date", "year", "y_10y", "y_3y"]].copy()
    out["dy10_1d"] = p["dy10_1d"]
    out["dy3_1d"] = p["dy3_1d"]
    out["daily_pnl_man"] = daily_pnl
    out["pos_10F"] = daily_pos10
    out["pos_3F"] = daily_pos3
    out["cum_pnl_man"] = daily_pnl.cumsum()
    out["peak"] = out["cum_pnl_man"].cummax()
    out["drawdown_man"] = out["cum_pnl_man"] - out["peak"]
    return out, pd.DataFrame(trades), cur_tbl


def perf_summary(daily, trades):
    s_nz = daily["daily_pnl_man"][daily["daily_pnl_man"] != 0]
    mu = s_nz.mean() if len(s_nz) else 0
    sd = s_nz.std() if len(s_nz) else 1
    sh = mu / sd * np.sqrt(TRADING_DAYS) if sd > 0 else 0
    total = daily["daily_pnl_man"].sum()
    nyrs = len(daily) / TRADING_DAYS
    mdd = daily["drawdown_man"].min()
    if len(trades):
        win = (trades["trade_pnl_man"] > 0).sum()
        loss = (trades["trade_pnl_man"] < 0).sum()
        hit = win / (win + loss) * 100 if (win + loss) else 0
        avg_win = trades.loc[trades["trade_pnl_man"] > 0, "trade_pnl_man"].mean()
        avg_loss = trades.loc[trades["trade_pnl_man"] < 0, "trade_pnl_man"].mean()
    else:
        win = loss = hit = avg_win = avg_loss = 0
    return {
        "Backtest period (start)": str(daily["price_date"].min().date()),
        "Backtest period (end)": str(daily["price_date"].max().date()),
        "Total days": len(daily),
        "Total trades": len(trades),
        "Total P&L (만)": round(total, 0),
        "Per_yr (만)": round(total / nyrs, 0) if nyrs > 0 else 0,
        "Sharpe (annualized)": round(sh, 3),
        "Max Drawdown (만)": round(mdd, 0),
        "Hit rate (%)": round(hit, 2),
        "Avg win (만)": round(avg_win, 2) if win else 0,
        "Avg loss (만)": round(avg_loss, 2) if loss else 0,
        "Win/Loss ratio": round(avg_win / -avg_loss, 2) if loss and avg_loss < 0 else None,
        "Refit frequency (days)": REFIT_FREQ,
        "Warm-up days": WARM_UP,
        "Hold days": HOLD,
        "Threshold (bp)": THRESHOLD_BP,
        "Size cap": SIZE_CAP,
        "DV01 KTB10F (만/bp)": DV01_KTB10F,
        "DV01 KTB3F (만/bp)": DV01_KTB3F,
    }


def main():
    print("[load] panel ...")
    p = load_panel("2020-01-01")
    print(f"  {len(p):,} rows  {p['price_date'].min().date()} ~ {p['price_date'].max().date()}\n")

    print("[backtest] V5-B walk-forward expanding window (refit every 63d, warm-up 252d) ...")
    daily, trades, final_tbl = backtest_v5b_walkforward(p)
    print(f"  Trades: {len(trades):,}")
    print(f"  Active days (P&L != 0): {(daily['daily_pnl_man'] != 0).sum():,}")
    print()

    s = perf_summary(daily, trades)
    print("=" * 80)
    print("V5-B Walk-forward Summary (no look-ahead in sizing)")
    print("=" * 80)
    for k, v in s.items():
        if isinstance(v, float):
            print(f"  {k:>32s}: {v:>+12,.2f}")
        else:
            print(f"  {k:>32s}: {v}")
    print()

    # 연도별
    daily["year"] = daily["price_date"].dt.year
    yr = daily.groupby("year").agg(
        N_days=("daily_pnl_man", lambda x: (x != 0).sum()),
        total_man=("daily_pnl_man", "sum"),
        sharpe=("daily_pnl_man", lambda x: x[x != 0].mean() / x[x != 0].std() * np.sqrt(TRADING_DAYS)
                if len(x[x != 0]) > 1 and x[x != 0].std() > 0 else 0),
    ).round(2)
    print("연도별:")
    print(yr.to_string())
    print()

    # ── Excel 저장 ──
    CHART_DIR.mkdir(exist_ok=True)
    xlsx_path = CHART_DIR / "V5B_track_record.xlsx"
    print(f"[save] {xlsx_path}")

    audit_rows = [
        ("1. Signal input (f10, f3, b10F, b3F)",
         "t 시점까지 5d cum, t 정보만 사용", "OK (no look-ahead)"),
        ("2. Cell mean (sizing)",
         "매 REFIT_FREQ=63영업일 마다 [start, t-HOLD-1] 까지 만 학습",
         "OK (expanding window, forward fwd_dy_21 결측 회피)"),
        ("3. Daily P&L",
         "dy_1d[i+1] = y(t+1) - y(t), entry 후 t+1 ~ t+21 변동만 사용",
         "OK"),
        ("4. Warm-up",
         f"첫 {WARM_UP} 영업일 (~1년) 은 시그널 없음 (cell N 부족)",
         "OK"),
        ("5. Cell min N",
         f"학습 데이터 cell N < {MIN_CELL_N} 인 cell 은 trade skip",
         "OK"),
        ("6. Entry timing",
         "시그널 t 일 close → t+1 부터 P&L 시작 (T+1 진입 가정)",
         "OK"),
    ]
    audit_df = pd.DataFrame(audit_rows, columns=["Item", "Description", "Status"])

    summary_df = pd.DataFrame(list(s.items()), columns=["Metric", "Value"])

    yearly_df = yr.reset_index()
    yearly_df.columns = ["Year", "N_active_days", "Total_man", "Sharpe"]

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xl:
        summary_df.to_excel(xl, sheet_name="Summary", index=False)
        yearly_df.to_excel(xl, sheet_name="Yearly", index=False)
        trades_out = trades.copy()
        for c in trades_out.select_dtypes(include=["float64"]).columns:
            trades_out[c] = trades_out[c].round(3)
        trades_out.to_excel(xl, sheet_name="Trade_log", index=False)
        daily_out = daily[["price_date", "year", "y_10y", "y_3y", "dy10_1d", "dy3_1d",
                            "pos_10F", "pos_3F", "daily_pnl_man",
                            "cum_pnl_man", "drawdown_man"]].copy()
        daily_out["price_date"] = daily_out["price_date"].dt.strftime("%Y-%m-%d")
        for c in daily_out.select_dtypes(include=["float64"]).columns:
            daily_out[c] = daily_out[c].round(3)
        daily_out.to_excel(xl, sheet_name="Daily_PnL", index=False)
        if not final_tbl.empty:
            final_tbl.reset_index().to_excel(xl, sheet_name="Cell_table_final", index=False)
        audit_df.to_excel(xl, sheet_name="Audit", index=False)

    print(f"  OK -> {xlsx_path}")
    print()

    # 차트도 추가
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    for fname in ["Malgun Gothic", "NanumGothic", "AppleGothic"]:
        try:
            plt.rcParams["font.family"] = fname
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True,
                              gridspec_kw={"height_ratios": [2.2, 1, 1]})
    axes[0].fill_between(daily["price_date"], 0, daily["cum_pnl_man"],
                          where=daily["cum_pnl_man"] >= 0, alpha=0.25, color="#2a9d8f")
    axes[0].fill_between(daily["price_date"], 0, daily["cum_pnl_man"],
                          where=daily["cum_pnl_man"] < 0, alpha=0.25, color="#e76f51")
    axes[0].plot(daily["price_date"], daily["cum_pnl_man"], color="#264653", lw=1.8)
    axes[0].axhline(0, color="gray", lw=0.7, ls="--")
    axes[0].set_title(f"V5-B Walk-forward Cumulative P&L (no look-ahead)  Final {s['Total P&L (만)']:+,.0f}만  Sharpe {s['Sharpe (annualized)']:+.2f}",
                      fontsize=13, weight="bold")
    axes[0].set_ylabel("Cumulative P&L (만원)")
    axes[0].grid(alpha=0.3)

    axes[1].fill_between(daily["price_date"], 0, daily["drawdown_man"],
                          color="#e76f51", alpha=0.35)
    axes[1].plot(daily["price_date"], daily["drawdown_man"], color="#a8331b", lw=1.2)
    axes[1].set_title(f"Drawdown (MaxDD {s['Max Drawdown (만)']:,.0f}만)")
    axes[1].set_ylabel("DD (만)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(daily["price_date"], daily["pos_10F"], color="#e76f51", lw=0.8,
                  label="KTB10F net (계약)")
    axes[2].plot(daily["price_date"], daily["pos_3F"], color="#2a9d8f", lw=0.8,
                  label="KTB3F net (계약)")
    axes[2].axhline(0, color="gray", lw=0.5, ls="--")
    axes[2].set_title("Net positions (계약 수)")
    axes[2].set_ylabel("Contracts")
    axes[2].legend()
    axes[2].xaxis.set_major_locator(mdates.YearLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "27_v5b_walkforward.png", bbox_inches="tight")
    plt.close(fig)
    print("  OK 27_v5b_walkforward.png")

    print("\n[done]")


if __name__ == "__main__":
    main()
