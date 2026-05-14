"""
22 - V5 16-cell cross-tenor backtest.

V5 시그널:
  각 거래일에 4 카테고리 sign (f10, f3, b10F, b3F) 으로 16 cell 중 하나 lookup.
  Cell 별 in-sample mean (ΔY_10Y_21, Δslope_21) 으로 시그널 강도 결정.

Variant:
  V5-A : KTB10F 단독 short/long, target = mean ΔY_10Y_21 of cell
  V5-B : 선물 커브 트레이드 (KTB3F + KTB10F, DV01 균형), target = mean Δslope_21
         curve steepener 예측 (Δslope > 0) → KTB10F short + KTB3F long
         curve flattener 예측 (Δslope < 0) → KTB10F long + KTB3F short
         사이즈: DV01 균형 (KTB10F 1 계약 ≡ KTB3F 약 3 계약)

Sizing:
  대수: sign(cell_mean) × |cell_mean|  (강한 cell = 큰 사이즈)
  단, abs(cell_mean) 이 threshold 이상인 cell 만 활성, 외 flat
  hold = 21d (cell mean 의 horizon)

Threshold sweep:
  cell_mean abs ≥ 0, 2, 3, 5 bp 별 성과 비교

OOS 검증:
  - 전체 in-sample (단순 backtest)
  - Walk-forward (첫 3년 train cell-mean → 후 3년 적용)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
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
    for h in [3, 10, 21]:
        p[f"dy3_fwd_{h}"] = p["y_3y"].shift(-h) - p["y_3y"]
        p[f"dy10_fwd_{h}"] = p["y_10y"].shift(-h) - p["y_10y"]
        p[f"dslope_fwd_{h}"] = p[f"dy10_fwd_{h}"] - p[f"dy3_fwd_{h}"]

    # Cell sign label
    p["s_f10"] = (p["f10"] > 0).astype(int)
    p["s_f3"] = (p["f3"] > 0).astype(int)
    p["s_b10F"] = (p["b10F"] > 0).astype(int)
    p["s_b3F"] = (p["b3F"] > 0).astype(int)
    p["cell"] = (p["s_f10"].astype(str) + p["s_f3"].astype(str)
                  + p["s_b10F"].astype(str) + p["s_b3F"].astype(str))
    return p


def build_cell_table(train_p: pd.DataFrame):
    """Cell 별 mean 을 train 데이터에서 학습."""
    tbl = train_p.dropna(subset=["dy10_fwd_21", "dslope_fwd_21"]).groupby("cell").agg(
        N=("dy10_fwd_21", "size"),
        mean_dy10=("dy10_fwd_21", "mean"),
        mean_dslope=("dslope_fwd_21", "mean"),
        hit_up_dy10=("dy10_fwd_21", lambda x: (x > 0).mean() * 100),
        hit_up_dslope=("dslope_fwd_21", lambda x: (x > 0).mean() * 100),
    ).round(3)
    return tbl


def backtest_v5(p, cell_tbl, mode="A_10y", threshold_bp=0.0,
                  size_scale=1.0, use_dv01_balanced=True):
    """
    mode='A_10y'   : KTB10F 단독, target = mean_dy10
                     pos_10F = -sign(mean_dy10) × min(abs(mean_dy10)/scale_norm, cap)
                     daily pnl_10F = pos_10F × (-dy10_1d) × DV01_KTB10F
    mode='B_slope' : KTB10F + KTB3F slope trade
                     curve_signal = -sign(mean_dslope) × ...
                     pos_10F = -sign × size_10F   (short 10F if steepener expected)
                     pos_3F  = +sign × size_3F   (long 3F if steepener)
                     DV01 균형: size_3F = size_10F × DV01_KTB10F / DV01_KTB3F (≈ 3)
                     daily pnl = pos_10F × (-dy10_1d) × DV01_10F + pos_3F × (-dy3_1d) × DV01_3F
    """
    n = len(p)
    daily_pnl = np.zeros(n)
    daily_pos10 = np.zeros(n)
    daily_pos3 = np.zeros(n)
    dy10_1d = p["dy10_1d"].fillna(0.0).values
    dy3_1d = p["dy3_1d"].fillna(0.0).values
    cells = p["cell"].values
    log = []

    SCALE_NORM_DY10 = 5.0       # mean_dy10 / 5 = signal strength unit
    SCALE_NORM_DSLOPE = 3.0
    SIZE_CAP = 3.0              # max |size_unit| = 3

    for i in range(n):
        c = cells[i]
        if c not in cell_tbl.index:
            continue
        row = cell_tbl.loc[c]
        if mode == "A_10y":
            m = row["mean_dy10"]
            if abs(m) < threshold_bp:
                continue
            sz = min(abs(m) / SCALE_NORM_DY10, SIZE_CAP) * np.sign(m) * size_scale
            pos_10 = -sz   # short 10F if expected ΔY_10Y > 0 (yield up = price down)
            pos_3 = 0.0
        elif mode == "B_slope":
            m = row["mean_dslope"]
            if abs(m) < threshold_bp:
                continue
            sz = min(abs(m) / SCALE_NORM_DSLOPE, SIZE_CAP) * np.sign(m) * size_scale
            # expected Δslope > 0 (steepener) → 10Y 약세, 3Y 강세
            # → short 10F + long 3F
            pos_10 = -sz   # 10F short if steepener
            pos_3 = +sz * (DV01_KTB10F / DV01_KTB3F)   # 3F long DV01 balanced
        else:
            continue
        log.append({"date": p.loc[i, "price_date"], "cell": c,
                     "pos_10": pos_10, "pos_3": pos_3, "signal": m})
        # active for hold days
        for d in range(i + 1, min(i + HOLD + 1, n)):
            daily_pos10[d] += pos_10
            daily_pos3[d] += pos_3
            daily_pnl[d] += pos_10 * (-dy10_1d[d]) * DV01_KTB10F + \
                            pos_3 * (-dy3_1d[d]) * DV01_KTB3F   # 만원
    out = p[["price_date", "year"]].copy()
    out["pnl_man"] = daily_pnl
    out["pos_10"] = daily_pos10
    out["pos_3"] = daily_pos3
    return out, pd.DataFrame(log)


def perf(series, name):
    s = series.dropna()
    s_nz = s[s != 0]
    mu = s_nz.mean() if len(s_nz) else 0
    sd = s_nz.std() if len(s_nz) else 1
    sh = mu / sd * np.sqrt(TRADING_DAYS) if sd > 0 else 0
    cum = s.cumsum()
    mdd = (cum - cum.cummax()).min()
    total = s.sum()
    nyrs = len(s) / TRADING_DAYS
    return {"name": name, "total": total, "per_yr": total / nyrs if nyrs > 0 else 0,
            "sharpe": sh, "mdd": mdd, "N_active": len(s_nz)}


def yr_table(dp):
    r = dp[dp["pnl_man"] != 0].copy()
    return r.groupby("year")["pnl_man"].sum().round(1)


def main():
    print("[load] panel ...")
    p = load_panel("2020-01-01")
    print(f"  {len(p):,} rows  {p['price_date'].min().date()} ~ {p['price_date'].max().date()}\n")

    # ── A) Cell table (전체 기간 in-sample) ──
    print("=" * 100)
    print("A) Cell table (전체 기간 in-sample)")
    print("=" * 100)
    tbl_full = build_cell_table(p)
    tbl_full = tbl_full.sort_values("mean_dy10", ascending=False)
    print("\n" + tbl_full.to_string())
    print()

    # ── B) V5-A (KTB10F 단독) backtest, threshold sweep ──
    print("=" * 100)
    print("B) V5-A (KTB10F 단독) - threshold sweep (in-sample)")
    print("=" * 100)
    print(f"\n  {'thr':>5s} {'N_entries':>10s} {'sharpe':>8s} {'per_yr':>12s} {'total':>14s} {'mdd':>14s}")
    for thr in [0.0, 1.0, 2.0, 3.0, 5.0]:
        dp, log = backtest_v5(p, tbl_full, mode="A_10y", threshold_bp=thr)
        m = perf(dp["pnl_man"], f"V5-A thr={thr}")
        print(f"  {thr:>5.1f} {len(log):>10d} {m['sharpe']:>+8.2f} "
              f"{m['per_yr']:>+12,.0f} {m['total']:>+14,.0f} {m['mdd']:>+14,.0f}")
    print()

    # ── C) V5-B (slope trade) backtest, threshold sweep ──
    print("=" * 100)
    print("C) V5-B (Slope: KTB10F + KTB3F DV01 균형) - threshold sweep (in-sample)")
    print("=" * 100)
    print(f"\n  {'thr':>5s} {'N_entries':>10s} {'sharpe':>8s} {'per_yr':>12s} {'total':>14s} {'mdd':>14s}")
    for thr in [0.0, 0.5, 1.0, 1.5, 2.5]:
        dp, log = backtest_v5(p, tbl_full, mode="B_slope", threshold_bp=thr)
        m = perf(dp["pnl_man"], f"V5-B thr={thr}")
        print(f"  {thr:>5.1f} {len(log):>10d} {m['sharpe']:>+8.2f} "
              f"{m['per_yr']:>+12,.0f} {m['total']:>+14,.0f} {m['mdd']:>+14,.0f}")
    print()

    # ── D) 연도별 (best variant) ──
    print("=" * 100)
    print("D) 연도별 P&L (V5-A thr=2, V5-B thr=1)")
    print("=" * 100)
    dp_a, _ = backtest_v5(p, tbl_full, mode="A_10y", threshold_bp=2.0)
    dp_b, _ = backtest_v5(p, tbl_full, mode="B_slope", threshold_bp=1.0)
    yr_a = yr_table(dp_a)
    yr_b = yr_table(dp_b)
    df = pd.DataFrame({"V5-A (10F)": yr_a, "V5-B (slope)": yr_b})
    print("\n" + df.to_string())
    print()

    # ── E) Walk-forward OOS test ──
    print("=" * 100)
    print("E) Walk-forward OOS: train 2020-01 ~ 2023-12, test 2024-01 ~ 2026-05")
    print("=" * 100)
    cutoff = pd.Timestamp("2024-01-01")
    train_p = p[p["price_date"] < cutoff].copy()
    test_p = p[p["price_date"] >= cutoff].copy().reset_index(drop=True)

    tbl_train = build_cell_table(train_p)
    print(f"\n  train period: {train_p['price_date'].min().date()} ~ {train_p['price_date'].max().date()} "
          f"({len(train_p):,} days)")
    print(f"  test period:  {test_p['price_date'].min().date()} ~ {test_p['price_date'].max().date()} "
          f"({len(test_p):,} days)")

    print(f"\n  -> OOS test 결과 (cell mean is from train only):")
    print(f"  {'mode':>12s} {'thr':>5s} {'N':>5s} {'sharpe':>8s} {'per_yr':>12s} {'total':>12s} {'mdd':>12s}")
    for mode in ["A_10y", "B_slope"]:
        for thr in [0.0, 1.0, 2.0, 3.0]:
            dp, log = backtest_v5(test_p, tbl_train, mode=mode, threshold_bp=thr)
            m = perf(dp["pnl_man"], f"{mode} thr={thr}")
            print(f"  {mode:>12s} {thr:>5.1f} {len(log):>5d} {m['sharpe']:>+8.2f} "
                  f"{m['per_yr']:>+12,.0f} {m['total']:>+12,.0f} {m['mdd']:>+12,.0f}")
    print()

    # ── F) 5/11 기준 시그널 ──
    print("=" * 100)
    print("F) 5/11 기준 V5 시그널")
    print("=" * 100)
    latest = p.iloc[-1]
    cell_now = latest["cell"]
    print(f"\n  date: {latest['price_date'].strftime('%Y-%m-%d')}")
    print(f"  cell: {cell_now}")
    if cell_now in tbl_full.index:
        row = tbl_full.loc[cell_now]
        f10_s = "BUY" if int(cell_now[0]) else "SELL"
        f3_s = "BUY" if int(cell_now[1]) else "SELL"
        b10_s = "BUY" if int(cell_now[2]) else "SELL"
        b3_s = "BUY" if int(cell_now[3]) else "SELL"
        print(f"  combo: f10={f10_s}, f3={f3_s}, b10F={b10_s}, b3F={b3_s}")
        print(f"\n  Cell stats (in-sample):")
        print(f"    N            : {int(row['N']):,}")
        print(f"    mean ΔY_10Y_21: {row['mean_dy10']:+.2f} bp")
        print(f"    mean Δslope_21: {row['mean_dslope']:+.2f} bp")
        print(f"    hit ΔY_10Y>0  : {row['hit_up_dy10']:.1f}%")
        print(f"    hit Δslope>0  : {row['hit_up_dslope']:.1f}%")
        print(f"\n  -> V5-A: KTB10F {'short' if row['mean_dy10'] > 0 else 'long'}")
        print(f"  -> V5-B: {'steepener' if row['mean_dslope'] > 0 else 'flattener'} "
              f"= KTB10F {'short' if row['mean_dslope'] > 0 else 'long'} + KTB3F {'long' if row['mean_dslope'] > 0 else 'short'}")
    print()

    # ── Charts ──
    print("=" * 100)
    print("차트 ...")
    print("=" * 100)
    CHART_DIR.mkdir(exist_ok=True)
    # Cumulative
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(dp_a["price_date"], dp_a["pnl_man"].cumsum(), color="#2a9d8f", lw=2,
             label=f"V5-A (10F, thr=2)  final={dp_a['pnl_man'].sum():+,.0f}만")
    ax.plot(dp_b["price_date"], dp_b["pnl_man"].cumsum(), color="#e76f51", lw=2,
             label=f"V5-B (slope, thr=1) final={dp_b['pnl_man'].sum():+,.0f}만")
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.axvline(cutoff, color="purple", lw=1, ls=":", alpha=0.6, label="OOS cutoff (2024-01)")
    ax.set_title("V5 cross-tenor backtest 누적 P&L (in-sample)", fontsize=13, weight="bold")
    ax.set_ylabel("Cumulative P&L (만원)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "26_v5_cumulative.png", bbox_inches="tight")
    plt.close(fig)
    print("  OK 26_v5_cumulative.png")

    print("\n[done]")


if __name__ == "__main__":
    main()
