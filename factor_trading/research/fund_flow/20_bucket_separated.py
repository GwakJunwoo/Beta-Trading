"""
20 — 만기별 bucket 분리:
  - V4b-10Y (재검증): KTB10F + **7-13Y bucket** 현물 외국인 + FX → ΔY_10Y
  - V4b-3Y  (재설계): KTB3F  + **2-4Y bucket**  현물 외국인 + FX → ΔY_3Y

기존 V4b 는 전체 KTB aggregate (만기 무관) 를 현물로 사용 — 분리 안 됨.
이번에 각 모델이 자기 만기 bucket 의 현물만 사용하도록 재설계.

또한 V4b-3Y 는 04~06 stage 분해에서 발견된 (buy+sell, KRW强) +11.65 bp 시그널을
활용한 별도 매핑 시도.

Plan:
  A) Bucket 별 현물 외국인 sum_5d daily aggregate 만들기
  B) 4 조합 × FX matrix 재산출 (V4b-10Y bucket-separated, V4b-3Y bucket-separated)
  C) V4b-10Y 재검증 vs original
  D) V4b-3Y 재설계 (buy+sell 시그널 활성)
  E) 백테스트 비교 + 차트
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
DV01_KTB3F = 2.8
DV01_KTB10F = 8.5
TRADING_DAYS = 252

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


def load_bucket_cash_flows(start="2020-01-01"):
    """잔존만기 bucket 별 외국인 sum_5d daily aggregate."""
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
              FROM ktb
              WHERE category='국고채' AND remain_year IS NOT NULL
              GROUP BY bond_code
            ) k ON f.bond_code = k.bond_code
            WHERE f.bond_code IS NOT NULL AND f.bond_code != ''
              AND f.price_date >= %s
            GROUP BY f.price_date, bucket
            ORDER BY f.price_date
        """, (start,))
        rows = cur.fetchall()
    df = pd.DataFrame(rows)
    df["price_date"] = pd.to_datetime(df["price_date"])
    df["for_s5"] = pd.to_numeric(df["for_s5"], errors="coerce")
    return df.pivot_table(index="price_date", columns="bucket", values="for_s5",
                           aggfunc="sum").reset_index()


def load_futures_flows(start="2020-01-01"):
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""SELECT price_date, tenor, foreigner FROM ktbf_netbuy
                       WHERE price_date >= %s AND tenor IN ('KTB3F','KTB10F')""", (start,))
        rows = cur.fetchall()
    df = pd.DataFrame(rows)
    df["price_date"] = pd.to_datetime(df["price_date"])
    df["foreigner"] = pd.to_numeric(df["foreigner"], errors="coerce").fillna(0)
    return df.pivot_table(index="price_date", columns="tenor", values="foreigner").reset_index()


def build_panel(start="2020-01-01"):
    cash = load_bucket_cash_flows(start)
    fut = load_futures_flows(start)
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

    # 5d rolling sums
    p["f3_s5"] = p["KTB3F"].rolling(5, min_periods=1).sum()
    p["f10_s5"] = p["KTB10F"].rolling(5, min_periods=1).sum()
    # bucket 별 현물 (없으면 0)
    for b in ["b3F", "b5F", "b10F", "b30F"]:
        if b not in p.columns:
            p[b] = 0.0
        p[b] = p[b].fillna(0)
    p["dfx_past_5"] = p["fx"] - p["fx"].shift(5)
    p["dy3_1d"] = p["y_3y"].diff()
    p["dy10_1d"] = p["y_10y"].diff()
    p["year"] = p["price_date"].dt.year
    for h in [3, 5, 7, 10, 14, 21]:
        p[f"dy3_fwd_{h}"] = p["y_3y"].shift(-h) - p["y_3y"]
        p[f"dy10_fwd_{h}"] = p["y_10y"].shift(-h) - p["y_10y"]
    return p


def ic(x, y):
    s = pd.DataFrame({"x": x, "y": y}).dropna()
    s = s[(s["x"] != 0) | (s["y"] != 0)]
    if len(s) < 30:
        return np.nan
    rho, _ = spearmanr(s["x"], s["y"])
    return float(rho)


# ── 4 조합 matrix ──
def combo_matrix(p, fut_col, cash_col, target_col):
    sub = p[[fut_col, cash_col, "dfx_past_5", target_col]].dropna().copy()
    sub["fut_buy"] = sub[fut_col] > 0
    sub["cash_buy"] = sub[cash_col] > 0
    sub["krw_strong"] = sub["dfx_past_5"] < 0
    return sub.groupby(["fut_buy", "cash_buy", "krw_strong"]).agg(
        n=(target_col, "size"),
        mean_dy=(target_col, "mean"),
        median_dy=(target_col, "median"),
    ).round(2)


# ── V4b-10Y (bucket-separated): same scheme, b10F cash ──
def signal_v4b_10y_bsep(row):
    """KTB10F + b10F (7-13Y bucket) 현물 + FX."""
    fb = row["f10_s5"] > 0
    cb = row["b10F"] > 0
    krw_strong = row["dfx_past_5"] < 0
    if not fb and not cb:                # SELL+SELL
        return ((-1.5 if krw_strong else -0.7), 21)
    if not fb and cb:                    # SELL+BUY
        return ((-1.0 if krw_strong else -0.4), 3)
    return (0.0, 0)


# ── V4b-3Y bucket-separated (matrix 검증 후 매핑 결정) ──
def signal_v4b_3y_bsep_v1(row):
    """Same scheme as V4b-10Y, 단 KTB3F + b3F (2-4Y bucket) 현물.
    hold: SELL+SELL 21d, SELL+BUY 3d (V4b 와 동일)."""
    fb = row["f3_s5"] > 0
    cb = row["b3F"] > 0
    krw_strong = row["dfx_past_5"] < 0
    if not fb and not cb:
        return ((-1.5 if krw_strong else -0.7), 21)
    if not fb and cb:
        return ((-1.0 if krw_strong else -0.4), 3)
    return (0.0, 0)


# ── V4b-3Y v2: (buy+sell, KRW强) 추가 활용 + hold 10d ──
def signal_v4b_3y_bsep_v2(row):
    """V4b-3Y 재설계: 04~06 의 패턴 반영.

    KTB3F: 매트릭스 결과 따라 매핑 (after 검증):
      - SELL+SELL/KRW强 → -1.0, hold=21d (강도 약함, 사이즈 작게)
      - SELL+SELL/KRW弱 → -0.4, hold=21d
      - SELL+BUY/KRW强  → -0.7, hold=10d (KTB3F 는 10d sweet spot)
      - SELL+BUY/KRW弱  → -0.5, hold=10d (KRW弱 일 때 더 강함, mean +6.26)
      - BUY+SELL/KRW强  → -1.5, hold=10d  ★ (가장 강한 +11.65 시그널)
      - BUY+SELL/KRW弱  → -0.5, hold=10d
      - BUY+BUY → flat
    """
    fb = row["f3_s5"] > 0
    cb = row["b3F"] > 0
    krw_strong = row["dfx_past_5"] < 0
    if not fb and not cb:                  # SELL+SELL
        return ((-1.0 if krw_strong else -0.4), 21)
    if not fb and cb:                      # SELL+BUY
        return ((-0.7 if krw_strong else -0.5), 10)
    if fb and not cb:                      # BUY+SELL (새 활성)
        return ((-1.5 if krw_strong else -0.5), 10)
    return (0.0, 0)


def daily_sim(p, sig_fn, target_dy1d_col):
    n = len(p)
    daily_pnl = np.zeros(n)
    daily_pos = np.zeros(n)
    dy1d = p[target_dy1d_col].fillna(0.0).values
    rows = p.to_dict("records")
    for i, row in enumerate(rows):
        s, h = sig_fn(row)
        if s == 0 or h == 0:
            continue
        for d in range(i + 1, min(i + h + 1, n)):
            daily_pnl[d] += s * (-dy1d[d])
            daily_pos[d] += s
    out = p[["price_date", "year"]].copy()
    out["pnl_bp"] = daily_pnl
    out["pos"] = daily_pos
    return out


def metrics(dp):
    r = dp[dp["pos"] != 0].copy()
    if len(r) == 0:
        return None
    mu = r["pnl_bp"].mean()
    sd = r["pnl_bp"].std()
    sh = mu / sd * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan
    cum = r["pnl_bp"].cumsum()
    mdd = (cum - cum.cummax()).min()
    total = r["pnl_bp"].sum()
    nyrs = (r["price_date"].max() - r["price_date"].min()).days / 365.25
    return {"N": len(r), "total": total, "per_yr": total / nyrs if nyrs > 0 else 0,
            "sharpe": sh, "maxDD": mdd}


def yr_table(dp):
    r = dp[dp["pos"] != 0].copy()
    return r.groupby("year")["pnl_bp"].sum().round(1)


def main():
    print("[load] panel (bucket-separated cash flows) ...")
    p = build_panel("2020-01-01")
    print(f"  {len(p):,} rows {p['price_date'].min().date()} ~ {p['price_date'].max().date()}")
    print(f"  bucket coverage (non-zero days):")
    for b in ["b3F", "b5F", "b10F", "b30F"]:
        if b in p.columns:
            print(f"    {b}: {(p[b] != 0).sum():,} days, mean abs = {p[b].abs().mean():.0f}")
    print()

    # ── A) IC by bucket ──
    print("=" * 90)
    print("A) IC by bucket (signal → forward Δy)")
    print("=" * 90)
    print(f"\n  10Y target (ΔY_10Y_21):")
    for fcol in ["f10_s5", "b10F", "b3F"]:
        for tgt in ["dy10_fwd_21", "dy3_fwd_21"]:
            r = ic(p[fcol], p[tgt])
            print(f"    {fcol:8s} -> {tgt}: IC = {r:+.3f}")
    print()

    # ── B) Matrix V4b-10Y (bucket-separated) ──
    print("=" * 90)
    print("B) V4b-10Y (bucket-separated): KTB10F + b10F + FX -> ΔY_10Y_21")
    print("=" * 90)
    m10 = combo_matrix(p, "f10_s5", "b10F", "dy10_fwd_21")
    print("\n" + m10.to_string())
    print()

    # ── C) Matrix V4b-3Y (bucket-separated) ──
    print("=" * 90)
    print("C) V4b-3Y (bucket-separated): KTB3F + b3F + FX -> ΔY_3Y_21")
    print("=" * 90)
    m3 = combo_matrix(p, "f3_s5", "b3F", "dy3_fwd_21")
    print("\n" + m3.to_string())
    print()

    # ── D) 비교: 기존 V4b (전체 aggregate) vs bucket-separated ──
    print("=" * 90)
    print("D) 기존 V4b (전체 aggregate) - 참고용")
    print("=" * 90)
    # 전체 aggregate cash flow 임시 계산
    p["total_cash_s5"] = p[["b3F", "b5F", "b10F", "b30F"]].sum(axis=1)
    m_old = combo_matrix(p, "f10_s5", "total_cash_s5", "dy10_fwd_21")
    print("\n  (KTB10F + 전체 KTB aggregate + FX -> ΔY_10Y_21)")
    print(m_old.to_string())
    print()

    # ── E) 백테스트 비교 ──
    print("=" * 90)
    print("E) 백테스트 비교")
    print("=" * 90)
    print(f"\n  {'Model':40s} {'N':>8s} {'sharpe':>8s} {'per_yr':>10s} {'total':>10s} {'maxDD':>10s}")
    print("  " + "-" * 86)
    runs = [
        ("V4b-10Y (bucket-separated)", signal_v4b_10y_bsep, "dy10_1d"),
        ("V4b-3Y v1 (bsep, V4b 스킴)",  signal_v4b_3y_bsep_v1, "dy3_1d"),
        ("V4b-3Y v2 (재설계, buy+sell)", signal_v4b_3y_bsep_v2, "dy3_1d"),
    ]
    results = {}
    for name, fn, tgt in runs:
        dp = daily_sim(p, fn, tgt)
        m = metrics(dp)
        results[name] = dp
        print(f"  {name:40s} {m['N']:>8,d} {m['sharpe']:>+8.2f} {m['per_yr']:>+10.0f} "
              f"{m['total']:>+10.0f} {m['maxDD']:>10.0f}")
    print()

    # ── F) 연도별 ──
    print("=" * 90)
    print("F) 연도별 P&L (bp)")
    print("=" * 90)
    yr_df = pd.DataFrame({name: yr_table(dp) for name, dp in results.items()})
    print(yr_df.to_string())
    print()

    # ── G) V4b-3Y v2 의 조합별 trade-attribution ──
    print("=" * 90)
    print("G) V4b-3Y v2 조합별 trade-attribution")
    print("=" * 90)
    rows = []
    for _, row in p.iterrows():
        s, h = signal_v4b_3y_bsep_v2(row)
        if s == 0 or h == 0:
            continue
        fwd = row[f"dy3_fwd_{h}"]
        if pd.isna(fwd):
            continue
        fb = row["f3_s5"] > 0
        cb = row["b3F"] > 0
        krw = row["dfx_past_5"] < 0
        fut = "BUY" if fb else "SELL"
        cash = "BUY" if cb else "SELL"
        fxr = "KRW强" if krw else "KRW弱"
        combo = f"{fut}+{cash}/{fxr}"
        rows.append({"year": row["year"], "combo": combo, "hold": h,
                      "sig": s, "fwd_dy": fwd, "pnl_bp": s * (-fwd)})
    t = pd.DataFrame(rows)
    cb_t = t.groupby(["combo", "hold"]).agg(
        N=("pnl_bp", "size"),
        hit_pct=("pnl_bp", lambda x: (x > 0).mean() * 100),
        total_bp=("pnl_bp", "sum"),
        avg_bp=("pnl_bp", "mean"),
    ).round(2)
    print(cb_t.to_string())
    print()

    # ── H) 5/11 V4b-3Y v2 시그널 ──
    print("=" * 90)
    print("H) 5/11 기준 V4b-3Y v2 시그널 (직전 15 영업일)")
    print("=" * 90)
    recent = p.tail(15)
    for _, r in recent.iterrows():
        s, h = signal_v4b_3y_bsep_v2(r)
        fb = r["f3_s5"] > 0
        cb = r["b3F"] > 0
        krw = r["dfx_past_5"] < 0
        fut = "BUY" if fb else "SELL"
        cash = "BUY" if cb else "SELL"
        fxr = "KRW强" if krw else "KRW弱"
        combo = f"{fut}+{cash}/{fxr}"
        print(f"  {r['price_date'].strftime('%Y-%m-%d')}: y3={r['y_3y']:.1f}  "
              f"f3_s5={int(r['f3_s5']):+,}  b3F_s5={int(r['b3F']):+,}  "
              f"fx5d={r['dfx_past_5']:+.1f}  combo={combo:18s} sig={s:+.2f} hold={h}d")
    print()

    # ── 차트 ──
    print("=" * 90)
    print("차트 생성 ...")
    print("=" * 90)
    CHART_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True,
                              gridspec_kw={"height_ratios": [2.5, 1]})
    for name, dp in results.items():
        r = dp.copy()
        r["cum"] = r["pnl_bp"].cumsum()
        m = metrics(dp)
        final = r["cum"].iloc[-1]
        axes[0].plot(r["price_date"], r["cum"], lw=1.8,
                      label=f"{name}: final={final:+.0f}, sharpe={m['sharpe']:+.2f}")
    axes[0].axhline(0, color="gray", lw=0.7, ls="--")
    axes[0].set_title("Bucket-separated V4b 모델 누적 P&L 비교", fontsize=13, weight="bold")
    axes[0].set_ylabel("Cumulative P&L (bp)"); axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper left")

    # drawdown of best
    best_name = "V4b-3Y v2 (재설계, buy+sell)"
    dp = results[best_name]
    r = dp[dp["pos"] != 0].copy()
    r["cum"] = r["pnl_bp"].cumsum()
    r["dd"] = r["cum"] - r["cum"].cummax()
    axes[1].fill_between(r["price_date"], 0, r["dd"], color="#e76f51", alpha=0.35)
    axes[1].plot(r["price_date"], r["dd"], color="#a8331b", lw=1.2)
    axes[1].axhline(0, color="gray", lw=0.7, ls="--")
    axes[1].set_title(f"Drawdown ({best_name}, MaxDD {r['dd'].min():.0f}bp)", fontsize=11)
    axes[1].set_ylabel("DD (bp)")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "22_bucket_separated.png", bbox_inches="tight")
    plt.close(fig)
    print("  OK 22_bucket_separated.png")

    # 연도별 비교
    fig, ax = plt.subplots(figsize=(12, 5))
    yr_df.plot(kind="bar", ax=ax, width=0.8, edgecolor="white")
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.set_title("Bucket-separated 모델 연도별 P&L (bp)", fontsize=13, weight="bold")
    ax.set_ylabel("P&L (bp)"); ax.set_xlabel("Year")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "23_bucket_yearly.png", bbox_inches="tight")
    plt.close(fig)
    print("  OK 23_bucket_yearly.png")

    print("\n[done]")


if __name__ == "__main__":
    main()
