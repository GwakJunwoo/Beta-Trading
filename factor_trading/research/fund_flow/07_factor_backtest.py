"""
07 — 독립 팩터 전략 백테스트.

전략:
  매일 close 에서 시그널 평가:
    - fut_buy  = sign(KTB10F 외국인 5d cum)
    - cash_buy = sign(현물 외국인 sum_5d aggregate)
    - fx_weak  = sign(USDKRW past 5d 변동)  (양수 = KRW 약세, 음수 = KRW 강세)

  포지션 (10Y duration 방향, 단위: bp 효과):
    Score = +1 (long bond, yield 하락 기대)
          = -1 (short bond, yield 상승 기대)
          =  0 (중립)

  3 가지 시그널 variant 비교:
    V1) 단순 4 조합 (FX 무관)
    V2) 4 조합 × FX regime (8 조합)
    V3) (sell+sell, sell+buy) × KRW 강세 → SHORT; (buy+buy) × KRW 강세 → LONG; 나머지 0

  진입/엑싯:
    - 매일 close 진입 (T+1 부터 P&L)
    - hold 21 영업일 (overlapping positions 허용)
    - daily P&L = -position × ΔY_10Y_1d (bp 단위, long=yield 하락 시 + 익)

  성능 측정:
    - average daily P&L (bp), volatility, sharpe (annualized)
    - hit rate
    - max DD
    - 연도별 일관성
"""
from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

BETA_ROOT = Path(__file__).resolve().parents[3]
FULL_ROOT = Path(r"C:\Users\infomax\Desktop\fullstackjunior")
for p in (BETA_ROOT, FULL_ROOT, FULL_ROOT / "server"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from beta_trading.db import get_connection
from app.routers.beta import _load_label_series

FX_PATH = r"C:\Users\infomax\Desktop\USDKRW_INFOMAX.xlsx"
HOLD = 21
TRADING_DAYS = 252


def load_fx():
    df = pd.read_excel(FX_PATH, sheet_name="Sheet1", header=None, skiprows=2, usecols=[0, 1])
    df.columns = ["price_date", "usdkrw"]
    df["price_date"] = pd.to_datetime(df["price_date"], errors="coerce")
    df["usdkrw"] = pd.to_numeric(df["usdkrw"], errors="coerce")
    return df.dropna().set_index("price_date")["usdkrw"].sort_index()


def load_panel(start="2020-01-01"):
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT price_date, foreigner
            FROM ktbf_netbuy
            WHERE price_date >= %s AND tenor='KTB10F'
        """, (start,))
        f10 = pd.DataFrame(cur.fetchall()).rename(columns={"foreigner": "f10_for"})
        cur.execute("""
            SELECT price_date, foreigner
            FROM ktbf_netbuy
            WHERE price_date >= %s AND tenor='KTB3F'
        """, (start,))
        f3 = pd.DataFrame(cur.fetchall()).rename(columns={"foreigner": "f3_for"})
        cur.execute("""
            SELECT price_date, SUM(foreigner_sum_5d) AS for_s5
            FROM ktb_trade_flow_features
            WHERE price_date >= %s AND bond_code IS NOT NULL AND bond_code != ''
            GROUP BY price_date
        """, (start,))
        cash = pd.DataFrame(cur.fetchall())

    for df in (f10, f3, cash):
        df["price_date"] = pd.to_datetime(df["price_date"])
        for c in df.columns:
            if c != "price_date":
                df[c] = pd.to_numeric(df[c], errors="coerce")

    s10 = _load_label_series("10년지표", days=2200)
    s10.index = pd.to_datetime(s10.index)
    fx = load_fx()

    p = f10.merge(f3, on="price_date", how="outer").merge(cash, on="price_date", how="outer")
    p = p.sort_values("price_date").reset_index(drop=True)
    p["y_10y"] = p["price_date"].map(s10) * 100.0   # bp
    p["fx"] = p["price_date"].map(fx)
    p = p.dropna(subset=["y_10y", "fx"]).reset_index(drop=True)

    p["f10_s5"] = p["f10_for"].rolling(5, min_periods=1).sum()
    p["dy10_1d"] = p["y_10y"].diff()
    p["dfx_past_5"] = p["fx"] - p["fx"].shift(5)
    return p


# ── 시그널 generator ──
def signal_v1(row):
    """V1: 4 조합 단순 (FX 무관)
    sell+sell → -1, sell+buy → -1, buy+sell → -0.5, buy+buy → +1"""
    fb = row["f10_s5"] > 0
    cb = row["for_s5"] > 0
    if not fb and not cb:
        return -1.0    # 강한 short
    if not fb and cb:
        return -0.8    # short (sell+buy 정점 의심)
    if fb and not cb:
        return -0.3    # 약한 short
    return +1.0        # buy+buy → long


def signal_v2(row):
    """V2: 4 조합 × FX regime (KRW 강세 진행중 = 더 강한 시그널)"""
    fb = row["f10_s5"] > 0
    cb = row["for_s5"] > 0
    krw_strong = row["dfx_past_5"] < 0   # KRW 강세 진행
    if not fb and not cb:
        return -1.5 if krw_strong else -0.7   # 강력 short × 강세 정점
    if not fb and cb:
        return -1.0 if krw_strong else -0.4   # short 정점 의심
    if fb and not cb:
        return -0.3 if krw_strong else 0.0
    return +0.8 if krw_strong else +0.3        # buy+buy = 안전 carry


def signal_v3(row):
    """V3: 가장 sharp 한 conditions 만 trade (sparse signal)"""
    fb = row["f10_s5"] > 0
    cb = row["for_s5"] > 0
    krw_strong = row["dfx_past_5"] < 0
    if krw_strong and (not fb) and (not cb):
        return -2.0    # 최강 short
    if krw_strong and (not fb) and cb:
        return -1.5    # 강한 short
    if krw_strong and fb and cb:
        return +1.0    # carry long
    return 0.0


def daily_pnl_with_overlap(p: pd.DataFrame, sig_fn, hold=HOLD) -> pd.DataFrame:
    """Overlapping positions: 매일 새 포지션 진입, hold 일간 유지.

    Daily P&L = sum_{positions still alive} pos_i × (-ΔY_10Y_1d_t)
                                                    ^^ long bond gains when yield ↓
    """
    sig = p.apply(sig_fn, axis=1).fillna(0.0)
    n = len(p)
    pos_active = np.zeros(n)
    # rolling sum of last `hold` signals (각 signal 이 진입 후 hold 일간 유지)
    sig_arr = sig.values
    for i in range(n):
        # T 일 진입한 포지션은 T+1 ~ T+hold 일까지 효과 (forward holding)
        # daily P&L at date t uses sum of signals from (t-hold) to (t-1)
        lo = max(0, i - hold)
        pos_active[i] = sig_arr[lo:i].sum()
    # daily pnl: position × (-ΔY_10Y bp)
    dy = p["dy10_1d"].fillna(0.0).values
    daily_pnl = pos_active * (-dy)   # bp 단위 (per 1 unit of signal, per 1 bp yield change)
    out = p[["price_date", "y_10y", "dy10_1d"]].copy()
    out["signal"] = sig_arr
    out["position"] = pos_active
    out["pnl_bp"] = daily_pnl
    return out


def summarize(name: str, res: pd.DataFrame):
    r = res.dropna(subset=["pnl_bp"]).copy()
    r = r[r["position"] != 0]    # active 기간만
    if len(r) < 50:
        print(f"  {name}: insufficient data")
        return
    mu = r["pnl_bp"].mean()
    sd = r["pnl_bp"].std()
    sharpe = (mu / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan
    hit = (r["pnl_bp"] > 0).mean() * 100
    # cumulative
    cum = r["pnl_bp"].cumsum()
    peak = cum.cummax()
    dd = (cum - peak)
    max_dd = dd.min()
    total = cum.iloc[-1] if len(cum) else 0
    n_yrs = (r["price_date"].max() - r["price_date"].min()).days / 365.25
    per_yr = total / n_yrs if n_yrs > 0 else 0
    print(f"  {name}: N={len(r):,d}  mean={mu:+.3f}bp/d  sd={sd:.3f}  "
          f"sharpe(ann)={sharpe:+.2f}  hit={hit:.1f}%  "
          f"total={total:+.0f}bp  per_yr={per_yr:+.0f}bp/y  maxDD={max_dd:.0f}bp")


def yearly_breakdown(name: str, res: pd.DataFrame):
    r = res.dropna(subset=["pnl_bp"]).copy()
    r["year"] = r["price_date"].dt.year
    print(f"\n  ▶ 연도별 ({name})")
    yg = r.groupby("year").agg(
        N=("pnl_bp", "size"),
        total_bp=("pnl_bp", "sum"),
        sharpe=("pnl_bp", lambda x: x.mean() / x.std() * np.sqrt(TRADING_DAYS) if x.std() > 0 else np.nan),
        hit=("pnl_bp", lambda x: (x > 0).mean() * 100),
    ).round(2)
    print(yg.to_string())


def main():
    print("[load] panel ...")
    p = load_panel("2020-01-01")
    print(f"  panel: {len(p):,} rows  {p['price_date'].min().date()} ~ {p['price_date'].max().date()}\n")

    print("=" * 78)
    print("Strategy: long/short 10Y duration, daily entry, 21d hold, overlap")
    print("Unit: 1 signal unit × 1 bp yield change = 1 bp P&L")
    print("=" * 78)

    for name, fn in [("V1 (4 조합)", signal_v1),
                     ("V2 (4 조합 × FX)", signal_v2),
                     ("V3 (sparse: KRW강세 × sell모드만)", signal_v3)]:
        res = daily_pnl_with_overlap(p, fn)
        print(f"\n[{name}]")
        summarize(name, res)
        yearly_breakdown(name, res)

    # ── V3 의 signal distribution 분석 ──
    print("\n=== V3 시그널 distribution ===")
    sig = p.apply(signal_v3, axis=1)
    print(sig.value_counts().sort_index().to_string())

    # ── 단순 sign-test: 시그널 = -1 일 때 forward 21d ΔY_10Y 분포 ──
    print("\n=== Direct forward 21d test (in-sample sanity check) ===")
    p["y10_fwd_21"] = p["y_10y"].shift(-21) - p["y_10y"]
    p["sig_v3"] = sig
    by = p[p["sig_v3"] != 0].groupby("sig_v3").agg(
        n=("y10_fwd_21", "size"),
        mean_fwd_dy=("y10_fwd_21", "mean"),
        median=("y10_fwd_21", "median"),
        hit_pos=("y10_fwd_21", lambda x: (x > 0).mean() * 100),
    ).round(2)
    print(by.to_string())
    print("\n  (sig<0 일 때 mean_fwd_dy>0 이면 short 포지션 익)")

    print("\n[done]")


if __name__ == "__main__":
    main()
