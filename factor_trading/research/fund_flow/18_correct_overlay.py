"""
18 — 사용자 의도에 정확히 맞춘 RV + V4b overlay 재검토.

진입 조건 (둘 다 만족):
  (A) RV 가 flattener bias: net 10Y DV01 ≥ threshold (만원/bp, 기본 50)
  (B) V4b SHORT 시그널 발생 (|sig| ≥ 1.0 — 강한 시그널만; weak 시그널 -0.4 등은 제외)

진입 액션:
  KTB10F short, 사이즈 = floor(RV_net_10Y_DV01 / KTB10F_DV01)
  --> 전체 portfolio 의 net 10Y DV01 = 0 (curve neutral)

거래비용:
  KTB10F round trip 0.12 bp × DV01 × 사이즈
  RV 페어: 1bp/round trip (기존 backtest 에 내재)

또한 검증:
  - RV V2 단독 backtest 의 정확한 사이즈/per_yr/sharpe (trade-level + daily 둘 다)
  - 동시 평균 active face (RV 의 "5000억 델타" 와 비교)
  - V4b overlay 의 marginal contribution
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BETA_ROOT = Path(__file__).resolve().parents[3]
FT_ROOT = Path(__file__).resolve().parents[2]
FULL_ROOT = Path(r"C:\Users\infomax\Desktop\fullstackjunior")
for p in (BETA_ROOT, FT_ROOT, FT_ROOT / "scripts", FULL_ROOT, FULL_ROOT / "server"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from beta_trading.db import get_connection
from app.routers.beta import _load_label_series

# Bypass stdout reassign
import io as _io
_orig_wrapper = _io.TextIOWrapper
class _Noop:
    def __new__(cls, *args, **kwargs):
        return sys.stdout
_io.TextIOWrapper = _Noop
try:
    from pair_backtest_level_v2 import build_universe_v2, backtest_v2, summarize_v2
finally:
    _io.TextIOWrapper = _orig_wrapper

FX_PATH = r"C:\Users\infomax\Desktop\USDKRW_INFOMAX.xlsx"
DV01_KTB10F = 8.5     # 만원/bp/계약
TC_KTB10F_BP = 0.12   # round trip 비용 bp (per contract)
TRADING_DAYS = 252


def load_fx():
    df = pd.read_excel(FX_PATH, sheet_name="Sheet1", header=None, skiprows=2, usecols=[0, 1])
    df.columns = ["price_date", "usdkrw"]
    df["price_date"] = pd.to_datetime(df["price_date"], errors="coerce")
    df["usdkrw"] = pd.to_numeric(df["usdkrw"], errors="coerce")
    return df.dropna().set_index("price_date")["usdkrw"].sort_index()


def load_v4b_panel(start="2020-01-01"):
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""SELECT price_date, foreigner FROM ktbf_netbuy
                       WHERE price_date >= %s AND tenor='KTB10F'""", (start,))
        f10 = pd.DataFrame(cur.fetchall()).rename(columns={"foreigner": "f10_for"})
        cur.execute("""SELECT price_date, SUM(foreigner_sum_5d) AS for_s5
                       FROM ktb_trade_flow_features
                       WHERE price_date >= %s AND bond_code IS NOT NULL AND bond_code != ''
                       GROUP BY price_date""", (start,))
        cash = pd.DataFrame(cur.fetchall())
    for df in (f10, cash):
        df["price_date"] = pd.to_datetime(df["price_date"])
        for c in df.columns:
            if c != "price_date":
                df[c] = pd.to_numeric(df[c], errors="coerce")
    s10 = _load_label_series("10년지표", days=2200)
    s10.index = pd.to_datetime(s10.index)
    fx = load_fx()
    p = f10.merge(cash, on="price_date", how="outer").sort_values("price_date").reset_index(drop=True)
    p["y_10y"] = p["price_date"].map(s10) * 100.0
    p["fx"] = p["price_date"].map(fx)
    p = p.dropna(subset=["y_10y", "fx"]).reset_index(drop=True)
    p["f10_s5"] = p["f10_for"].rolling(5, min_periods=1).sum()
    p["dfx_past_5"] = p["fx"] - p["fx"].shift(5)
    p["dy10_1d"] = p["y_10y"].diff()
    return p


def signal_v4b(row):
    """V4b: returns (sig, hold)."""
    fb = row["f10_s5"] > 0
    cb = row["for_s5"] > 0
    krw_strong = row["dfx_past_5"] < 0
    if not fb and not cb:
        return ((-1.5 if krw_strong else -0.7), 21)
    if not fb and cb:
        return ((-1.0 if krw_strong else -0.4), 3)
    return (0.0, 0)


def reconstruct_rv_daily(trades, ytm_panel, rem_panel, dates, tc_bp=1.0,
                          remain_10y_threshold=7.0):
    """RV daily P&L (만원) + net 10Y DV01 (만원/bp) + active face sum (억)."""
    daily_pnl = pd.Series(0.0, index=dates)
    daily_net10y = pd.Series(0.0, index=dates)
    daily_active_face_eok = pd.Series(0.0, index=dates)
    daily_pair_count = pd.Series(0, index=dates)
    if trades is None or trades.empty:
        return daily_pnl, daily_net10y, daily_active_face_eok, daily_pair_count
    for _, tr in trades.iterrows():
        l, s = tr["long_key"], tr["short_key"]
        e_date = pd.Timestamp(tr["entry_date"])
        x_date = pd.Timestamp(tr["exit_date"])
        dv01_l = float(tr["dv01_long_won"])
        dv01_s = float(tr["dv01_short_won"])
        face_l_eok = float(tr["long_face_eok"])
        face_s_eok = float(tr["short_face_eok"])
        active = dates[(dates > e_date) & (dates <= x_date)]
        for t in active:
            pos = dates.get_loc(t)
            if pos == 0:
                continue
            prev_t = dates[pos - 1]
            try:
                y_l_t = ytm_panel.loc[t, l]; y_l_p = ytm_panel.loc[prev_t, l]
                y_s_t = ytm_panel.loc[t, s]; y_s_p = ytm_panel.loc[prev_t, s]
            except Exception:
                continue
            if not all(pd.notna(v) for v in [y_l_t, y_l_p, y_s_t, y_s_p]):
                continue
            dy_l = (float(y_l_t) - float(y_l_p)) * 100.0
            dy_s = (float(y_s_t) - float(y_s_p)) * 100.0
            daily_pnl.loc[t] += -dv01_l * dy_l + dv01_s * dy_s
            daily_active_face_eok.loc[t] += (face_l_eok + face_s_eok)
            daily_pair_count.loc[t] += 1
            # net 10Y DV01 (만원/bp): 잔존 ≥ 7 인 leg 의 DV01 (long: +, short: -)
            try:
                rem_l = rem_panel.loc[t, l]; rem_s = rem_panel.loc[t, s]
            except Exception:
                rem_l = rem_s = np.nan
            if pd.notna(rem_l) and rem_l >= remain_10y_threshold:
                daily_net10y.loc[t] += dv01_l
            if pd.notna(rem_s) and rem_s >= remain_10y_threshold:
                daily_net10y.loc[t] -= dv01_s
        if x_date in dates:
            daily_pnl.loc[x_date] -= tc_bp * (dv01_l + dv01_s) / 2.0
    return (daily_pnl / 10000.0,         # 만원
            daily_net10y / 10000.0,      # 만원/bp
            daily_active_face_eok,        # 억
            daily_pair_count)


def overlay_v4b_strict(panel, rv_net10y_man_per_bp,
                        flattener_threshold_man=50.0,   # net 10Y DV01 ≥ 50만원/bp 면 flattener
                        sig_strength_min=1.0,            # |sig| ≥ 1.0 강한 시그널만
                        with_cost=True):
    """
    정확한 진입 조건:
      RV net 10Y DV01 ≥ threshold (flattener) AND |V4b sig| ≥ sig_strength_min AND V4b SHORT
      --> KTB10F short, 사이즈 = floor(RV_net_10Y / DV01_KTB10F)
    """
    n = len(panel)
    dy1d = panel["dy10_1d"].fillna(0.0).values
    rows = panel.to_dict("records")
    pos_ctr = np.zeros(n)
    log = []
    cost_total = 0.0
    for i, row in enumerate(rows):
        s, h = signal_v4b(row)
        if s == 0 or h == 0:
            continue
        if abs(s) < sig_strength_min:
            continue
        t = pd.Timestamp(row["price_date"])
        net10y = rv_net10y_man_per_bp.get(t, 0.0)
        if net10y < flattener_threshold_man:
            continue   # RV 가 flattener 아님 --> skip
        # 사이즈: RV 노출을 0 으로 만드는 KTB10F short
        size_ctr = -round(net10y / DV01_KTB10F)
        if size_ctr == 0:
            continue
        log.append({"date": t, "sig": s, "hold": h,
                    "rv_net10y_man": net10y,
                    "ktb10f_size": size_ctr})
        # 거래비용 (entry + exit)
        if with_cost:
            cost = abs(size_ctr) * TC_KTB10F_BP * DV01_KTB10F   # 만원
            cost_total += cost
        for d in range(i + 1, min(i + h + 1, n)):
            pos_ctr[d] += size_ctr
    daily = pos_ctr * (-dy1d) * DV01_KTB10F   # 만원
    # 비용을 균등 분배 (단순화)
    if with_cost and len(log):
        daily_cost = cost_total / max(1, (pos_ctr != 0).sum())
        daily = np.where(pos_ctr != 0, daily - daily_cost, daily)
    out = pd.Series(daily, index=pd.to_datetime(panel["price_date"]))
    return out, pd.DataFrame(log), cost_total


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


def main():
    print("[1/5] RV V2 backtest (정확한 사이즈 base 확인) ...")
    u = build_universe_v2(days=2200)
    rv = backtest_v2(u, entry_threshold=5.0, target_pnl_bp=3.0, stop_pnl_bp=-3.0,
                      max_holding_days=90, max_issue_age_years=5.0)
    print(f"  N_trades={len(rv)}")
    s = summarize_v2(rv)
    print(f"  summarize_v2:")
    print(f"    per_yr_man = {s.get('per_yr_man'):,.0f}만/y (사이즈는 페어당 face_l + face_s)")
    print(f"    mean_man    = {s.get('mean_man'):,.0f}만/trade")
    print(f"    win_pct     = {s.get('win_pct'):.1f}%")
    print(f"    sharpe(trade) = {s.get('sharpe'):.3f}  (★ trade-level, NOT daily annualized)")
    print(f"    hold_d      = {s.get('hold_d'):.1f}일")
    print()

    # face size 분포
    if not rv.empty:
        face_sum = (rv["long_face_eok"] + rv["short_face_eok"])
        print(f"  페어당 face (long+short, 억): "
              f"mean={face_sum.mean():.0f}, median={face_sum.median():.0f}, "
              f"min={face_sum.min():.0f}, max={face_sum.max():.0f}")
        avg_dv01_per_trade = (rv["dv01_long_won"] + rv["dv01_short_won"]) / 2 / 1e4
        print(f"  페어당 avg DV01 (만원/bp): mean={avg_dv01_per_trade.mean():.0f}, "
              f"min={avg_dv01_per_trade.min():.0f}, max={avg_dv01_per_trade.max():.0f}")
    print()

    print("[2/5] daily reconstruction + 평균 동시 active face/pair ...")
    dates = u["eps_panel"].index
    rv_daily, rv_net10y, active_face, pair_count = reconstruct_rv_daily(
        rv, u["ytm_panel"], u["rem_panel"], dates
    )
    active_mask = pair_count > 0
    print(f"  RV daily P&L: total = {rv_daily.sum():+,.0f}만 over {len(rv_daily):,} days")
    print(f"  active days  : {active_mask.sum():,} / {len(dates):,} "
          f"({active_mask.mean()*100:.1f}%)")
    print(f"  평균 동시 active 페어 (활성일만): {pair_count[active_mask].mean():.2f}")
    print(f"  평균 동시 active face (활성일만): {active_face[active_mask].mean():.0f}억")
    print(f"  최대 동시 active face            : {active_face.max():.0f}억")
    nyrs = len(dates) / TRADING_DAYS
    per_yr_man = rv_daily.sum() / nyrs
    print(f"  per_yr_man (daily-based)         : {per_yr_man:+,.0f}만/y")
    # daily-based sharpe annualized
    s_nz = rv_daily[rv_daily != 0]
    sh_daily = s_nz.mean() / s_nz.std() * np.sqrt(TRADING_DAYS) if len(s_nz) > 1 else np.nan
    print(f"  sharpe (daily, annualized)       : {sh_daily:+.2f}")
    print()
    print("  -> 비교:")
    print(f"    summarize_v2 의 sharpe (trade-level): {s.get('sharpe'):+.2f}")
    print(f"    daily annualized sharpe              : {sh_daily:+.2f}")
    print(f"    --> 두 정의가 다름. trade-level 은 단일 trade variance 기준 (long horizon).")
    print()

    print("[3/5] RV net 10Y DV01 분포 (잔존 ≥7Y leg DV01 sum, long+ / short-) ...")
    nz = rv_net10y[rv_net10y != 0]
    if len(nz):
        print(f"  N nonzero days: {len(nz):,}")
        print(f"  분포 (만원/bp):")
        print(f"    mean abs : {nz.abs().mean():.1f}")
        print(f"    mean signed: {nz.mean():+.1f}")
        print(f"    >0 (flattener): {(nz > 0).sum():,} days ({(nz > 0).mean()*100:.1f}%)")
        print(f"    <0 (steepener): {(nz < 0).sum():,} days ({(nz < 0).mean()*100:.1f}%)")
        print(f"    threshold ≥50만/bp (의미있는 flattener): {(nz >= 50).sum():,} days")
        print(f"    threshold ≥100만/bp: {(nz >= 100).sum():,} days")
        print(f"    threshold ≥200만/bp: {(nz >= 200).sum():,} days")
    print()

    print("[4/5] V4b strict overlay (조건 두 가지 만족 시만) ...")
    panel = load_v4b_panel("2020-01-01")
    common = dates.intersection(pd.to_datetime(panel["price_date"]))
    rv_daily = rv_daily.reindex(common).fillna(0)
    rv_net10y = rv_net10y.reindex(common).fillna(0)
    panel = panel.set_index("price_date").reindex(common).reset_index()

    # 조건 sweep: threshold 50, 100, 200, sig_min 1.0 / 0.4
    print(f"\n  {'flattener_thr':>15s} {'sig_min':>8s} {'V4b N_trades':>14s} "
          f"{'V4b per_yr':>12s} {'V4b sharpe':>12s} {'Combined per_yr':>16s} "
          f"{'Combined sharpe':>16s} {'Combined MDD':>15s}")
    rows = []
    for thr in [50.0, 100.0, 200.0]:
        for sig_min in [1.0, 0.4]:
            v_daily, log, cost = overlay_v4b_strict(
                panel, rv_net10y,
                flattener_threshold_man=thr, sig_strength_min=sig_min, with_cost=True)
            combined = rv_daily + v_daily
            m_v = perf(v_daily, f"V4b thr={thr} sig≥{sig_min}")
            m_c = perf(combined, f"Combined thr={thr} sig≥{sig_min}")
            rows.append({
                "thr": thr, "sig_min": sig_min, "N_v4b_entries": len(log),
                "v4b_per_yr": m_v["per_yr"], "v4b_sharpe": m_v["sharpe"],
                "comb_per_yr": m_c["per_yr"], "comb_sharpe": m_c["sharpe"],
                "comb_mdd": m_c["mdd"], "cost_total": cost,
                "v4b_total": m_v["total"], "v4b_mdd": m_v["mdd"],
            })
            print(f"  {thr:>15.0f} {sig_min:>8.1f} {len(log):>14d} "
                  f"{m_v['per_yr']:>+12,.0f} {m_v['sharpe']:>+12.2f} "
                  f"{m_c['per_yr']:>+16,.0f} {m_c['sharpe']:>+16.2f} {m_c['mdd']:>+15,.0f}")

    m_rv = perf(rv_daily, "RV only")
    print(f"\n  Baseline RV only:")
    print(f"    per_yr={m_rv['per_yr']:+,.0f}만  sharpe={m_rv['sharpe']:+.2f}  MDD={m_rv['mdd']:+,.0f}만")

    print()
    print("[5/5] 사용자 의문 '5,000억 델타 = 7.x억' 환산 ...")
    # 5,000억 face base 환산: 우리 평균 동시 face 대비 scale factor
    avg_face = active_face[active_mask].mean()
    scale_5000 = 5000 / avg_face if avg_face > 0 else 0
    scaled_per_yr = m_rv["per_yr"] * scale_5000
    print(f"  우리 RV 평균 동시 face: {avg_face:.0f}억")
    print(f"  scale factor to 5,000억: ×{scale_5000:.2f}")
    print(f"  per_yr × scale = {scaled_per_yr:+,.0f}만 = {scaled_per_yr/1e4:+.2f}억/y")
    print(f"  사용자 기억치 (7.x억) 와 비교: {'근접' if abs(scaled_per_yr/1e4 - 7) < 3 else '괴리'}")
    print()
    print("[done]")


if __name__ == "__main__":
    main()
