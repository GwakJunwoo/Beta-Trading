"""
16 - RV 페어 + V4b overlay (동적 사이즈 KTB10F short).

전략 결합:
  Base: RV V2 페어 포트폴리오 (duration neutral, level ε mean revert)
  Overlay: V4b 시그널 발생일 → KTB10F short
           사이즈 = RV 포트의 net 10Y DV01 / KTB10F DV01 (8.5만원/bp)
           hold = V4b 의 hold (3d / 21d)

매일 시뮬:
  1. 활성 RV 페어 trades 의 daily P&L (mark-to-market, 만원)
  2. RV 포트의 net 10Y DV01 (만원/bp) - long bond DV01 - short bond DV01 (잔존 ≥ 7Y 만)
  3. V4b 시그널 daily 평가
  4. V4b 시그널 시 KTB10F short, 사이즈 = round(net_10Y_DV01 / 8.5)
  5. daily P&L (V4b) = -KTB10F_pos × Δy_10Y_1d_bp × 8.5만원
  6. 결합 = RV + V4b

차트:
  cumulative RV / V4b overlay / Combined
  drawdown 3종
  monthly heatmap (combined)
  scatter (RV daily vs V4b daily)
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
# Note: 일부 의존 모듈이 stdout 을 변경할 수 있어 reassign 하지 않음

BETA_ROOT = Path(__file__).resolve().parents[3]
FT_ROOT = Path(__file__).resolve().parents[2]  # factor_trading
FULL_ROOT = Path(r"C:\Users\infomax\Desktop\fullstackjunior")
for p in (BETA_ROOT, FT_ROOT, FT_ROOT / "scripts", FULL_ROOT, FULL_ROOT / "server"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from beta_trading.db import get_connection
from app.routers.beta import _load_label_series

# pair_backtest_level_optionA 가 module-level 에서 sys.stdout = io.TextIOWrapper(...) 를 호출함.
# 이 reassign 이 pipe/jupyter 환경에서 stdout 을 closed 시키므로 우회.
import io as _io
_orig_wrapper = _io.TextIOWrapper
class _NoopTextIOWrapper:
    """No-op replacement during import (prevents sys.stdout reassign side effects)."""
    def __new__(cls, *args, **kwargs):
        return sys.stdout
_io.TextIOWrapper = _NoopTextIOWrapper
try:
    from pair_backtest_level_v2 import build_universe_v2, backtest_v2
finally:
    _io.TextIOWrapper = _orig_wrapper

FX_PATH = r"C:\Users\infomax\Desktop\USDKRW_INFOMAX.xlsx"
CHART_DIR = Path(__file__).parent / "charts"
DV01_KTB10F = 8.5      # 만원/bp/계약
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


# ── V4b 시그널 panel ──
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
    fb = row["f10_s5"] > 0
    cb = row["for_s5"] > 0
    krw_strong = row["dfx_past_5"] < 0
    if not fb and not cb:
        return ((-1.5 if krw_strong else -0.7), 21)
    if not fb and cb:
        return ((-1.0 if krw_strong else -0.4), 3)
    return (0.0, 0)


# ── RV daily reconstruction ──
def reconstruct_rv_daily(trades: pd.DataFrame, ytm_panel: pd.DataFrame,
                          rem_panel: pd.DataFrame, dates: pd.DatetimeIndex,
                          tc_bp: float = 1.0,
                          remain_10y_threshold: float = 7.0):
    """
    매일 RV 페어 portfolio 의:
      - daily P&L (만원, mark-to-market)
      - net 10Y DV01 (만원/bp, long_dv01 - short_dv01 where rem >= 7)
    """
    daily_pnl = pd.Series(0.0, index=dates)
    daily_net10y = pd.Series(0.0, index=dates)
    if trades is None or trades.empty:
        return daily_pnl, daily_net10y

    for _, tr in trades.iterrows():
        l, s = tr["long_key"], tr["short_key"]
        e_date = pd.Timestamp(tr["entry_date"])
        x_date = pd.Timestamp(tr["exit_date"])
        dv01_l_won = float(tr["dv01_long_won"])
        dv01_s_won = float(tr["dv01_short_won"])
        # 활성 일자: entry+1 ~ exit
        active = dates[(dates > e_date) & (dates <= x_date)]
        if len(active) == 0:
            continue
        # daily mark-to-market
        for i, t in enumerate(active):
            # prev trading day in dates
            pos = dates.get_loc(t)
            if pos == 0:
                continue
            prev_t = dates[pos - 1]
            y_l_t = ytm_panel.loc[t, l] if (t in ytm_panel.index and l in ytm_panel.columns) else np.nan
            y_l_p = ytm_panel.loc[prev_t, l] if (prev_t in ytm_panel.index and l in ytm_panel.columns) else np.nan
            y_s_t = ytm_panel.loc[t, s] if (t in ytm_panel.index and s in ytm_panel.columns) else np.nan
            y_s_p = ytm_panel.loc[prev_t, s] if (prev_t in ytm_panel.index and s in ytm_panel.columns) else np.nan
            if not (pd.notna(y_l_t) and pd.notna(y_l_p) and pd.notna(y_s_t) and pd.notna(y_s_p)):
                continue
            dy_l = (float(y_l_t) - float(y_l_p)) * 100.0   # bp
            dy_s = (float(y_s_t) - float(y_s_p)) * 100.0
            pnl_d = -dv01_l_won * dy_l + dv01_s_won * dy_s
            daily_pnl.loc[t] += pnl_d
            # 10Y exposure (잔존 ≥ 7)
            rem_l = rem_panel.loc[t, l] if (t in rem_panel.index and l in rem_panel.columns) else np.nan
            rem_s = rem_panel.loc[t, s] if (t in rem_panel.index and s in rem_panel.columns) else np.nan
            if pd.notna(rem_l) and rem_l >= remain_10y_threshold:
                daily_net10y.loc[t] += dv01_l_won
            if pd.notna(rem_s) and rem_s >= remain_10y_threshold:
                daily_net10y.loc[t] -= dv01_s_won
        # 거래비용: exit 일
        if x_date in dates:
            cost = tc_bp * (dv01_l_won + dv01_s_won) / 2.0
            daily_pnl.loc[x_date] -= cost
    # 만원 단위로
    daily_pnl = daily_pnl / 10000.0
    daily_net10y = daily_net10y / 10000.0
    return daily_pnl, daily_net10y


# ── V4b overlay daily P&L ──
def overlay_v4b(panel: pd.DataFrame, daily_net10y_man: pd.Series,
                  mode: str = "dynamic_hedge"):
    """
    V4b 시그널 발생일 → KTB10F 진입.
      mode='dynamic_hedge'  : 사이즈 = |net 10Y DV01| / 8.5 (∝ RV 노출). 방향 = V4b sig 부호
      mode='dynamic_offset' : 사이즈 = net 10Y DV01 / 8.5 (RV 노출 부호 그대로 헤지)
                                       즉 RV 가 long bias 면 KTB10F short 으로 0 으로 만듬
                                       단 V4b 시그널 sig != 0 일 때만 활성화
      mode='fixed_1ctr'     : 사이즈 = 1 계약 (단순 baseline)
    """
    n = len(panel)
    dates = panel["price_date"].values
    dy1d = panel["dy10_1d"].fillna(0.0).values
    rows = panel.to_dict("records")
    pos_arr = np.zeros(n)        # KTB10F net 계약수 (음수=short, 양수=long)
    contract_log = []
    for i, row in enumerate(rows):
        s, h = signal_v4b(row)
        if s == 0 or h == 0:
            continue
        # 진입 시점 RV 10Y DV01 (만원/bp) → 사이즈 결정
        t = pd.Timestamp(dates[i])
        net10y = daily_net10y_man.get(t, 0.0)
        if mode == "fixed_1ctr":
            size = -np.sign(s) * 1.0   # short -1, long +1
            size_contracts = -1.0      # 1 계약 short
        elif mode == "dynamic_hedge":
            # |net 10Y| / 8.5 → 계약수, V4b 시그널 부호 따라
            n_ctr = abs(net10y) / DV01_KTB10F
            size_contracts = np.sign(s) * n_ctr   # V4b sig 부호 (음수면 short)
        elif mode == "dynamic_offset":
            # RV net10y 부호 반대로 hedge (long bias 면 short hedge)
            n_ctr = net10y / DV01_KTB10F
            size_contracts = -n_ctr   # offset, but ignore V4b sig direction
            # V4b sig 활성화 조건만 적용
        else:
            raise ValueError(mode)

        contract_log.append({"date": t, "sig": s, "hold": h,
                              "rv_net10y_man_per_bp": net10y,
                              "ktb10f_contracts": size_contracts})

        # active for next h days
        for d in range(i + 1, min(i + h + 1, n)):
            pos_arr[d] += size_contracts

    daily_pnl_v4b_man = pos_arr * (-dy1d) * DV01_KTB10F   # 만원
    out = panel[["price_date"]].copy()
    out["v4b_pos_ctr"] = pos_arr
    out["v4b_pnl_man"] = daily_pnl_v4b_man
    return out, pd.DataFrame(contract_log)


# ── Metrics ──
def perf(series: pd.Series, name: str):
    s = series.dropna()
    s_nz = s[s != 0]
    mu = s_nz.mean()
    sd = s_nz.std()
    sh = mu / sd * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan
    cum = s.cumsum()
    mdd = (cum - cum.cummax()).min()
    total = s.sum()
    days = len(s)
    nyrs = days / TRADING_DAYS
    return {
        "name": name,
        "total_man": total,
        "per_yr_man": total / nyrs if nyrs > 0 else 0,
        "sharpe": sh,
        "maxDD_man": mdd,
        "N_active": len(s_nz),
        "mean_man": mu, "std_man": sd,
    }


def main():
    print("[1/5] RV V2 universe build ...")
    u = build_universe_v2(days=2200)
    print(f"  eps shape: {u['eps_panel'].shape}, "
          f"dates: {u['eps_panel'].index.min().date()} ~ {u['eps_panel'].index.max().date()}")

    print("\n[2/5] RV V2 baseline backtest (entry=5, target=+3, stop=-3, hold=90, issue≤5y) ...")
    rv_trades = backtest_v2(
        u,
        entry_threshold=5.0, target_pnl_bp=3.0, stop_pnl_bp=-3.0,
        max_holding_days=90, max_issue_age_years=5.0,
    )
    print(f"  N_trades={len(rv_trades)}")
    if rv_trades.empty:
        print("  No trades. exit.")
        return

    print("\n[3/5] RV daily mark-to-market 재구성 + net 10Y DV01 ...")
    dates = u["eps_panel"].index
    rv_daily_man, rv_net10y_man = reconstruct_rv_daily(
        rv_trades, u["ytm_panel"], u["rem_panel"], dates
    )
    print(f"  RV daily P&L: {len(rv_daily_man):,} days  total={rv_daily_man.sum():+,.0f}만")
    print(f"  RV net 10Y DV01 (만원/bp) - mean abs: {rv_net10y_man.abs().mean():.1f}, "
          f"max abs: {rv_net10y_man.abs().max():.1f}")

    print("\n[4/5] V4b 시그널 panel 로드 + overlay ...")
    panel = load_v4b_panel("2020-01-01")
    # align panel dates ⇆ rv dates (intersect)
    panel = panel.set_index("price_date")
    common = dates.intersection(panel.index)
    print(f"  common dates: {len(common):,}")
    rv_daily_man = rv_daily_man.reindex(common).fillna(0)
    rv_net10y_man = rv_net10y_man.reindex(common).fillna(0)
    panel = panel.reindex(common).reset_index()

    # 3 변형
    print("\n  variant comparison:")
    variants = {}
    for mode in ["fixed_1ctr", "dynamic_hedge"]:
        v_daily, v_log = overlay_v4b(panel, rv_net10y_man, mode=mode)
        v_daily = v_daily.set_index("price_date")["v4b_pnl_man"].reindex(common).fillna(0)
        combined = rv_daily_man + v_daily
        variants[mode] = {
            "v4b": v_daily, "combined": combined, "log": v_log,
        }
        m_rv = perf(rv_daily_man, "RV only")
        m_v = perf(v_daily, f"V4b ({mode})")
        m_c = perf(combined, f"Combined ({mode})")
        print(f"\n  ▶ mode={mode}")
        for m in [m_rv, m_v, m_c]:
            print(f"    {m['name']:35s} total={m['total_man']:>+12,.0f}만  per_yr={m['per_yr_man']:>+10,.0f}만/y  "
                  f"sharpe={m['sharpe']:+.2f}  MDD={m['maxDD_man']:+,.0f}만")
    # correlation
    print("\n  Correlation:")
    print(f"    RV vs V4b (dynamic_hedge): {rv_daily_man.corr(variants['dynamic_hedge']['v4b']):+.3f}")
    print(f"    RV vs V4b (fixed_1ctr)   : {rv_daily_man.corr(variants['fixed_1ctr']['v4b']):+.3f}")

    print("\n[5/5] 차트 + log 저장 ...")
    CHART_DIR.mkdir(exist_ok=True)
    df_combined = pd.DataFrame({
        "date": common,
        "rv_daily_man": rv_daily_man.values,
        "rv_net10y_man_per_bp": rv_net10y_man.values,
        "v4b_dyn_man": variants["dynamic_hedge"]["v4b"].values,
        "v4b_fix_man": variants["fixed_1ctr"]["v4b"].values,
        "combined_dyn_man": variants["dynamic_hedge"]["combined"].values,
        "combined_fix_man": variants["fixed_1ctr"]["combined"].values,
    })
    df_combined.to_csv(CHART_DIR / "rv_v4b_daily.csv", index=False, encoding="utf-8-sig")

    log = variants["dynamic_hedge"]["log"]
    log.to_csv(CHART_DIR / "rv_v4b_trade_log.csv", index=False, encoding="utf-8-sig")
    print(f"  log size: {len(log):,} entries")
    if not log.empty:
        print("  log sample:")
        print(log.head(10).to_string(index=False))
        print(f"\n  KTB10F 계약수 분포 (dynamic_hedge):")
        print(log["ktb10f_contracts"].describe().round(1).to_string())

    # ── 차트들 ──
    rv_cum = rv_daily_man.cumsum()
    v_cum_dyn = variants["dynamic_hedge"]["v4b"].cumsum()
    v_cum_fix = variants["fixed_1ctr"]["v4b"].cumsum()
    cmb_cum_dyn = variants["dynamic_hedge"]["combined"].cumsum()
    cmb_cum_fix = variants["fixed_1ctr"]["combined"].cumsum()

    # 1: cumulative 비교
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(common, rv_cum, color="#e76f51", lw=2, label=f"RV V2 only (총 {rv_cum.iloc[-1]:+,.0f}만)")
    ax.plot(common, v_cum_dyn, color="#2a9d8f", lw=2, label=f"V4b overlay (dynamic) ({v_cum_dyn.iloc[-1]:+,.0f}만)")
    ax.plot(common, cmb_cum_dyn, color="#264653", lw=2.5, label=f"Combined (dynamic) ({cmb_cum_dyn.iloc[-1]:+,.0f}만)")
    ax.fill_between(common, 0, cmb_cum_dyn, alpha=0.1, color="#264653")
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.set_title("RV V2 + V4b Overlay 누적 P&L (단위: 만원)", fontsize=13, weight="bold")
    ax.set_ylabel("Cumulative P&L (만원)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)
    fig.savefig(CHART_DIR / "10_rv_v4b_cumulative.png", bbox_inches="tight")
    plt.close(fig)
    print("  OK 10_rv_v4b_cumulative.png")

    # 2: net 10Y DV01 시계열
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(common, rv_net10y_man, color="#e76f51", lw=1.2)
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.fill_between(common, 0, rv_net10y_man,
                     where=rv_net10y_man > 0, color="#2a9d8f", alpha=0.3, label="long 10Y bias")
    ax.fill_between(common, 0, rv_net10y_man,
                     where=rv_net10y_man < 0, color="#e76f51", alpha=0.3, label="short 10Y bias")
    ax.set_title("RV 포트의 net 10Y DV01 (만원/bp) 시계열", fontsize=13, weight="bold")
    ax.set_ylabel("net 10Y DV01 (만원/bp)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(CHART_DIR / "11_rv_net10y_dv01.png", bbox_inches="tight")
    plt.close(fig)
    print("  OK 11_rv_net10y_dv01.png")

    # 3: drawdown 비교
    fig, ax = plt.subplots(figsize=(13, 4.5))
    for cum, color, name in [
        (rv_cum, "#e76f51", "RV only"),
        (cmb_cum_dyn, "#264653", "Combined (dynamic)"),
    ]:
        dd = cum - cum.cummax()
        ax.fill_between(common, 0, dd, alpha=0.25, color=color)
        ax.plot(common, dd, color=color, lw=1.2, label=f"{name} (MDD {dd.min():,.0f}만)")
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.set_title("Drawdown 비교", fontsize=13, weight="bold")
    ax.set_ylabel("Drawdown (만원)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.savefig(CHART_DIR / "12_rv_v4b_drawdown.png", bbox_inches="tight")
    plt.close(fig)
    print("  OK 12_rv_v4b_drawdown.png")

    # 4: scatter
    fig, ax = plt.subplots(figsize=(7, 6))
    v_daily = variants["dynamic_hedge"]["v4b"]
    mask = (rv_daily_man != 0) | (v_daily != 0)
    ax.scatter(rv_daily_man[mask], v_daily[mask], alpha=0.4, s=12, color="#264653")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    corr = rv_daily_man.corr(v_daily)
    ax.set_title(f"Daily P&L: RV vs V4b (corr={corr:+.3f})", fontsize=12, weight="bold")
    ax.set_xlabel("RV daily P&L (만원)")
    ax.set_ylabel("V4b overlay daily P&L (만원)")
    ax.grid(alpha=0.3)
    fig.savefig(CHART_DIR / "13_rv_v4b_scatter.png", bbox_inches="tight")
    plt.close(fig)
    print("  OK 13_rv_v4b_scatter.png")

    print("\n[done]")


if __name__ == "__main__":
    main()
