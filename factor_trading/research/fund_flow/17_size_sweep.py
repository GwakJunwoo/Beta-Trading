"""
17 — V4b 사이즈 매칭 시뮬 (KTB10F N 계약 short, RV 와 결합).

사용자 의문: "RV 100억 base vs V4b 1 계약 = 100배 차이인데 어떻게 P&L 비슷하냐?"
→ V4b 의 IR (per_yr / DV01) 가 RV 의 40배. 작은 사이즈에도 큰 alpha.
→ 사이즈 매칭 시 V4b 단독 P&L 이 RV 의 ~4 배가 됨 (계산상)

본 스크립트:
  V4b 사이즈 sweep (1, 10, 24, 50, 100 계약) × RV combined 비교
  각 사이즈 별 sharpe, per_yr, MDD
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
FT_ROOT = Path(__file__).resolve().parents[2]
FULL_ROOT = Path(r"C:\Users\infomax\Desktop\fullstackjunior")
for p in (BETA_ROOT, FT_ROOT, FT_ROOT / "scripts", FULL_ROOT, FULL_ROOT / "server"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from beta_trading.db import get_connection
from app.routers.beta import _load_label_series

# Bypass pair_backtest_level_optionA stdout reassign
import io as _io
_orig_wrapper = _io.TextIOWrapper
class _NoopWrapper:
    def __new__(cls, *args, **kwargs):
        return sys.stdout
_io.TextIOWrapper = _NoopWrapper
try:
    from pair_backtest_level_v2 import build_universe_v2, backtest_v2
finally:
    _io.TextIOWrapper = _orig_wrapper

FX_PATH = r"C:\Users\infomax\Desktop\USDKRW_INFOMAX.xlsx"
CHART_DIR = Path(__file__).parent / "charts"
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


def reconstruct_rv_daily(trades, ytm_panel, rem_panel, dates, tc_bp=1.0):
    daily_pnl = pd.Series(0.0, index=dates)
    if trades is None or trades.empty:
        return daily_pnl
    for _, tr in trades.iterrows():
        l, s = tr["long_key"], tr["short_key"]
        e_date = pd.Timestamp(tr["entry_date"])
        x_date = pd.Timestamp(tr["exit_date"])
        dv01_l = float(tr["dv01_long_won"])
        dv01_s = float(tr["dv01_short_won"])
        active = dates[(dates > e_date) & (dates <= x_date)]
        for t in active:
            pos = dates.get_loc(t)
            if pos == 0:
                continue
            prev_t = dates[pos - 1]
            try:
                y_l_t = ytm_panel.loc[t, l]
                y_l_p = ytm_panel.loc[prev_t, l]
                y_s_t = ytm_panel.loc[t, s]
                y_s_p = ytm_panel.loc[prev_t, s]
            except Exception:
                continue
            if not all(pd.notna(v) for v in [y_l_t, y_l_p, y_s_t, y_s_p]):
                continue
            dy_l = (float(y_l_t) - float(y_l_p)) * 100.0
            dy_s = (float(y_s_t) - float(y_s_p)) * 100.0
            daily_pnl.loc[t] += -dv01_l * dy_l + dv01_s * dy_s
        if x_date in dates:
            daily_pnl.loc[x_date] -= tc_bp * (dv01_l + dv01_s) / 2.0
    return daily_pnl / 10000.0   # 만원


def v4b_daily_pnl_manwon(panel, n_contracts=1.0, scale_by_sig_strength=False):
    """KTB10F 매일 short, hold 따라 active. P&L 만원."""
    n = len(panel)
    dy1d = panel["dy10_1d"].fillna(0.0).values
    rows = panel.to_dict("records")
    pos_ctr = np.zeros(n)
    for i, row in enumerate(rows):
        s, h = signal_v4b(row)
        if s == 0 or h == 0:
            continue
        # sig 부호로 방향 (모두 음수 short, 그러므로 +n_contracts short → pos = -n_contracts)
        # scale_by_sig_strength=True 면 |s| 곱
        size = np.sign(s) * n_contracts * (abs(s) if scale_by_sig_strength else 1.0)
        for d in range(i + 1, min(i + h + 1, n)):
            pos_ctr[d] += size
    daily = pos_ctr * (-dy1d) * DV01_KTB10F   # 만원
    return pd.Series(daily, index=pd.to_datetime(panel["price_date"]))


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
            "sharpe": sh, "mdd": mdd, "N": len(s_nz)}


def main():
    print("[1/4] RV V2 backtest ...")
    u = build_universe_v2(days=2200)
    rv = backtest_v2(u, entry_threshold=5.0, target_pnl_bp=3.0, stop_pnl_bp=-3.0,
                      max_holding_days=90, max_issue_age_years=5.0)
    print(f"  N_trades={len(rv)}")

    print("\n[2/4] RV daily reconstruction ...")
    rv_daily = reconstruct_rv_daily(rv, u["ytm_panel"], u["rem_panel"], u["eps_panel"].index)
    print(f"  RV daily P&L: total={rv_daily.sum():+,.0f}만")

    print("\n[3/4] V4b panel + size sweep ...")
    panel = load_v4b_panel("2020-01-01")
    common = u["eps_panel"].index.intersection(pd.to_datetime(panel["price_date"]))
    rv_daily = rv_daily.reindex(common).fillna(0)
    panel = panel.set_index("price_date").reindex(common).reset_index()

    sizes = [1, 10, 24, 50, 100]
    summary = []
    cumulatives = {}
    m_rv = perf(rv_daily, "RV only")
    summary.append({"sizing": "-", "n_ctr": "-", **m_rv})
    cumulatives["RV only"] = rv_daily.cumsum()
    for n_ctr in sizes:
        v_daily = v4b_daily_pnl_manwon(panel, n_contracts=n_ctr).reindex(common).fillna(0)
        combined = rv_daily + v_daily
        m_v = perf(v_daily, f"V4b {n_ctr}계약")
        m_c = perf(combined, f"Combined +V4b {n_ctr}")
        summary.extend([
            {"sizing": "V4b", "n_ctr": n_ctr, **m_v},
            {"sizing": "Combined", "n_ctr": n_ctr, **m_c},
        ])
        cumulatives[f"V4b {n_ctr}"] = v_daily.cumsum()
        cumulatives[f"Combined +V4b {n_ctr}"] = combined.cumsum()

    # 출력 표
    print("\n" + "=" * 100)
    print(f"{'Strategy':35s} {'Total(만)':>12s} {'Per_yr(만)':>12s} {'Sharpe':>8s} {'MDD(만)':>12s}")
    print("=" * 100)
    for s in summary:
        nm = s["name"]
        print(f"{nm:35s} {s['total']:>+12,.0f} {s['per_yr']:>+12,.0f} {s['sharpe']:>+8.2f} {s['mdd']:>+12,.0f}")

    print("\n[4/4] 차트 ...")
    CHART_DIR.mkdir(exist_ok=True)

    # cumulative comparison: RV vs Combined (각 사이즈)
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.plot(common, cumulatives["RV only"], color="#e76f51", lw=2.5,
             label=f"RV only ({cumulatives['RV only'].iloc[-1]:+,.0f}만)")
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sizes)))
    for n_ctr, c in zip(sizes, colors):
        cum = cumulatives[f"Combined +V4b {n_ctr}"]
        ax.plot(common, cum, color=c, lw=1.8,
                 label=f"Combined +V4b {n_ctr}계약 ({cum.iloc[-1]:+,.0f}만)")
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.set_title("RV + V4b 사이즈별 누적 P&L 비교", fontsize=13, weight="bold")
    ax.set_ylabel("Cumulative P&L (만원)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(CHART_DIR / "14_size_sweep_cumulative.png", bbox_inches="tight")
    plt.close(fig)
    print("  OK 14_size_sweep_cumulative.png")

    # Sharpe / MDD 사이즈 바차트
    df = pd.DataFrame([s for s in summary if s["sizing"] == "Combined"])
    df["n_ctr"] = df["n_ctr"].astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(df["n_ctr"], df["per_yr"], "o-", color="#264653", lw=2, ms=8)
    axes[0].axhline(m_rv["per_yr"], color="#e76f51", ls="--", label=f"RV only {m_rv['per_yr']:,.0f}")
    axes[0].set_title("Per_yr P&L (만원)", weight="bold")
    axes[0].set_xlabel("V4b 사이즈 (계약)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(df["n_ctr"], df["sharpe"], "o-", color="#2a9d8f", lw=2, ms=8)
    axes[1].axhline(m_rv["sharpe"], color="#e76f51", ls="--", label=f"RV only {m_rv['sharpe']:.2f}")
    axes[1].set_title("Sharpe", weight="bold")
    axes[1].set_xlabel("V4b 사이즈 (계약)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(df["n_ctr"], df["mdd"], "o-", color="#e76f51", lw=2, ms=8)
    axes[2].axhline(m_rv["mdd"], color="#e76f51", ls="--", alpha=0.5,
                     label=f"RV only {m_rv['mdd']:,.0f}")
    axes[2].set_title("Max Drawdown (만원)", weight="bold")
    axes[2].set_xlabel("V4b 사이즈 (계약)")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    fig.suptitle("V4b 사이즈에 따른 Combined 성능", fontsize=13, weight="bold")
    fig.savefig(CHART_DIR / "15_size_sweep_metrics.png", bbox_inches="tight")
    plt.close(fig)
    print("  OK 15_size_sweep_metrics.png")

    print("\n[done]")


if __name__ == "__main__":
    main()
