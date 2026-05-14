"""
15 — V4b 최종 백테스트 + 보고서용 차트 생성.

V4b spec:
  SELL+SELL/KRW强 → -1.5, hold=21d
  SELL+SELL/KRW弱 → -0.7, hold=21d
  SELL+BUY/KRW强  → -1.0, hold=3d
  SELL+BUY/KRW弱  → -0.4, hold=3d
  BUY+SELL, BUY+BUY → 제거 (short-only)

차트 출력 (charts/):
  01_cumulative_pnl.png     : 누적 P&L 시계열
  02_yearly_pnl.png         : 연도별 bar chart
  03_drawdown.png           : drawdown timeline
  04_monthly_heatmap.png    : 월별 returns heatmap
  05_trade_pnl_hist.png     : trade P&L 분포
  06_metrics_summary.png    : 핵심 메트릭스 표
  07_pnl_vs_y10.png         : 누적 P&L vs Y_10Y
  08_signal_breakdown.png   : 시그널/조합별 P&L
"""
from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

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
CHART_DIR = Path(__file__).parent / "charts"
TRADING_DAYS = 252
DV01 = 8.5  # 만원/bp/계약

# Korean font
for fname in ["Malgun Gothic", "NanumGothic", "AppleGothic"]:
    try:
        plt.rcParams["font.family"] = fname
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 140
plt.rcParams["savefig.bbox"] = "tight"


def load_fx():
    df = pd.read_excel(FX_PATH, sheet_name="Sheet1", header=None, skiprows=2, usecols=[0, 1])
    df.columns = ["price_date", "usdkrw"]
    df["price_date"] = pd.to_datetime(df["price_date"], errors="coerce")
    df["usdkrw"] = pd.to_numeric(df["usdkrw"], errors="coerce")
    return df.dropna().set_index("price_date")["usdkrw"].sort_index()


def load_panel(start="2020-01-01"):
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
    p["year"] = p["price_date"].dt.year
    for h in [3, 21]:
        p[f"dy10_fwd_{h}"] = p["y_10y"].shift(-h) - p["y_10y"]
    return p


def classify_combo(row):
    fb = row["f10_s5"] > 0
    cb = row["for_s5"] > 0
    krw_strong = row["dfx_past_5"] < 0
    fut = "BUY" if fb else "SELL"
    cash = "BUY" if cb else "SELL"
    fx = "KRW强" if krw_strong else "KRW弱"
    return f"{fut}+{cash}/{fx}"


def signal_v4b(row):
    fb = row["f10_s5"] > 0
    cb = row["for_s5"] > 0
    krw_strong = row["dfx_past_5"] < 0
    if not fb and not cb:                  # SELL+SELL
        return ((-1.5 if krw_strong else -0.7), 21)
    if not fb and cb:                      # SELL+BUY
        return ((-1.0 if krw_strong else -0.4), 3)
    return (0.0, 0)


def daily_simulation(p):
    n = len(p)
    daily_pnl = np.zeros(n)
    daily_pos = np.zeros(n)
    dy1d = p["dy10_1d"].fillna(0.0).values
    rows = p.to_dict("records")
    for i, row in enumerate(rows):
        s, h = signal_v4b(row)
        if s == 0 or h == 0:
            continue
        for d in range(i + 1, min(i + h + 1, n)):
            daily_pnl[d] += s * (-dy1d[d])
            daily_pos[d] += s
    out = p[["price_date", "year", "y_10y", "fx"]].copy()
    out["pnl_bp"] = daily_pnl
    out["pos"] = daily_pos
    out["cum_pnl"] = daily_pnl.cumsum()
    out["peak"] = out["cum_pnl"].cummax()
    out["drawdown"] = out["cum_pnl"] - out["peak"]
    return out


def trade_attribution(p):
    rows = []
    for _, row in p.iterrows():
        s, h = signal_v4b(row)
        if s == 0 or h == 0:
            continue
        fwd = row[f"dy10_fwd_{h}"]
        if pd.isna(fwd):
            continue
        rows.append({
            "price_date": row["price_date"],
            "year": row["year"],
            "sig": s, "hold": h,
            "combo": row["combo"],
            "y_entry": row["y_10y"],
            "fwd_dy": fwd,
            "pnl_bp": s * (-fwd),
        })
    return pd.DataFrame(rows)


# ── Plot functions ──
def plot_cumulative_pnl(dp, out):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.fill_between(dp["price_date"], 0, dp["cum_pnl"],
                     where=dp["cum_pnl"] >= 0, alpha=0.25, color="#2a9d8f")
    ax.fill_between(dp["price_date"], 0, dp["cum_pnl"],
                     where=dp["cum_pnl"] < 0, alpha=0.25, color="#e76f51")
    ax.plot(dp["price_date"], dp["cum_pnl"], color="#264653", lw=1.8)
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.set_title("V4b 누적 P&L (단위: bp, 1 unit × 1bp = 1bp)", fontsize=13, weight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative P&L (bp)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.3)
    # final marker
    final = dp.iloc[-1]
    ax.annotate(f"  Final: +{final['cum_pnl']:.0f} bp",
                xy=(final["price_date"], final["cum_pnl"]),
                fontsize=11, color="#264653", weight="bold",
                xytext=(8, 0), textcoords="offset points", va="center")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_yearly_pnl(dp, out):
    yr = dp.groupby("year")["pnl_bp"].sum()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2a9d8f" if v >= 0 else "#e76f51" for v in yr.values]
    bars = ax.bar(yr.index.astype(str), yr.values, color=colors, edgecolor="#264653")
    for b, v in zip(bars, yr.values):
        ax.text(b.get_x() + b.get_width() / 2, v + (15 if v >= 0 else -25),
                f"{v:+.0f}", ha="center", fontsize=10, weight="bold",
                color="#264653")
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.set_title("V4b 연도별 P&L (bp)", fontsize=13, weight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("P&L (bp)")
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_drawdown(dp, out):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(dp["price_date"], 0, dp["drawdown"],
                     color="#e76f51", alpha=0.35)
    ax.plot(dp["price_date"], dp["drawdown"], color="#a8331b", lw=1.2)
    mdd = dp["drawdown"].min()
    mdd_date = dp.loc[dp["drawdown"].idxmin(), "price_date"]
    ax.scatter([mdd_date], [mdd], color="darkred", zorder=5, s=60)
    ax.annotate(f"MaxDD: {mdd:.0f} bp\n({mdd_date.strftime('%Y-%m-%d')})",
                xy=(mdd_date, mdd), fontsize=10,
                xytext=(-90, -25), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="darkred"),
                color="darkred", weight="bold")
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.set_title("V4b Drawdown (bp)", fontsize=13, weight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (bp)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.3)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_heatmap(dp, out):
    dp2 = dp.copy()
    dp2["ym"] = dp2["price_date"].dt.to_period("M")
    monthly = dp2.groupby("ym")["pnl_bp"].sum().reset_index()
    monthly["year"] = monthly["ym"].dt.year
    monthly["month"] = monthly["ym"].dt.month
    pivot = monthly.pivot(index="year", columns="month", values="pnl_bp")
    pivot = pivot.reindex(columns=range(1, 13))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                    vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(str))
    for i in range(len(pivot.index)):
        for j in range(12):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.0f}", ha="center", va="center",
                         fontsize=8.5, color="black")
    ax.set_title("V4b 월별 P&L (bp) Heatmap", fontsize=13, weight="bold")
    fig.colorbar(im, ax=ax, label="P&L (bp)")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_trade_hist(t, out):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(t["pnl_bp"], bins=60, color="#264653", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", lw=1.2, ls="--", alpha=0.7)
    mean = t["pnl_bp"].mean()
    ax.axvline(mean, color="#2a9d8f", lw=1.5, label=f"평균 {mean:+.2f}bp")
    ax.set_title(f"V4b Trade P&L 분포 (N={len(t):,} trades)", fontsize=13, weight="bold")
    ax.set_xlabel("Trade P&L (bp)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_summary(dp, t, out):
    r = dp[dp["pos"] != 0]
    mu = r["pnl_bp"].mean()
    sd = r["pnl_bp"].std()
    sharpe = mu / sd * np.sqrt(TRADING_DAYS)
    total = dp["cum_pnl"].iloc[-1]
    nyrs = (dp["price_date"].max() - dp["price_date"].min()).days / 365.25
    per_yr = total / nyrs
    mdd = dp["drawdown"].min()
    win = (t["pnl_bp"] > 0).sum()
    loss = (t["pnl_bp"] < 0).sum()
    hit = win / (win + loss) * 100
    avg_win = t.loc[t["pnl_bp"] > 0, "pnl_bp"].mean()
    avg_loss = t.loc[t["pnl_bp"] < 0, "pnl_bp"].mean()
    wl = avg_win / -avg_loss
    trades_per_yr = len(t) / nyrs
    cost = trades_per_yr * t["sig"].abs().mean() * 0.12
    net_per_yr = per_yr - cost
    calmar = per_yr / abs(mdd)

    metrics = [
        ("기간", f"{dp['price_date'].min().date()} ~ {dp['price_date'].max().date()}"),
        ("거래일", f"{len(dp):,} days"),
        ("Total P&L", f"+{total:.0f} bp"),
        ("Per Year (gross)", f"+{per_yr:.0f} bp"),
        ("Per Year (net of 0.12bp cost)", f"+{net_per_yr:.0f} bp"),
        ("Sharpe (annualized)", f"+{sharpe:.2f}"),
        ("Max Drawdown", f"{mdd:.0f} bp"),
        ("Calmar (per_yr/|MDD|)", f"{calmar:.2f}"),
        ("Trades total", f"{len(t):,}"),
        ("Trades / year", f"{trades_per_yr:.0f}"),
        ("Hit rate", f"{hit:.1f}%"),
        ("Avg win / loss", f"+{avg_win:.2f} / {avg_loss:.2f} bp"),
        ("Win/Loss ratio", f"{wl:.2f}"),
        ("Avg position size (gross)", f"{r['pos'].abs().mean():.2f} unit"),
        ("100 계약 사이즈 환산 (per year, gross)",
         f"≈ {per_yr * DV01 * 100 * 10000:,.0f} 원"),
        ("100 계약 사이즈 환산 (per year, net)",
         f"≈ {net_per_yr * DV01 * 100 * 10000:,.0f} 원"),
    ]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")
    ax.set_title("V4b 핵심 메트릭스", fontsize=15, weight="bold", pad=15)
    # table
    cell_text = [[k, v] for k, v in metrics]
    tbl = ax.table(cellText=cell_text, colLabels=["Metric", "Value"],
                    loc="center", cellLoc="left", colWidths=[0.45, 0.45])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.7)
    # header
    for i in range(2):
        tbl[(0, i)].set_facecolor("#264653")
        tbl[(0, i)].set_text_props(color="white", weight="bold")
    # rows alternating
    for i in range(1, len(metrics) + 1):
        for j in range(2):
            tbl[(i, j)].set_facecolor("#f5f5f5" if i % 2 else "white")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_pnl_vs_y10(dp, out):
    fig, ax1 = plt.subplots(figsize=(12, 5.5))
    ax1.plot(dp["price_date"], dp["cum_pnl"], color="#2a9d8f", lw=2,
              label="V4b cumulative P&L")
    ax1.set_ylabel("Cumulative P&L (bp)", color="#2a9d8f")
    ax1.tick_params(axis="y", labelcolor="#2a9d8f")
    ax1.set_xlabel("Date")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(dp["price_date"], dp["y_10y"], color="#e76f51", lw=1.2, alpha=0.7,
              label="10Y yield (bp)")
    ax2.set_ylabel("10Y Yield (bp)", color="#e76f51")
    ax2.tick_params(axis="y", labelcolor="#e76f51")

    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.set_title("V4b 누적 P&L vs 10Y Yield", fontsize=13, weight="bold")
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.95))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_signal_breakdown(t, out):
    # combo 별 + 연도별
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # left: combo total P&L
    cb = t.groupby("combo")["pnl_bp"].agg(["sum", "size", "mean"]).round(2)
    cb = cb.sort_values("sum", ascending=True)
    colors = ["#2a9d8f" if v >= 0 else "#e76f51" for v in cb["sum"]]
    axes[0].barh(cb.index, cb["sum"], color=colors, edgecolor="#264653")
    for i, (combo, row) in enumerate(cb.iterrows()):
        axes[0].text(row["sum"] + 10, i,
                      f" N={int(row['size'])} ({row['mean']:+.2f}/tr)",
                      va="center", fontsize=9)
    axes[0].axvline(0, color="gray", lw=0.7)
    axes[0].set_title("시그널 조합 별 총 P&L", fontsize=12, weight="bold")
    axes[0].set_xlabel("Total P&L (bp)")
    axes[0].grid(alpha=0.3, axis="x")

    # right: 연도 × short sig 강도
    pivot = t.pivot_table(index="year", columns="sig", values="pnl_bp", aggfunc="sum").fillna(0)
    pivot.plot(kind="bar", ax=axes[1], width=0.8, edgecolor="white",
                colormap="RdYlGn_r")
    axes[1].axhline(0, color="gray", lw=0.7, ls="--")
    axes[1].set_title("연도별 P&L (sig 강도별)", fontsize=12, weight="bold")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("P&L (bp)")
    axes[1].legend(title="sig", loc="upper left", fontsize=9)
    axes[1].grid(alpha=0.3, axis="y")

    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    print("[load] panel ...")
    p = load_panel("2020-01-01")
    p["combo"] = p.apply(classify_combo, axis=1)
    print(f"  {len(p):,} rows  {p['price_date'].min().date()} ~ {p['price_date'].max().date()}\n")

    print("[backtest] V4b daily simulation ...")
    dp = daily_simulation(p)
    t = trade_attribution(p)
    print(f"  daily P&L points: {len(dp):,}")
    print(f"  trades: {len(t):,}\n")

    # summary
    r = dp[dp["pos"] != 0]
    mu = r["pnl_bp"].mean()
    sd = r["pnl_bp"].std()
    sharpe = mu / sd * np.sqrt(TRADING_DAYS)
    total = dp["cum_pnl"].iloc[-1]
    mdd = dp["drawdown"].min()
    hit = (t["pnl_bp"] > 0).mean() * 100
    nyrs = (dp["price_date"].max() - dp["price_date"].min()).days / 365.25
    print(f"  Total: +{total:.0f} bp  per_yr={total/nyrs:+.0f} bp  sharpe={sharpe:+.2f}  "
          f"MDD={mdd:.0f} bp  hit={hit:.1f}%\n")

    CHART_DIR.mkdir(exist_ok=True)
    print("[charts] generating ...")

    out_paths = [
        ("01_cumulative_pnl.png",   lambda p: plot_cumulative_pnl(dp, p)),
        ("02_yearly_pnl.png",       lambda p: plot_yearly_pnl(dp, p)),
        ("03_drawdown.png",         lambda p: plot_drawdown(dp, p)),
        ("04_monthly_heatmap.png",  lambda p: plot_monthly_heatmap(dp, p)),
        ("05_trade_pnl_hist.png",   lambda p: plot_trade_hist(t, p)),
        ("06_metrics_summary.png",  lambda p: plot_metrics_summary(dp, t, p)),
        ("07_pnl_vs_y10.png",       lambda p: plot_pnl_vs_y10(dp, p)),
        ("08_signal_breakdown.png", lambda p: plot_signal_breakdown(t, p)),
    ]
    for fname, fn in out_paths:
        full = CHART_DIR / fname
        fn(full)
        print(f"  ✓ {fname}")

    print(f"\n[done] charts → {CHART_DIR}")


if __name__ == "__main__":
    main()
