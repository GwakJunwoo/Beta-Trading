"""
21 — Cross-tenor flow 패턴 딥 리서치.

가설: 외국인이 만기 간 rotation 하는 패턴이 있을 것
  예) f10 매도 + b3F 매수 = 듀레이션 축소 (bear flattener 시그널)
  예) f3 매도 + b10F 매수 = 듀레이션 확대 (bull steepener)
  예) 4 카테고리 모두 매도 = total unwind

4 카테고리:
  f10  = KTB10F 외국인 5d cum (선물)
  f3   = KTB3F  외국인 5d cum (선물)
  b10F = 잔존 7-13Y 현물 외국인 5d cum
  b3F  = 잔존 2-4Y  현물 외국인 5d cum

각 카테고리 부호 (+/-) 의 16 조합 → forward ΔY_10Y, ΔY_3Y, Δslope IC + 평균.

추가 분석:
  - Tenor rotation index: b3F − b10F (cash), f3 − f10 (fut)
  - 단순 pair-wise correlation
  - 가장 informed combinations 식별
  - 차트
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

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
        # bucket 별 daily aggregate
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
    # b3F, b10F 는 이미 5d sum (DB sum_5d)

    p["dfx_past_5"] = p["fx"] - p["fx"].shift(5)
    p["slope"] = p["y_10y"] - p["y_3y"]
    p["year"] = p["price_date"].dt.year
    for h in [3, 5, 10, 21]:
        p[f"dy3_fwd_{h}"] = p["y_3y"].shift(-h) - p["y_3y"]
        p[f"dy10_fwd_{h}"] = p["y_10y"].shift(-h) - p["y_10y"]
        p[f"dslope_fwd_{h}"] = p["dy10_fwd_" + str(h)] - p["dy3_fwd_" + str(h)]
    return p


def ic(x, y):
    s = pd.DataFrame({"x": x, "y": y}).dropna()
    s = s[(s["x"] != 0) | (s["y"] != 0)]
    if len(s) < 30:
        return np.nan
    rho, _ = spearmanr(s["x"], s["y"])
    return float(rho)


def main():
    print("[load] panel ...")
    p = load_panel("2020-01-01")
    print(f"  {len(p):,} rows  {p['price_date'].min().date()} ~ {p['price_date'].max().date()}\n")

    # 4 카테고리
    cats = ["f10", "f3", "b10F", "b3F"]
    targets = ["dy3_fwd_21", "dy10_fwd_21", "dslope_fwd_21"]

    # ── A) Pair-wise correlation between 4 categories ──
    print("=" * 100)
    print("A) 4 카테고리 간 동시점 correlation (sign-level)")
    print("=" * 100)
    corr_mat = p[cats].corr().round(3)
    print("\n  Pearson:")
    print(corr_mat.to_string())
    print()
    # spearman
    sp_mat = p[cats].apply(lambda x: x.rank()).corr().round(3)
    print("\n  Spearman:")
    print(sp_mat.to_string())
    print()

    # ── B) IC matrix (each category vs each target) ──
    print("=" * 100)
    print("B) 단일 카테고리 IC matrix (각 카테고리 → forward Δy)")
    print("=" * 100)
    ic_mat = []
    for c in cats:
        row = [c]
        for t in targets:
            row.append(f"{ic(p[c], p[t]):+.3f}")
        ic_mat.append(row)
    print(f"\n  {'category':>10s} {'ΔY_3Y_21':>10s} {'ΔY_10Y_21':>11s} {'Δslope_21':>11s}")
    for row in ic_mat:
        print(f"  {row[0]:>10s} {row[1]:>10s} {row[2]:>11s} {row[3]:>11s}")
    print()

    # ── C) Tenor rotation index ──
    print("=" * 100)
    print("C) Tenor rotation indexes")
    print("=" * 100)
    # 정규화 (z-score) 후 차이로
    for c in cats:
        p[f"{c}_z"] = (p[c] - p[c].rolling(63, min_periods=20).mean()) / \
                       p[c].rolling(63, min_periods=20).std()
    p["cash_rot"] = p["b3F_z"] - p["b10F_z"]    # 양수 = 단기로 rotation
    p["fut_rot"] = p["f3_z"] - p["f10_z"]
    p["mix_short_long"] = (p["b3F_z"] + p["f3_z"]) - (p["b10F_z"] + p["f10_z"])  # short tenor 강세 - long tenor 강세

    print(f"\n  Rotation index IC vs forward:")
    print(f"  {'index':>20s} {'ΔY_3Y_21':>10s} {'ΔY_10Y_21':>11s} {'Δslope_21':>11s}")
    for idx in ["cash_rot", "fut_rot", "mix_short_long"]:
        line = f"  {idx:>20s}"
        for t in targets:
            line += f" {ic(p[idx], p[t]):>+10.3f}"
        print(line)
    print()
    print("  해석:")
    print("    cash_rot > 0  = 외국인 현물에서 단기(b3F) 매수, 장기(b10F) 매도 rotation")
    print("    Δslope > 0   = 10Y 가 3Y 보다 약세 (curve steepening)")
    print("    cash_rot vs Δslope IC 양수 = rotation 이 steepening 예측")
    print()

    # ── D) 4 카테고리 sign 조합 매트릭스 (16 조합) ──
    print("=" * 100)
    print("D) 4 카테고리 sign 조합 16개 매트릭스 → forward Δy_10Y_21")
    print("=" * 100)
    sub = p[cats + targets].dropna().copy()
    sub["s_f10"]  = (sub["f10"]  > 0).astype(int)
    sub["s_f3"]   = (sub["f3"]   > 0).astype(int)
    sub["s_b10F"] = (sub["b10F"] > 0).astype(int)
    sub["s_b3F"]  = (sub["b3F"]  > 0).astype(int)

    for tgt in targets:
        g = sub.groupby(["s_f10", "s_f3", "s_b10F", "s_b3F"]).agg(
            n=(tgt, "size"),
            mean_dy=(tgt, "mean"),
            hit_up=(tgt, lambda x: (x > 0).mean() * 100),
        ).round(2)
        # readability: rename to BUY/SELL
        idx_new = []
        for s_f10, s_f3, s_b10F, s_b3F in g.index:
            label = f"f10={'BUY' if s_f10 else 'SELL'} f3={'BUY' if s_f3 else 'SELL'} " \
                    f"b10F={'BUY' if s_b10F else 'SELL'} b3F={'BUY' if s_b3F else 'SELL'}"
            idx_new.append(label)
        g.index = idx_new
        g = g.sort_values("mean_dy", ascending=False)
        print(f"\n  ▶ Target: {tgt}")
        print(g.to_string())
    print()

    # ── E) Cross-tenor 특정 패턴 (사용자 가설) ──
    print("=" * 100)
    print("E) 사용자 가설 specific patterns")
    print("=" * 100)
    patterns = [
        ("10F sell + 3현물 buy",       (sub["s_f10"] == 0) & (sub["s_b3F"] == 1)),
        ("10F sell + 10F현물 sell",    (sub["s_f10"] == 0) & (sub["s_b10F"] == 0)),
        ("10F sell + 3F sell + 3현물 buy", (sub["s_f10"] == 0) & (sub["s_f3"] == 0) & (sub["s_b3F"] == 1)),
        ("10F sell + 3F buy + 3현물 buy",  (sub["s_f10"] == 0) & (sub["s_f3"] == 1) & (sub["s_b3F"] == 1)),
        ("3F sell + 10F현물 buy",      (sub["s_f3"] == 0) & (sub["s_b10F"] == 1)),
        ("3F sell + 10F현물 sell",     (sub["s_f3"] == 0) & (sub["s_b10F"] == 0)),
        ("4 카테고리 모두 매도",            (sub["s_f10"] == 0) & (sub["s_f3"] == 0) &
                                          (sub["s_b10F"] == 0) & (sub["s_b3F"] == 0)),
        ("4 카테고리 모두 매수",            (sub["s_f10"] == 1) & (sub["s_f3"] == 1) &
                                          (sub["s_b10F"] == 1) & (sub["s_b3F"] == 1)),
        ("선물 매도 + 현물 매수 (정점)",     (sub["s_f10"] == 0) & (sub["s_f3"] == 0) &
                                          (sub["s_b10F"] == 1) & (sub["s_b3F"] == 1)),
        ("선물 매수 + 현물 매도",            (sub["s_f10"] == 1) & (sub["s_f3"] == 1) &
                                          (sub["s_b10F"] == 0) & (sub["s_b3F"] == 0)),
    ]
    print(f"\n  {'Pattern':40s} {'N':>5s} {'ΔY_3Y':>8s} {'ΔY_10Y':>9s} {'Δslope':>8s}")
    print("  " + "-" * 80)
    for name, mask in patterns:
        s = sub[mask]
        n = len(s)
        if n < 10:
            continue
        mean_3 = s["dy3_fwd_21"].mean()
        mean_10 = s["dy10_fwd_21"].mean()
        mean_sl = s["dslope_fwd_21"].mean()
        print(f"  {name:40s} {n:>5d} {mean_3:>+8.2f} {mean_10:>+9.2f} {mean_sl:>+8.2f}")
    print()

    # ── F) 패턴 × FX regime ──
    print("=" * 100)
    print("F) Top 패턴 × FX regime (KRW强 / KRW弱)")
    print("=" * 100)
    sub["krw_strong"] = sub.index.map(lambda i: p.loc[i, "dfx_past_5"] < 0 if i in p.index else False)
    # safer mapping
    sub2 = p[cats + targets + ["dfx_past_5"]].dropna().copy()
    sub2["s_f10"]  = (sub2["f10"]  > 0).astype(int)
    sub2["s_f3"]   = (sub2["f3"]   > 0).astype(int)
    sub2["s_b10F"] = (sub2["b10F"] > 0).astype(int)
    sub2["s_b3F"]  = (sub2["b3F"]  > 0).astype(int)
    sub2["krw_strong"] = sub2["dfx_past_5"] < 0

    top_patterns = [
        ("10F sell + 3현물 buy",       (sub2["s_f10"] == 0) & (sub2["s_b3F"] == 1)),
        ("3F sell + 10현물 buy",       (sub2["s_f3"] == 0) & (sub2["s_b10F"] == 1)),
        ("선물 sell + 현물 buy (정점)", (sub2["s_f10"] == 0) & (sub2["s_f3"] == 0) &
                                       (sub2["s_b10F"] == 1) & (sub2["s_b3F"] == 1)),
        ("선물 buy + 현물 sell",       (sub2["s_f10"] == 1) & (sub2["s_f3"] == 1) &
                                       (sub2["s_b10F"] == 0) & (sub2["s_b3F"] == 0)),
    ]
    print(f"\n  {'Pattern':38s} {'KRW':>5s} {'N':>5s} {'ΔY_3Y':>8s} {'ΔY_10Y':>9s} {'Δslope':>8s}")
    print("  " + "-" * 85)
    for name, mask in top_patterns:
        for kr_lbl, kr_mask in [("强", sub2["krw_strong"]), ("弱", ~sub2["krw_strong"])]:
            s = sub2[mask & kr_mask]
            n = len(s)
            if n < 10:
                continue
            print(f"  {name:38s} {kr_lbl:>5s} {n:>5d} "
                  f"{s['dy3_fwd_21'].mean():>+8.2f} "
                  f"{s['dy10_fwd_21'].mean():>+9.2f} "
                  f"{s['dslope_fwd_21'].mean():>+8.2f}")
    print()

    # ── G) 5/11 기준 — 4 카테고리 현재 상태 ──
    print("=" * 100)
    print("G) 5/11 기준 4 카테고리 sign 확인")
    print("=" * 100)
    recent = p.tail(10)
    print(f"\n  {'date':>12s} {'f10':>10s} {'f3':>10s} {'b10F':>10s} {'b3F':>10s} {'fx5d':>8s}")
    for _, r in recent.iterrows():
        f10_s = "BUY" if r["f10"] > 0 else "SELL"
        f3_s = "BUY" if r["f3"] > 0 else "SELL"
        b10_s = "BUY" if r["b10F"] > 0 else "SELL"
        b3_s = "BUY" if r["b3F"] > 0 else "SELL"
        print(f"  {r['price_date'].strftime('%Y-%m-%d'):>12s} "
              f"{int(r['f10']):>+5,d}({f10_s:>4s}) "
              f"{int(r['f3']):>+5,d}({f3_s:>4s}) "
              f"{int(r['b10F']):>+5,d}({b10_s:>4s}) "
              f"{int(r['b3F']):>+5,d}({b3_s:>4s}) "
              f"{r['dfx_past_5']:>+8.1f}")
    print()

    # ── H) IC heatmap chart ──
    print("=" * 100)
    print("차트 ...")
    print("=" * 100)
    CHART_DIR.mkdir(exist_ok=True)

    # Full IC matrix (single + rotation indexes)
    all_signals = cats + ["cash_rot", "fut_rot", "mix_short_long"]
    horizons = [5, 10, 21]
    ic_data = []
    for sig in all_signals:
        row = []
        for h in horizons:
            for tgt_label, tgt_col in [("3Y", f"dy3_fwd_{h}"), ("10Y", f"dy10_fwd_{h}"),
                                         ("slope", f"dslope_fwd_{h}")]:
                row.append(ic(p[sig], p[tgt_col]))
        ic_data.append(row)
    col_labels = []
    for h in horizons:
        for t in ["3Y", "10Y", "slope"]:
            col_labels.append(f"{t}\n{h}d")
    ic_arr = np.array(ic_data)

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(ic_arr, cmap="RdBu_r", vmin=-0.20, vmax=0.20, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(all_signals)))
    ax.set_yticklabels(all_signals)
    for i in range(len(all_signals)):
        for j in range(len(col_labels)):
            v = ic_arr[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                     fontsize=8,
                     color="white" if abs(v) > 0.13 else "black")
    ax.set_title("Cross-tenor 시그널 IC matrix (forward Δy)", fontsize=12, weight="bold")
    plt.colorbar(im, ax=ax, label="Spearman IC")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "24_cross_tenor_ic.png", bbox_inches="tight")
    plt.close(fig)
    print("  OK 24_cross_tenor_ic.png")

    # 4 카테고리 시계열
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    for ax, c, lbl in zip(axes, cats, ["KTB10F", "KTB3F", "10년현물 (b10F)", "3년현물 (b3F)"]):
        ax.plot(p["price_date"], p[c], color="#264653", lw=0.8)
        ax.fill_between(p["price_date"], 0, p[c],
                         where=p[c] > 0, color="#2a9d8f", alpha=0.3)
        ax.fill_between(p["price_date"], 0, p[c],
                         where=p[c] < 0, color="#e76f51", alpha=0.3)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_title(f"외국인 {lbl} 5d cum", fontsize=11)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "25_cross_tenor_flows.png", bbox_inches="tight")
    plt.close(fig)
    print("  OK 25_cross_tenor_flows.png")

    print("\n[done]")


if __name__ == "__main__":
    main()
