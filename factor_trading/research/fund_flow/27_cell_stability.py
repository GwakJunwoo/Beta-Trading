"""
27 — Cell sign stability across time periods.

질문: V7 의 8 cell 의 mean Δslope_21 부호가 시기 무관 일관된가?

검증:
  A) 3 sub-period 분할 (각 ~2년) 으로 cell mean Δslope_21 비교
       Period 1: 2020-05 ~ 2022-04
       Period 2: 2022-05 ~ 2024-04
       Period 3: 2024-05 ~ 2026-05
  B) Cell 별 부호 일관성 표 + flip 여부
  C) Walk-forward 학습 길이 sweep (1년, 2년, 3년, 4년)
       → 학습 데이터 길어질수록 sharpe 안정화 되는가?
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
DV01_KTB10F = 8.5
DV01_KTB3F = 2.8
RATIO = DV01_KTB10F / DV01_KTB3F
TRADING_DAYS = 252
HOLD = 21

# V7 priori 의 8 cells
V7_CELLS = ["1001", "1100", "1101", "1000", "0011", "0111", "1011", "0101"]
V7_EXPECTED_SIGN = {  # priori rule 의 부호
    "1001": "+", "1100": "+", "1101": "+", "1000": "+",
    "0011": "-", "0111": "-", "1011": "-", "0101": "-",
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


def cell_stats(sub):
    g = sub.dropna(subset=["dslope_fwd_21"]).groupby("cell").agg(
        N=("dslope_fwd_21", "size"),
        mean=("dslope_fwd_21", "mean"),
        hit_up=("dslope_fwd_21", lambda x: (x > 0).mean() * 100),
    ).round(2)
    return g


def main():
    print("[load] panel ...")
    p = load_panel("2020-01-01")
    print(f"  {len(p):,} rows  {p['price_date'].min().date()} ~ {p['price_date'].max().date()}\n")

    # ── A) 3 sub-period 분할 ──
    print("=" * 100)
    print("A) Cell sign across 3 sub-periods")
    print("=" * 100)
    periods = [
        ("P1: 2020-05~2022-04", p[p["price_date"] < "2022-05-01"]),
        ("P2: 2022-05~2024-04", p[(p["price_date"] >= "2022-05-01") & (p["price_date"] < "2024-05-01")]),
        ("P3: 2024-05~2026-05", p[p["price_date"] >= "2024-05-01"]),
    ]
    stats = {name: cell_stats(sub) for name, sub in periods}
    full = cell_stats(p)

    # Cell 별 부호 비교
    print(f"\n  {'cell':>6s} | {'expected':>9s} | "
          f"{'P1 mean':>9s} {'P1 sign':>8s} {'P1 N':>6s} | "
          f"{'P2 mean':>9s} {'P2 sign':>8s} {'P2 N':>6s} | "
          f"{'P3 mean':>9s} {'P3 sign':>8s} {'P3 N':>6s} | "
          f"{'Full mean':>10s} {'consistent?':>12s}")
    print("  " + "-" * 140)
    rows_audit = []
    for c in V7_CELLS:
        exp = V7_EXPECTED_SIGN[c]
        cells_signs = []
        cells_means = []
        cells_n = []
        for name, st in stats.items():
            if c in st.index:
                m = st.loc[c, "mean"]
                n = int(st.loc[c, "N"])
                sg = "+" if m > 0 else "-"
            else:
                m, n, sg = np.nan, 0, "N/A"
            cells_signs.append(sg)
            cells_means.append(m)
            cells_n.append(n)
        full_m = full.loc[c, "mean"] if c in full.index else np.nan
        consistent = all(s == exp for s in cells_signs if s != "N/A")
        print(f"  {c:>6s} | {exp:>9s} | "
              f"{cells_means[0]:>+9.2f} {cells_signs[0]:>8s} {cells_n[0]:>6d} | "
              f"{cells_means[1]:>+9.2f} {cells_signs[1]:>8s} {cells_n[1]:>6d} | "
              f"{cells_means[2]:>+9.2f} {cells_signs[2]:>8s} {cells_n[2]:>6d} | "
              f"{full_m:>+10.2f} {'YES' if consistent else 'NO ***':>12s}")
        rows_audit.append({
            "cell": c, "expected_sign": exp,
            "P1_mean": cells_means[0], "P1_sign": cells_signs[0], "P1_N": cells_n[0],
            "P2_mean": cells_means[1], "P2_sign": cells_signs[1], "P2_N": cells_n[1],
            "P3_mean": cells_means[2], "P3_sign": cells_signs[2], "P3_N": cells_n[2],
            "full_mean": full_m, "consistent": consistent,
        })
    print()

    # ── B) 모든 16 cell 의 sub-period 비교 ──
    print("=" * 100)
    print("B) 모든 16 cells 의 mean Δslope_21 across sub-periods")
    print("=" * 100)
    all_cells = sorted(set().union(*[st.index for st in stats.values()]))
    print(f"\n  {'cell':>6s} | {'P1 mean':>9s} {'P1 N':>6s} | {'P2 mean':>9s} {'P2 N':>6s} | "
          f"{'P3 mean':>9s} {'P3 N':>6s} | {'sign flips':>10s}")
    print("  " + "-" * 95)
    for c in all_cells:
        rows = []
        for name, st in stats.items():
            if c in st.index:
                rows.append((st.loc[c, "mean"], int(st.loc[c, "N"])))
            else:
                rows.append((np.nan, 0))
        signs = [("+" if m > 0 else "-") if pd.notna(m) else "N/A" for m, _ in rows]
        unique_signs = set(s for s in signs if s != "N/A")
        flips = "stable" if len(unique_signs) <= 1 else "FLIPPED"
        line = f"  {c:>6s} | "
        for m, n in rows:
            line += f"{m:>+9.2f} {n:>6d} | "
        line += f"{flips:>10s}"
        print(line)
    print()

    # ── C) Walk-forward sharpe vs 학습 기간 길이 ──
    print("=" * 100)
    print("C) Walk-forward sharpe: 학습 기간 길이 sweep")
    print("=" * 100)
    print(f"\n  Cell sign learning: [start ~ start + train_years] -> [train_years+ ~ end] 적용")
    print(f"  Trade hold: 21d fixed, sizing: cell sign (+1 / -1, fixed)")
    print()
    print(f"  {'train_years':>13s} {'train_end':>12s} {'test_days':>10s} "
          f"{'cells_active':>13s} {'trades':>8s} {'total':>10s} "
          f"{'sharpe':>8s} {'per_yr':>10s}")
    print("  " + "-" * 90)

    for train_years in [1.0, 2.0, 3.0, 4.0]:
        train_days = int(train_years * TRADING_DAYS)
        train_p = p.iloc[:train_days].copy()
        test_p = p.iloc[train_days:].copy().reset_index(drop=True)
        if len(test_p) < 30:
            continue
        # 학습: cell mean Δslope (with N>=5)
        tbl = train_p.dropna(subset=["dslope_fwd_21"]).groupby("cell").agg(
            N=("dslope_fwd_21", "size"),
            mean=("dslope_fwd_21", "mean"),
        )
        rule = {}
        for c, row in tbl.iterrows():
            if row["N"] >= 5 and abs(row["mean"]) >= 1.0:
                rule[c] = float(np.sign(row["mean"]))   # fixed size ±1
        # 백테스트 (test_p)
        n = len(test_p)
        daily_pnl = np.zeros(n)
        dy10 = test_p["dy10_1d"].fillna(0.0).values
        dy3 = test_p["dy3_1d"].fillna(0.0).values
        cells = test_p["cell"].values
        n_trades = 0
        for i in range(n):
            if cells[i] not in rule:
                continue
            size = rule[cells[i]]
            pos_10 = -size
            pos_3 = +size * RATIO
            n_trades += 1
            for d in range(i + 1, min(i + HOLD + 1, n)):
                daily_pnl[d] += pos_10 * (-dy10[d]) * DV01_KTB10F \
                                + pos_3 * (-dy3[d]) * DV01_KTB3F
        active = daily_pnl[daily_pnl != 0]
        sh = active.mean() / active.std() * np.sqrt(TRADING_DAYS) if len(active) > 1 and active.std() > 0 else 0
        total = daily_pnl.sum()
        nyrs = n / TRADING_DAYS
        train_end_date = train_p["price_date"].iloc[-1].date()
        print(f"  {train_years:>13.1f} {str(train_end_date):>12s} {n:>10d} "
              f"{len(rule):>13d} {n_trades:>8d} {total:>+10,.0f} "
              f"{sh:>+8.2f} {total/nyrs:>+10,.0f}")
    print()

    # ── D) 사용자 결론 요약 ──
    consistent_count = sum(1 for r in rows_audit if r["consistent"])
    print("=" * 100)
    print("D) 결론")
    print("=" * 100)
    print(f"\n  V7 priori 8 cell 중 {consistent_count}/8 cell 이 3 sub-period 전체에서 부호 일관")
    print(f"  → 일관성 높을수록 V7 priori 의 in-sample 통계 활용이 정당화됨")
    print(f"  → walk-forward sharpe 가 학습 기간 길이에 따라 어떻게 변하는지 확인 (C 표)")
    print()

    # ── Excel 저장 ──
    CHART_DIR = Path(__file__).parent / "charts"
    CHART_DIR.mkdir(exist_ok=True)
    xlsx = CHART_DIR / "V7_cell_stability.xlsx"
    audit_df = pd.DataFrame(rows_audit)
    period_full = pd.concat({name: st for name, st in stats.items()}, axis=1)
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xl:
        audit_df.to_excel(xl, sheet_name="V7_cell_audit", index=False)
        period_full.to_excel(xl, sheet_name="Cell_means_by_period")
        full.reset_index().to_excel(xl, sheet_name="Full_period", index=False)
    print(f"[save] {xlsx}\n[done]")


if __name__ == "__main__":
    main()
