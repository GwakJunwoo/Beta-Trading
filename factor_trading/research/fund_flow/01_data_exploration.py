"""
01 — Fund Flow 데이터 탐색.

확인할 것:
  1. 단위 (억원? 백만원? 매매수량?)
  2. 종목 커버리지 (어떤 만기/종류가 tracked?)
  3. 결측 / zero 비율 (sparse?)
  4. 주체별 분포 (외국인 활발도 vs 은행 등)
  5. 윈도우 간 일관성 (sum_3d 가 diff_1d × 3 근사?)
  6. ktb 본체 yield 데이터와 join 가능성
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BETA_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BETA_ROOT))

from beta_trading.db import get_connection

ENTITIES = ["foreigner", "insurance", "asset_mgmt", "bank"]
WINDOWS = ["diff_1d", "sum_3d", "sum_5d", "sum_10d"]


def load_flow(start: str = "2022-01-01") -> pd.DataFrame:
    cols = ", ".join(
        ["bond_code", "bond_name", "price_date"]
        + [f"{e}_{w}" for e in ENTITIES for w in WINDOWS]
    )
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(f"""
            SELECT {cols}
            FROM ktb_trade_flow_features
            WHERE price_date >= %s AND bond_code IS NOT NULL AND bond_code != ''
            ORDER BY price_date, bond_code
        """, (start,))
        rows = cur.fetchall()
    df = pd.DataFrame(rows)
    df["price_date"] = pd.to_datetime(df["price_date"])
    for col in df.columns:
        if col not in ("bond_code", "bond_name", "price_date"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_ktb(start: str = "2022-01-01") -> pd.DataFrame:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT price_date, bond_code, bond_name, category, label, ytm, remain_year, issue_date
            FROM ktb
            WHERE category='국고채' AND price_date >= %s AND ytm > 0
              AND bond_code IS NOT NULL AND bond_code != ''
        """, (start,))
        rows = cur.fetchall()
    df = pd.DataFrame(rows)
    df["price_date"] = pd.to_datetime(df["price_date"])
    df["ytm"] = pd.to_numeric(df["ytm"], errors="coerce")
    df["remain_year"] = pd.to_numeric(df["remain_year"], errors="coerce")
    return df


def main():
    print("[load] flow ...")
    flow = load_flow(start="2023-01-01")
    print(f"  rows={len(flow):,}, bonds={flow['bond_code'].nunique()}, dates={flow['price_date'].nunique()}")
    print(f"  range: {flow['price_date'].min().date()} ~ {flow['price_date'].max().date()}\n")

    # 1) 분포 (각 4 주체의 diff_1d)
    print("=== 주체별 1일 net buy 분포 (단위 확인) ===")
    desc_rows = []
    for e in ENTITIES:
        s = flow[f"{e}_diff_1d"].dropna()
        desc_rows.append({
            "entity": e, "n": len(s),
            "mean": s.mean(), "std": s.std(),
            "min": s.min(), "p25": s.quantile(0.25),
            "p50": s.median(), "p75": s.quantile(0.75),
            "max": s.max(),
            "nonzero_pct": float((s != 0).mean() * 100),
        })
    desc = pd.DataFrame(desc_rows).set_index("entity")
    print(desc.round(2).to_string())

    print()
    print("→ 단위 추정: max 수십~수백 → 억원 net buy 가능성 큼 (참고)")
    print()

    # 2) 종목 커버리지 (최근 1년)
    print("=== 최근 1년 종목별 커버리지 (활발도) ===")
    recent = flow[flow["price_date"] >= flow["price_date"].max() - pd.Timedelta(days=365)]
    per_bond = recent.groupby("bond_code").agg(
        n_days=("price_date", "nunique"),
        foreigner_active=("foreigner_diff_1d", lambda x: (x != 0).sum()),
        insurance_active=("insurance_diff_1d", lambda x: (x != 0).sum()),
        asset_mgmt_active=("asset_mgmt_diff_1d", lambda x: (x != 0).sum()),
        bank_active=("bank_diff_1d", lambda x: (x != 0).sum()),
    )
    per_bond["any_flow_active"] = per_bond[[f"{e}_active" for e in ENTITIES]].max(axis=1)
    per_bond["active_pct"] = per_bond["any_flow_active"] / per_bond["n_days"] * 100
    print(f"  최근 1년 covered bonds: {len(per_bond)}")
    print(f"  활발도 분포 (any 주체 활동일 / 전체일):")
    print(per_bond["active_pct"].describe().round(1).to_string())
    print(f"\n  활발도 25% 미만 종목 수: {(per_bond['active_pct'] < 25).sum()} (대부분 off-the-run)")
    print(f"  활발도 75%+ 종목 수: {(per_bond['active_pct'] >= 75).sum()} (active 종목)")

    # 3) Active 종목 top 10
    top = per_bond.sort_values("active_pct", ascending=False).head(10)
    print("\n  최근 1년 가장 활발한 종목 top 10:")
    print(top[["n_days", "any_flow_active", "active_pct"]].to_string())

    # 4) ktb 본체와 join 가능성
    print("\n=== ktb 본체와 cross-check ===")
    ktb = load_ktb(start="2025-01-01")
    flow_bonds = set(flow["bond_code"].unique())
    ktb_bonds = set(ktb["bond_code"].unique())
    print(f"  flow 종목 (전기간): {len(flow_bonds)}")
    print(f"  ktb 국고채 (2025+): {len(ktb_bonds)}")
    print(f"  교집합: {len(flow_bonds & ktb_bonds)}")
    print(f"  ktb 에 있지만 flow 에 없음: {len(ktb_bonds - flow_bonds)} (지표/통안 등)")
    print(f"  flow 에 있지만 ktb 에 없음: {len(flow_bonds - ktb_bonds)} (만기/생산 차이)")

    # 5) 잔존만기 bucket 별 flow 활동도
    latest_t = ktb["price_date"].max()
    latest = ktb[ktb["price_date"] == latest_t][["bond_code", "remain_year"]].set_index("bond_code")
    per_bond_join = per_bond.join(latest, how="inner")
    per_bond_join["rem_bucket"] = pd.cut(
        per_bond_join["remain_year"], bins=[0, 3, 5, 7, 10, 13, 30],
        labels=["2-3Y", "3-5Y", "5-7Y", "7-10Y", "10-13Y", "13Y+"],
    )
    bucket = per_bond_join.groupby("rem_bucket", observed=True).agg(
        n_bonds=("active_pct", "size"),
        mean_active=("active_pct", "mean"),
        median_active=("active_pct", "median"),
    ).round(1)
    print(f"\n  잔존만기 bucket 별 활동도 (최근 1년 평균):")
    print(bucket.to_string())

    # 6) Window 간 일관성 (sum_3d vs sum_5d vs sum_10d cumulative correlation 등)
    print("\n=== 윈도우 일관성 check (foreigner) ===")
    # 같은 (bond, date) 의 sum_3d 와 sum_5d 의 관계
    fr = flow[["foreigner_diff_1d", "foreigner_sum_3d", "foreigner_sum_5d", "foreigner_sum_10d"]].dropna()
    print(fr.corr().round(2).to_string())
    print("\n  (sum_3d 와 sum_5d 가 0.8+ 상관 = 누적 윈도우는 서로 redundant 한 측면 있음)")

    # 7) 최근 활발 종목의 시계열 sample
    print("\n=== 활발 종목 1개의 flow 시계열 sample ===")
    top1_code = top.index[0]
    sample = flow[flow["bond_code"] == top1_code].sort_values("price_date").tail(10)
    cols_show = ["price_date"] + [f"{e}_diff_1d" for e in ENTITIES]
    print(f"  bond: {top1_code}")
    print(sample[cols_show].to_string(index=False))

    print("\n[done]")


if __name__ == "__main__":
    main()
