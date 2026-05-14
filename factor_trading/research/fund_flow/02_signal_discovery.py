"""
02 — Signal Discovery

핵심 질문: flow_t (오늘까지의 net buy) → ΔY_t+k (k일 뒤 yield 변화) 예측력?

각 (entity, window, horizon) 조합:
  - IC (Information Coefficient, Spearman corr of flow vs forward yield change)
  - 만기 bucket 별 분리
  - sign 안정성 (월/분기별)

기대 방향:
  - flow 가 buy (positive) → bond 가 매수 → 가격 ↑ / yield ↓
  - 즉, flow 와 ΔY 는 음의 상관 예상

Look-ahead 방지:
  - flow_t 의 정보는 t 일 close 후 발표 (또는 다음날 open)
  - ΔY 는 [t+1, t+k] 기간 (t 일 close → t+k 일 close)
"""
from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass
BETA_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BETA_ROOT))

from beta_trading.db import get_connection

ENTITIES = ["foreigner", "insurance", "asset_mgmt", "bank"]
WINDOWS = ["diff_1d", "sum_3d", "sum_5d", "sum_10d"]
HORIZONS = [1, 3, 5, 10, 21]  # 영업일 단위 forward yield change


def load_data(start: str = "2023-01-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    """flow + ktb yield panel."""
    flow_cols = ", ".join(
        ["bond_code", "price_date"] + [f"{e}_{w}" for e in ENTITIES for w in WINDOWS]
    )
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(f"""
            SELECT {flow_cols}
            FROM ktb_trade_flow_features
            WHERE price_date >= %s AND bond_code IS NOT NULL AND bond_code != ''
        """, (start,))
        flow_rows = cur.fetchall()

        cur.execute("""
            SELECT price_date, bond_code, AVG(ytm) AS ytm, AVG(remain_year) AS remain_year
            FROM ktb
            WHERE category='국고채' AND price_date >= %s AND ytm > 0
              AND bond_code IS NOT NULL AND bond_code != ''
            GROUP BY price_date, bond_code
        """, (start,))
        ktb_rows = cur.fetchall()

    flow = pd.DataFrame(flow_rows)
    flow["price_date"] = pd.to_datetime(flow["price_date"])
    for c in flow.columns:
        if c not in ("bond_code", "price_date"):
            flow[c] = pd.to_numeric(flow[c], errors="coerce")

    ktb = pd.DataFrame(ktb_rows)
    ktb["price_date"] = pd.to_datetime(ktb["price_date"])
    ktb["ytm"] = pd.to_numeric(ktb["ytm"], errors="coerce")
    ktb["remain_year"] = pd.to_numeric(ktb["remain_year"], errors="coerce")
    return flow, ktb


def compute_forward_yield_change(ktb: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """각 (date, bond) → t+k 일 yield change (bp) 계산.

    Returns long-form DataFrame with columns: price_date, bond_code, fwd_dy_{k}_bp
    """
    pivot = ktb.pivot_table(index="price_date", columns="bond_code", values="ytm",
                            aggfunc="mean").sort_index().ffill(limit=5)
    out = ktb[["price_date", "bond_code", "remain_year"]].copy()

    for h in horizons:
        shifted = pivot.shift(-h)  # t+h 일자의 yield
        diff_bp = (shifted - pivot) * 100.0  # bp 단위
        # stack to long form
        stk = diff_bp.stack().reset_index().rename(columns={"level_0": "price_date", 0: f"fwd_dy_{h}_bp"})
        stk.columns = ["price_date", "bond_code", f"fwd_dy_{h}_bp"]
        out = out.merge(stk, on=["price_date", "bond_code"], how="left")
    return out


def compute_ic(merged: pd.DataFrame, flow_col: str, fwd_col: str,
               bucket_col: str | None = None) -> dict:
    """Spearman IC (음의 상관 = flow 가 buy 면 yield 하락 = 예상 방향)."""
    df = merged[[flow_col, fwd_col]].dropna()
    df = df[(df[flow_col] != 0) | (df[fwd_col] != 0)]
    if len(df) < 30:
        return {"n": len(df), "ic": np.nan, "pval": np.nan}
    rho, p = spearmanr(df[flow_col], df[fwd_col])
    return {"n": len(df), "ic": float(rho), "pval": float(p)}


def main():
    print("[load] flow + ktb ...")
    flow, ktb = load_data(start="2023-01-01")
    print(f"  flow: {len(flow):,} rows, ktb: {len(ktb):,} rows\n")

    print("[compute forward yield changes] ...")
    fwd = compute_forward_yield_change(ktb, HORIZONS)
    print(f"  fwd rows: {len(fwd):,}\n")

    # Merge flow + fwd
    merged = flow.merge(fwd, on=["price_date", "bond_code"], how="inner")
    print(f"[merged] {len(merged):,} rows, {merged['bond_code'].nunique()} bonds\n")

    # remain 2-13Y 필터 (RV 페어 universe 와 align)
    merged = merged[(merged["remain_year"] >= 2) & (merged["remain_year"] <= 13)]
    print(f"[filtered remain 2-13Y] {len(merged):,} rows, {merged['bond_code'].nunique()} bonds\n")

    # IC matrix: (entity, window) × horizon
    print("=== IC matrix — Spearman corr (flow vs forward Δyield bp) ===")
    print("    [음수 = flow buy 시 yield 하락 = 예상 방향, 절대값 0.05+ 면 유의미]")
    print()
    rows = []
    for e in ENTITIES:
        for w in WINDOWS:
            row = {"entity": e, "window": w}
            flow_col = f"{e}_{w}"
            for h in HORIZONS:
                res = compute_ic(merged, flow_col, f"fwd_dy_{h}_bp")
                row[f"IC_{h}d"] = res["ic"]
            rows.append(row)
    ic_df = pd.DataFrame(rows).set_index(["entity", "window"])
    print(ic_df.round(3).to_string())

    print()
    print("=== Top 10 (entity, window, horizon) by |IC| ===")
    ic_flat = []
    for e in ENTITIES:
        for w in WINDOWS:
            for h in HORIZONS:
                res = compute_ic(merged, f"{e}_{w}", f"fwd_dy_{h}_bp")
                ic_flat.append({"entity": e, "window": w, "horizon": h,
                               "n": res["n"], "IC": res["ic"], "pval": res["pval"]})
    flat = pd.DataFrame(ic_flat)
    flat["abs_IC"] = flat["IC"].abs()
    flat = flat.sort_values("abs_IC", ascending=False).head(10)
    print(flat[["entity", "window", "horizon", "n", "IC", "pval"]].to_string(index=False))

    # 만기 bucket 별 IC (best 조합으로)
    if len(flat) > 0:
        best = flat.iloc[0]
        best_flow_col = f"{best['entity']}_{best['window']}"
        best_fwd_col = f"fwd_dy_{int(best['horizon'])}_bp"
        print(f"\n=== 잔존만기 bucket 별 IC ({best['entity']}_{best['window']} → {int(best['horizon'])}d) ===")
        merged["rem_bucket"] = pd.cut(merged["remain_year"], bins=[2, 3, 5, 7, 10, 13],
                                       labels=["2-3Y", "3-5Y", "5-7Y", "7-10Y", "10-13Y"])
        for bucket, sub in merged.groupby("rem_bucket", observed=True):
            res = compute_ic(sub, best_flow_col, best_fwd_col)
            print(f"  {bucket}: N={res['n']:,}  IC={res['ic']:+.3f}  pval={res['pval']:.4f}")

    # 연도별 IC 안정성 (best 조합)
    if len(flat) > 0:
        best = flat.iloc[0]
        bf = f"{best['entity']}_{best['window']}"
        bw = f"fwd_dy_{int(best['horizon'])}_bp"
        print(f"\n=== 연도별 IC 안정성 ({best['entity']}_{best['window']} → {int(best['horizon'])}d) ===")
        merged["year"] = merged["price_date"].dt.year
        for yr, sub in merged.groupby("year"):
            res = compute_ic(sub, bf, bw)
            print(f"  {yr}: N={res['n']:,}  IC={res['ic']:+.3f}")

    # 4 주체별 평균 |IC| (어느 주체가 가장 유효한가)
    print("\n=== 주체별 평균 |IC| (모든 window × horizon) ===")
    for e in ENTITIES:
        sub = pd.DataFrame(ic_flat)
        sub = sub[sub["entity"] == e]
        print(f"  {e:12s}: 평균 |IC|={sub['IC'].abs().mean():.3f}  최대 |IC|={sub['IC'].abs().max():.3f}")

    print("\n[done]")


if __name__ == "__main__":
    main()
