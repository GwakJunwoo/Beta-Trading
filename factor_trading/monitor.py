"""프로덕션 일일 모니터링 엔진.

산출물
------
1. 오늘의 signal: MOM ±1, CURVE ±1, RV quintile top/bottom N 종목 리스트
2. 종목별 dY 3팩터 분해표:
     dY_i,t = α_i + β_3·dY_3,t + β_10·dY_10,t + ε_i,t
     → 각 날짜의 systematic 노출 / 잔차 분리
3. 최근 N일 성과 (20/63/252d), 각 팩터 + 합성 portfolio
4. Regime state 경고 (bull강·flat 등 취약 regime)
5. 시각화 PNG + CSV
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .main import FactorPipeline
from .factors.mom_factor import mom_raw_cum, mom_raw_signal
from .portfolio.single_factor import (
    within_bucket_quintile, xsec_ls_pnl, curve_pnl,
)
from .portfolio.combiner import equal_weight, risk_parity, stats_table


REGIME_BINS = [-np.inf, -25, -5, 5, 25, np.inf]
REGIME_LABELS = ["bull강(-25↓)", "bull약(-25~-5)", "flat(±5)",
                 "bear약(5~25)", "bear강(25↑)"]
REGIME_WARNING = {
    "RV":    {"bull강(-25↓)": "⚠️ RV Achilles' heel"},
    "MOM":   {"flat(±5)":    "⚠️ MOM flat regime 취약",
              "bull약(-25~-5)": "⚠️ MOM OOS weak"},
    "CURVE": {},  # all regimes robust
}


@dataclass
class DailyMonitor:
    """일일 스냅샷을 생성하는 모니터."""

    start: Optional[str] = None
    end:   Optional[str] = None
    categories: Sequence[str] = ("국고채",)
    out_dir: Optional[Path] = None

    # 파라미터 (spec §11)
    rv_horizon: str = "1m"
    rv_bucket_edges: list[float] = field(default_factory=lambda: [0, 5, 10, 100])
    rv_long_q: int = 5
    rv_short_q: int = 1
    mom_cum: int = 63
    curve_cum: int = 21

    # 외부 loader 주입 (예: CachedDataLoader for Streamlit Cloud). None → DB 기반.
    loader: object = None

    # 결과 (run 후 채워짐)
    pipe: Optional[FactorPipeline] = field(default=None, init=False, repr=False)
    as_of: Optional[pd.Timestamp] = field(default=None, init=False, repr=False)
    today_signals: Optional[dict] = field(default=None, init=False, repr=False)
    decomposition: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    recent_perf: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    warnings_: Optional[list[str]] = field(default=None, init=False, repr=False)
    pnls: Optional[dict[str, pd.Series]] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    def run(self, verbose: bool = True) -> "DailyMonitor":
        if verbose: print("[monitor] FactorPipeline …")
        self.pipe = FactorPipeline(start=self.start, end=self.end,
                                    categories=list(self.categories),
                                    rv_horizon=self.rv_horizon,
                                    loader=self.loader).run(verbose=False)
        dy3  = self.pipe.dl.dY_3y()
        dy10 = self.pipe.dl.dY_10y()
        dyP  = self.pipe.dl.dy_panel()
        ytmP = self.pipe.dl.ytm_panel(unit="bp")
        rem  = self.pipe.dl.remain_panel()
        eps  = self.pipe.residual
        beta3  = self.pipe.betas["beta_3y"]
        beta10 = self.pipe.betas["beta_10y"]
        meta = self.pipe.dl.instrument_meta()
        rv_score = self.pipe.rv_score

        self.as_of = dyP.index[-1]
        if verbose: print(f"[monitor] as_of = {self.as_of.date()}")

        # --- Signal 생성 ---
        slope = dy10.reindex(dy3.index) - dy3
        cum63 = mom_raw_cum(dy3, cum_window=self.mom_cum)
        mom_sig = mom_raw_signal(cum63, dead_zone_bp=0.0)
        cum21_slope = slope.rolling(self.curve_cum, min_periods=self.curve_cum // 2).sum()
        curve_sig = (-np.sign(cum21_slope)).rename("CURVE_sig")

        # --- 오늘의 signal ---
        today = self.as_of
        labels = within_bucket_quintile(rv_score, rem,
                                         bucket_edges=self.rv_bucket_edges, n_bins=5)
        today_labels = labels.loc[today].dropna()
        rv_long_bonds  = today_labels[today_labels == self.rv_long_q].index.tolist()
        rv_short_bonds = today_labels[today_labels == self.rv_short_q].index.tolist()

        # ranking within bucket (score 순)
        today_score = rv_score.loc[today].reindex(today_labels.index)
        rv_long_df = pd.DataFrame({
            "bond_code": rv_long_bonds,
            "rv_score_bp": today_score.reindex(rv_long_bonds).values,
            "remain_y":    rem.loc[today].reindex(rv_long_bonds).values,
            "ytm_bp":      ytmP.loc[today].reindex(rv_long_bonds).values,
        }).sort_values("rv_score_bp", ascending=False)
        rv_short_df = pd.DataFrame({
            "bond_code": rv_short_bonds,
            "rv_score_bp": today_score.reindex(rv_short_bonds).values,
            "remain_y":    rem.loc[today].reindex(rv_short_bonds).values,
            "ytm_bp":      ytmP.loc[today].reindex(rv_short_bonds).values,
        }).sort_values("rv_score_bp", ascending=True)

        # bond_name 매핑
        for df_ in [rv_long_df, rv_short_df]:
            df_["bond_name"] = df_["bond_code"].map(meta["bond_name"])

        self.today_signals = {
            "as_of": today,
            "MOM": {
                "signal": int(mom_sig.loc[today]) if pd.notna(mom_sig.loc[today]) else 0,
                "cum_dY_3Y_63d_bp": float(cum63.loc[today]) if pd.notna(cum63.loc[today]) else np.nan,
                "action": _mom_action(mom_sig.loc[today]),
            },
            "CURVE": {
                "signal": int(curve_sig.loc[today]) if pd.notna(curve_sig.loc[today]) else 0,
                "cum_slope_21d_bp": float(cum21_slope.loc[today]) if pd.notna(cum21_slope.loc[today]) else np.nan,
                "action": _curve_action(curve_sig.loc[today]),
            },
            "RV": {
                "long_bonds": rv_long_df,
                "short_bonds": rv_short_df,
            },
        }

        # --- 3팩터 분해표 (모든 종목 × 최근 지정 일) ---
        self.decomposition = _build_decomposition(
            dyP, dy3, dy10, beta3, beta10, eps, rem, meta,
            as_of=today, window=5,
        )

        # --- PnL 시계열 + 최근 성과 ---
        rv_pnl_daily = xsec_ls_pnl(rv_score, dyP, rem,
                                    bucket_edges=self.rv_bucket_edges,
                                    long_quintile=self.rv_long_q,
                                    short_quintile=self.rv_short_q, lag=1)["ls_bp"]
        mom_pnl = (mom_sig.shift(1) * (-dy3)).rename("MOM_pnl")
        curve_pnl_ = (curve_sig.shift(1) * slope).rename("CURVE_pnl")

        self.pnls = {"RV": rv_pnl_daily.dropna(),
                     "MOM": mom_pnl.dropna(),
                     "CURVE": curve_pnl_.dropna()}
        # 합성 (rolling σ로 표준화 — DYN과 동일 단위로 맞추어 공정 비교)
        ew = equal_weight(self.pnls, standardize=True, vol_window=63)
        rp = risk_parity(self.pnls, vol_window=63)
        self.pnls["EW_combo"] = ew.dropna()
        self.pnls["RP_combo"] = rp.dropna()

        # 최근 성과 (20d / 63d / 252d / all)
        windows = {"20d": 20, "63d": 63, "252d": 252, "all": None}
        rows = []
        for w_name, w in windows.items():
            for f, s in self.pnls.items():
                if w is None:
                    ss = s
                else:
                    ss = s.iloc[-w:] if len(s) >= w else s
                if len(ss) < 2: continue
                mu, sd = float(ss.mean()), float(ss.std(ddof=1))
                rows.append({
                    "factor": f, "window": w_name, "n": len(ss),
                    "cum_bp": float(ss.sum()),
                    "mean_bp": mu,
                    "sharpe_ann": mu / sd * np.sqrt(252) if sd > 0 else np.nan,
                    "hit%": float((ss > 0).mean()) * 100,
                })
        self.recent_perf = pd.DataFrame(rows)

        # --- Regime state 경고 ---
        f3m = dy3.rolling(63, min_periods=30).sum().loc[today]
        regime_idx = np.digitize([f3m], REGIME_BINS) - 1
        regime_idx = int(np.clip(regime_idx[0], 0, len(REGIME_LABELS) - 1))
        regime_now = REGIME_LABELS[regime_idx]
        warns = [f"현재 rate regime: {regime_now}  (3M 누적 dY_3Y = {f3m:+.1f}bp)"]
        for fct, flags in REGIME_WARNING.items():
            if regime_now in flags:
                warns.append(f"  {fct}: {flags[regime_now]}")
        self.warnings_ = warns

        if verbose:
            print(f"[monitor] signals: MOM={self.today_signals['MOM']['signal']}, "
                  f"CURVE={self.today_signals['CURVE']['signal']}")
            print(f"[monitor] RV Q{self.rv_long_q}: {len(rv_long_bonds)}개, "
                  f"Q{self.rv_short_q}: {len(rv_short_bonds)}개")
            for w in warns: print(f"[monitor] {w}")

        if self.out_dir:
            self.save(self.out_dir)

        return self

    # ------------------------------------------------------------------
    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        date_tag = self.as_of.strftime("%Y%m%d")

        # signals
        sig = self.today_signals
        sig["RV"]["long_bonds"].to_csv(
            out_dir / f"{date_tag}_RV_long.csv", index=False, encoding="utf-8-sig")
        sig["RV"]["short_bonds"].to_csv(
            out_dir / f"{date_tag}_RV_short.csv", index=False, encoding="utf-8-sig")

        # decomposition table
        self.decomposition.to_csv(
            out_dir / f"{date_tag}_decomposition.csv", index=False, encoding="utf-8-sig")

        # recent perf
        self.recent_perf.to_csv(
            out_dir / f"{date_tag}_recent_perf.csv", index=False, encoding="utf-8-sig")

        # PnL timeseries
        pd.DataFrame(self.pnls).to_csv(
            out_dir / f"{date_tag}_factor_pnls.csv", encoding="utf-8-sig")

        # text summary
        with open(out_dir / f"{date_tag}_summary.txt", "w", encoding="utf-8") as f:
            f.write(self.text_report())

    # ------------------------------------------------------------------
    def text_report(self) -> str:
        s = self.today_signals
        lines = []
        lines.append("="*70)
        lines.append(f"  채권 3팩터 일일 스냅샷 — as_of {self.as_of.date()}")
        lines.append("="*70)
        lines.append("")
        lines.append("## 경고 / 현재 regime")
        for w in self.warnings_:
            lines.append(f"  {w}")
        lines.append("")
        lines.append("## 오늘의 signal")
        lines.append(f"  MOM   : signal={s['MOM']['signal']:+d}  "
                     f"({s['MOM']['action']})  "
                     f"cum_dY_3Y_63d = {s['MOM']['cum_dY_3Y_63d_bp']:+.2f}bp")
        lines.append(f"  CURVE : signal={s['CURVE']['signal']:+d}  "
                     f"({s['CURVE']['action']})  "
                     f"cum_slope_21d = {s['CURVE']['cum_slope_21d_bp']:+.2f}bp")
        lines.append("")
        lines.append(f"## RV Long (Q{self.rv_long_q}) — 상위 10종목 (cheap LONG 후보)")
        lines.append(s["RV"]["long_bonds"].head(10).to_string(index=False,
            float_format=lambda v: f"{v:+,.2f}" if abs(v) < 1000 else f"{v:,.0f}"))
        lines.append("")
        lines.append(f"## RV Short (Q{self.rv_short_q}) — 상위 10종목 (rich SHORT 후보)")
        lines.append(s["RV"]["short_bonds"].head(10).to_string(index=False,
            float_format=lambda v: f"{v:+,.2f}" if abs(v) < 1000 else f"{v:,.0f}"))
        lines.append("")
        lines.append("## 최근 성과 (daily PnL 기준)")
        pivot = self.recent_perf.pivot(index="factor", columns="window", values="sharpe_ann")
        lines.append("  Sharpe(ann):")
        lines.append(pivot.to_string(float_format=lambda v: f"{v:+,.2f}"))
        lines.append("")
        lines.append("## 3팩터 분해표 (최근 5영업일, 선별 종목)")
        lines.append(self.decomposition.head(30).to_string(index=False,
            float_format=lambda v: f"{v:+,.2f}" if abs(v) < 1000 else f"{v:,.0f}"))
        return "\n".join(lines)


def _mom_action(signal) -> str:
    if pd.isna(signal) or signal == 0: return "중립"
    return "LONG 3Y bond (rate 하락 기대)" if signal > 0 else "SHORT 3Y bond (rate 상승 기대)"


def _curve_action(signal) -> str:
    if pd.isna(signal) or signal == 0: return "중립"
    return ("STEEPEN (3Y LONG / 10Y SHORT dur-neutral)" if signal > 0 else
            "FLATTEN (3Y SHORT / 10Y LONG dur-neutral)")


def _build_decomposition(
    dyP: pd.DataFrame,
    dy3: pd.Series,
    dy10: pd.Series,
    beta3: pd.DataFrame,
    beta10: pd.DataFrame,
    eps: pd.DataFrame,
    rem: pd.DataFrame,
    meta: pd.DataFrame,
    as_of: pd.Timestamp,
    window: int = 5,
) -> pd.DataFrame:
    """종목별 dY 분해 테이블 (최근 window 일, 모든 종목).

    각 (date, bond)에 대해:
        dY_i,t = sys_3Y_i,t + sys_10Y_i,t + ε_i,t
         sys_3Y_i,t  = β_3Y,i,t · dY_3Y,t
         sys_10Y_i,t = β_10Y,i,t · dY_10Y,t
    """
    days = dyP.index[-window:]
    rows = []
    for dt in days:
        if dt not in beta3.index: continue
        b3  = beta3.loc[dt]
        b10 = beta10.loc[dt]
        e   = eps.loc[dt] if dt in eps.index else pd.Series(dtype=float)
        dy_all = dyP.loc[dt]
        r      = rem.loc[dt] if dt in rem.index else pd.Series(dtype=float)
        for code in dy_all.dropna().index:
            if pd.isna(b3.get(code)) or pd.isna(b10.get(code)):
                continue
            sys3  = float(b3[code]  * dy3.loc[dt])  if dt in dy3.index  and pd.notna(dy3.loc[dt])  else np.nan
            sys10 = float(b10[code] * dy10.loc[dt]) if dt in dy10.index and pd.notna(dy10.loc[dt]) else np.nan
            rows.append({
                "date": dt.date(),
                "bond_code": code,
                "bond_name": meta.loc[code, "bond_name"] if code in meta.index else "",
                "remain_y": float(r.get(code, np.nan)) if hasattr(r, "get") else np.nan,
                "dY_bp":    float(dy_all[code]),
                "sys_3Y_bp":  sys3,
                "sys_10Y_bp": sys10,
                "epsilon_bp": float(e.get(code, np.nan)) if hasattr(e, "get") else np.nan,
                "beta_3Y":  float(b3[code]),
                "beta_10Y": float(b10[code]),
            })
    df = pd.DataFrame(rows)
    # 최근 날짜 종목 정렬: |epsilon| 큰 순
    if not df.empty:
        latest = df[df["date"] == df["date"].max()].copy()
        latest["abs_eps"] = latest["epsilon_bp"].abs()
        latest = latest.sort_values("abs_eps", ascending=False)
        other = df[df["date"] != df["date"].max()]
        df = pd.concat([latest.drop(columns="abs_eps"), other], ignore_index=True)
    return df
