"""4팩터 RV 시스템 — 한 번에 계산하는 오케스트레이터.

사용 예::

    from factor_trading.main import FactorPipeline
    pipe = FactorPipeline(start="2022-01-01", end="2026-04-22").run()
    pipe.rv_score, pipe.vol_score, pipe.mom_z, pipe.curve_z   # 팩터 스코어
    pipe.rv_pnl, pipe.vol_pnl, pipe.mom_pnl_, pipe.curve_pnl_ # daily P&L (bp)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import pandas as pd

from .data_loader import DataLoader
from .beta_estimator import estimate_all_betas_2f, sanity_check
from .residual_builder import residual_panel_2f, accumulate_horizons, HORIZONS
from .factors import compute_rv, compute_vol, compute_mom, compute_curve
from .factors.mom_factor import mom_signal
from .factors.curve_factor import curve_signal
from .portfolio import xsec_ls_pnl, mom_pnl, curve_pnl


@dataclass
class FactorPipeline:
    """4팩터 점수 + daily P&L 전체 산출."""

    start: Optional[str] = None
    end:   Optional[str] = None
    categories: Sequence[str] = ("국고채",)
    rv_horizon: str = "1m"                      # 주력 (명세 §12)
    vol_window: int = 20
    mom_cum: int = 21
    mom_zwin: int = 252
    curve_resid_win: int = 63
    curve_zwin: int = 21
    curve_direction: str = "mean_rev"           # {mean_rev, momentum}
    bucket_edges: list[float] = field(default_factory=lambda: [0, 5, 10, 100])

    # 산출물
    dl: Optional[DataLoader]      = field(default=None, init=False, repr=False)
    betas: Optional[dict]         = field(default=None, init=False, repr=False)
    residual: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    rv_score:  Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    vol_score: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    mom_z:     Optional[pd.Series]    = field(default=None, init=False, repr=False)
    curve_z:   Optional[pd.Series]    = field(default=None, init=False, repr=False)

    rv_pnl:    Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    vol_pnl:   Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    mom_pnl_:  Optional[pd.Series]    = field(default=None, init=False, repr=False)
    curve_pnl_: Optional[pd.Series]   = field(default=None, init=False, repr=False)

    sanity: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    # 외부에서 loader 주입 가능 (예: CachedDataLoader). None 이면 DB 기반 DataLoader 생성.
    loader: object = None

    def run(self, verbose: bool = True) -> "FactorPipeline":
        if verbose: print("[1/5] DataLoader …")
        if self.loader is not None:
            self.dl = self.loader
        else:
            self.dl = DataLoader(start=self.start, end=self.end, categories=list(self.categories))
        dy3 = self.dl.dY_3y()
        dy10 = self.dl.dY_10y()
        dy_panel = self.dl.dy_panel()
        remain   = self.dl.remain_panel()
        rollover = self.dl.rollover_flag()

        if verbose: print(f"  universe={dy_panel.shape[1]}  days={dy_panel.shape[0]}  "
                          f"3Y_days={dy3.notna().sum()} 10Y_days={dy10.notna().sum()}  "
                          f"rollover_days={int(rollover.sum())}")

        if verbose: print("[2/5] 2-factor β (rolling OLS) …")
        self.betas = estimate_all_betas_2f(dy_panel, dy3, dy10, rollover=rollover)

        if verbose: print("[3/5] Residual ε + horizon 누적 …")
        self.residual = residual_panel_2f(dy_panel, dy3, dy10, self.betas)

        # Sanity check
        self.sanity = sanity_check(
            self.betas,
            bench_3y=self.dl.latest_3y_bench(),
            bench_10y=self.dl.latest_10y_bench(),
            remain=remain,
        )
        if verbose and not self.sanity.empty:
            print("  [sanity] β (mean over time):")
            print(self.sanity.to_string(index=False,
                    float_format=lambda v: f"{v:+,.3f}"))

        if verbose: print("[4/5] 4팩터 스코어 …")
        self.rv_score  = compute_rv(self.residual, horizon=self.rv_horizon)
        self.vol_score = compute_vol(self.residual, window=self.vol_window)
        self.mom_z     = compute_mom(dy3, cum_window=self.mom_cum, zscore_win=self.mom_zwin)
        self.curve_z   = compute_curve(dy3, dy10,
                                       resid_window=self.curve_resid_win,
                                       z_window=self.curve_zwin)

        if verbose: print("[5/5] 팩터별 daily P&L …")
        # RV: Q5 LONG / Q1 SHORT (cheap LONG / rich SHORT)
        self.rv_pnl = xsec_ls_pnl(
            self.rv_score, dy_panel, remain,
            bucket_edges=self.bucket_edges,
            long_quintile=5, short_quintile=1, lag=1,
        )
        # VOL: 초기 가설 = low-vol LONG (Q1 LONG / Q5 SHORT)
        self.vol_pnl = xsec_ls_pnl(
            self.vol_score, dy_panel, remain,
            bucket_edges=self.bucket_edges,
            long_quintile=1, short_quintile=5, lag=1,
        )
        # MOM
        m_sig = mom_signal(self.mom_z, dead_zone=0.0)
        self.mom_pnl_ = mom_pnl(m_sig, dy3, lag=1)
        # CURVE (초기 가설: mean reversion)
        c_sig = curve_signal(self.curve_z, direction=self.curve_direction, dead_zone=0.0)
        self.curve_pnl_ = curve_pnl(c_sig, dy3, dy10, lag=1)

        if verbose: print("✓ pipeline done.")
        return self

    # ---------- 편의 ----------
    def pnl_frame(self) -> pd.DataFrame:
        """4팩터 daily P&L(bp)을 하나의 DataFrame으로."""
        return pd.DataFrame({
            "RV":    self.rv_pnl["ls_bp"]    if self.rv_pnl is not None else None,
            "VOL":   self.vol_pnl["ls_bp"]   if self.vol_pnl is not None else None,
            "MOM":   self.mom_pnl_,
            "CURVE": self.curve_pnl_,
        }).dropna(how="all")

    def summary(self) -> pd.DataFrame:
        pnl = self.pnl_frame()
        def _stats(s: pd.Series) -> dict:
            s = s.dropna()
            if s.empty:
                return dict(n=0, mean=float("nan"), std=float("nan"),
                            sharpe_1d_ann=float("nan"), hit=float("nan"))
            mu, sd = float(s.mean()), float(s.std(ddof=1))
            return dict(n=len(s), mean_bp=mu, std_bp=sd,
                        sharpe_1d_ann=mu/sd*252**0.5 if sd > 0 else float("nan"),
                        hit_pct=float((s > 0).mean()) * 100)
        return pd.DataFrame({c: _stats(pnl[c]) for c in pnl.columns}).T
