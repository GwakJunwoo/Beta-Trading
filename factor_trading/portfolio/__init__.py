"""Portfolio layer — factor score → daily P&L (bp).

- ``single_factor.py``: cross-section factor (RV, VOL) → within-bucket quintile LS
                        time-series factor (MOM, CURVE) → bang-bang 현물/페어 P&L
- ``duration_neutral.py``: 3Y vs 10Y 페어 BPV 비중 (CURVE 트레이드용)
"""
from .single_factor import (
    within_bucket_quintile,
    xsec_ls_pnl,
    mom_pnl,
    curve_pnl,
)
from .duration_neutral import dv01_weights
from .combiner import (
    equal_weight, risk_parity, target_vol, combine_summary, stats_table,
    zscore_standardize, satellite_overlay,
)
from .dynamic_combiner import dynamic_weight_backtest, rank_weights

__all__ = ["within_bucket_quintile", "xsec_ls_pnl",
           "mom_pnl", "curve_pnl", "dv01_weights",
           "equal_weight", "risk_parity", "target_vol", "combine_summary",
           "stats_table", "zscore_standardize",
           "dynamic_weight_backtest", "rank_weights"]
