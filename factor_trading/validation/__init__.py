"""Factor validation / diagnostics.

- ``rv_diagnostics`` : RV 팩터용 grid · within-bucket · drawdown · regime 분해
- (추후) ``orthogonality``  : 4팩터 공동 직교성 검증
"""
from .rv_diagnostics import (
    newey_west_long_run_var,
    forward_return_2f,
    quantile_labels_simple,
    quantile_labels_within_bucket,
    quantile_forward_returns,
    run_rv_grid,
    ls_stats,
    monotonicity,
    drawdown_stats,
    rate_regime_split,
    tail_event_impact,
)

__all__ = [
    "newey_west_long_run_var",
    "forward_return_2f",
    "quantile_labels_simple",
    "quantile_labels_within_bucket",
    "quantile_forward_returns",
    "run_rv_grid",
    "ls_stats",
    "monotonicity",
    "drawdown_stats",
    "rate_regime_split",
    "tail_event_impact",
]
