"""2팩터 잔차(ε) 계산 + horizon 누적 (명세 §3).

    ε_i,t = dY_i,t − β^3Y_i,t · dY_3Y,t − β^10Y_i,t · dY_10Y,t

RV_i,t(h) = Σ_{s=t-h+1..t} ε_i,s  (h ∈ {1d, 1w, 2w, 1m})
"""
from __future__ import annotations

import numpy as np
import pandas as pd


HORIZONS: dict[str, int] = {"1d": 1, "1w": 5, "2w": 10, "1m": 21}


def residual_panel_2f(
    dy_panel: pd.DataFrame,
    dy_3y:   pd.Series,
    dy_10y:  pd.Series,
    betas: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """ε_i,t. betas = {'beta_3y': DF, 'beta_10y': DF}."""
    idx = dy_panel.index
    P   = dy_panel.reindex(idx).astype(float)
    F3  = dy_3y.reindex(idx).astype(float)
    F10 = dy_10y.reindex(idx).astype(float)
    B3  = betas["beta_3y"].reindex(index=idx, columns=P.columns).astype(float)
    B10 = betas["beta_10y"].reindex(index=idx, columns=P.columns).astype(float)
    return P.sub(B3.mul(F3, axis=0)).sub(B10.mul(F10, axis=0))


def accumulate_horizons(
    residual: pd.DataFrame,
    horizons: dict[str, int] = HORIZONS,
    min_ratio: float = 0.8,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name, w in horizons.items():
        min_p = max(1, int(np.ceil(w * min_ratio)))
        out[name] = residual.rolling(w, min_periods=min_p).sum()
    return out
