"""RV 팩터 — 2팩터 잔차 ε의 horizon 누적 (명세 §3).

``RV_i,t(h) = Σ_{s=t-h+1..t} ε_i,s``

부호 해석
---------
- RV > 0 → 팩터 대비 yield 더 오름 (덜 내림) → 가격 underperform → **상대적으로 쌈** → LONG 후보
- RV < 0 → 가격 outperform → **상대적으로 비쌈** → SHORT 후보

포트폴리오 구성은 ``portfolio/single_factor.py``가 within-bucket quintile로 LS를 만든다.
"""
from __future__ import annotations

import pandas as pd

from ..residual_builder import accumulate_horizons, HORIZONS


def compute_rv(
    residual_panel: pd.DataFrame,
    horizon: str = "1m",
) -> pd.DataFrame:
    """잔차 패널로부터 RV score (horizon 누적 ε). 단위 bp.

    Parameters
    ----------
    residual_panel : DataFrame (index=date, columns=bond_code, values=ε_t)
    horizon        : '1d' | '1w' | '2w' | '1m'

    Returns
    -------
    DataFrame — RV_score (bp). 그대로 quintile 매김 인풋으로.
    """
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {list(HORIZONS)}, got {horizon!r}")
    out = accumulate_horizons(residual_panel, horizons={horizon: HORIZONS[horizon]})
    return out[horizon].rename_axis(columns="bond_code")
