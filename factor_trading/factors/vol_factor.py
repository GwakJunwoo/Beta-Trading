"""VOL 팩터 — 잔차의 20일 rolling std (명세 §4).

``VOL_i,t = std(ε_{i, t-19..t})``  (단위 bp, 20영업일 윈도우)

주의
----
- Lag **없음** (t 포함). RV와의 mechanical correlation 감시 필요.
- Within-bucket quintile 필수 (장기물이 자연히 vol 크므로).
- 방향 (low-vol LONG vs high-vol LONG) 은 백테스트로 결정 — 채권 시장에서
  어느 쪽이 premium 있는지 실증 필요.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_vol(
    residual_panel: pd.DataFrame,
    window: int = 20,
    min_periods: int = 15,
) -> pd.DataFrame:
    """잔차의 rolling std. score 단위 bp.

    Parameters
    ----------
    residual_panel : DataFrame (index=date, columns=bond_code, values=ε)
    window         : 20 (명세)
    min_periods    : 비결측 최소 (기본 15, =75%)

    Returns
    -------
    DataFrame — VOL_i,t (bp)
    """
    vol = residual_panel.rolling(window, min_periods=min_periods).std(ddof=1)
    return vol
