"""동적 비중 리밸런싱 합성 (rank-based momentum-of-factor).

아이디어
--------
매 리밸런싱 시점에 각 팩터의 최근 성과 (1w/2w/1m Sharpe 가중평균) 로 순위 매김.
Rank 1위 → w_max, 3위 → w_min, 2위 → w_mid (=1−max−min) 할당.
Rank 순위만 사용하므로 score 절대값 robust. 다음 리밸런싱까지 비중 유지.

주의
----
- 팩터 vol scale 차이 (MOM std 4bp vs RV std 1.2bp)로 인한 왜곡을 막으려면
  **표준화 후 가중합** 권장 (standardize=True).
- Look-ahead 방지: score 계산에 당일 PnL 제외, 표준화 σ도 lag=1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _sharpe_ann(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 2: return 0.0
    sd = float(s.std(ddof=1))
    if sd <= 0: return 0.0
    return float(s.mean()) / sd * np.sqrt(252.0)


def rank_weights(
    scores: dict[str, float],
    w_min: float, w_max: float,
) -> dict[str, float]:
    """3팩터 score → rank 기반 비중.

    - 1위: w_max
    - 3위: w_min
    - 2위: 1 − w_max − w_min
    동률 시 입력 순서대로 break.
    """
    assert 0 < w_min < 1/3 < w_max < 1, f"min/max 범위 이상: {w_min}, {w_max}"
    w_mid = 1.0 - w_max - w_min
    assert abs((w_min + w_mid + w_max) - 1.0) < 1e-9
    sorted_names = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    out = {sorted_names[0]: w_max, sorted_names[1]: w_mid, sorted_names[2]: w_min}
    return out


def dynamic_weight_backtest(
    pnls: dict[str, pd.Series],
    rebalance_days: int = 21,
    lookback_1w: int = 5,
    lookback_2w: int = 10,
    lookback_1m: int = 21,
    score_w1w: float = 0.5,
    score_w2w: float = 0.3,
    score_w1m: float = 0.2,
    w_min: float = 0.15,
    w_max: float = 0.55,
    vol_window: int = 63,
    standardize: bool = True,
    warmup: int | None = None,
) -> dict:
    """동적 rank-based 가중 리밸런싱 백테스트.

    Returns
    -------
    dict with keys:
        pnl_dyn   : pd.Series   동적 포트폴리오 daily PnL (standardize=True면 unit scale)
        weights   : pd.DataFrame (index=date, cols=factor) 시점별 비중
        rebalance_log : pd.DataFrame  리밸런싱 이력 (각 시점 scores + 비중)
    """
    # 0. 전처리
    df = pd.DataFrame(pnls).copy()
    df = df.dropna(how="all")
    names = list(df.columns)
    n_fac = len(names)
    assert n_fac == 3, f"현재는 정확히 3팩터 지원: got {names}"

    warmup = warmup if warmup is not None else max(vol_window, lookback_1m) + 5

    # 1. 표준화용 σ (shift로 look-ahead 방지)
    if standardize:
        vol = df.rolling(vol_window, min_periods=max(30, vol_window // 2)).std(ddof=1)
        vol = vol.shift(1)                         # t의 포지션엔 t−1까지의 vol만 사용
        z = df / vol.replace(0, np.nan)
    else:
        z = df.copy()
    z = z.fillna(0)

    # 2. daily 비중 — 리밸런싱 사이 유지
    w_df = pd.DataFrame(1.0 / n_fac, index=df.index, columns=names, dtype=float)
    current_w = {n: 1.0 / n_fac for n in names}

    rebal_rows = []
    last_rebal_idx = None

    for i, dt in enumerate(df.index):
        if i < warmup:
            continue
        trigger = (last_rebal_idx is None) or (i - last_rebal_idx >= rebalance_days)
        if trigger:
            # 당일 제외, 과거 1m window로 score
            hist = df.iloc[:i]                      # exclusive (i번째 포함 안 함)
            if len(hist) < lookback_1m:
                continue
            scores: dict[str, float] = {}
            s_parts = {}
            for n in names:
                s_1w = _sharpe_ann(hist[n].iloc[-lookback_1w:])
                s_2w = _sharpe_ann(hist[n].iloc[-lookback_2w:])
                s_1m = _sharpe_ann(hist[n].iloc[-lookback_1m:])
                scores[n] = score_w1w * s_1w + score_w2w * s_2w + score_w1m * s_1m
                s_parts[n] = (s_1w, s_2w, s_1m)
            current_w = rank_weights(scores, w_min, w_max)
            last_rebal_idx = i
            rebal_rows.append({
                "date": dt,
                **{f"Sh_1w_{n}": s_parts[n][0] for n in names},
                **{f"Sh_2w_{n}": s_parts[n][1] for n in names},
                **{f"Sh_1m_{n}": s_parts[n][2] for n in names},
                **{f"score_{n}": scores[n] for n in names},
                **{f"w_{n}": current_w[n] for n in names},
            })
        w_df.iloc[i] = [current_w[n] for n in names]

    # 3. PnL (표준화된 또는 raw)
    pnl_dyn = (z * w_df).sum(axis=1).rename("DYN")

    return {
        "pnl_dyn": pnl_dyn,
        "weights": w_df,
        "z_pnl": z,
        "rebalance_log": pd.DataFrame(rebal_rows),
        "params": {
            "rebalance_days": rebalance_days,
            "lookback_1w": lookback_1w, "lookback_2w": lookback_2w, "lookback_1m": lookback_1m,
            "score_w1w": score_w1w, "score_w2w": score_w2w, "score_w1m": score_w1m,
            "w_min": w_min, "w_max": w_max,
            "vol_window": vol_window, "standardize": standardize, "warmup": warmup,
        },
    }
