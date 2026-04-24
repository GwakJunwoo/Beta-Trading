"""Cached DataLoader — parquet 기반 (사내 DB 없이 Streamlit Cloud 용).

`DataLoader` 와 동일한 메서드 시그니처를 제공. parquet 파일에서 읽어온다.

사용
----
    from factor_trading.data_loader_cached import CachedDataLoader
    dl = CachedDataLoader(cache_dir="data_cache")
    dl.dy_panel()  # DataLoader 와 동일 결과

Export
------
    python factor_trading/scripts/export_for_cloud.py
    → data_cache/*.parquet 생성 (사내 네트워크에서만 실행)

파일 목록 (data_cache/)
-----------------------
- dy_3y.parquet          : dY_3Y_bp 시계열 (index=price_date, 1 col)
- dy_10y.parquet         : dY_10Y_bp 시계열
- dy_panel.parquet       : wide dY(bp) 패널 (index=date, cols=bond_code)
- ytm_panel_bp.parquet   : wide YTM(bp) 패널
- remain_panel.parquet   : wide 잔존만기(년) 패널
- rollover_flag.parquet  : 롤오버 플래그 시계열 (bool)
- instrument_meta.parquet: 종목 메타 (bond_code index, [category, label, bond_name] cols)
- meta.json              : as_of_date, latest benchmark codes
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class CachedDataLoader:
    """DataLoader 인터페이스 호환 (parquet 기반)."""

    cache_dir: str | Path = "data_cache"
    start: Optional[str] = None                     # 추가 slicing (원본 parquet 위)
    end:   Optional[str] = None
    categories: Sequence[str] = ("국고채",)          # 필터 옵션 (현재 단일 category 권장)

    # internal
    _meta_: dict = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        if not self.cache_dir.exists():
            raise FileNotFoundError(
                f"{self.cache_dir} 가 없습니다. 먼저 `python factor_trading/scripts/export_for_cloud.py` 실행."
            )

    # ---------------- helper ----------------
    def _load_series(self, fname: str, col: str) -> pd.Series:
        df = pd.read_parquet(self.cache_dir / fname)
        if isinstance(df, pd.Series):
            s = df
        else:
            s = df[col] if col in df.columns else df.iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()
        s = self._slice(s)
        return s

    def _load_frame(self, fname: str) -> pd.DataFrame:
        df = pd.read_parquet(self.cache_dir / fname)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return self._slice(df)

    def _slice(self, x):
        if self.start:
            x = x.loc[x.index >= pd.Timestamp(self.start)]
        if self.end:
            x = x.loc[x.index <= pd.Timestamp(self.end)]
        return x

    def meta(self) -> dict:
        if self._meta_ is None:
            with open(self.cache_dir / "meta.json", "r", encoding="utf-8") as f:
                self._meta_ = json.load(f)
        return self._meta_

    # ---------------- DataLoader 인터페이스 ----------------
    def dY_3y(self) -> pd.Series:
        s = self._load_series("dy_3y.parquet", "dY_3Y_bp")
        return s.rename("dY_3Y_bp").astype(float)

    def dY_10y(self) -> pd.Series:
        s = self._load_series("dy_10y.parquet", "dY_10Y_bp")
        return s.rename("dY_10Y_bp").astype(float)

    def rollover_flag(self) -> pd.Series:
        s = self._load_series("rollover_flag.parquet", "rollover")
        return s.astype(bool).rename("rollover")

    def dy_panel(self) -> pd.DataFrame:
        return self._load_frame("dy_panel.parquet").astype(float)

    def ytm_panel(self, unit: str = "bp") -> pd.DataFrame:
        df = self._load_frame("ytm_panel_bp.parquet").astype(float)
        if unit == "bp":
            return df
        if unit == "pct":
            return df / 100.0
        raise ValueError(f"unit must be 'bp' or 'pct', got {unit}")

    def remain_panel(self) -> pd.DataFrame:
        return self._load_frame("remain_panel.parquet").astype(float)

    def instrument_meta(self) -> pd.DataFrame:
        df = pd.read_parquet(self.cache_dir / "instrument_meta.parquet")
        return df

    def universe(self) -> pd.DataFrame:
        """Long-format universe가 필요한 경우만 로드. dy_panel/ytm_panel로 대부분 충분."""
        f = self.cache_dir / "universe.parquet"
        if f.exists():
            return self._slice(pd.read_parquet(f))
        # fallback: dy_panel → long format 재구성
        dy = self.dy_panel()
        ytm = self.ytm_panel(unit="pct")
        rem = self.remain_panel()
        long = (dy.stack().rename("dY_bp").reset_index()
                  .rename(columns={"level_0": "price_date", "level_1": "bond_code"}))
        return long

    def latest_3y_bench(self) -> Optional[str]:
        return self.meta().get("latest_3y_bench")

    def latest_10y_bench(self) -> Optional[str]:
        return self.meta().get("latest_10y_bench")

    # bench_3y / bench_10y 는 FactorPipeline 에서 직접 사용 안 함 → 생략
