"""4팩터 모듈: RV / VOL / MOM / CURVE.

공통 출력 규약
--------------
- Cross-section 팩터 (RV, VOL): ``pd.DataFrame`` (index=date, columns=bond_code, values=score)
- Time-series 팩터 (MOM, CURVE): ``pd.Series`` (index=date, values=z-score)

포트폴리오 변환은 ``portfolio/single_factor.py``가 담당.
"""
from .rv_factor import compute_rv
from .vol_factor import compute_vol
from .mom_factor import compute_mom
from .curve_factor import compute_curve

__all__ = ["compute_rv", "compute_vol", "compute_mom", "compute_curve"]
