"""채권 4팩터 RV 트레이딩 시스템 (RV / VOL / MOM / CURVE).

명세: ``factor_trading/bond_4factor_framework_spec.md``

핵심 회귀:
    dY_i,t = α_i + β_i^3Y · dY_3Y,t + β_i^10Y · dY_10Y,t + ε_i,t

여기서 4팩터를 추출:
    RV    = ε 수준 (cross-section)         → quintile LS within 만기 버킷
    VOL   = rolling_std(ε) (cross-section)  → quintile LS within 만기 버킷
    MOM   = z-score(21d 누적 dY_3Y)         → 3Y 현물 bang-bang
    CURVE = (dY_10Y − dY_3Y)을 dY_3Y에 회귀한 잔차의 z-score → 3Y/10Y duration-neutral 페어
"""

__all__: list[str] = []
