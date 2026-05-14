# Fund Flow Alpha Research

> 목적: 개별 국채 종목의 수급 주체별 매매 flow → 미래 yield 변동 예측력 검증.
> 최종 목표: RV 페어 모델 (근접만기 long/short) 의 알파 소스 추가.

## 데이터

- 테이블: `ktb_trade_flow_features` (영문 컬럼)
- 기간: 2014-01-02 ~ 현재 (12년+)
- 종목: ~443 bond_codes (history 전체)
- 주체 (4): `foreigner`, `insurance`, `asset_mgmt`, `bank`
- 윈도우 (4): `diff_1d`, `sum_3d`, `sum_5d`, `sum_10d`
- 단위: 추정 억원 net buy (음수=net sell)

## 리서치 단계 (단계별로 합격해야 다음)

| Stage | 스크립트 | 목적 | 합격 기준 |
|---|---|---|---|
| **01** | `01_data_exploration.py` | 데이터 정합성, 분포, 커버리지 | 결측/이상치 식별, 단위 확인 |
| **02** | `02_signal_discovery.py` | flow ↔ 미래 yield 변동 상관 | 의미있는 IC (>0.05) 가진 (주체, window) 발견 |
| **03** | `03_predictive_test.py` | 시그널의 OOS 예측력 | 시그널 sharpe > 0.2 (단독) |
| **04** | `04_combine_with_eps.py` | RV ε 신호 + flow 결합 | combined sharpe > RV 단독 |
| **05** | `05_backtest_full.py` | 페어 진입에 flow 활용한 백테스트 | per_yr 개선 입증 |

각 단계 PASS 시 메인 시스템으로 promote (이 폴더 외부로 이동/통합).

## 검증 방법론

1. **Look-ahead 방지**: flow_t (오늘 종가 후 발표 가정) → predict ΔY_t+1, t+5, t+10
2. **분모 정규화**: 발행잔액 또는 일평균 거래량 대비 비율로 표준화 (단위 dependency 제거)
3. **잔여만기 bucket 분리**: 2-3Y / 3-5Y / 5-7Y / 7-10Y / 10-13Y
4. **종목별 z-score**: 종목 마다 자기 historical mean/std 로 표준화
5. **RV 페어 호환**: 같은 universe (잔존 2-13Y, issue ≤5y), 같은 신호 frequency

## 가설

1. **외국인 (foreigner)**: 대규모 매매, 정보우위 가정 → flow 후 yield 같은 방향 움직임
2. **연기금/보험 (insurance)**: 장기 buy-and-hold → 매수 시 점진적 강세
3. **자산운용 (asset_mgmt)**: 단기 매매, momentum-following → flow 후 follow-through
4. **은행 (bank)**: 보유 만기 매칭 위주 → 만기별 다른 패턴

## 기록

- 2026-05-12: 폴더 생성, exploration 시작
