# 팩터 모델 → 실거래 포트폴리오 복제 — 방법론 연구

## 0. 문서 위치 & 목적

본 문서는 **현재 프로젝트의 본업(팩터 모델링)** 과 **실거래 운용**의 경계를 명확히 한다.

- 현재 프로젝트 = **팩터 분석 + 시장 변동 분해 + 직교성 검증** (`bond_3factor_final_spec.md`)
- 본 문서 = **팩터 모델 출력을 실거래로 옮길 때의 별도 연구 영역**

방법론 후보를 정리하고 단계적 접근을 제시한다. 실제 구현은 별도 sub-project (`factor_trading/replication/`) 에서 진행 예정.

---

## 1. 문제 정의

### 1.1 갭 (Gap)
- **이상적 팩터 포트폴리오** (모델 산출물)
  - RV: 109개 종목 cross-section 중 within-bucket Q5 16개 LONG / Q1 14개 SHORT, 균등 비중
  - MOM: 3Y 지표 single instrument 100% LONG/SHORT
  - CURVE: 3Y/10Y duration-neutral 페어, 매일 비중 재계산
- **실거래 가능한 포트폴리오** (현실)
  - 종목당 lot ≥ 100억 KRW (호가 단위)
  - SHORT은 RP 대차 가능 종목으로 한정 + 대차료 발생
  - 매일 30종목 페어 진입/청산 = 운용 inviable
  - 호가 깊이가 만기별로 다름 (지표 deep vs 비지표 thin)

→ **이상 포트폴리오를 실거래 가능한 소수 종목 + 라운드된 수량으로 변환** 필수.
이 변환 과정 = "복제 (replication)".

### 1.2 채권시장 특수성 (vs 주식 시장)
| 항목 | 주식 | KTB 현물 |
|---|---|---|
| 거래소 | 중앙집중 (KRX) | OTC 중심 (장외) |
| 호가 단위 | 1주 (수십 KRW) | 액면 100억 KRW 표준 lot |
| 종목 수 | ~2000 (KOSPI+KOSDAQ) | ~110 (active KTB) |
| SHORT | 차입공매도 (제한적) | RP 대차 (지표 외 어려움) |
| 호가 depth | 대형주 균질 | 지표 vs 비지표 격차 큼 |
| 거래비용 | 0.05~0.3% | 호가 1~3bp + 대차 비용 |
| Cross-section 헤지 | 가능 (수많은 종목) | 사실상 proxy 종목 선택 강제 |
| Dealer 시장조성 | 없음 | 핵심 (자체 inventory 운용) |

채권은 **dealer market** 이라 우리(트레이더)가 이미 inventory 보유 + 시장조성 운용 중. **순수 알고리즘 매매로 복제 X, dealer's book 위에 overlay** 가 현실 모델.

### 1.3 주식과 다른 결정적 제약
1. **Lot indivisibility**: 100억 단위로만 매매. 0.5종목 비중 같은 fractional 불가능.
2. **Universe 좁음**: 109종목 → top-N 선택 시 통계적 다양성 빠르게 줄어듦
3. **SHORT 비용**: 대차 가능 종목 일부만, 비용 연 5~30bp
4. **Inventory dependence**: 현재 보유 종목 raise/cut 부담이 새 종목 진입보다 작음
5. **Carry / Roll**: 보유 자체가 수익원. 매매로 잠시 비우는 것도 cost

---

## 2. 현재 시스템 산출물 vs 운용 가능 포지션

| Layer | 산출물 (이상) | 운용 변환 시 issue |
|---|---|---|
| **RV** | within-bucket Q5/Q1, 균등 비중 LS | 30종목 동시 매매 무리. SHORT 비용. lot 라운딩. |
| **MOM** | sign(-cum_dY_3Y_63d) × 3Y 지표 | 1종목이라 단순. 단 SHORT 시 대차. 매일 flip 가능 (turnover). |
| **CURVE** | sign(-cum_slope_21d) × 3Y/10Y 페어 dur-neutral | **연 34회 페어 flip**. 페어 round-trip × 2 leg → 비용 폭발. |
| **DYN/EW** | 위 3개 가중 합산 | 가중치 변경 시 미세 비중 조정 (lot 단위 라운딩). |

특히 CURVE 는 gross Sharpe 1.49 → **net 0.4 이하**로 추락 (이미 검증). 복제 단계에서 turnover 제약이 결정적.

---

## 3. 복제 방법론 후보

### A. Rule-based Heuristic (단순, 즉시)
사람이 정한 규칙으로 mapping:

```
# RV
Q5 LONG → 만기버킷별 top-3 (RV score 큰) 종목, 각 100억
Q1 SHORT → SHORT 가능 (대차 OK) 한정 + score top-3
total notional 600억

# MOM
signal=+1 → 3Y 지표 (현재 25-10) 500억 LONG, hedge 없음
signal=-1 → 3Y 지표 500억 SHORT (대차 가능 시) or skip

# CURVE
signal=+1 → 3Y 지표 LONG 300억 / 10Y 지표 SHORT 100억 (BPV 비율)
signal=-1 → 반대
주: 매일 flip 시 비용 폭발 → flip 시 hysteresis (3일 유지 후만 변경)

# 합성
DYN/EW 가중 → 위 3 portfolio를 가중 결합
```

**장점**: 즉시 운용 가능, 트레이더 직관 부합
**단점**: 추적 오차 큰 편, parameter 임의

### B. Constrained Optimization (정식 QP/MILP)

목적함수:
```
min ||w_actual - w_factor||²
  s.t.
    n_i ∈ ℤ × 100억               (lot 단위)
    Σ |Δn_i| × cost_i ≤ TC_budget  (turnover 한도)
    n_i ≥ 0  for i ∈ NoShort       (대차 불가 종목)
    Σ |n_i| ≤ Capital              (자본 한도)
    DV01_total ≤ DV01_limit        (리스크 한도)
```

**장점**: 추적 오차 minimize, 모든 제약 명시적
**단점**: MILP solver 필요, 매일 푸는 건 무리, parameter sensitive

### C. Factor-Mimicking Portfolio (FMP, 학계 표준)

각 팩터에 대해:
```
w_FMP = argmin Var(r_p − r_factor)
        s.t. β_p^factor = 1, β_p^other = 0
```

회귀 기반 가중치로 K개 종목 선별. 일반화된 sparse FMP는:
```
w = argmin Var(w'r − r_factor) + λ‖w‖₁     (LASSO regularization)
```

→ 자동 sparsity. 결과 0이 아닌 5~10 종목으로 충분히 추적.

**장점**: 학계 검증 풍부, 종목 수 자동 결정
**단점**: 채권의 lot 제약 정수화 미반영, 대차 제약 별도 처리 필요

### D. Active Inventory Management (dealer 통합)
이미 보유 중인 inventory + 시장 흐름 + 팩터 signal 을 **결합 optimization**:

```
maximize: factor_alpha + carry + market_making_spread − turnover_cost
  s.t.
    inventory_i ≥ 0  (long-only book)
    Δinventory ∈ ℤ × lot
    Risk constraints (DV01, VaR, concentration)
```

**장점**: 채권 dealer 현실 부합 (이미 보유 활용)
**단점**: dealer book + factor signal + market data 모두 통합 필요. 가장 복잡.

### E. Replication via Sleeves (복합 기법)
- **Core sleeve**: 지표 종목 (3Y, 5Y, 10Y, 30Y) 만으로 만기 부족분 충당
- **Satellite sleeve**: 비지표 RV signal 강한 top-3 만 추가
- 비중 = factor portfolio 의 만기 버킷 별 합

**장점**: 거래 부담 작음 (지표 위주), bid-ask 작음, 대차 용이
**단점**: 비지표 idiosyncratic alpha 일부 포기

---

## 4. 추적 오차 (Tracking Error) 분해

### 4.1 정의
```
TE_t = pnl_actual_t − pnl_factor_t
```

### 4.2 분해 (Performance attribution 풍)
```
TE = TE_selection + TE_timing + TE_size + TE_cost + TE_basis
```

- **TE_selection**: 종목 선택 차이 (factor portfolio 모든 종목 못 담아 발생)
- **TE_timing**: 매매 시점 차이 (실거래는 다음날 open or 점진 분할)
- **TE_size**: lot 라운딩으로 비중 차이
- **TE_cost**: bid-ask + market impact + 대차료
- **TE_basis**: 페어 BPV 비율 라운딩 (CURVE 같은 dur-neutral)

### 4.3 평가 지표
```
Information Ratio = mean(TE) / std(TE)
TE_total = annualized std(TE)
hit ratio = P(sign(pnl_actual) == sign(pnl_factor))
```

목표:
- TE_total < 0.5 × σ(pnl_factor)
- IR ≥ −0.3 (음수면 운용 손실 발생, 0이면 비용만큼 손실 정상)

---

## 5. 평가 Framework

### 5.1 multi-stage evaluation
```
Stage 0: Factor PnL (gross, 이상)               ← 현재 시스템
Stage 1: Factor PnL − 가정된 cost (1bp/round)   ← 명세 §13 "거래비용 미반영" 항목
Stage 2: Replicated portfolio gross             ← 본 연구 핵심
Stage 3: Replicated portfolio + 실제 비용
Stage 4: dealer inventory 효과 추가              ← 운용 단계
```

### 5.2 비교 metric
| metric | Factor (S0) | Replicated (S2) | 갭 분석 |
|---|---|---|---|
| Sharpe | 1.16 | 0.7~0.9 (예상) | TE_size + TE_selection |
| Max DD | -185bp | TE 추가로 더 깊을 가능성 | TE_timing |
| Hit% | 65% | 55~60% | TE_cost 영향 |
| Turnover | n/a | 측정 필수 | 직접 비용 척도 |
| Capital | n/a | 명시 필요 | book size 결정 |

---

## 6. 한국 KTB 특화 고려사항

### 6.1 종목 type별 특성
- **지표 종목 (25-10, 25-11 등)**: 호가 deep, 대차 OK, 발행잔액 큼 → **core sleeve 우선**
- **비지표 (구지표·재발행)**: 호가 thin, 대차 어려움, RV 신호 자주 → **satellite로 한정**
- **장기물 (20-30Y)**: 호가 매우 thin, 보험사 buyer 위주 → **단방향 LONG 만 현실적**
- **단기물 (1-3Y)**: 호가 OK, 단 RP 비용 vs MOM signal 비교 필요

### 6.2 시간적 cycle
- **월 25일 입찰**: 새 발행 종목 진입 → 신규 RV opportunity
- **분기 말 BS 마감**: SHORT unwind 압력
- **BOK 결정일 (매월 셋째주 목)**: vol 폭발, signal 신뢰도 일시 낮음
- **외국인 포지션**: 월말 / 분기말 flow 영향

### 6.3 운용 제약
- **자본 한도**: 트레이더별 book limit (예: 1000억 KRW)
- **DV01 한도**: 절대 듀레이션 노출 (예: ±50백만/bp)
- **단일종목 한도**: 한 종목 30% 미만
- **VaR 한도**: 95% 1-day VaR ≤ X bp
- **Carry 의무**: book 일부는 carry 목적 보유 (factor signal 무관)

---

## 7. 단계별 접근 권고 (구현 우선순위)

### Phase 1 — Heuristic Rule-Based (1~2주)
- 위 §3.A 규칙 코드화
- 실거래 환경 시뮬레이터 (lot, bid-ask, 대차료 가정)
- TE 측정 → factor S0 대비 gap 정량화
- **목표**: Replicated Sharpe ≥ 0.5

### Phase 2 — Sleeves Architecture (2~4주)
- §3.E Core/Satellite 분리
- Core: 지표 4~5종목 (3Y/5Y/10Y/30Y)
- Satellite: RV top-3, MOM 단일, CURVE 페어
- DYN 비중 → 두 sleeve 자본 배분으로 변환

### Phase 3 — Optimization-based (1~2개월)
- §3.B QP/MILP solver
- Cost-aware rebalancing (Almgren-Chriss style)
- Backtest sklearn or `cvxpy` 활용

### Phase 4 — Inventory Integration (장기)
- §3.D dealer book + factor 결합
- carry + spread + factor + risk multi-objective
- 실거래 OMS 연동 prerequisite

---

## 8. 부수 연구 주제

### 8.1 매매 손익 vs 팩터 손익 분리 (Performance Attribution)
딜러 입장에서:
```
Total P&L = Carry + Roll + Factor alpha + Market making spread + TE
          ─────────────  ─────────────  ──────────────────  ───
              passive       active           operational
```
각 component 정량화 → bonus / 평가 / 자본배분 기준.

### 8.2 시장조성 spread 와 factor signal 의 상호작용
- 우리가 호가 제출하는 종목 = 비대칭 정보 노출
- factor signal SHORT 종목에서 BID 호가 줄임 → spread 갭
- 자체 시장조성 행위가 factor 실현률에 영향

### 8.3 Lot-aware rebalancing
- factor 비중 변화 ≤ lot 미만이면 trade skip (no-trade band)
- 누적 deviation tracking → 임계 도달 시 일괄 rebalance
- → turnover 50% 감축 가능 (실증적으로)

### 8.4 RP 대차료 동적 모델링
- 종목별 대차 가능 잔량 + 비용 시계열
- factor SHORT signal 발동 시 비용 비교 후 substitute 종목 선택
- 비용 spike 시 SHORT 포지션 자동 reduce

### 8.5 OOS 추적 (Live Tracking)
- 실거래 시작 후 매일 (Replicated) vs (Factor S0) 차이 dashboard
- TE 누적 시계열, IR
- TE > 임계치면 알림 → 복제 모델 재튜닝

---

## 9. 결정 사항 / 논의 필요

- [ ] Phase 1 시작 시 자본 규모 (예: 500억? 1000억?)
- [ ] Long-only proxy vs LS — 회사 정책
- [ ] 대차 가능 종목 list 어떻게 받아올지 (RP 데스크 연동)
- [ ] 평가 단위: gross alpha vs net P&L vs IR
- [ ] Factor weight ↔ KRW notional 변환 공식 (target vol? target DV01?)
- [ ] Rebalance 주기: 일별 / 주별 / 월별 / signal-flip 기반

---

## 10. 본 문서 vs 현재 프로젝트 관계

```
현재 프로젝트 (factor_trading/)
├── 팩터 모델: RV / MOM / CURVE 정의·검증·합성
├── Daily monitor + Streamlit GUI
├── 이론 P&L (gross, lot/cost 무관)
└── ⬇ 출력 = "이상적 포트폴리오 비중 W*"

──────────  본 연구 영역 (factor_trading/replication/, 미구현)  ──────────

복제 모듈 (Phase 1~4)
├── lot/cost/SHORT/inventory 제약 적용
├── Realized portfolio = f(W*, market_state, dealer_book)
├── Tracking Error 측정·분해
└── ⬇ 출력 = "실제 매매 instruction"
```

본 연구는 **별도 sub-project로 분리**해서 진행. 현재 모니터링 시스템(Streamlit) 은 **이상 P&L 표시까지만 담당**. 복제 결과는 별도 운용 dashboard에서 (또는 같은 GUI에 새 탭 추가) 표시.

---

## 11. 참고 문헌·자료

- Almgren & Chriss (1999) "Optimal Execution of Portfolio Transactions"
- Grinold & Kahn "Active Portfolio Management" — Information Ratio 분해
- Connor & Korajczyk "A Test for the Number of Factors in an Approximate Factor Model"
- Garleanu & Pedersen (2013) "Dynamic Trading with Predictable Returns and Transaction Costs"
- 한국 KTB 시장 microstructure 자료: 한국채권평가원·KIS 보고서

---

*본 문서는 방법론 정리이며, 구체 구현은 별도 진행. 명세 변경 시 본 문서도 함께 업데이트.*
