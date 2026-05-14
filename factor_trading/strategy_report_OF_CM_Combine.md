# 채권 상대가치 트레이딩 전략 종합 보고서 — OF / CM / Combine

작성: 2026-05-13
대상: 한국 국고채 (KTB) 비지표 ↔ 지표 페어 LS 시스템
백테스트 구간: 2017-01 ~ 2026-04-30 (9년)

---

## 0. Executive Summary

세 가지 alpha source를 결합한 통합 페어 LS 시스템 확정.

| 모델 | 신호 source | 단독 net alpha | 운용 status |
|---|---|---|---|
| **OF** | monthly_z (계절 conditional spread) | **+73 bp/y** | ✅ 운용 채택 |
| **CM** | abn_z (panel OLS residual rolling z) | +34 bp/y (v2) / −91 (v4 정직) | ⚠️ 보조 layer only |
| **RV** | Level cum_ε_21d (2-factor 잔차 21d 누적) | Sharpe 7.21 (단일 팩터) | ✅ Combine 입력 |
| **Combine** | RV·OF·CM Sharpe-prop 가중 cross-section rank | **+30.7 bp/y (LONG only)** | ✅ 백테스트 확정 |

**핵심 결정**: Combine 모델에서 **비지표 LONG only** 방향만 채택. SHORT side는 9년 평균 −1.24 bp/trade, 승률 22%로 실증 손실 영역.

---

## 1. 시장 미시구조 분석 — OF 모델의 동기

OF (On-Off) 모델은 두 가지 한국 국고채 시장 고유 현상에 기반.

### 1.1 명제 A — 비지표 spread의 강한 계절성

**관측**: 비지표(non-bench) 종목의 보간 yield curve 대비 spread는 **월별로 평균 수준이 체계적으로 다름**.

| 월 | 비지표 평균 spread (bp) | 비지표 평균 spread 표준편차 | 거래 net 평균 (bp/trade) |
|---|---|---|---|
| 1월 | +3.1 | 2.5 | **+1.70** (강한 mean-rev) |
| 3월 | +3.4 | 2.6 | **+1.81** |
| 6월 | +3.8 | 2.7 | **+2.36** (가장 강함) |
| 7월 | +3.5 | 2.5 | +0.22 |
| 8월 | +2.8 | 2.4 | −0.54 |
| 9월 | +2.5 | 2.3 | −0.55 |
| 10월 | +2.2 | 2.2 | **−0.86** (약함) |
| 11월 | +2.3 | 2.2 | **+0.14** (가장 약함, 일부 분석 −1.09) |
| 12월 | +2.8 | 2.4 | +0.33 |

**해석**:
- **1·3·6월** 은 분기·반기 자금 흐름, 새 발행 calendar 영향으로 비지표가 일시적으로 cheap 한 빈도가 높음 → mean-rev 신호 강함
- **10·11월** 은 연말 portfolio 재조정, 신지표 직전 narrative 활성화로 mean-rev 약함

**활용**: spread 자체를 절대값 기준으로 비교하면 평소 +3 bp 인 6월과 평소 +2 bp 인 10월을 동일하게 평가하게 됨 → 그 월의 expanding pool로 정규화한 **monthly z-score** 필요.

### 1.2 명제 B — 신규지표 편입·탈락 이벤트 효과

**Event study (2015~2026, 93건 지표 교체)**:

#### B-1. **신지표 편입 직전 spread 가속 압축 (richer → richest)**

새 지표로 편입되기 직전 60영업일간 spread 평균 경로:

| T (영업일) | n | mean spread (bp) | 해석 |
|---|---|---|---|
| T−60 | 10 | −0.66 | 약간 rich |
| T−30 | 35 | −1.21 | rich 가속 |
| T−15 | 66 | **−1.81** | 매우 rich |
| T−10 | 75 | −2.18 | |
| T−5 | 77 | **−2.38** | |
| **T−1** | **78** | **−2.61** | **최대 rich** |
| T0 | — | (편입 후 spread 정의 X) | |

**해석**:
- 시장이 다음 지표 후보를 미리 알고 매수 → 편입 직전 spread가 −0.66 → −2.61 bp로 가속 압축
- 이 종목들에 **mean-rev를 베팅하면 (rich → fair) 신지표 narrative가 깨질 때까지 손실**
- → OF 블랙리스트 + Combine에서 SHORT 차단의 근거

#### B-2. **지표 탈락 직후 spread 즉시 점프 + 느린 회복**

지표에서 탈락한 직후 60영업일간 spread 평균 경로:

| T (영업일) | n | mean spread (bp) | 해석 |
|---|---|---|---|
| T0 | 71 | **+2.82** | **즉시 +2.8 bp 점프** |
| T+1 | 71 | +2.95 | |
| T+5 | 71 | +2.80 | |
| T+10 | 71 | +2.69 | |
| T+21 | 71 | +2.65 | |
| T+30 | 71 | +2.31 | 점진 감소 |
| T+45 | 69 | +2.11 | |
| T+60 | 69 | +2.36 | **여전히 +2.3 bp 잔존** |

**해석**:
- 지표에서 빠진 즉시 +2.82 bp jump (rich → fair → cheap 가속)
- mean-rev는 느림 (60일 후도 +2.3 bp 유지) → **LONG 베팅 mean-rev 신호로 활용 가능, 단 hold 짧으면 일부만 회수**
- → OF 모델 event_drop overlay (탈락 후 30bd 내 LONG bonus) 근거

### 1.3 명제 A + B → OF 신호 설계

| 명제 | 신호 활용 |
|---|---|
| A. 계절성 | spread를 **그 월의 expanding pool 평균/표준편차로 정규화** → monthly_z |
| B-1. 신지표 편입 직전 | **블랙리스트 (LONG 베팅 차단)** — 신3년·5년·10년 직전 후보 자동 감지 |
| B-2. 탈락 직후 | LONG 신호 boost (event overlay, T+0 ~ T+30bd) |

---

## 2. OF 모델 — On-Off Trading (Strategy D 2-10y 4-anchor)

### 2.1 Concept

각 비지표 종목의 spread 가 **그 월의 historical cross-section 분포**에서 얼마나 극단적인지 (= monthly z-score)를 신호로 활용 → mean-rev 페어 트레이드.

### 2.2 Universe

- 비지표 국고채, **잔존만기 2 ~ 10 년**
- DB `ktb` 테이블, `category = '국고채'`
- 비지표 = 매일 5대 지표 (2/3/5/10년/20년/30년지표) bond_code 집합 외 종목

### 2.3 Yield Curve Anchors

| Label | Remain (y) |
|---|---|
| 2년지표 | 2.0 |
| 3년지표 | 3.0 |
| 5년지표 | 5.0 |
| 10년지표 | 10.0 |

매일 anchor 4종의 (remain, ytm) 으로 piecewise linear 보간 curve 구성.

### 2.4 Spread 정의

```
spread_bp_i,t = (ytm_i,t − interp_ytm_i,t) × 100   [bp]

interp_ytm_i,t = 위 4 anchor 의 (remain, ytm) 으로 linear 보간
                  evaluated at remain_i,t
```

- spread > 0: 보간선 위 (yield 높음 = 가격 낮음 = **cheap**, LONG 후보)
- spread < 0: 보간선 아래 (yield 낮음 = 가격 높음 = **rich**, SHORT 후보)

### 2.5 Monthly Z-Score (look-ahead-free)

매일 t, 각 비지표 i:

```
month_t = t.month   (1~12)

month_mean_t = Σ spread_bp_j,s for all j,s where s<t AND s.month==month_t
month_std_t  = std of same pool
              (cumulative obs ≥ 100 required, else NaN)

monthly_z_i,t = (spread_bp_i,t − month_mean_t) / month_std_t
```

→ t 시점 이전의 그 월 데이터만 사용 (expanding past-pool).

### 2.6 Pair 구성 (BPV-neutral)

각 비지표 i의 nearest anchor label:
- remain ∈ [2, 2.5): 2y
- remain ∈ [2.5, 4): 3y or 5y (가까운 쪽)
- remain ∈ [4, 7.5): 5y
- remain ∈ [7.5, 10]: 10y

→ 그 anchor의 그 일자 `bench_code` 와 페어:

```
direction = sign(monthly_z)
  z > 0 (cheap)  → LONG nonbench / SHORT bench
  z < 0 (rich)   → SHORT nonbench / LONG bench

Sizing (BPV-neutral):
  notional_bench = notional_nonbench × dur_nonbench / dur_bench
  → DV01 = 0
```

### 2.7 Entry / Exit

| 항목 | 값 |
|---|---|
| Entry threshold | `|monthly_z| > 2.0` |
| 일별 max 신규 페어 | 4 |
| 종목별 동시 active | ≤ 3 |
| Target | +5 bp |
| Stop | −5 bp |
| Time exit | 21 영업일 |
| Cost | 1 bp/trade (진+청 합산) |

### 2.8 백테스트 성과 (2017~2026, 9년)

| 지표 | 값 |
|---|---|
| N trades | 1,463 |
| trades/yr | 129 |
| win % (gross / **net**) | 60.8% / **56.5%** |
| mean (gross / **net**) | +1.56 / **+0.56** bp |
| **ann gross** | +202 bp/y |
| **ann net** | **+73 bp/y** |
| avg hold | 17.2 영업일 |
| exit (target / stop / time) | 23 / 4 / 66 % |

### 2.9 Bucket 기여도

| Bucket | n | mean net | total net | win% |
|---|---|---|---|---|
| 2-3y | 211 | +0.61 | +128 | 55% |
| 3-5y | 451 | +0.46 | +210 | 51% |
| **5-10y** | **801** | **+0.61** | **+487** | **60%** (절반 이상 기여) |

### 2.10 Bond Blacklist

신지표 편입 직전 종목 대형 손실 사례:
- KR103501GFC3 (25-10, 신3년지표): **−195 bp total**
- KR103503GG37 (26-3, 신5년지표): **−150 bp total**
- KR103502GAC4 (25-4): **−307 bp** (3 trades)

→ 사전 제외 권장. Combine 모델은 cross-section rank로 자동 흡수되지만 별도 alarm은 필요.

### 2.11 운용 도구

- spec: `factor_trading/of_cm_models_spec.md` (section 1)
- script: `factor_trading/scripts/of_monthly_extreme_2_10y.py`
- 산출: `data/factor_trading/phase33_of_monthly_2_10y/spread_2_10y_4anchors.parquet`

---

## 3. CM 모델 — Curve Mismatching

### 3.1 Concept

OF 와 달리 시즌·이벤트 효과를 **panel OLS 안에 흡수**시킨 후 남은 종목별 idiosyncratic mispricing noise 의 cross-section mean-rev.

### 3.2 Panel OLS Model

```
spread_bp_i,t = β · features_i,t + month_dummies + ε_i,t

features = remain_year, nearest_anchor_dist, age_y, coupon_pct,
            was_ever_bench, log_gross, dominant_share,
            foreign_share, inst_share
```

**features 효과** (within bond FE + month FE):
- `log_gross` (발행잔액 log) → +0.074 bp/log unit (유동성 ↑ → 약간 cheap)
- `dominant_share` (집중도) → −1.01 bp (집중 ↑ → rich)
- `inst_share` (기관 보유) → +0.195 bp/% (기관 ↑ → cheap)
- `foreign_share` → +0.051 bp/% (외인 ↑ → 약간 cheap)
- within R² = 0.18

### 3.3 신호 산출 (Walk-forward OOS)

```
expected_spread_i,t = β · features_i,t + month_dummies   (OOS, expanding by year)
abnormal_i,t       = actual_spread_i,t − expected_i,t
abn_z_i,t          = (abnormal_i,t − rolling_mean_252d) / rolling_std_252d
```

- abn_z > 0: 모델 예상보다 cheap (LONG 후보)
- abn_z < 0: 모델 예상보다 rich (SHORT 후보)
- 시즌 효과는 month FE 로 흡수 → monthly_z 와 부분 직교

### 3.4 Trading Variants 실증 (단독 백테스트)

| Variant | Universe | 페어 | ann net bp/y | 비고 |
|---|---|---|---|---|
| v1 | 3-10y | single bench, cap=2 | +12 | conservative |
| **v2** | 3-10y | single bench, cap=4, max3/bond | **+34** | best (단, anchor 매일 재계산의 noise illusion 포함) |
| v3 | 3-10y | BPV both anchors | +13 | |
| v4 (정직) | 3-10y | leg-based, anchor fixed at entry | **−91** | 실거래 가까운 정직 시뮬레이션 |
| v5 | 3-10y | 비지표 ↔ 비지표 | +0.2 | break-even |

### 3.5 평가

- v2의 +34 bp/y 는 일부 anchor 재선택의 illusion
- **v4의 −91 이 진정한 OOS 정직 결과 → CM 단독 trading은 net 음수**
- 단 **RV + CM double-confirm 결합** 시 (phase23): **+168 bp/y** (RV ε≥+3 AND abn_z≥+1.5 → LONG, win 76%)

### 3.6 활용 방향

| 활용 | 비고 |
|---|---|
| **단독 trading** | ❌ 비권장 (정직 −91 bp/y) |
| **RV/OF double-confirm filter** | ✅ 결합 효과 |
| **Combine 모델 입력 (rank 가중)** | ✅ 본 보고서 4장 |
| Streamlit GUI daily indicator 보조 | bond 별 ranking 표시 |

### 3.7 운용 도구

- spec: `factor_trading/of_cm_models_spec.md` (section 2)
- script: `factor_trading/scripts/bench_spread_walkforward.py` (phase22)
- 산출: `data/factor_trading/phase22_bench_walkforward/oos_abnormal_with_z.parquet`

---

## 4. RV 모델 — Level cum_ε_21d (Combine 입력)

### 4.1 Concept

각 종목의 일일 ytm을 3Y / (10Y−3Y) 두 시장 팩터로 회귀해 잔차 ε를 산출. 잔차의 **level 누적 21일 평균** 이 fair-value gap 시계열. cross-section 분포에서 mean-rev.

### 4.2 신호 산출 (No-intercept 2-factor rolling regression)

```
Y_i,t = β_lvl_i,t · Y_3Y,t + β_slope_i,t · (Y_10Y,t − Y_3Y,t) + ε_i,t

window = 63d, min_periods = 20, no intercept
```

→ ε_i,t = fair-value gap (bp)

### 4.3 Level vs Diff Mode 비교 (결정적)

기존(phase23)은 **diff mode** 사용:
```
diff:  ΔY_i = β_lvl·ΔY_3Y + β_slope·Δ(10Y−3Y) + ε
       cum_ε_21d = Σ ε_t (21일 누적)
       Sharpe = 4.90
```

신규(phase36)는 **level mode**:
```
level: Y_i = β_lvl·Y_3Y + β_slope·(10Y−3Y) + ε
       cum_ε_21d = mean ε_t (21일 평균, fair value gap)
       Sharpe = 7.21  (+47% 개선)
```

→ Combine 모델은 **level mode** 채택 (`server/app/routers/beta.py:744` 의 production 로직과 일치).

### 4.4 단독 LS 백테스트 (cross-section quintile)

- Q5 (가장 cheap) LONG / Q1 (가장 rich) SHORT, daily LS basket
- ann net bp/y ≈ +150~200 (cost=1 bp/trade)
- 본 Combine에서는 단일 팩터로 직접 사용

### 4.5 산출 도구

- script: `factor_trading/scripts/three_factor_pure_level_rv.py` (phase36)
- 산출: `data/factor_trading/phase36_three_factor_level_rv/level_cum_eps_21d_panel.parquet`

---

## 5. Combine 모델 — RV + OF + CM (Sharpe-prop, LONG only)

세 신호를 단일 page로 결합한 실거래 모델 (phase37 Replication Trading Model).

### 5.1 Combined Score 산출

매일 t, 각 비지표 i:

```
# Cross-section rank normalization (각 일자 내 [-1, +1] uniform)
for each signal s in {OF, CM, RV}:
  rank_s_i,t = (rank(s_i,t) − 1) / (n − 1) × 2 − 1

# Sharpe-prop 가중 (단일 팩터 Sharpe 비례)
W_OF = 14.19 / 33.51 = 0.423
W_CM = 12.11 / 33.51 = 0.361
W_RV =  7.21 / 33.51 = 0.215

combined_base_i,t = W_OF·rank_OF + W_CM·rank_CM + W_RV·rank_RV

# Event overlay (명제 B 활용)
event_new_i,t  = 1 if age_bd < 60 else 0   (신규 발행 후 60bd 내)
event_drop_i,t = 1 if 0 ≤ bd_since_drop < 30 else 0
combined_i,t   = combined_base + 0.30·event_new + 0.30·event_drop
```

### 5.2 Sharpe-prop 가중 근거

| 단일 팩터 | Sharpe (cross-section quintile LS) | Weight |
|---|---|---|
| OF (monthly_z, 2-10y 4-anchor) | 14.19 | 0.423 |
| CM (panel OLS abn_z) | 12.11 | 0.361 |
| RV (level cum_ε_21) | 7.21 | 0.215 |

→ Kelly 분석 (phase35): L1-norm Full Kelly에서도 OF·CM 가중 > RV 확인. RV는 negative weight 영역도 있음 (Kelly L1 [−0.043, +0.774, +0.182]) → Sharpe-prop이 robust.

### 5.3 페어 구성 룰

| 항목 | 값 |
|---|---|
| 비지표 size | 100억 (고정 운용 단위) |
| 지표 size | BPV-neutral 후 10억 단위 반올림 (보통 80~130억) |
| 진입 비용 | 1 bp/페어 (진+청 합산) |
| Hold 한도 | 21 영업일 |
| DV01 한도 (한 방향) | 5,000 만원/bp |
| 일별 최대 신규 진입 | 4 페어 |
| 종목별 동시 active | ≤ 3 |
| Cutoff (백테스트) | 2026-04-30 |
| 잔존만기 | 2 ~ 10 년 (발행연한 5년 제한 제거) |
| 주간 거래 빈도 | 평균 ≥ 1회 |

### 5.4 Grid Search 결과 (LONG+SHORT 초기 버전)

3 × 6 × 6 = 108 조합 (θ_entry × target × stop):

```
Best: θ=0.40, target=2bp, stop=5bp
  N = 711, trades/yr = 78
  ann_net (전체) = +14.4 bp/y
  → forced exit 제외 = −6.6 bp/y   ❌ 실제 성과는 negative
```

**Forced exit artifact 발견**: 백테스트 종료 시점 강제 청산 32건 (+190 bp 일시 추가) → cum PnL 차트 막판 spike. 분석에서 제외해야 진실 보임.

### 5.5 PnL 분해 (LONG+SHORT, forced 제외) — **SHORT side 함정 발견**

| Side | N | mean (bp) | win % | total (bp) | 9년 ann_net (bp/y) |
|---|---|---|---|---|---|
| **LONG** (비지표 LONG / 지표 SHORT) | 454 | **+0.57** | **59%** | +260 | +29 |
| **SHORT** (비지표 SHORT / 지표 LONG) | 257 | **−1.24** | **22%** | **−319** | **−35** |

**SHORT side 실패 메커니즘** (명제 B-1과 정합):
- Rich 비지표가 mean-revert하지 않고 **오히려 더 rich 해짐**
- 메커니즘: rich 비지표 = 다음 지표 후보일 가능성 ↑ → **신지표 편입 직전 추가 compression** 활성화
- 즉 강한 SHORT 신호일수록 더 큰 손실 (raw signal과 실제 PnL 음의 상관)

### 5.6 LONG only 결정 근거

> **"비지표 LONG / 지표 SHORT" 방향만 채택. 비지표 SHORT 신호는 실증적으로 손실 영역.**

**이유**:
1. **백테스트 9년 실증**: SHORT 257건 mean −1.24 bp, win 22%, total −319 bp
2. **명제 B-1 메커니즘**: rich 비지표가 새 지표 narrative 활성화로 추가 compression
3. **이론적 정합**: 비지표 LONG / 지표 SHORT는 "지표 프리미엄 (liquidity premium) mean-rev"라는 명확한 source 가 있음. 반대 방향은 source 가 약함
4. **Kelly 분석**: walk-forward Kelly에서도 SHORT 방향 weight가 음수로 일관

**Entry rule (LONG only 적용)**:
```
매일 t, 각 비지표 i:
  if combined_i,t > θ_entry:
    direction = +1  (비지표 LONG / nearest 지표 SHORT)
  else: skip   # combined < +θ_entry 인 종목은 진입하지 않음 (SHORT 차단)
```

### 5.7 Grid Search 결과 (LONG only — 최종)

| 조합 | N | trades/yr | trades/wk | win % | mean net | **ann_net** | ann_net (억/y) |
|---|---|---|---|---|---|---|---|
| **θ=0.40 / T=2 / S=5** ⭐ | **608** | **67** | **1.3** | **60.2%** | **+0.69** | **+46.5 bp/y** | **+2.40 억/y** |
| θ=0.20 / T=2 / S=5 | 536 | 59 | 1.1 | 59.5% | +0.72 | +42.7 | +2.06 |
| θ=0.20 / T=2 / S=None | 482 | 53 | 1.0 | 59.8% | +0.76 | +40.6 | +1.94 |
| θ=0.30 / T=2 / S=7 | 543 | 60 | 1.2 | 57.8% | +0.67 | +40.1 | +2.02 |
| θ=0.30 / T=1 / S=7 | 677 | 75 | 1.4 | 63.5% | +0.51 | +37.7 | +1.90 |

**Forced 제외 clean** (최종):
- N = 593, ann_net = **+30.7 bp/y**, win 61.4%
- mean net = +0.47 bp, avg hold = 16일

### 5.8 Best 조합 분해

#### Exit reason (forced 포함)
| 청산 사유 | n | mean | total |
|---|---|---|---|
| target | 219 | +2.83 | **+620.8 bp** (100% 승) |
| stop | 37 | −6.08 | −225.0 |
| time | 337 | −0.35 | −117.4 |
| forced | 15 | +9.58 | +143.7 (분석 제외) |

#### Bucket 기여도
| Remain bucket | n | mean | total | win % |
|---|---|---|---|---|
| **3-5y** | **321** | **+0.63** | **+202.7** | **65.7%** ← 가장 강함 |
| 5-10y | 272 | +0.28 | +75.7 | 56.3% |

#### Yearly
| 연도 | n | total (bp) | win % |
|---|---|---|---|
| 2017 | 76 | +21.8 | 55% |
| 2018 | 99 | +52.4 | 60% |
| 2019 | 77 | −7.1 | 47% |
| 2020 | 62 | +58.7 | 66% |
| 2021 | 50 | +46.4 | 74% |
| 2022 | 84 | +51.8 | 69% |
| 2023 | 45 | +21.5 | 62% |
| 2024 | 41 | **−56.0** | **34%** ← 유일 큰 적자 |
| 2025 | 42 | +55.6 | 81% |
| 2026 | 17 | +33.3 | 88% |

→ 2024년 한 해를 제외하면 매년 흑자. 2024는 시장 regime shift 가능성 (조사 필요).

### 5.9 운용 도구

- script: `factor_trading/scripts/three_factor_replication.py` (phase37)
- 산출: `data/factor_trading/phase37_three_factor_replication/`
  - `three_factor_replication_dashboard.png` (11-panel)
  - `signals_2026_05_01_to_12.csv` (실시간 신호)
  - `trades_*.csv` (grid 각 조합 trade list)

---

## 6. 일일 운용 Routine

### 6.1 데이터 갱신 (자동)

매일 09:00 DB 갱신 후:

```
Stage 1 — phase19 bench_nonbench_spread.py    (spread_long.parquet)
Stage 2 — phase20 bench_spread_individual.py  (abnormal_panel.parquet)
Stage 3 — phase21 bench_spread_liquidity.py   (abnormal_with_liquidity.parquet)
Stage 4 — phase22 bench_spread_walkforward.py (oos_abnormal_with_z.parquet)
Stage 5 — phase33 of_monthly_extreme_2_10y.py (spread_2_10y_4anchors.parquet)
Stage 6 — phase36 three_factor_pure_level_rv.py (level_cum_eps_21d_panel.parquet)
Stage 7 — phase37 three_factor_replication.py (combined score & signals)
```

⚠️ **CM 파이프라인 (Stage 1~4) 가 누락되면 통합 신호 산출 불가**. 일일 자동 점검 필요.

### 6.2 신호 모니터링

```
1. Combined score top-4 (combined > +0.40) 추출
2. 종목별 active count < 3 확인
3. 신지표 블랙리스트 제외 (auto-detect: age_bd < 30 의 신규 발행 + nearest_label 매칭)
4. BPV-neutral sizing 계산 (비지표 100억 / 지표 10억 단위)
5. DV01 한도 (한 방향 5천만원/bp) 점검
6. 진입
```

### 6.3 Active Trade 관리

```
매일 mark-to-market:
  cum_pnl_bp = −direction × [(ytm_n − ytm_n_entry) − (ytm_b − ytm_b_entry)] × 100

Exit (LONG only 기준, best 조합):
  target: +2 bp → 익절
  stop:   −5 bp → 손절
  time:   21 영업일 → 자동 청산
```

### 6.4 리스크 한도

| 항목 | 한도 |
|---|---|
| 한 방향 DV01 | 5,000 만원/bp |
| 동시 active 페어 | ≤ 12 (daily_cap 4 × max_hold 21 / avg ~16 = 약 10) |
| 종목별 동시 active | ≤ 3 |
| 주간 신규 거래 | ≥ 1 (평균 1.3) |

---

## 7. 모델 한계 / 향후 작업

### 7.1 알려진 한계

1. **SHORT side 차단으로 alpha 절반 손실 가능**
   - 현재 비지표 LONG only → 비지표 자체가 rich 한 regime 에서는 기회 부재
   - 향후 비지표↔비지표 spread (intra-nonbench) 페어 검토 필요

2. **2024 regime shift**
   - 9년 중 유일하게 win 34%, −56 bp 손실
   - 원인 미확인 (한은 통화정책 변경 시기?)
   - Walk-forward weight 조정 검토

3. **CM 파이프라인 수동 빌드 의존**
   - OF/RV는 자동, CM은 phase19~22 수동 실행 필요
   - 자동화 스크립트 (`run_daily_pair_signal.bat`) 에 통합 필요

4. **신지표 블랙리스트 수동 관리**
   - 현재 코드의 EVENT_BOOST는 +30% 이지만 블랙리스트 자동 감지는 미구현
   - vintage 패턴 (issue_year, issue_month, 잔존 ≈ 다음 지표 후보 거리) auto-detect 필요

### 7.2 향후 작업 우선순위

| 우선 | 작업 |
|---|---|
| 🔥 | OF blacklist 자동 감지 (신지표 후보 vintage 패턴 학습) |
| 🔥 | 일일 자동 갱신 시스템 (DB → 7-stage pipeline → signal → 알람) |
| 🔥 | 2024 손실 regime 분석 + 시즌/금리 regime conditional weight |
| 중 | CM walk-forward 매년 재학습 자동화 |
| 중 | RV diff vs level mode 시기별 dominance 분석 |
| 중 | Streamlit GUI에 combined score daily indicator 탭 추가 |
| 낮 | Kelly weight walk-forward + DV01 fractional allocation |

---

## 8. 핵심 통계 요약

| 항목 | 값 |
|---|---|
| **백테스트 구간** | 2017-01 ~ 2026-04-30 (9년) |
| **운용 universe** | 비지표 KTB 2-10y (~56 종목 동시 panel) |
| **거래 방향** | **LONG only** (비지표 LONG / 지표 SHORT) |
| **신호** | RV·OF·CM Sharpe-prop 가중 cross-section rank + event overlay |
| **Best 조합** | θ_entry=0.40, target=+2bp, stop=−5bp |
| **N trades** | 608 (forced 포함) / 593 (forced 제외) |
| **trades/year** | 67 |
| **trades/week** | 1.3 |
| **win %** | 60.2% |
| **mean net / trade** | +0.69 bp |
| **ann net** | **+46.5 bp/y (전체) / +30.7 bp/y (clean)** |
| **ann net (억/y)** | +2.40 억/y |
| **avg hold** | 15.9 영업일 |
| **유일 적자 연도** | 2024 (−56 bp, win 34%) |

---

## 9. 참고 문서

| 문서 | 내용 |
|---|---|
| `factor_trading/bond_3factor_final_spec.md` | RV 3팩터 시스템 (RV/MOM/CURVE) 확정 명세 |
| `factor_trading/of_cm_models_spec.md` | OF / CM 단독 모델 명세 |
| `factor_trading/three_strategies_comparison.md` | 3 모델 1차 비교 (RV vs OF vs CM) |
| `factor_trading/factor_methodology.md` | RV 발표용 정리 |
| `factor_trading/research_2026_05.md` | 전체 리서치 로그 |
| `factor_trading/strategy_report_OF_CM_Combine.md` | **본 보고서 (OF + CM + Combine 통합)** |

---

작성: 2026-05-13
다음 갱신: 자동 일일 갱신 시스템 구축 후 매주
