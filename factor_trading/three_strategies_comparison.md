# 상대가치 Trading 3 모델 — RV / OF / CM 종합 비교 (2026-05-13 확정)

채권 시장 상대가치 트레이딩 시스템의 세 가지 핵심 모델. 각자 다른 alpha source 활용, 결합 시 diversification 가능.

---

## 0. Overview — 한눈에 비교

| 항목 | **RV** (3팩터) | **OF** (On-Off) | **CM** (Curve Mismatching) |
|---|---|---|---|
| Full name | Relative Value 3팩터 시스템 | On-Off (지표-비지표) Trading | Curve Mismatching trade |
| 핵심 신호 | dY 2팩터 잔차 ε quintile | **monthly_z** (시즌 conditional) | **abn_z** (panel OLS residual rolling z) |
| Universe | 전체 국고채 (3 bucket) | 비지표 2-10y | 비지표 3-10y |
| 방향성 | Cross-section LS within-bucket | Pair LS (BPV-neutral) | Pair LS (BPV-neutral) |
| Hold | 21 영업일 | 21 영업일 | 21 영업일 |
| 페어 구성 | Q5 LONG basket / Q1 SHORT basket | nonbench ↔ nearest single bench | nonbench ↔ nearest single bench (or BPV both) |
| **ann net bp/y** | **~+200~600** (검증 구간) | **+73** (cost=1bp/trade) | **+34 (v2) / −91 (v4 정직)** |
| Trades/yr | 매일 LS rebalance | 129 | 200~700 (variant) |
| 확정 status | ✅ 운용 | ✅ 운용 채택 | ⚠️ 보조 layer (단독 trading 비권장) |
| 운용 스크립트 | `monitor.py`, `app.py` | `of_monthly_extreme_2_10y.py` | `bench_spread_walkforward.py`, `of_pair_strategy_v4.py` |

---

## 1. RV — Relative Value (3팩터)

### 1.1 Concept
한국 국채 시장의 상대가치를 3 독립 alpha source 로 분해:
- **RV factor**: cross-section 잔차 mispricing
- **MOM factor**: 시장 모멘텀 (3Y rate)
- **CURVE factor**: 3Y/10Y slope mean-rev

### 1.2 신호 산출

**2-factor rolling β**:
```
dY_i,t = α_i + β_3Y_i,t × dY_3Y,t + β_10Y_i,t × dY_10Y,t + ε_i,t
window = 63d, min_periods = 40
```

**RV score**:
```
RV_i,t(h) = Σ_{s=t-h+1..t} ε_i,s   (h=21d default)
```

**MOM signal** (3Y only):
```
MOM_v2 = −sign(cum_dY_3Y_63d)
```

**CURVE signal** (3Y/10Y pair):
```
CURVE_v2 = −sign(cum_slope_21d), slope = dY_10Y − dY_3Y
```

### 1.3 Portfolio 구성
- **RV**: within-bucket quintile (≤5y / 5~10y / >10y), Q5 LONG / Q1 SHORT, daily LS, 21d hold
- **MOM**: 3Y bench 단일 종목 bang-bang 매매
- **CURVE**: 3Y vs 10Y BPV-neutral pair, daily flip

### 1.4 성과 (검증 구간 2023-07 ~ 2026-04)

| Factor | gross ann bp/y | net (cost=1bp) | Sharpe | 비고 |
|---|---|---|---|---|
| RV | ~+200 | +150 | 1.0+ | 가장 robust |
| MOM | +349 | (단일 종목, 매일 reset) | 1.43 | regime 의존 |
| CURVE_v2 | +1,540 (gross) | −60 | 1.49 | 매일 flip 거래비 부담 |
| **Hybrid B** (CURVE) | +243 (net) | — | — | LOW vol regime = ML, MID/HIGH = CR_v2 |

### 1.5 운용 도구
- spec: `factor_trading/bond_3factor_final_spec.md`
- methodology: `factor_trading/factor_methodology.md`
- pipeline: `factor_trading/main.py` (FactorPipeline)
- daily monitor: `factor_trading/monitor.py`
- GUI: `factor_trading/app.py` (Streamlit, 4 tab)
- CURVE Hybrid signal: `factor_trading/scripts/curve_hybrid_signal.py`

---

## 2. OF — On-Off Trading (확정)

### 2.1 Concept
**명제 A (월별 spread 계절성)** 직접 활용. 각 종목의 spread 가 **그 월의 historical 분포에서 얼마나 극단적인지** (= monthly z-score) 평가 → mean-rev trade.

### 2.2 신호 산출

```
spread_bp_i,t = (ytm_i,t − interp_ytm_i,t) × 100
interp_ytm    = 2y/3y/5y/10y 지표 4 anchor piecewise linear

monthly_z_i,t = (spread_i,t − month_mean_t) / month_std_t
  where month_mean/std = "그 month-of-year" 의 expanding past pool 통계
  (look-ahead-free, min 100 obs)
```

### 2.3 Pair 구성
- 각 비지표의 nearest anchor (2/3/5/10y) bench 와 페어
- BPV-neutral sizing (DV01 = 0)

### 2.4 Entry / Exit
- Entry: `|monthly_z| > 2.0`, 매일 cross-section top score 4 페어, max 3/bond
- Exit: target +5 bp / stop −5 bp / time 21bd
- Cost: 1 bp/trade (진+청)

### 2.5 성과 (2017~2026, 9년)

| 항목 | 값 |
|---|---|
| N trades | 1,463 |
| trades/yr | 129 |
| win % gross / net | 60.8% / **56.5%** |
| mean gross / net | +1.56 / **+0.56** bp |
| **ann gross** | +202 bp/y |
| **ann net** | **+73 bp/y** |
| avg hold | 17.2 bd |
| exit (tgt/stp/time) | 23 / 4 / 66 % |

### 2.6 Bucket / 시즌 패턴

| Bucket | n | mean net | total net | win% |
|---|---|---|---|---|
| 2-3y | 211 | +0.61 | +128 | 55% |
| 3-5y | 451 | +0.46 | +210 | 51% |
| **5-10y** | **801** | **+0.61** | **+487** | **60%** |

**Strong months**: 1 (+1.70), 3 (+1.81), 6 (+2.36)
**Weak months**: 8 (−0.35), 10 (−0.36), **11 (−1.09)**

### 2.7 Blacklist
신지표 편입 직전 종목 큰 손실:
- 25-10 (신3년지표): −195 bp total
- 26-3 (신5년지표): −150 bp total
- 25-4: −307 bp (3 trades)

### 2.8 운용 도구
- spec: `factor_trading/of_cm_models_spec.md` (section 1)
- script: `factor_trading/scripts/of_monthly_extreme_2_10y.py`
- 산출: `data/factor_trading/phase33_of_monthly_2_10y/`

---

## 3. CM — Curve Mismatching (보조 layer)

### 3.1 Concept
**보간 yield curve 와 개별 종목의 fitted residual** 의 cross-section mean-rev. OF 와 달리 시즌·이벤트 효과가 panel OLS 모델에 흡수된 후 남은 **종목별 idiosyncratic mispricing noise** mean-rev.

### 3.2 신호 산출

**Panel OLS expected spread**:
```
spread_bp_i,t = β · features_i,t + month_dummies + ε_i,t

features = remain_year, nearest_anchor_dist, age_y, coupon_pct,
            was_ever_bench, log_gross, dominant_share,
            foreign_share, inst_share
```

**abnormal spread** (OOS, walk-forward expanding by year):
```
abnormal_i,t = actual_spread_i,t − expected_spread_i,t
abn_z_i,t    = (abnormal_i,t − rolling_mean_252d) / rolling_std_252d
```

### 3.3 Trading Variants 실증

| Variant | Universe | 페어 | ann net bp/y |
|---|---|---|---|
| v1 | 3-10y | single bench, cap=2, no re-enter | +12 |
| **v2** | 3-10y | single bench, cap=4, max 3/bond | **+34** |
| v3 | 3-10y | BPV both anchors | +13 |
| v4 (정직) | 3-10y | leg-based, anchor fixed at entry | **−91** |
| v5 | 3-10y | 비지표 ↔ 비지표 | +0.2 |

→ v2 의 +34 는 일부 noise illusion. v4 의 −91 이 진정한 OOS 정직 시뮬레이션.

### 3.4 결합 효과 (RV + CM = double-confirm)

```
RV ε ≥ +3 bp AND CM abn_z ≥ +1.5 → LONG  (both cheap)
RV ε ≤ −3 bp AND CM abn_z ≤ −1.5 → SHORT (both rich)
```

성과 (Combined |z|>1.5):
- N = 892, win 76%, **mean net +1.65 bp/trade**, **ann net +168 bp/y**

→ CM 단독은 약하나 **RV double-confirm filter** 로 강함.

### 3.5 운용 권고
- ❌ **CM 단독 trading 비권장** (v4 정직 simulation 시 적자)
- ✅ **RV double-confirm filter** 로 활용 (RV+CM 결합 +168 bp/y)
- ✅ **OF 신호와 직교성 layer** (시즌 흡수 후 noise mean-rev → OF 와 보완)

### 3.6 운용 도구
- spec: `factor_trading/of_cm_models_spec.md` (section 2)
- abn_z 산출: `factor_trading/scripts/bench_spread_walkforward.py` (phase22)
- RV+CM combined: `factor_trading/scripts/bench_spread_rv_combo.py` (phase23)
- 정직 leg-based: `factor_trading/scripts/of_pair_strategy_v4.py` (phase28)

---

## 4. 직교성 / 결합 가능성

### 4.1 신호 source 직교성 표

| 모델 | Cross-section vs Time-series | 시즌 정보 | 미시구조 |
|---|---|---|---|
| **RV** | dY 잔차 cross-section quintile (within-bucket) | ✗ (제거) | ✗ |
| **OF** | 보간 spread cross-section | ✅ (직접 사용) | ✗ |
| **CM** | panel OLS residual cross-section | ✗ (month FE 흡수) | ✅ (liquidity features) |

→ **3 모델 alpha source 가 서로 다른 차원**. 결합 시 직교 alpha 가능.

### 4.2 결합 전략 (실험적)

```
LAYER 1 — RV main (cross-section ε quintile, daily LS)
   ↓ 자본 50%
LAYER 2 — OF overlay (monthly_z extreme pair)
   ↓ 자본 35%
LAYER 3 — CM double-confirm filter
   ↓ 자본 15% (RV/OF 신호 confirm 시 사이즈 ↑)

→ 총 net alpha 추정 (가산 가정 + 직교성 hold 시):
   RV +150 × 0.5  =  +75
   OF +73  × 0.35 =  +26
   CM +30  × 0.15 =  +5
   합계         ≈ +106 bp/y
```

(가산은 단순 추정, 실제는 corr / volatility 측정 후 weighted)

### 4.3 자본 배분 권고

| 모델 | 비중 | DV01 한도 | 이유 |
|---|---|---|---|
| **RV** | 50% | 25M/bp (전체 50M 가정) | main, 가장 검증된 alpha |
| **OF** | 35% | 17.5M/bp | net +73 bp/y robust |
| **CM** | 15% | 7.5M/bp | 보조 layer, double-confirm |

---

## 5. Universe / Pair / 거래비용 정리

| 측면 | RV | OF | CM |
|---|---|---|---|
| Universe | 전 국고채 (110+) | 비지표 2-10y (100) | 비지표 3-10y (86) |
| 분류 | within-bucket (3 bucket) | nearest anchor pair | nearest anchor pair |
| Hedge instrument | within-bucket cross-section | single bench bond | single bench bond |
| BPV neutral | within-bucket 자동 균등 | 명시적 BPV-neutral | 명시적 BPV-neutral |
| 거래비용 | 0~0.5 bp/side (LS basket) | **1 bp/trade** | **1 bp/trade** |
| 거래 횟수 | 일별 LS rebalance | 129/yr (페어) | 200~700/yr (variant) |

---

## 6. 핵심 인사이트 — 3 모델 비교

### 6.1 각 모델의 "진짜" 가치

- **RV**: cross-section dY 잔차 → 시장 dY 영향 제거한 순수 종목별 mispricing
- **OF**: 그 월의 평소 spread 분포 대비 극단치 → 시즌 + 종목 lifetime 정보 종합
- **CM**: 보간선 vs 실제의 잔차 mean-rev → 시즌·feature 흡수 후 noise

### 6.2 정직한 평가

| 모델 | net alpha viability | 권장 활용 |
|---|---|---|
| **RV** | ✅ 강함 (ann +150~200 bp/y) | 단독 main |
| **OF** | ✅ 의미 있음 (ann +73 bp/y, 운용 채택) | 단독 + RV overlay |
| **CM** | ⚠️ 약함 (단독 ≈0~음수) | RV/OF double-confirm filter only |

### 6.3 사용자 직관 (2026-05-13 확정)

> "CM 은 사실 지표-비지표 패턴 trade 가 아니라, 보간 yield curve 와 개별 종목 갭의 mismatching mean-rev"
> → OF 와 CM 명확히 분리. OF = 명제 A (시즌) 직접 활용, CM = 잔차 noise mean-rev.

---

## 7. 운용 routine (3 모델 통합)

```
매일 09:00:
  1. DB 데이터 갱신 (자동)
  2. RV factor pipeline 실행
       → ε quintile 매김, 신호 산출
       → monitor.py / app.py 로 GUI 표시
  3. OF signal 산출
       → 4-anchor 보간 yield curve
       → monthly_z 계산
       → |z|>2.0 페어 후보 list
       → blacklist 제외 (신지표 편입 직전 종목)
  4. CM signal (선택)
       → abn_z 산출, RV/OF confirm filter 로 사용
  5. 운용 결정
       → RV 신호 + OF top 페어 + CM filter 결합
       → DV01 한도 내 진입
       → 활성 trade target/stop/21d 점검
       → 7-11월 약한 시즌은 사이즈 감축
```

---

## 8. 향후 작업

| 우선순위 | 작업 |
|---|---|
| 🔥 | OF + RV 결합 백테스트 (실제 alpha 가산성 검증) |
| 🔥 | OF blacklist 자동 감지 (신지표 후보 vintage 패턴) |
| 🔥 | Streamlit GUI 에 OF daily indicator tab 추가 |
| 중 | CM walk-forward 매년 expanding 재학습 |
| 중 | OF + CM signal corr 측정 + portfolio Kelly weight |
| 낮 | 일별 자동 갱신 + 알람 시스템 |

---

## 9. 참고 문서

| 문서 | 내용 |
|---|---|
| `factor_trading/bond_3factor_final_spec.md` | RV 확정 명세 |
| `factor_trading/factor_methodology.md` | RV 발표용 정리 |
| `factor_trading/of_cm_models_spec.md` | OF / CM 확정 명세 |
| `factor_trading/research_2026_05.md` | 전체 리서치 로그 |
| `factor_trading/three_strategies_comparison.md` | **본 문서 (3-way 통합 비교)** |

작성: 2026-05-13
