# OF / CM Models — 확정 명세서 (2026-05-13)

채권 운용 시스템의 두 가지 추가 모델 (RV 3팩터 시스템과 별도) 확정 명세.

| 모델 | Full Name | 핵심 신호 | 확정 status |
|---|---|---|---|
| **OF** | On-Off Trading | monthly_z (시즌 conditional) | **운용 채택 (2026-05-13)** |
| **CM** | Curve Mismatching | bond-rolling z of panel OLS residual | **별도 분리, 보조 layer** |

---

## 1. OF Model — On-Off Trading (확정)

### 1.1 Concept

**명제 A (월별 spread 계절성)** 을 신호 자체에 직접 반영. 각 종목의 spread 가 **그 월의 historical cross-section 분포** 에서 얼마나 극단적인지 (= monthly z-score) 평가 → mean-rev trade.

### 1.2 Universe

- 비지표 국고채, 잔존만기 **2 ~ 10 년**
- DB `ktb` 테이블, `category = '국고채'`
- 비지표 = 그 일자의 5대 지표 (label = 2년지표/3년지표/5년지표/10년지표) 의 bond_code 집합 외 모든 국고채

### 1.3 Yield Curve Anchors (4개)

| Label | Remain (y) |
|---|---|
| 2년지표 | 2.0 |
| 3년지표 | 3.0 |
| 5년지표 | 5.0 |
| 10년지표 | 10.0 |

매일 5대 label 중 위 4개의 `bond_code` 추출 → 그 종목의 (remain_year, ytm) 으로 piecewise linear 보간 yield curve 구성.

### 1.4 Spread 정의

```
spread_bp_i,t = (ytm_i,t − interp_ytm_i,t) × 100   [bp]

interp_ytm_i,t = linear interpolation of anchor (remain, ytm)
                  evaluated at remain_i,t
```

- spread > 0: 보간선 위 (yield 높음 = 가격 낮음 = cheap)
- spread < 0: 보간선 아래 (yield 낮음 = 가격 높음 = rich)

### 1.5 Monthly Z-Score Signal

매일 t, 각 비지표 i 에 대해:

```
month_t = t.month   (1~12)

month_mean_t = mean(spread_bp_j,s) for all j, s
               where s < t AND s.month == month_t
               AND past-pool 누적 ≥ 100 obs

month_std_t  = std(...)  (동일 조건)

monthly_z_i,t = (spread_bp_i,t − month_mean_t) / month_std_t
```

**look-ahead-free**: t 시점 이전의 그 월(M) 데이터만 사용. 누적 (expanding) 통계.

### 1.6 Pair Construction (BPV-neutral)

각 비지표 i 의 `nearest anchor label`:
- remain ∈ [2, 2.5): 2y
- remain ∈ [2.5, 4): 3y or 5y (가까운 쪽)
- remain ∈ [4, 7.5): 5y
- remain ∈ [7.5, 10]: 10y

→ 그 anchor 의 그 일자 `bench_code` 와 페어:

```
position +1 (LONG): nonbench LONG / single bench SHORT
position −1 (SHORT): nonbench SHORT / single bench LONG

DV01 (BPV) 비례 sizing:
  notional_bench = notional_nonbench × duration_nonbench / duration_bench
  → DV01 = 0 (BPV-neutral)
```

### 1.7 Entry Rule

매 영업일 t (cross-section):
1. `monthly_z_i,t` 모든 비지표 계산
2. `|monthly_z| > 2.0` 종목 추출
3. `|monthly_z|` 큰 순 정렬
4. **이미 active 중 종목 (동시 ≥ 3 개) 제외**
5. **하루 최대 4 개 새 페어 진입** (top score)
6. 각 페어: `direction = sign(monthly_z)`
   - z > 0 (cheap) → LONG nonbench / SHORT nearest bench
   - z < 0 (rich) → SHORT nonbench / LONG nearest bench

### 1.8 Exit Rule

각 trade 별 매일 mark-to-market:
```
cum_pnl_bp = direction × (−1) × [(ytm_n_t − ytm_n_0) − (ytm_b_t − ytm_b_0)] × 100
```

(BPV-neutral 가정: 두 leg dY 차이의 negation)

청산 조건 (첫 도달):
- **target hit**: `cum_pnl_bp ≥ +5 bp` → 익절
- **stop hit**: `cum_pnl_bp ≤ −5 bp` → 손절
- **time exit**: `hold ≥ 21 영업일` (한 달)
- **forced**: 백테스트 종료 또는 데이터 없음

### 1.9 거래비용

**1 bp/trade (한 페어 진입+청산 합산)** — 슬리피지 가정.

### 1.10 백테스트 성과 (2017~2026, walk-forward + monthly_z look-ahead-free)

| 지표 | 값 |
|---|---|
| N trades | 1,463 |
| trades/year | 129 |
| win % (gross) | 60.8% |
| **win % (net)** | **56.5%** |
| mean (gross) | +1.56 bp |
| **mean (net)** | **+0.56 bp** |
| **ann gross** | +202 bp/y |
| **ann net** | **+73 bp/y** |
| avg hold | 17.2 영업일 |
| exit (target / stop / time) | 23 / 4 / 66 % |

### 1.11 Per-Bucket Contribution

| Bucket | n | mean net | total net | win% |
|---|---|---|---|---|
| 2-3y | 211 | +0.61 | +128 | 55% |
| 3-5y | 451 | +0.46 | +210 | 51% |
| **5-10y** | **801** | **+0.61** | **+487** | **60%** ← 절반 이상 |

### 1.12 Seasonal Pattern (entry month 별 net PnL)

**Strong months**: 1 (+1.70), 3 (+1.81), 6 (+2.36)
**Weak months**: 8 (−0.35), 10 (−0.36), 11 (−1.09)

### 1.13 Bond Blacklist (큰 손실 종목)

신지표 편입 직전·후 종목들이 큰 손실. 사전 제외 권장:
- 신3년지표 직전 후보 (remain ≈ 2.5y 부근의 최신 vintage)
- 신5년지표 직전 후보 (remain ≈ 4.5y 부근)
- 신10년지표 직전 후보 (remain ≈ 9.5y 부근)

실증 손실 사례:
- KR103501GFC3 (25-10, 신3년지표): −195 bp total
- KR103503GG37 (26-3, 신5년지표): −150 bp total

### 1.14 운용 도구

스크립트: `factor_trading/scripts/of_monthly_extreme_2_10y.py`

산출물: `data/factor_trading/phase33_of_monthly_2_10y/`
- `spread_2_10y_4anchors.parquet`: 4-anchor spread panel (60,506 rows)
- `grid_results.csv`: 48 grid combinations
- `trades_thz2_T5_S5.csv`: best 조합 trade list
- `by_bucket.csv`, `monthly_best.csv`, `per_bond.csv`
- `of_monthly_2_10y_dashboard.png`

### 1.15 운용 권고

```
일일 routine:
  1. spread_bp 계산 (4-anchor 보간)
  2. monthly_z 계산 (그 월 expanding pool)
  3. |monthly_z| > 2.0 종목 ranking
  4. 위 종목 중 blacklist 제외 + active count < 3 확인
  5. 사이즈: 페어 BPV-neutral, DV01 1억/bp/페어 (사용자 조정)
  6. 매일 active trade target/stop/21d 점검

리스크 한도:
  동시 active 페어 수 ≤ 8 ~ 12
  종목별 동시 active ≤ 3 페어
  부정적 시즌 (10/11월) 사이즈 50%
  강한 시즌 (1/3/6월) 사이즈 100~120%
```

---

## 2. CM Model — Curve Mismatching (보조 layer)

### 2.1 Concept

**보간 yield curve 와 개별 종목의 fitted residual mismatching** 의 cross-section mean-rev. OF 와 달리 시즌·이벤트 효과가 panel OLS 모델 안에 흡수된 후 남은 **idiosyncratic mispricing noise**.

### 2.2 Panel OLS Model

```
spread_bp_i,t = β · features_i,t + month_dummies + ε_i,t

features:
  remain_year, nearest_anchor_dist, age_y, coupon_pct, was_ever_bench
  + liquidity (log_gross, dominant_share, foreign_share, inst_share)
```

→ `abnormal_spread_i,t = ε_i,t` (OOS, walk-forward expanding by year)

### 2.3 Signal

```
abn_z_i,t = (abnormal_spread_i,t − rolling_mean_252d)
             / rolling_std_252d
```

종목별 평소 분포의 z-score. 시즌 효과는 month dummies 로 모델에 흡수됨.

### 2.4 Trading Variants (실증)

| Variant | Universe | 페어 | ann net bp/y |
|---|---|---|---|
| v1 | 3-10y | single bench, cap=2 | +12 |
| **v2** | 3-10y | single bench, cap=4, max3/bond | **+34** |
| v3 | 3-10y | BPV both anchors | +13 |
| v4 (정직 leg-based) | 3-10y | anchor fixed at entry | **−91** ← 실거래 정직 |
| v5 (non-bench pair) | 3-10y | 비지표 ↔ 비지표 | +0.2 |

### 2.5 평가

- v2 의 +34 bp/y 는 일부 noise 운 (anchor 매일 재계산의 illusion 포함)
- v4 의 −91 은 진정한 OOS leg-based 시뮬레이션 결과 (적자)
- → **CM 단독 trading 은 net break-even 또는 약한 음수**
- 단 RV ε 와 결합 시 (phase23) net +66 bp/y 가능 (Combined |z|>1.5 + |ε|>3)

### 2.6 활용 방향

| 활용 | 비고 |
|---|---|
| **단독 trading** | ❌ 권장 안 함 |
| RV 신호 + CM **double-confirm filter** | ✅ 결합 효과 |
| OF 신호와 직교성 layer | abn_z (CM) ↔ monthly_z (OF) corr 측정 후 결합 |
| Streamlit GUI daily indicator 보조 | bond 별 ranking 표시 |

### 2.7 운용 도구

스크립트:
- `factor_trading/scripts/bench_spread_walkforward.py` (phase22): abn_z 산출
- `factor_trading/scripts/bench_spread_rv_combo.py` (phase23): RV + CM combined
- `factor_trading/scripts/of_pair_strategy_v4.py` (phase28): 정직 leg-based

---

## 3. 모델 간 관계 정리

### 3.1 신호 source 의 직교성

| 모델 | Signal source | 시간 스케일 |
|---|---|---|
| **RV** | dY 2팩터 잔차 ε 의 cross-section quintile | daily, 21d hold |
| **OF** | monthly_z (시즌 conditional spread) | daily, 21d hold |
| **CM** | abn_z (panel OLS residual rolling z) | daily, 21d hold |

`abn_z` 는 month dummies 로 시즌 흡수 → `monthly_z` 와 부분 직교적.

### 3.2 결합 가능성

```
RV (ε quintile)  ─┐
                  ├── Cross-sectional alpha (within-bucket)
CM (abn_z)       ─┘

OF (monthly_z)   ─── Seasonal/temporal alpha (when extreme)

→ 세 모델 결합 시 직교성 + diversification 가능
   단 자본·DV01 분리 + 사이즈 조절 필요
```

### 3.3 자본 배분 권장 (실험적)

| 모델 | 자본 비중 | 이유 |
|---|---|---|
| RV | 50% | 가장 검증된 main alpha |
| OF | 35% | net +73 bp/y, robust |
| CM | 15% | 보조 layer, double-confirm 용 |

---

## 4. 향후 검증·확장

| 항목 | 우선순위 |
|---|---|
| OF + RV 신호 결합 (월별 entry month overlay) | 🔥 |
| OF blacklist 자동화 (신지표 후보 자동 감지) | 🔥 |
| CM walk-forward (매년 panel OLS 재학습) | 중 |
| OF + CM 신호 corr 측정 | 중 |
| Streamlit GUI에 OF daily indicator 추가 | 중 |
| 일별 자동 갱신 시스템 (DB → signal → 알람) | 낮 |

---

작성: 2026-05-13
관련 문서:
- `factor_trading/bond_3factor_final_spec.md` (RV)
- `factor_trading/research_2026_05.md` (전체 리서치 로그)
- `factor_trading/factor_methodology.md` (RV 발표용 정리)
