# 채권 운용 리서치 노트 — 2026-05

3팩터 RV 시스템 (RV / MOM / CURVE) 확정 이후 두 갈래로 진행한 리서치 정리.

1. **CURVE 팩터 발전** — ML 기반 hybrid 모델 (vol regime switching)
2. **지표-비지표 RV 분석** — 보간 spread + 계절성 + 이벤트 + 개별 종목 drivers

---

## 0. 데이터·환경

| 항목 | 값 |
|---|---|
| DB | SCON MySQL, table `ktb` + `ktb_trade_flow` |
| 기간 | 2015-01-02 ~ 2026-05-11 (11.4년, 2,787 영업일) |
| Universe | 국고채 149 종목 (3-10y 분석 시 86종) |
| 5대 지표 | 3y / 5y / 10y / 20y / 30y (label 컬럼) |
| 거래주체 | 외국인 / 금융투자 / 보험 / 자산운용 / 은행 / 기타금융 / 기금공제 / 국가지자체 / 기타법인 / 개인 |

---

## 1. CURVE 팩터 발전 — ML 기반 hybrid

### 1.1 출발점

기존 `CURVE_v2`: `signal = -sign(cum_slope_21d)` 매일 flip → gross Sharpe +1.5, 그러나 거래비용 1bp/trade 부담 시 net 음수. 이를 개선하는 다양한 접근 시도.

### 1.2 시도한 접근들 (전체 비교)

| Phase | 접근 | 결과 | 결론 |
|---|---|---|---|
| 9 | Kelly criterion 3팩터 결합 | OOS rolling Sharpe **−0.81** | β-projection cross-section 실패 |
| 10 | Forward 5d slope prediction (rolling OLS) | **IC = +0.015** | 5d ahead 예측 사실상 불가 |
| 11 | slope = CM (momentum) + CR (mean-rev) 선형 분해 | OLS 학습 `w_M = -0.137` (음수) | momentum component 존재 X |
| 12 | Multi-output ML (Y1=1d, Y2=21d) | Y2 OOS IC +0.19 | 신호 자체는 가치 있음 |
| 13 | Binary classifier (5일 내 ±T 도달 확률) | 단일 split 에서 Yup 실패 | regime artifact 의심 |
| 14 | **Walk-forward 11년 (10 folds)** | 모든 AUC 0.55~0.62 | 양방향 모두 작동 확인 |
| 15 | 5d ML signal backtest | Yup 음수, Ydn 양수, baseline 압도 | 단일 trading 부적합 |
| 16 | 30d horizon 재학습 + backtest | **Yup 양수, Ydn 음수** (5d 정반대) | horizon 별 잡는 effect 다름 |
| 17 | Vol regime × horizon switching | LOW=Yup_20, MID=Yup_20, HIGH=Ydn_3 | pure ML +65 bp/y (약함) |
| 18 | **Hybrid: LOW=ML, MID/HIGH=CR_v2** | **+243 bp/y** (baseline +215 대비 +13%) | **채택** |

### 1.3 최종 Hybrid B 모델

**룰**:

```
vol_regime = expanding tercile of sigma_63d(dslope)

if regime == "LOW":
    if ML_Yup_20 prob >= top20 threshold (in-sample 80%ile):
        signal = +1 (STEEPEN)
    else:
        signal = 0
elif regime in ("MID", "HIGH"):
    signal = -sign(cum_slope_21d)   # CR_v2 baseline
```

**성과 (검증 구간 2017~2026, 9년)**:

| 구분 | N | win% | mean/trade | ann bp/y |
|---|---|---|---|---|
| **Hybrid B** | 1,684 | 56.4% | +1.34 bp | **+243** |
| CR_v2 baseline only | 2,283 | 54.9% | +0.88 bp | +215 |
| Pure ML regime-switch | 369 | 55.6% | +1.44 bp | +65 |

**Per-regime PnL 분해**:

| regime | days | CR_v2 alone | Hybrid B |
|---|---|---|---|
| LOW (저변동) | 653 | **−2 bp/y** ← 약점 | **+36 bp/y** ← ML 채움 |
| MID (중변동) | 818 | +99 bp/y | +99 bp/y |
| HIGH (고변동) | 817 | +119 bp/y | +119 bp/y |

→ **ML 의 핵심 가치 = baseline 의 LOW regime 빈틈 보완**. 독립 trading 도구가 아닌 supplement.

### 1.4 운용 도구

**[`factor_trading/scripts/curve_hybrid_signal.py`](scripts/curve_hybrid_signal.py)**: 날짜 인풋 → `-1` (FLATTEN) / `0` (NO POSITION) / `+1` (STEEPEN) 출력.

```bash
# CLI
python factor_trading/scripts/curve_hybrid_signal.py 2026-05-11           # → -1
python factor_trading/scripts/curve_hybrid_signal.py 2026-05-11 --verbose
python factor_trading/scripts/curve_hybrid_signal.py                       # DB 최신

# Python import
from factor_trading.scripts.curve_hybrid_signal import hybrid_signal
sig = hybrid_signal("2026-05-11")              # int
sig, info = hybrid_signal("2026-05-11", verbose=True)  # (int, dict)
```

내부 safety:
- Train cutoff = `target − 30 영업일` (label boundary leak 차단)
- Regime tercile = expanding window (target 이전만)
- ML threshold = 매번 train 데이터 자체 predict 분포의 80% (post-hoc 9년 분포 사용 X)

### 1.5 한계와 caveat

| 항목 | 내용 |
|---|---|
| Look-ahead 안전성 | Signal 자체 OK. 단 학습 label 의 30일 boundary leak ~1.2% (수정됨) |
| Variant 선정 | Hybrid B 가 best 라는 결정 자체가 9년 전체 본 후 → post-hoc bias 존재 |
| Calibration | ML prob 절댓값 과대평가. ranking (top 20% / 10%) 으로만 사용 |
| Regime stability | fold AUC std 0.06~0.21 — 일부 해는 noise |
| 결합 백테스트 시 | 2025-01-02 등 historic 시점 결합 OK. 단 결합 weights 도 expanding 으로 |

---

## 2. 지표-비지표 RV 분석

### 2.1 방법론

**핵심 spread 정의**:

매 `price_date` 마다 5개 지표 (3/5/10/20/30y) 의 `(remain_year, ytm)` 으로 piecewise linear 보간 yield curve 구성. 비지표 종목 각각의:

```
spread_bp = (actual_ytm − interp_ytm) × 100
```

여기서 `interp_ytm` 은 인접 두 지표 anchor 의 **잔존만기 거리 가중평균**:

```
interp_ytm = y_L · (x_R − remain) / (x_R − x_L) + y_R · (remain − x_L) / (x_R − x_L)
```

- **spread > 0**: 비지표가 보간선 위 → 가격 underperform → 디스카운트
- **spread < 0**: 비지표가 보간선 아래 → 가격 outperform → 프리미엄

**Panel 규모**: 92,628 rows × 113 unique non-bench bonds, 보간 가능 구간 (3y ≤ remain ≤ 30y).

**전체 통계**:

| 통계 | 값 |
|---|---|
| 평균 | **+2.39 bp** (비지표 평균적으로 디스카운트) |
| median | +2.16 bp |
| std | 3.01 bp |

**Bucket 별 mean spread**:

| Bucket | mean (bp) |
|---|---|
| 3-5y | +3.48 |
| **5-10y** | **+4.01** (가장 wide) |
| 10-20y | +1.06 |
| 20-30y | -0.09 (사실상 zero) |

### 2.2 명제 A — 계절성 (확인, KW p ≈ 0)

**월별 mean spread**:

| 월 | mean | 월 | mean |
|---|---|---|---|
| 1월 | +3.41 | 7월 | +1.83 |
| 2월 | +2.63 | **8월** | **+1.46** ← tightest |
| 3월 | +2.78 | 9월 | +2.13 |
| 4월 | +2.22 | 10월 | +2.32 |
| 5월 | +1.82 | 11월 | +2.21 |
| 6월 | +2.20 | **12월** | **+3.55** ← widest |

**핵심**:
- **연말 (12월) vs 한여름 (8월) 차이 ≈ 2 bp** — 운용 가능 magnitude
- KW p < 0.001 모든 bucket
- 운용 함의: 연말~연초 매수 / 한여름 청산 (basic seasonal trade)
- 2024-12 / 2025-1 = +7.1 / +6.3 bp (극단치, regime spike — 발행 이슈 관련 가능)

### 2.3 명제 B — 이벤트 임팩트 (매우 강한 통계 증거)

**탈락 종목 event study (N=71)**:

| Offset | mean spread (bp) |
|---|---|
| T-30 ~ T-1 | NaN (지표 시기 → spread undefined) |
| **T = 0** | **+2.82** ← 즉시 점프 |
| T+1 | +2.95 |
| T+10 | +2.69 |
| T+30 | +2.31 |
| T+60 | +2.36 |

drift T+1~T+30 vs T+31~T+60: **−0.44 bp**, t=6.86, **p<1e-11**

**신규 종목 event study (N=78)**:

| Offset | mean spread (bp) |
|---|---|
| T-60 | NaN |
| T-30 | −1.54 |
| T-10 | −2.18 |
| T-5 | −2.38 |
| **T-1** | **−2.61** ← max compression |
| T = 0 이후 | NaN (지표 시기) |

compression T-60~T-31 vs T-5~T-1: **−1.28 bp**, t=10.45, **p<1e-23**

**한 종목의 평생 spread 궤적**:

```
[비지표 신규]  편입 30d 전부터 -1.21 → 점진 압축 → T-1 max -2.61
[지표 시기]   spread undefined (보간 anchor)
[탈락 즉시]   T=0  +2.82 jump
[탈락 이후]   T+30 +2.31, T+60 +2.36 점진 수렴 (장기 비지표 평균)
```

**비대칭**: 신규 compression magnitude (−1.28 bp) > 탈락 drift (−0.44 bp).

→ **시장은 신규 편입을 ~30일 전부터 가격에 미리 반영, 탈락은 사후 (T=0) 즉시 반영**.

### 2.4 개별 종목 driver 분석 (3-10y 심화)

**Cross-sectional regression** — bond level mean_spread 를 features 로 회귀:

| Model | R² |
|---|---|
| 기본 features (remain, age, coupon, anchor_dist, was_ever_bench, nominal_tenor) | 0.673 |
| **+ 유동성 features (log_gross, dominant_share, foreign_share, inst_share)** | **0.817** |

**Cross-section OLS coefficients (with liquidity, R²=0.82, N=81)**:

| Feature | coef | t | 의미 |
|---|---|---|---|
| `nearest_anchor_dist` | **+1.76** | +4.51 | anchor 멀수록 wider (가장 강한 driver) |
| `dominant_mean` | **−41.77** | **−2.94** | concentration ↑ → tighter |
| `foreign_share_mean` | **+11.55** | +2.39 | 외국인 비중 ↑ → wider |
| `log_gross_mean` | +0.42 | +2.01 | 거래량 ↑ → 약하게 wider |
| `remain_mean` | +0.42 | +2.76 | 만기 ↑ → wider |

**Panel within (bond FE) + month FE**:

| Feature | coef | t |
|---|---|---|
| `log_gross` | **+0.073** | **+6.26** |
| `dominant_share` | −1.01 | −3.17 |
| `nearest_anchor_dist` | +0.87 | +4.81 |
| `inst_share` | +0.19 | +2.28 |

**해석**:

1. **`nearest_anchor_dist`** 가 모든 모델에서 가장 robust한 driver (3y 또는 10y anchor 와의 거리). 5y/10y 사이 가운데 ~7-8y 위치가 spread 가장 wide.
2. **High liquidity → slightly wider spread** (직관 반대): 거래 활발 = price discovery active = 보간선 대비 실제 가격 편차 큼 (인과 reverse: real mispricing 이 있어서 거래가 활발한 것)
3. **Concentration (dominant_share) ↑ → tighter**: 단일 매수자가 가격 끌어내림, effective volatility ↓
4. **외국인 비중 ↑ → wider**: 외국인 매매가 비지표 가격 변동 증폭

**Top wide bonds** (3-10y, mean spread):
- 모두 remain 7-9y (5y/10y anchor 사이 가운데)
- KR103502GE63 (23-5, 8.79y) +7.01 bp
- KR103502GDC6 (23-11, 8.55y) +6.50 bp
- 모두 AC1 0.95+ (매우 persistent, half-life ~14-45일)

**Top tight bonds** (most negative):
- 모두 remain ≈ 3.1y (3y 편입 직전 신규 종목)
- mean spread −5 ~ −8 bp (편입 compression 효과)
- KR103501GAC4 (3.09y) −5.04 bp, KR103501GA68 −4.89 bp

### 2.5 RV 모델 통합 활용 방안 (3-Layer)

```
Layer 1 — RV residual (기존)
  ε_i = dY_i − β_3Y·dY_3Y − β_10Y·dY_10Y
  → cross-section within-bucket quintile Q5 LONG / Q1 SHORT

Layer 2 — Bench-curve spread (신규)
  spread = ytm_i − interp_ytm
  → bond 별 "보간 대비 비싼/싼" 직관적 지표

Layer 3 — Abnormal spread (신규, 가장 정교)
  abnormal = spread − expected_spread (panel model)
  expected_spread = f(remain, anchor_dist, age, coupon, log_gross, dominant_share, ...)
  → anchor 거리·유동성 보정 후 진짜 mispricing
```

**활용 룰**:

| 시나리오 | 권고 |
|---|---|
| abnormal_spread 큰 양수 + RV ε 양수 (둘 다 cheap) | **double-confirm LONG** |
| abnormal_spread 큰 음수 + RV ε 음수 (둘 다 rich) | **double-confirm SHORT** |
| 두 신호 충돌 | size down 또는 skip |
| Persistence (AC1 0.95+) | 시그널 안정, mean-rev 느림 (half-life 14~45d) |

**이벤트 overlay**:
- 다음 편입 후보 종목 (3y, 5y, 10y, 20y, 30y 만기 부근의 vintage) 식별 → 발표 30일 전부터 LONG bias
- 막 탈락한 종목 → +2.82bp jump 후 점진 수렴 → SHORT (or 다른 비지표 LONG / 탈락 종목 SHORT pair)

**계절 overlay**:
- 12-1월: 비지표 LONG / 지표 SHORT 사이즈 up
- 7-8월: 사이즈 down 또는 close
- 5-10y bucket 집중 (mean +4 bp 가장 wide)

---

## 3. 산출물 정리

### 3.1 분석 scripts

| Phase | 파일 | 내용 |
|---|---|---|
| 14 | `scripts/curve_ml_walkforward.py` | 5d binary classifier walk-forward |
| 16 | `scripts/curve_ml_30d_full.py` | 30d binary classifier |
| 17 | `scripts/curve_ml_regime_switch.py` | vol regime × horizon 매핑 |
| 18 | `scripts/curve_ml_hybrid.py` | Hybrid B backtest |
| 18 | **`scripts/curve_hybrid_signal.py`** | **운용 CLI** |
| 19 | `scripts/bench_nonbench_spread.py` | 보간 + spread panel 산출 |
| 19 | `scripts/bench_spread_seasonality.py` | 계절성 분석 |
| 19 | `scripts/bench_spread_event_study.py` | 지표 변경 이벤트 study |
| 19 | `scripts/bench_spread_dashboard.py` | 종합 dashboard |
| 20 | `scripts/bench_spread_individual.py` | 개별 종목 cross-section + panel |
| 21 | `scripts/bench_spread_liquidity.py` | 유동성 통합 |

### 3.2 데이터 산출물 (`data/factor_trading/`)

| 위치 | 핵심 파일 |
|---|---|
| `phase14_ml_walkforward/` | `oos_predictions.csv`, `overall_oos_metrics.csv`, `fold_metrics.csv` |
| `phase16_ml_30d/` | `oos_predictions_30d.csv`, `overall_oos_metrics_30d.csv` |
| `phase17_ml_regime_switch/` | `regime_x_label.csv`, `regime_switch.png` |
| `phase18_ml_hybrid/` | `hybrid_results.csv`, `hybrid_results.png` |
| `phase19_bench_spread/` | `spread_long.parquet`, `bench_history.csv`, `dashboard.png`, `seasonality/`, `event_study/` |
| `phase20_bench_individual/` | `per_bond_descriptive.csv`, `per_bond_full.csv`, `abnormal_panel.parquet`, `individual_dashboard.png` |
| `phase21_bench_liquidity/` | `abnormal_with_liquidity.parquet`, `liquidity_dashboard.png` |

### 3.3 운용 indicator 단일 호출

```python
# 1. CURVE direction signal (Hybrid B)
from factor_trading.scripts.curve_hybrid_signal import hybrid_signal
direction = hybrid_signal("2026-05-11")  # -1 / 0 / +1

# 2. Bench-Non-bench abnormal spread (Layer 3)
import pandas as pd
abn = pd.read_parquet("data/factor_trading/phase21_bench_liquidity/abnormal_with_liquidity.parquet")
today_abn = abn[abn["price_date"] == "2026-05-11"]   # bond_code × abnormal_spread_liq
```

---

## 4. 핵심 인사이트 종합

### 4.1 CURVE 측면

1. **한국 KTB 10y-3y slope = pure mean-reverting** (모든 horizon, AR(2) ρ_1=-0.067 ρ_2=-0.081, momentum 가설 모두 기각)
2. **5d / 30d horizon 잡는 effect 다름**: 5d Ydn / 30d Yup 양수
3. **단일 ML 으로는 CR_v2 baseline 못 이김** (+65 bp/y < +215 bp/y)
4. **ML의 진짜 역할 = baseline 의 LOW vol regime 약점 보완** (Hybrid B +243 bp/y, +13%)

### 4.2 지표-비지표 측면

1. **비지표 평균 +2.39 bp 디스카운트**, 5-10y bucket 가장 wide (+4.01)
2. **계절성 강함**: 12월/1월 widest (+3.5bp), 7월/8월 tightest (+1.5bp), 진폭 ~2bp
3. **이벤트 impact 비대칭**:
   - 신규 편입: 30일 전부터 미리 압축 (~-1.28bp)
   - 탈락: T=0 즉시 +2.82bp jump
4. **개별 종목 spread driver** (R²=0.82):
   - `nearest_anchor_dist` 가 가장 강력
   - 유동성 features 추가가 결정적 (R² 0.67 → 0.82)
   - High liquidity / foreign share ↑ → wider
   - High concentration → tighter

### 4.3 통합 인사이트

- 한국 채권 RV 시장에서 **시장 미시구조 (유동성/주체)** 가 가격 deviation 의 큰 부분 설명
- 단순 보간 spread (Layer 2) → abnormal spread (Layer 3) 변환으로 진짜 mispricing 추출 가능
- CURVE (slope) 와 bench-non-bench spread 는 **독립적인 alpha source** — 결합 가능

---

## 5. 정직한 caveat / 한계

| 영역 | 한계 |
|---|---|
| Hybrid B 의 +243 bp/y | Post-hoc variant 선정 + 30일 label boundary leak 1.2% 포함 (이미 fix). 진정한 forward OOS 는 보수적 추정 필요 |
| ML probability 절댓값 | Calibration 안 됨. ranking-based 만 신뢰 |
| 거래 비용 | CURVE 분석은 cost=0 (사용자 지정), 실거래 1bp/trade 시 일부 약화 |
| Panel regression in-sample fit | Phase 21 의 abnormal_spread 도 전체 panel OLS 잔차. expanding window 재학습으로 진정한 OOS abnormal_spread 산출 권장 |
| 결합 신호 동시 사용 | 두 alpha (CURVE direction + abnormal_spread) 결합 시 weights 도 expanding 으로 update 해야 추가 leak 방지 |

---

## 6. 다음 step (미진행)

### 6.1 즉시 가능 (낮은 비용)

| 작업 | 효과 |
|---|---|
| Streamlit GUI 에 daily Hybrid B signal + abnormal_spread top 10 표시 | 운용 의사결정 보조 indicator 노출 |
| Abnormal spread walk-forward 재학습 (panel OLS, 매년) | 진정한 OOS abnormal_spread series |
| `curve_hybrid_signal.py` import 가 다른 alpha 와 결합 시 wrapper 작성 | RV pipeline 에 layer 추가 |

### 6.2 중기 (의미 있는 확장)

1. **Walk-forward variant selection**: Hybrid 의 LOW regime 모델 (Yup_20 vs Ydn_20 vs none) 자체도 매년 재선정 → 진정한 forward-safe hybrid
2. **다음 편입 후보 자동 식별**: 만기/vintage 패턴으로 다음 5대 지표 후보 종목 추정 → 발표 ~30일 전부터 LONG 매수 백테스트
3. **RV 모델 + Layer 3 결합 백테스트**: ε 잔차 + abnormal_spread z-score 합성 신호의 ann bp/y 측정
4. **계절 overlay backtest**: 12-1월 size up / 7-8월 size down 룰 적용 시 PnL 변화

### 6.3 장기 (research-grade)

1. **Stacking meta-model**: CURVE signal + bench-spread signal + RV ε → input feature 로 사용하는 meta classifier (every-year walk-forward)
2. **Liquidity regime ML**: dominant_share, foreign_share 등을 input 으로 미래 spread compression 확률 예측
3. **Cross-asset overlay**: UST 10y dY, KRW/USD change 를 features 로 추가 → 외생 driver 식별

---

## 7. 메모리 / 운용 참고

| 항목 | 값 |
|---|---|
| 현재 운용 가능 indicator | `curve_hybrid_signal.py` (CURVE direction), Hybrid B |
| 추가 가능 indicator | `abnormal_with_liquidity.parquet` 에서 bond × date 조회 |
| 데이터 갱신 주기 | 매일 (DB 자동 update 가정) |
| 최신 데이터 확인 | DB 의 `MAX(price_date) FROM ktb` 또는 `ktb_trade_flow` 의 `MAX(기준일자)` |

---

작성: 2026-05-12
이전 spec: [`bond_3factor_final_spec.md`](bond_3factor_final_spec.md), [`factor_methodology.md`](factor_methodology.md), [`portfolio_replication_research.md`](portfolio_replication_research.md)
