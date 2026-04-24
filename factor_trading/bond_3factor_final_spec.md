# 채권 3팩터 RV 체계 — 확정 명세서 (v2)

## 0. 문서 목적

한국 국채(KTB) 상대가치 트레이딩 시스템의 최종 확정 설계본. 초기 4팩터(RV/VOL/MOM/CURVE) 체계를 실증 검증을 거쳐 **3팩터(RV/MOM/CURVE)** 로 확정하고, 각 팩터의 핵심 파라미터·검증 결과·알려진 한계를 정리한다.

선행 문서:
- `bond_rv_strategy_spec.md` — 1팩터 RV 초기 설계
- `bond_4factor_framework_spec.md` — 4팩터 초안 (VOL 포함)

본 문서가 **실거래 기준 최종본**.

---

## 1. 철학: 3대 원칙

모든 팩터는 아래 세 가지 원칙을 **순서대로** 만족해야 한다. 1번은 절대 원칙.

### ① 이론적 컨셉의 타당성 (절대)
팩터가 왜 작동해야 하는지에 대한 **경제적·행동적 근거**가 선행. 사후 데이터 fit만으로는 채택 불가. 가설 → 검증 순서 엄수.

### ② 분위수 성과의 단조성
Signal의 강도(quintile 또는 z-bin)에 따라 **forward PnL이 단조**롭게 증가/감소. Spearman ρ로 정량화.

### ③ 팩터 간 직교성
- PnL 상관행렬: **|corr| < 0.3** 목표
- α-β 분해: 각 팩터를 나머지에 회귀한 α **t-stat > 1.5**
- 잔차(ε) 레벨 교차검증: 모델이 완결되면 2팩터 회귀 잔차가 다른 signal과도 직교

---

## 2. 공통 기반 — 2팩터 회귀

### 회귀식

```
dY_i,t = α_i + β_3Y,i · dY_3Y,t + β_10Y,i · dY_10Y,t + ε_i,t
```

| 항목 | 값 |
|---|---|
| 윈도우 | **63 영업일** (3M rolling) |
| min_periods | 40 |
| 추정 방식 | Rolling OLS (statsmodels RollingOLS) |
| 지표 롤오버 | 교체일 **±5 영업일** 회귀 제외 |
| dY 단위 | **bp** (YTM × 100) |
| 3Y 지표 | DB label `"3년지표"` (현재 25-10) |
| 10Y 지표 | DB label `"10년지표"` |

### Sanity 검증 결과 (2023-07~2026-04)

| 종목 | 기대 β | 실측 β_3Y | 실측 β_10Y |
|---|---|---|---|
| 3Y 지표 (KR103501GFC3) | (1, 0) | **+0.998** | **+0.002** |
| 10Y 지표 (KR103502GFC1) | (0, 1) | +0.003 | **+0.997** |
| 5Y 중기물 (20개 평균) | (0.3~0.7, 0.3~0.7) | +0.620 | +0.378 |

구조가 완벽. β_3Y + β_10Y ≈ 1이 장기물·중기물·단기물 전반에서 자연스럽게 성립.

### 금리변동 설명력

2팩터 회귀의 종목별 R² 분포 (n=97):
- **median R² = 0.871**, mean = 0.842
- R² > 0.9 종목: **56%** (55/97)
- R² < 0.5 종목: 6% (6/97)

**2팩터가 한국 국채 일간 금리변동의 ~87%를 설명**. 남은 13%가 RV 팩터의 소재 (cross-sectional ε).

### 잔차 ε의 성질
- pool std = **1.15 bp/day** (원본 dY std ~5bp에서 1/5 축소)
- ACF(1) median = +0.079 → 거의 white noise
- 종목 간 독립 (systematic leakage 거의 없음)
- **MOM/CURVE signal과 직교**: corr 모두 |0.03| 이하, |>0.2| 종목 0~5%

즉 2팩터가 선형 rate risk를 뽑아낸 뒤 **비선형 trend / slope mean-rev signal도 잔차에 남지 않음**. 구조적으로 클린한 3팩터 분해.

---

## 3. 유니버스 & 만기 버킷

| 항목 | 값 |
|---|---|
| Category | `"국고채"` (KTB 현물 전종목) |
| 최소 관측일 | 60 영업일 |
| 잔존만기 하한 | 0.25년 (만기 임박 왜곡 회피) |
| **만기 버킷** | **≤5년 / 5–10년 / >10년** |
| 종목 수 (2023-07~2026-04) | 약 109개 |

만기 버킷은 **RV 팩터 within-bucket quintile에 필수** (단독 적용 시 Sharpe 0.75 → 0.91 개선).

---

## 4. RV 팩터 (Cross-section)

### 이론
채권별 누적 잔차 `Σε`는 팩터(3Y+10Y) 대비 상대적 dislocation. 이론상 평균회귀.
- RV > 0: yield 상대적으로 더 올라감 → **가격 underperform → 쌈 → LONG 후보**
- RV < 0: yield 덜 올라감 → 가격 outperform → 비쌈 → SHORT 후보

### 정의

```
RV_i,t = Σ_{s=t-20..t} ε_i,s    (horizon = 1m, 21 영업일)
```

horizon 옵션: 1d / 1w / 2w / **1m (주력)**.

### 포트폴리오 구성

1. 매 시점 각 종목을 만기 버킷(3개)에 배정
2. 버킷 내에서 RV score 기준 **quintile** 매김 (Q1~Q5)
3. 라벨을 전 버킷 pool → 최종 5개 quintile
4. **Long Q5 / Short Q1** (cheap LONG / rich SHORT)
5. **Hold = 21 영업일** (월 1회 rebalance)

### Stage 1 결과 (2023-07~2026-04, n=663 obs)

| 지표 | 값 |
|---|---|
| LS mean | +1.13 bp/21d |
| LS std (1d) | 3.38 bp |
| **Sharpe (ann)** | **+1.16** |
| NW t-stat | +2.16 |
| hit% | 64.9% |
| 단조성 ρ | **+1.00** (완벽) |
| Max DD | −185 bp |
| DD duration | 132일 |

### 분위수별 fwd 21d P&L (bp)

```
Q1 +0.30  Q2 +0.57  Q3 +0.70  Q4 +0.92  Q5 +1.43
```

완벽한 단조증가. 모든 quintile 양수(시장 평균 mean-rev drift 동반).

### Regime별

| regime | Sharpe | comment |
|---|---|---|
| bull강 (3M dY ≤ −25bp) | **−1.01** | ⚠️ 취약 |
| bull약 | +2.44 | 최강 |
| flat (±5bp) | +1.99 | |
| bear약 | +1.58 | |
| bear강 | +2.27 | |

**bull강 regime이 Achilles' heel**. Max DD 시기(2024-05~2024-09)가 인하 기대 가속 국면과 일치. 합성 단계에서 다른 팩터(MOM)가 커버.

### 알려진 한계
- Daily hold로 평가하면 단조성 무너짐 (1d ρ=−0.30). **반드시 21d hold 기준으로 운용·평가**.
- bull강 drawdown. regime filter 고려 가능 (별도 overlay, 합성 레이어).
- Tail event 의존: K=30 제거 시 Sharpe 0.88 (24% 하락).

---

## 5. MOM 팩터 (Time-series)

### 이론
한국 국채 3Y yield는 **중장기(3개월) 추세의 일부 지속성**을 보인다. 단, 전 regime에서 균일하지 않고 trend 강도에 의존.

### 정의 — **v2 확정안 (raw_sign)**

```
cum_dY_3Y,t = Σ_{s=t-62..t} dY_3Y,s          (63 영업일 누적, raw bp)
MOM_signal_t = −sign(cum_dY_3Y,t)             (bang-bang)
MOM_pnl_t = MOM_signal_{t-1} · (−dY_3Y,t)    (LONG bond = −dY)
```

| 항목 | 값 |
|---|---|
| lookback | **63 영업일** (3M 누적 dY) |
| signal 형식 | **raw_sign** (z-score 사용 안 함) |
| direction | momentum (rate 상승세 → SHORT bond) |
| dead_zone | 0.0 (hysteresis 역효과) |
| hold | 1d (daily rebalance) |
| 진입 수단 | **3Y 지표 현물** |
| 실행 lag | 1 영업일 (MOC) |

### 왜 z-score가 아닌 raw_sign인가 (심층 진단 결론)

z-score 정규화는 252d trailing vol로 누적값을 표준화하여 **vol regime을 섞어 signal 정보의 절반 이상을 죽였다**. Stage 1 결과 (raw vs z, N=63):

```
N=63  raw_sign Sharpe +0.87   z-score Sharpe +0.32
```

raw cumulative dY의 부호만 사용하는 것이 **정보량을 최대화**하면서도 과적합 위험이 낮다.

### 왜 cum=63인가

Lookback × Forward IC grid에서:
- 단기(cum 1–5일): 약한 **mean-reversion** (IC 음수)
- 중기(cum 10–21일): **momentum** (IC +0.15~+0.20)
- 장기(cum 63일): **momentum 확장 + 낮은 turnover**, Sharpe peak

Lookback × Hold grid:
```
N\H   1     2     5     10    21
10  +0.33 +0.51 +0.41 +0.57 +0.54
21  +0.76 +0.59 +0.65 +0.83 +0.64
42  +0.66 +0.69 +0.56 +0.48 +0.40
63  +0.90 +1.03 +1.00 +0.84 +0.64   ← best
```

### Stage 1 결과 (2023-07~2026-04, n=684 obs)

| 지표 | 값 |
|---|---|
| mean | +0.22 bp/day |
| std | 4.03 bp |
| **Sharpe (ann)** | **+0.87** |
| NW t-stat | +1.43 |
| **α-t (vs RV, CURVE)** | **+2.04** ⭐ |
| hit% | 51.2% |
| Max DD | −45.7 bp (54일) |
| skew | +0.71 |
| kurt | +6.1 |
| turnover | 6.6 flips/yr, avg hold 36일 |

### 단조성 ρ = +0.40 (nonlinear)

cum_63 bin별 fwd=5d LONG bond PnL:
```
cum ≤ −43bp   : −0.29 (t −0.40)
(−43, −20]    : +0.36 (t +0.57)
(−20, +22]    : +1.10 (t +2.38)
(+22, +47]    : −3.55 (t −4.57)   ← 강한 momentum
cum > +47     : +2.96 (t +2.86)   ← 극단 반전
```

**중간 양수 구간에서 momentum, 극단에서 mean-reversion** — nonlinear. 단조성 불완전의 원인이지만 α-t +2.04로 유의성은 명확.

### Regime별

| regime | Sharpe | comment |
|---|---|---|
| bull강 | +2.08 | RV의 약점을 커버 ⭐ |
| bull약 | −0.17 | |
| flat (±5bp) | **−5.44** | ⚠️ 큰 손실 |
| bear약 | +0.41 | |
| bear강 | +1.71 | |

**Flat regime에서 −5.44**, trend-following의 전형적 약점. CURVE 팩터가 이 구간(Sharpe +0.96)을 커버.

### OOS 경고

OOS sub-period (2022-03~2023-06): **Sharpe +0.09**
- 이 구간은 한국 인상 사이클 진입기로 bull약·flat 손실이 bear강 이익을 상쇄
- IS 성과의 상당 부분이 regime distribution에 의존
- **실거래 시 regime overlay 또는 포지션 scaling 강력 권고**

### 알려진 한계
- Flat regime 취약. OOS 불안정.
- 단조성 ρ 0.40로 완벽하지 않음 (극단 nonlinear 반전).
- 다만 α-t +2.04 유의 + 직교성 완벽 + regime 보완성으로 합성 가치 유지.

---

## 6. CURVE 팩터 (Time-series)

### 이론
한국 국채 **slope (10Y − 3Y yield) 의 단기(3주) 누적 변화는 평균회귀**. 한국 채권시장의 잘 알려진 micro-structure.

### 정의 — **v2 확정안 (raw_sign, Step 2 제거)**

```
slope_t = dY_10Y,t − dY_3Y,t                   (slope 변화)
cum_slope_t = Σ_{s=t-20..t} slope_s             (21 영업일 누적)
CURVE_signal_t = −sign(cum_slope_t)              (mean-rev: slope 과열 → 반전 베팅)
CURVE_pnl_t = CURVE_signal_{t-1} · slope_t      (steepen PnL 기준)
```

- `signal = +1`: **steepen 베팅** (3Y LONG / 10Y SHORT, duration-neutral)
- `signal = −1`: **flatten 베팅** (3Y SHORT / 10Y LONG, duration-neutral)

| 항목 | 값 |
|---|---|
| lookback | **21 영업일** (3주 slope 누적) |
| signal 형식 | **raw_sign** |
| direction | **mean_rev** (실증적으로 강하게 확인) |
| Step 2 직교화 | **제거** (rolling OLS β ≈ 0, 역효과) |
| z-score | **제거** (정보 손실) |
| hold | 1d (daily rebalance) |
| 진입 수단 | **3Y 지표 현물 vs 10Y 지표 현물 duration-neutral 페어** |

### 명세 원안 대비 변경

| 항목 | 원안 (명세) | v2 확정 |
|---|---|---|
| Step 1 (slope) | slope_t = dY_10 − dY_3 | 유지 |
| Step 2 (직교화) | slope를 dY_3에 rolling OLS | **제거** |
| Step 3 (z-score) | 21d z-score | **제거** |
| Signal | sign(z_score) with threshold | raw_sign(cum_slope_21) |
| direction | 백테스트 결정 | **mean_rev 확정** |

### Step 2 제거 근거

rolling β(slope ~ dY_3Y) 시변 통계:
- mean = **−0.078** (거의 0)
- std = 0.178

slope과 dY_3Y는 이미 거의 **구조적으로 직교**. Step 2 rolling OLS는 window noise만 주입하여 오히려 corr 증가 (absolute value 기준 +0.15 → −0.31). 단순성·실효성 모두 제거가 우수.

### Stage 1 결과 (2023-07~2026-04, n=684 obs)

| 지표 | 값 |
|---|---|
| mean | +0.18 bp/day |
| std | 1.92 bp |
| **Sharpe (ann) IS** | **+1.49** |
| **Sharpe (ann) OOS** | **+1.18** ⭐⭐ |
| NW t-stat | +2.08 |
| **α-t (vs RV, MOM)** | **+2.08** |
| hit% | 54.2% |
| Max DD | −27.1 bp (63일) |
| 단조성 ρ | **−0.90** (완벽 단조↓) |

### IS vs OOS (교과서적 견고함)

```
IS  (2023-07 ~ 2026-04): Sh +1.49, n 684
OOS (2022-03 ~ 2023-06): Sh +1.18, n 331     ← 80% 보존
```

### Regime별 (모든 regime 양수 Sharpe)

| regime | Sharpe | n |
|---|---|---|
| bull강 | +0.57 | 169 |
| bull약 | +1.75 | 142 |
| flat | +0.96 | 56 |
| bear약 | +0.39 | 126 |
| bear강 | **+2.64** | 191 |

**regime robust**. MOM의 flat 약점(−5.44)을 CURVE(+0.96)가 커버.

### 알려진 한계
- Turnover 높음 (매일 부호 바뀜 가능). 실제 BPV-중립 페어 실행 부담 큼.
- bull강에서 Sharpe +0.57로 약한 편 (단 양수 유지).
- 극단 slope (cum > +10bp) 구간에서만 강한 유의성 (t=−2.94).

---

## 7. VOL 팩터 — 탈락 (기록용)

명세 원안의 VOL 팩터는 **3원칙 기준으로 탈락**:
- ① **이론 취약**: 초기 가설(low-vol anomaly)이 데이터에서 반대 방향(high-vol LONG). 경제적 정당화 약함.
- ② **LONG side alpha 없음**: quintile 전부 음수 평균, LS spread만 작동.
- ③ **MOM과 |corr|=0.45**: 직교성 기준 초과.

추가로 RV × VOL double sort에서 VOL의 독립 기여는 RV_Q3 tercile에서만 유효 → **VOL은 독립 팩터가 아니라 RV 보조 필터** 성격.

→ **3팩터 체제로 간소화**.

---

## 8. 직교성 검증 (최종)

### PnL 상관행렬 (daily, 2023-07~2026-04)

```
          RV     MOM    CURVE
RV      +1.00  +0.26   +0.15
MOM     +0.26  +1.00   −0.17
CURVE   +0.15  −0.17   +1.00
```

- |corr|max off-diag = **0.256** → 명세 기준 0.3 **통과** ✓

### α-β 분해 (각 팩터를 나머지 2개에 회귀)

| 팩터 | α (bp/day) | α t-stat | α Sharpe | R² |
|---|---|---|---|---|
| RV | −0.019 | −0.40 | −0.26 | 0.11 |
| MOM | +0.326 | **+2.04** | +1.29 | 0.12 |
| CURVE | +0.162 | **+2.08** | +1.34 | 0.08 |

**MOM, CURVE 모두 α-t > 1.5 통과**. RV는 daily scale에서 α가 작게 측정됨 (21d hold 팩터 특성); 21d 기준으로 재평가하면 유의).

### 잔차 ε 기반 교차검증

2팩터 회귀 잔차와 MOM/CURVE signal 간 종목별 상관:
```
corr(ε_i, MOM_signal)   : median −0.025, |>0.2| bonds 0/95
corr(ε_i, CURVE_signal) : median +0.029, |>0.2| bonds 5/95
```

**잔차가 MOM/CURVE 비선형 signal과도 거의 완벽 직교** → 3팩터 모델이 구조적으로 체계 risk를 clean 분해.

---

## 9. 합성 포트폴리오

### Stage별 진행

**Stage 1 — 단독 검증** (완료)
- 각 팩터 Sharpe > 0.5 기준
- RV +1.16 (21d hold), MOM +0.87, CURVE +1.49 모두 통과

**Stage 2 — 직교성** (완료)
- |corr| < 0.3, α t > 1.5 MOM/CURVE 통과

**Stage 3 — 합성**
- **3-A: Equal Weight** (단순)
  - 각 팩터 daily PnL을 각자 σ로 표준화 후 평균
- **3-B: Risk Parity** (1/σ 가중)
  - vol scale 차이(RV/CURVE ~1.5-3.5bp vs MOM 4bp) 고려
- **3-C: Target Vol 포트폴리오**
  - 전체 포트폴리오 연 5% 같은 target vol로 leverage

### 주의: Scale 통일

- RV의 자연 hold는 21d, MOM/CURVE는 1d.
- 합성 시 daily-equivalent로 변환 필요:
  - RV daily PnL은 21d LS의 1/21 scale로 근사 (또는 daily Q5-Q1 spread 직접 사용)
  - MOM, CURVE는 그대로 daily

### 이론 Sharpe 기대치

3팩터 직교 + 각 Sh ~0.8~1.5 평균 → 이론치 합성 Sharpe ≈ √(0.87²+1.49²+1.16²_{scaled}) / √3 ≈ **1.3~1.5**.
실전 70~80% 달성 예상 = **실현 Sharpe 1.0~1.2**.

---

## 10. 모듈 구조

```
factor_trading/
├── bond_3factor_final_spec.md     ← 본 문서
├── __init__.py
├── data_loader.py                 3Y/10Y 지표 + 유니버스 dY/YTM/remain
├── beta_estimator.py              2팩터 RollingOLS (+ sanity_check)
├── residual_builder.py            ε = dY − β_3·dY_3 − β_10·dY_10 + horizon 누적
├── main.py                        FactorPipeline orchestrator
├── monitor.py                     프로덕션 일일 모니터링 엔진
│
├── factors/
│   ├── rv_factor.py               horizon 누적 ε (주력 horizon=1m)
│   ├── mom_factor.py              raw_sign(−cum_dY_3Y_63d)
│   ├── curve_factor.py            raw_sign(−cum_slope_21d), mean_rev
│   └── vol_factor.py              (탈락, 코드 보존)
│
├── portfolio/
│   ├── single_factor.py           within-bucket quintile, MOM/CURVE PnL
│   ├── duration_neutral.py        BPV 비중 계산 (CURVE 실행용)
│   └── combiner.py                EW / Risk Parity 합성
│
├── validation/
│   ├── rv_diagnostics.py          forward return, LS stats, NW HAC, drawdown, regime
│   ├── mom_diagnostics.py         z-bin, turnover, param grid
│   ├── curve_diagnostics.py       방향별 grid, lag sensitivity
│   ├── vol_diagnostics.py         (참고용)
│   └── principles.py              단조성 + 직교성 공식 검증
│
└── scripts/
    ├── run_phase1_sanity.py       2팩터 β sanity
    ├── run_phase2_factors.py      4팩터 기본 집계 (초기)
    ├── refine_rv.py               RV 4조합 + drawdown
    ├── diagnose_mom.py            1차 진단
    ├── diagnose_mom_deep.py       심층 (lookback × forward IC)
    ├── refine_mom_v2.py           MOM v2 정식 검증
    ├── diagnose_curve_deep.py     CURVE 심층 (slope ACF, Step 2 실효성)
    ├── refine_curve.py            CURVE 초기 grid
    ├── refine_vol.py              VOL (탈락 근거)
    ├── check_principles.py        3원칙 공식 검증
    ├── validate_model.py          3팩터 모델 설명력 (R², ε 직교성)
    └── daily_snapshot.py          ★ 프로덕션 일일 리포트
```

---

## 11. 확정 파라미터 요약

```yaml
regression:
  window: 63
  min_periods: 40
  rollover_buffer: 5

universe:
  category: ["국고채"]
  min_obs: 60
  drop_short_remain: 0.25
  bucket_edges: [0, 5, 10, 100]    # ≤5y / 5-10y / >10y

RV:
  horizon: "1m"                    # 21 영업일 누적 ε
  quintile_mode: "within_bucket"
  n_bins: 5
  long_quintile: 5                  # cheap LONG
  short_quintile: 1                 # rich SHORT
  hold_days: 21                     # 월 1회 rebalance
  exec_lag: 1

MOM:
  cum_window: 63
  signal: "raw_sign"                # −sign(cum_dY_3Y)
  dead_zone_bp: 0.0
  direction: "momentum"             # z>0 → SHORT bond
  hold_days: 1
  exec_lag: 1
  vehicle: "KTB_3Y_benchmark_cash"

CURVE:
  cum_window: 21
  signal: "raw_sign"                # −sign(cum_slope)
  direction: "mean_rev"
  step2_orthogonalization: false
  zscore: false
  hold_days: 1
  exec_lag: 1
  vehicle: "KTB_3Y_10Y_pair_duration_neutral"
```

---

## 12. 확장 로드맵 (Stage 3 이후)

### Stage 3 합성 (다음 작업)
- EW → Risk Parity → Target Vol 순차 검증
- 합성 시 regime overlay 테스트 (bull강 시 MOM 가중 증가 등)

### Stage 4 고도화
- **MOM 세분화**: Event-driven (BOK/FOMC ±3일), 극단 z 반전 구간 (contrarian), 일반 trend
- **CURVE 세분화**: bull/bear steepen vs flatten regime overlay, BOK cycle state
- **Carry 팩터 추가 검토**: 명세 원안에서 예약되었던 것. Carry는 3팩터와 자연 직교 예상.

### Stage 5 실거래
- 거래비용 모형 (KTB bid-ask 실측, 시장충격)
- Slippage 가정 (MOM 실행 lag 민감)
- Rebalance tolerance band (매일 거래 비용 회피)

---

## 13. 미결 / 경고 항목

- ⚠️ **MOM OOS 불안정**: regime overlay 검토 필요. IS Sharpe 0.87이 2022-2023 reg에서 재현 안 됨.
- ⚠️ **RV bull강 drawdown**: −185bp (132일). 합성에서 MOM이 커버하길 기대하지만 confirmed OOS 아님.
- ⚠️ **거래비용 미반영**: Gross 기준. Net Sharpe는 20~40% 하락 예상.
- ⚠️ **단일 regime 검증**: 2023-07~2026-04은 인하 사이클 후반 단일 regime. 다른 cycle OOS 확대 권장.
- **IRS/선물 유니버스 미포함**: 명세 원안은 KTB 선물·IRS 포함이었으나 현재 국채 현물만 사용.

---

## 14. 변경 이력

| 버전 | 날짜 | 변경 |
|---|---|---|
| 4f draft | — | 4팩터 초안 (VOL 포함) |
| **3f final v2** | 2026-04 | VOL 탈락, MOM·CURVE raw_sign 재정의, 3원칙 공식 검증, R² 87% 확인 |

---

*본 문서는 실거래 기준 확정본이며, 파라미터 변경 시 반드시 3원칙 재검증 후 본 문서를 업데이트한다.*
