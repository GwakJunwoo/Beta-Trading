# CV1T — Curve V1 with TP/SL + DCA Cap (확정 명세서)

작성: 2026-05-13
모델 유형: Daily Curve Composite Pair Trading (8 instruments)
백테스트 구간: 2015-01 ~ 2026-05 (11.4년)

---

## 0. Executive Summary

| 항목 | 값 |
|---|---|
| **ann_net** | **+112.6 bp/y** |
| **Sharpe** | **+2.15** |
| **win %** | **66.4%** |
| **MDD** | **−233 bp** |
| Low vol ann | +91.3 bp/y |
| High vol ann | +176.8 bp/y |
| N trades | 2,370 (11.4yr) |
| trades/yr | 207 |

핵심 특징:
- **5 Strategy composite** (A' regime / B event / C seasonal / D basis / E cross-section)
- **TP=+5 / SL=−15 bp** (A', B, D)
- **Max 3 concurrent per (strategy, signal, sign)** — DCA 차단
- **Rolling 42d half-Kelly sizing** (walk-forward)
- **DV01 한도 5,000 만/bp (한 방향)**

---

## 1. Universe (8 Instruments)

### 1.1 현물 (KTB, 지표 6종)
- 2년지표 (anchor 2.0y)
- 3년지표 (anchor 3.0y)
- 5년지표 (anchor 5.0y)
- 10년지표 (anchor 10.0y)
- 20년지표 (anchor 20.0y)
- 30년지표 (anchor 30.0y)

### 1.2 선물 (KTBF, 2종)
- **KTB3F** (3년 선물, duration ≈ 2.77)
- **KTB10F** (10년 선물, duration ≈ 7.91)
- (KTB30F 는 2024+ 데이터만, 보조)

---

## 2. Derived Signals (Pair / Butterfly / Basis)

### 2.1 Slopes (bp)
```
slope_3_2  = (3y - 2y) × 100
slope_5_2  = (5y - 2y) × 100
slope_10_3 = (10y - 3y) × 100    ← 핵심
slope_10_5 = (10y - 5y) × 100
slope_30_10 = (30y - 10y) × 100   ← 핵심
slope_20_10 = (20y - 10y) × 100
... (총 11종)
```

### 2.2 Butterflies (mid-anchored)
```
bf_2_5_10  = (2·5y - 2y - 10y) × 100
bf_3_10_30 = (2·10y - 3y - 30y) × 100   ← 핵심
bf_5_10_30 = (2·10y - 5y - 30y) × 100   ← 핵심
bf_3_5_10  = (2·5y - 3y - 10y) × 100
```

### 2.3 Spot-Futures Basis
```
basis_3y_KTB3F  = (3y_spot - KTB3F_implied_yield) × 100
basis_10y_KTB10F = (10y_spot - KTB10F_implied_yield) × 100
```

---

## 3. Macro Regime (BOK Policy Rate-Based)

```
코로나-제로금리: policy ≤ 0.75%             (high vol)
인상기      : policy_6m_chg ≥ +0.40%       (high vol)
인하기      : policy_6m_chg ≤ −0.30%       (low vol)
고금리-동결  : policy ≥ 3.0% (정상시기)      (low vol)
저금리-mild : policy ≥ 1.5% (정상시기)      (low vol)
저금리-동결  : 나머지 (가장 평탄)             (low vol)
```

## 4. Strategy Detail

### 4.1 A' — Regime-Conditional Level Deviation

**Concept**: 각 slope/BF 의 현 regime 의 expanding mean/std 대비 z-score, 평균회귀 진입.

**Universe (slope_10_3 제외 — low vol 실패)**:
- bf_3_10_30, bf_5_10_30, bf_2_5_10, slope_20_10, slope_30_10

**Signal**:
```python
for each regime r:
    regime_mean = expanding(min=30).mean()
    regime_std  = expanding(min=30).std()
z_value = (current - regime_mean) / regime_std

Entry: |z| > 1.5
  z > 1.5  → spread 평소보다 cheap → mean-rev 베팅 sign = +1
  z < -1.5 → spread 평소보다 rich  → sign = -1
```

**Exit**:
- TP: +5 bp
- **SL: −15 bp**
- max hold: 10 영업일
- max concurrent (signal, sign): 3

**성과 (단독, SL=15 적용)**:
- ann_net +138 bp/y, Sharpe 1.57, win 59%

---

### 4.2 B — Event Pre-drift Mean-Rev

**Concept**: 국채 발행 / FOMC 직전 21d 동안의 slope/BF drift 가 발행 후 mean-rev 회복 (명제 P3/P3'/P4').

**Universe (event × signal cell)**:
- 발행_30y × bf_3_10_30
- 발행_30y × bf_5_10_30
- 발행_3y × bf_3_10_30
- 발행_3y × bf_5_10_30
- 발행_3y × slope_30_10
- 발행_5y × slope_30_10

**Signal**:
```python
pre_drift = signal_value[T-1] - signal_value[T-22]   # 21bd pre
if abs(pre_drift) < 1.0 bp: skip
sign = +1 if pre_drift > 0 else -1   # mean-rev direction
score = abs(pre_drift)
```

**Exit**:
- TP: +5 bp / SL: −15 bp / max hold: 42 영업일 / cap 3

**성과**: ann +74 bp/y, Sharpe 0.98, win 70%

---

### 4.3 C — June Belly Seasonal

**Concept**: 명제 P5 — 6월 belly outperform (bf_3_10_30 −0.49 bp/day, p=0.015).

**Signal**:
- 매년 5월 마지막 영업일 진입
- sign = +1 (bf narrowing 베팅)
- max hold: 21 영업일 (6월 중순까지)
- TP/SL 없음 (time exit only)

**성과**: ann +6.6 bp/y, win 64% (sparse, 연 1회)

---

### 4.4 D — Basis Carry Drift

**Concept**: 현물−선물 basis 의 policy_rate 회귀 fair value mean-rev (명제 P7).

**Universe**:
- basis_3y_KTB3F
- basis_10y_KTB10F

**Signal**:
```python
fair_basis = expanding_OLS(basis ~ policy_rate)
resid = current_basis - fair_basis
resid_std = rolling_63d(resid).std()
z = resid / resid_std

Entry: |z| > 1.5
```

**Exit**:
- TP: +5 bp / SL: −15 bp (실제 hit 0%) / max hold: 42 / cap 3

**성과**: ann +208 bp/y 단독, Sharpe 4.65, win 65%
- **SL=15 까지 hit 0%** — basis mean-rev 가 항상 15bp 안에서 회복

---

### 4.5 E — Cross-Section Top-1 Ranking

**Concept**: 매일 8 signal 의 rolling 252d z-score 중 |z| 가장 큰 1건 진입 (vol-agnostic).

**Universe**:
- slope_10_5, slope_30_10, slope_20_10
- bf_3_10_30, bf_5_10_30, bf_2_5_10
- basis_3y_KTB3F, basis_10y_KTB10F

**Signal**:
```python
for each col:
    z = (current - rolling_252d_mean) / rolling_252d_std (shift(1))
top1 = max |z| signal of that day
Entry: |top1_z| > 1.5
sign = +1 if z > 0 else -1
```

**Exit**: max hold 21bd, time-only

**성과**: ann +14 bp/y, win 53% (보조 layer)

---

### 4.6 F — Calendar Overlay (제거)

분기말 5d trade 검증 결과 ann −6.5 bp/y, win 47% → **CV1T 에서 제거**.

---

## 5. Portfolio Management (M4 + Kelly + Cap)

### 5.1 Walk-Forward Half-Kelly Sizing

매월 (21bd 마다):
```python
for each strategy s in last_42_bd_history:
    if len(trades) < 5: weight[s] = 1.0 (default)
    else:
        mu  = trades_pnl_net.mean()
        var = trades_pnl_net.var()
        if mu <= 0 or var < 1e-6: weight[s] = 0.2 (min clamp)
        else:
            kelly = 0.5 * mu / var   (half-Kelly)
            weight[s] = sqrt(max(0, kelly)) clamped to [0.2, 2.5]
```

### 5.2 DCA Cap (Max Concurrent)
```
max concurrent (strategy, signal, sign) = 3
```
같은 페어 같은 방향 4번째 진입 차단 → DCA 위험 차단.

### 5.3 DV01 Limit
```
max long  bpv:   5,000 만/bp (한 방향)
max short bpv:   5,000 만/bp
```

새 trade 진입 시 사이즈 추가 후 한도 초과 → 진입 거부.

### 5.4 Entry Priority
DV01 한도 안에서 score 높은 trade 부터 진입.

---

## 6. PnL Calculation

### 6.1 단일 Trade
```
pnl_gross_bp = sign × (entry_val - exit_val)
              (sign=+1: narrow 베팅, entry > exit 면 양수)
pnl_net_bp = size × (pnl_gross_bp - cost)
```

### 6.2 Cost (페어당, 진입+청산 합산)
```
slope (현물 2 leg): 0.25 × 2 × 2 = 1.0 bp
butterfly (현물 3 leg): 0.25 × 3 × 2 = 1.5 bp
basis_3y_KTB3F (현물 + KTB3F):  0.25 + 0.9 (2.5 tick) → ×2 = 2.3 bp
basis_10y_KTB10F (현물 + KTB10F): 0.25 + 0.32 → ×2 = 1.14 bp
```

### 6.3 절대 금액 환산 (100억 명목 기준)
```
현물 100억 1bp = DV01 × 100만원/100억 = duration × 100만원/bp
  → 5y 100억:  500만원/bp
  → 10y 100억: 1,000만원/bp
선물 100계약 1tick = 100만원
  → KTB3F 100계약: 280만원/bp
  → KTB10F 100계약: 791만원/bp
```

---

## 7. Performance (2015-2026, 11.4 yr)

### 7.1 종합

| 지표 | 값 |
|---|---|
| N trades | 2,370 |
| trades/year | 207 |
| trades/week | 4.0 |
| ann_net (gross 후 cost) | **+112.6 bp/y** |
| Sharpe (annualized) | **+2.15** |
| Win % (net) | **66.4%** |
| Mean PnL/trade | +0.54 bp |
| Max Drawdown | **−233 bp** |
| Yearly std | ~200 bp |

### 7.2 Vol Regime

| Regime | N | ann | win % |
|---|---|---|---|
| **Low vol** | 1,510 | **+91.3** | **65.4%** |
| **High vol** | 860 | **+176.8** | **67.8%** |

### 7.3 Per-Strategy 기여 (ann bp/y)

| Strategy | N | ann | win % | TP hit % | SL hit % |
|---|---|---|---|---|---|
| A' (Regime-Cond) | 1,157 | +37.7 | 62% | 46% | 2.4% |
| B (Event) | 781 | +16.4 | 71% | 67% | 9.7% |
| **D** (Basis Carry) | 311 | **+41.9** | **73%** | 47% | **0%** |
| E (Cross-section) | 113 | +11.4 | 58% | — | — |
| C (June Belly) | 8 | +5.1 | 63% | — | — |

### 7.4 Yearly

```
2015: -71    2018: +191    2021: +80
2016: -108   2019: +56     2022: +372 ⭐
2017: +64    2020: +55     2023: +190    2024: +209
                                          2025: +175 ✅
                                          2026: +66
```

→ 2017+ 매년 양수. warmup (2015~2016) 제외 시 ann 약 +130 bp/y.

### 7.5 Direction Accuracy
- Overall: **65.92%** (random 50% 대비 p ≈ 6.5e−80)
- Strategy 별: D 74% > A' 64% > B 60%
- Vol regime: high 69%, low 65%
- Score quintile: Q1 63% → Q5 70% (monotonic)

---

## 8. Risk Management Trade-offs

| Setting | ann_net | Sharpe | MDD | 비고 |
|---|---|---|---|---|
| **No SL, no cap** | +171 | +2.95 | −350 | alpha 최대 but risk |
| **No SL, cap=3** | +131 | +2.50 | −246 | DCA 차단만 |
| **SL=15, cap=3** ⭐ | **+113** | **+2.15** | **−233** | **운용 채택** |
| SL=10, cap=3 | +85 | +1.60 | ~−200 | 보수적 |

**최종 채택 (SL=15, cap=3)**:
- alpha 35% 손실 vs MDD 33% 개선 + DCA 위험 차단
- Tail risk per trade = max −15 × kelly_size

---

## 9. 명제 P1~P12 References

CV1T 는 [CV1 SLOPE 명제](https://www.notion.so/CV1-SLOPE-P1-P12-2026-05-13-360f823b5dd081c69852da3ad8c10dd9) 12개 중 채택된 8개 (P1, P2, P3, P3', P4', P5, P6, P7) 를 운용 룰로 반영.

기각:
- P11 (BOK MPC sub-tenor) — 직접 효과 약함
- P12 (10y 발행) — 다른 tenor 와 다른 패턴
- P9 (regime shift leading) — n=11 한계

조건부:
- P10 (regime conditional) — A' 의 expanding fair value 로 반영

---

## 10. 실행

### 10.1 일일 운용 Routine
```
매일 09:30:
  1. DB 데이터 갱신 (ktb, ktbf_basis, ktbf_rollover, market_event)
  2. CV1 panel 빌드 → cv1_slopes_butterflies.parquet
  3. 각 strategy entry signal 산출 (A', B, C, D, E)
  4. Kelly weight 계산 (rolling 42d, 매월 갱신)
  5. DV01 한도 + cap=3 체크
  6. 진입 (sign + size = kelly_weight × base notional)
  7. 활성 trade TP/SL 점검 (daily mark-to-market)
  8. 만기 (max_hold) 도달 시 청산
```

### 10.2 스크립트
```
panel build:     factor_trading/scripts/cv1_panel_research.py
event study:     factor_trading/scripts/cv1_event_study.py
regime analysis: factor_trading/scripts/cv1_regime_analysis.py
hypothesis test: factor_trading/scripts/cv1_hypothesis_tests.py
TP/SL grid:      factor_trading/scripts/cv1t_tpsl_grid.py
SL + cap:        factor_trading/scripts/cv1t_v2_sl_maxcap.py
direction acc:   factor_trading/scripts/cv1_direction_accuracy.py
forward signal:  factor_trading/scripts/cv1_signal_now_v2.py
```

### 10.3 산출 데이터
```
data/factor_trading/phase40_cv1/             — panel + 명제 검증
data/factor_trading/phase47_kelly_grid/      — Kelly grid
data/factor_trading/phase50_cv1t_tpsl/       — TP/SL grid
data/factor_trading/phase51_cv1t_v2_sl_cap/  — 최종 (SL=15, cap=3)
data/factor_trading/phase48_direction_acc/   — direction accuracy
data/factor_trading/phase49_signal_513/      — forward signal
```

---

## 11. Open Questions / 향후 작업

| 항목 | 우선 |
|---|---|
| Instrument-level aggregation (8 instrument 합산 portfolio) | 🔥 |
| 매월 Kelly rebalance — 더 빠른 적응 시도 | 중 |
| 2024 regime shift 관련 adaptive sizing | 중 |
| CV2 (일중) 시작 + 통합 | 🔥 |
| Forward OOS 검증 (2026 H2) | 🔥 |
| Live trade execution 자동화 | 낮 |

---

작성: 2026-05-13
관련 문서:
- `factor_trading/strategy_report_OF_CM_Combine.md` (RV/OF/CM/Combine)
- 노션: CV1 SLOPE 명제 P1~P12
- 노션: CV1 Curve Daily Research 1차 ~ 3차 분석
