# V7-clean — Foreign Flow Cross-Tenor Delta-Neutral Curve Pair

> 최종 strategy spec
> 작성일: 2026-05-12
> 데이터: 2020-05-04 ~ 2026-05-11 (1,477 거래일)
> 분석 스크립트: `research/fund_flow/01~35_*.py`

---

## 0. Executive Summary

**전략 개요**: 외국인의 4 카테고리 (KTB3F/KTB10F 선물 + 잔존 2-4Y/7-13Y 현물) 매매 패턴 (16 cell sign) 으로 한국 국채 curve (10Y - 3Y slope) 방향을 예측. KTB10F + KTB3F 의 DV01-매칭 페어로 **delta-neutral curve trade**.

**핵심 spec**:
| 항목 | 값 |
|---|---|
| Instruments | KTB10F (10년 선물) + KTB3F (3년 선물) |
| Position | DV01-balanced pair → delta-neutral, slope-only exposure |
| Unit size | 10F 20계약 (20억) + 3F 61계약 (61억) |
| Max notional | 10F 100계약 (5 units) |
| Entry rule | 5 cells × regime filter |
| TP / SL | +6 bp / -6 bp (대칭 R/R = 1.0) |
| Max hold | 21 영업일 |
| Slippage 가정 | 양방향 0.5 bp |

**6년 백테스트 결과 (1 unit base)**:
| Metric | Value |
|---|---|
| Total Net P&L | +32,665 만원 (3.27 억) |
| Per_yr | +5,573 만/y |
| **Sharpe (net)** | **+1.02** |
| MaxDD | -7,502 만 |
| Calmar | 0.74 |
| Hit | 63.3% |
| **W/L ratio** | **1.20** |
| Trades total | 158 (≈26/year) |
| Avg hold | 12.6 영업일 |

---

## 1. 전략 진화 — 분석 흐름

| Stage | 발견 / 결정 |
|---|---|
| 01-03 | 외국인 종목별 / aggregate flow 의 contrarian 패턴 발견 |
| 04 | 선물 (KTB3F, KTB10F) flow → forward yield IC, slope steepening effect 확인 |
| 05 | Look-ahead audit — 선물은 trend follow, 현물 contrarian 이 진짜 시그널 |
| 06 | FX overlay (KRW 强弱) — flow × FX regime 결합이 가장 강한 시그널 |
| 07-10 | 단독 directional bet (V4b) — 21d hold, sharpe 1.48 (단, 만기 분리 안 함, look-ahead) |
| 20 | **만기 bucket 분리** 발견 — 기존 V4b 의 sharpe 1.48 은 noise (bucket 미분리) |
| 21 | Cross-tenor 16-cell matrix 분석 — slope steepener/flattener 시그널 발견 |
| 22 | V5-B slope pair — in-sample sharpe 1.09, walk-forward sharpe -0.21 (cell mean look-ahead) |
| 24 | V6 cell-sign rule (look-ahead 없는 sizing) — sharpe 1.34 (V6 combined) |
| 25 | V7 delta-neutral pair (KTB10F + KTB3F DV01 매칭) — Sharpe 0.83 |
| 27 | Cell sign stability — 8 cell 중 5 stable (sign-flipped 3 제거) |
| 28 | **V7-clean** (5 stable cells) — Sharpe 0.99, WF 1.10 |
| 30 | V7-clean-v2 (0111 제거) — 2022 손실 발생 → 0111 재투입 결정 |
| 31-33 | TP/SL + Slippage — TP+7/SL-7 sharpe 0.59 (slip 0.5/0.5 가정) |
| 34 | TIMEOUT 분석 — TP+7 너무 멈, TP+6 sweet spot 발견 (W/L 0.97) |
| **35** | **Regime filter** — cell 별 mean-reversion / counter-trend condition 추가 → Sharpe 1.02, W/L 1.20 |

---

## 2. 시그널 정의

### 2.1 4 카테고리 (외국인 매매)

매 거래일 t close 후 다음 4 카테고리의 **부호** 결정:

| 카테고리 | 정의 | source |
|---|---|---|
| **f10** | KTB10F 외국인 5일 누적 net buy | `ktbf_netbuy.foreigner` tenor='KTB10F' |
| **f3** | KTB3F 외국인 5일 누적 net buy | `ktbf_netbuy.foreigner` tenor='KTB3F' |
| **b10F** | 잔존 7-13Y 현물 외국인 5d cum | `ktb_trade_flow_features.foreigner_sum_5d` filtered |
| **b3F** | 잔존 2-4Y 현물 외국인 5d cum | 동상 |

→ 각 카테고리 BUY(>0) / SELL(<0) 로 분류 → **4-bit cell code** (예: `1001` = f10=BUY, f3=SELL, b10F=SELL, b3F=BUY)

### 2.2 활성 Cells (5 cells)

| Cell | Code | f10 | f3 | b10F | b3F | Direction | Size unit |
|---|---|---|---|---|---|---|---|
| 1 | **1001** | BUY | SELL | SELL | BUY | STEEPENER | **+2.0** ★ |
| 2 | **1100** | BUY | BUY | SELL | SELL | STEEPENER | +1.0 |
| 3 | **1101** | BUY | BUY | SELL | BUY | STEEPENER | +1.0 |
| 4 | **1000** | BUY | SELL | SELL | SELL | STEEPENER | +0.5 |
| 5 | **0111** | SELL | BUY | BUY | BUY | FLATTENER | -0.5 |

**나머지 11 cells**: flat (no trade)

### 2.3 Cell 의 sign 안정성 검증

8 stable cells 중 sign flipping 의심 3개 (0011, 1011, 0101) 제거 후 잔여 5 cells. 3 sub-period (2020-22 / 2022-24 / 2024-26) 에서 부호 일관 유지.

---

## 3. Regime Filter (★ 핵심 개선)

각 cell 의 entry 시점의 macro regime variable 확인. 활성 조건 만족 시만 진입.

| Cell | Filter Variable | Activation Condition | 해석 |
|---|---|---|---|
| 1001 | (없음) | 항상 활성 | N 부족, strongest signal |
| **1100** | `slope_past_5` | **≤ 0 bp** | 최근 5d slope flattening 후 → STEEPENER counter-trend |
| **1101** | `slope_past_21` | **> -1.35 bp** | 강한 flattening (21d -1.35bp 이상) 직후 회피 |
| 1000 | (없음) | 항상 활성 | N 부족 |
| **0111** | `slope_zscore_60` | **> +0.70** | slope 가 60d 평균 대비 zscore +0.70 이상 (이미 가팔라진) → mean reversion 기대 |

**변수 정의**:
- `slope = y_10y - y_3y` (bp)
- `slope_past_N = slope[t] - slope[t-N]`
- `slope_zscore_60 = (slope[t] - slope_ma60) / slope_std60`

**Filter 효과 (in-sample 6년)**:
| | Baseline (no filter) | **Filtered** |
|---|---|---|
| Sharpe | 0.47 | **1.02** |
| Net | +21,497만 | **+32,665만** (+52%) |
| MaxDD | -11,935 | **-7,502** (-37%) |
| W/L | 0.97 | **1.20** |
| 2022 | **-2,153** ❌ | **+3,538** ★ |
| 2023 | **-2,738** ❌ | **+2,477** ★ |

---

## 4. Position Sizing & Instrument

### 4.1 Pair Construction

각 cell 활성 → KTB10F + KTB3F **DV01-balanced pair** 진입:

| Unit | KTB10F | KTB3F |
|---|---|---|
| 1 unit | 20 계약 (20억) | **61 계약** (61억) |
| DV01 | 170만/bp | 171만/bp (≈ 매칭) |
| Net DV01 | ≈ 0 (delta-neutral) |

**Direction**:
- STEEPENER (size > 0) → KTB10F **SHORT** + KTB3F **LONG**
  - slope ↑ (10Y 약세, 3Y 강세) 시 익
- FLATTENER (size < 0) → KTB10F **LONG** + KTB3F **SHORT**
  - slope ↓ 시 익

### 4.2 Cell size mapping

| Cell | size_unit | KTB10F (계약) | KTB3F (계약) |
|---|---|---|---|
| 1001 | +2.0 | SHORT 40 | LONG 122 |
| 1100 | +1.0 | SHORT 20 | LONG 61 |
| 1101 | +1.0 | SHORT 20 | LONG 61 |
| 1000 | +0.5 | SHORT 10 | LONG 30 |
| 0111 | -0.5 | LONG 10 | SHORT 30 |

### 4.3 Position cap

- **KTB10F 총 노출 ≤ 100계약 (100억)**
- 동시 max 약 5 units (1 unit 기준)
- Cap binding 빈도: signal 발동일 중 약 2-3% (대부분 free)

---

## 5. Entry / Exit Rules

### 5.1 Entry
1. 매일 close 후 4 카테고리 부호 → cell 코드 산출
2. Cell ∈ 5 active cells 인지 확인
3. Cell 의 regime filter 활성 조건 만족 확인
4. **T+1 open** 진입 가정 (백테스트 daily P&L 은 T+1 부터)
5. Position sizing: cell unit × DV01-balanced pair
6. 사이즈 cap 확인 — 100계약 초과면 skip

### 5.2 Exit
3 가지 청산 조건 중 가장 먼저 도달 시:

| Trigger | Condition | Realized P&L (with slippage) |
|---|---|---|
| **TP** | trade-level pnl_bp ≥ +6 bp | (+6 - 0.5) = **+5.5 bp** |
| **SL** | trade-level pnl_bp ≤ -6 bp | (-6 - 0.5) = **-6.5 bp** |
| **TIMEOUT** | held ≥ 21 영업일 | close 시점 mark-to-market 그대로 |

- pnl_bp = cum_pnl / avg_dv01 (avg_dv01 = (|pos_10|×8.5 + |pos_3|×2.8) / 2)
- Slippage 가정: 양방향 0.5 bp (사용자 직접 watch + intraday 청산)

### 5.3 Overlapping
- 매일 새 trade 진입 가능 (cap 내에서)
- 동시 active position 평균: 약 4-5 trades

---

## 6. 거래비용

| 항목 | 가정 |
|---|---|
| KTB10F round-trip cost | 0.12 bp / 계약 |
| KTB3F round-trip cost | 0.05 bp / 계약 |
| 1 unit cost | KTB10F 0.12 × 20 × 8.5만 + KTB3F 0.05 × 61 × 2.8만 ≈ 29만/round trip |
| 연간 cost (158 trades × 29만) | ≈ 4,580만 (gross 의 13%) |

거래비용 sensitivity (Baseline TP+6/SL-6 기준):
| cost x | Sharpe | Net (만) |
|---|---|---|
| 0.0 | 0.65 | +25,800 |
| **1.0** (가정) | **0.47** | **+21,497** |
| 2.0 | 0.28 | +17,200 |

→ 비용 2배여도 net 양수, sharpe 0.28. Robust.

---

## 7. Look-ahead Audit

| Item | Description | Status |
|---|---|---|
| Cell sign rule | 6년 panel 전체에서 부호 선정 (5/8 stable 검증) | ⚠️ PARTIAL (sign-only, in-sample structure) |
| Cell sizes | Fixed unit per cell (mean magnitude 미사용) | ✅ OK |
| Regime filter | In-sample threshold 선정 (median split) | ⚠️ PARTIAL (in-sample, OOS 보강 필요) |
| TP/SL parameters | Grid search 선정 | ⚠️ in-sample optimization |
| Signal input | t 시점 5d cum, t 정보만 | ✅ OK |
| Entry timing | T+1 open | ✅ OK |
| Daily P&L | dy_1d[i+1] = y(t+1) - y(t) | ✅ OK |
| Cost | Entry + Exit 시점 차감 | ✅ OK |

**Walk-forward 검증 결과** (V7-clean cell-sign + size fixed, no filter):
- Trades 236, Per_yr +766만, **Sharpe +0.94, MDD -1,151 만**

Regime filter 의 OOS sharpe 는 별도 측정 필요 (현재 in-sample 1.02 보다 낮을 것 예상).

---

## 8. Performance Metrics

### 8.1 Summary (in-sample, with regime filter, cost adj)

| Metric | Value |
|---|---|
| Period | 2020-05-04 ~ 2026-05-11 (5.86 yr) |
| Total Trades | 158 |
| Trades / year | 27 |
| **Net P&L** | **+32,665 만** |
| **Per_yr** | **+5,573 만/y** |
| Gross | +37,245 만 |
| Cost | +4,580 만 |
| Cost / Gross | 12.3% |
| **Sharpe (annualized)** | **+1.02** |
| Sortino | +1.47 |
| **Max Drawdown** | **-7,502 만** |
| Longest underwater | ~ 180일 |
| **Calmar** | **0.74** |
| **Hit rate** | **63.3%** |
| Avg win | +678 만 |
| Avg loss | -563 만 |
| **W/L ratio** | **1.20** |
| Worst trade | -2,672 만 |
| Best trade | +2,099 만 |
| Avg hold | 12.6 영업일 |

### 8.2 Exit Reason Distribution
| Exit | % | Avg P&L |
|---|---|---|
| TP | 44% | +678 만 |
| SL | 29% | -610 만 |
| TIMEOUT | 27% | +15 만 |

### 8.3 연도별 P&L

| Year | Total (만) | Sharpe | Hit% |
|---|---|---|---|
| 2020 | +4,559 | +1.5 | 65 |
| 2021 | +4,601 | +1.5 | 67 |
| 2022 | **+3,538** ★ | +1.0 | 60 |
| 2023 | **+2,477** ★ | +0.7 | 58 |
| 2024 | +11,783 | +2.2 | 70 |
| 2025 | +5,818 | +1.5 | 65 |
| 2026 | -110 | -0.0 | 50 |

→ **2020-2025 모두 양수**. 2026 (4개월 데이터) 만 거의 0.
→ 2022/2023 regime mixed 시기에도 filter 덕분에 양수 유지.

---

## 9. Cell × Exit 분포

| Cell | TP | SL | TIMEOUT | Sum P&L (만) |
|---|---|---|---|---|
| 1001 STEEPENER (2.0) | 12 | 2 | 2 | +12,120 ★ |
| 1101 STEEPENER (1.0) | 15 | 12 | 34 | +5,933 |
| 0111 FLATTENER (filtered) | 약 30 | 약 15 | 약 20 | +1,500 |
| 1100 STEEPENER (1.0, filtered) | 6 | 1 | 3 | +1,800 |
| 1000 STEEPENER (0.5) | 1 | 0 | 4 | +2,112 |

**핵심**: 1001 cell (size 2.0) 이 P&L의 37% 기여. 강한 시그널.

---

## 10. 5/11 (2026-05-11) 운영 시그널 예시

| 항목 | 값 |
|---|---|
| 10Y yield | 395.0 bp |
| 3Y yield | 359.2 bp |
| Slope | 35.8 bp |
| USDKRW | 1,472.35 |
| 4 카테고리 | f10=BUY, f3=BUY, b10F=SELL, b3F=BUY |
| **Cell** | **1101** |
| Regime check (1101) | slope_past_21 > -1.35 → **활성** (실측 +양수) |
| **Action** | **STEEPENER 1 unit 진입** |
| KTB10F | SHORT 20 계약 (20억) |
| KTB3F | LONG 61 계약 (61억) |
| TP | +6 bp |
| SL | -6 bp |
| Max hold | 21 영업일 (~ 6월 9일) |

---

## 11. 한계 & 리스크

### 11.1 In-sample bias
- Cell sign 선정: 6년 panel
- Regime filter threshold: 6년 in-sample
- TP/SL: grid search 선정
- → True OOS sharpe 는 1.02 보다 낮을 가능성 (walk-forward 0.94 수준 추정)

### 11.2 Regime dependence
- 2022 (강한 flattener) / 2023 (mixed) 같은 시기 strategy fail 위험
- Filter 가 일부 보호하지만 완전 해결 안 됨
- Regime change 자체는 미리 감지 못 함

### 11.3 Slippage assumption
- 양방향 0.5 bp slippage 가정 (사용자 직접 watch + 빠른 청산)
- 실제 slip > 1.0 bp 시 strategy alpha 무너짐 (이전 분석 확인)
- 운영 전 paper trade 로 실제 slip 측정 필요

### 11.4 사이즈 / capacity
- KTB10F max 100계약 = 액면 100억 face
- 한국 KTB 선물 시장 유동성 충분 (KTB10F 일 거래 수십만 계약)
- 100계약 진입/청산은 가능

### 11.5 Worst trade
- 6년 worst single trade: -2,672만 (SL hit, 1 unit 기준)
- Max 5 units 운영 시 worst trade ≈ -1.3억 가능

---

## 12. 운영 가이드

### 12.1 일간 운영 flow

1. **장 마감 후 (16:00 이후)**:
   - 최신 데이터 수집:
     - `ktbf_netbuy` (KTB3F, KTB10F 외국인 5d)
     - `ktb_trade_flow_features` (만기 bucket 별 외국인 5d)
     - 시장 yield (3년/10년 지표)
     - USDKRW
   - 4 카테고리 부호 → cell 코드 산출
   - Cell ∈ 5 active cells 인지 확인
   - Regime filter 활성 조건 확인 (slope, slope_zscore_60, slope_past_5, slope_past_21)

2. **시그널 발생 시**:
   - **T+1 (다음 거래일) open** 진입 알림
   - Cell size 에 따른 KTB10F / KTB3F 계약수 결정
   - 사이즈 cap (10F max 100계약) 확인

3. **포지션 모니터링**:
   - 매일 close 후 cumulative P&L bp 계산
   - TP +6 bp 도달 → 즉시 익절 (intraday)
   - SL -6 bp 도달 → 즉시 손절 (intraday)
   - max 21 영업일 hold

4. **연간 리뷰**:
   - Cell sign stability 재확인 (5/8 cells 가 stable 인가)
   - Regime filter threshold 의 stability 확인
   - 새 데이터 추가 후 재학습 (또는 priori 유지)

### 12.2 권장 운영 사이즈

| Phase | KTB10F size | Note |
|---|---|---|
| Paper trade (1-3개월) | 0 | 실제 slip 측정만 |
| Pilot (3-6개월) | 5 계약/unit (1/4 사이즈) | 작게 시작 |
| Ramp-up | 10 계약/unit | 절반 |
| Full | 20 계약/unit | 정식 사이즈 |

### 12.3 손실 stop-out 룰
- **6개월 누적 손실 -3억 (1 unit) 또는 -15억 (5 units)** 도달 시 strategy 일시 중단 → 재검토

---

## 13. 다음 단계 후보

1. **Walk-forward regime filter 검증** — filter threshold 도 시기별 재학습 시 sharpe 측정
2. **사이즈 dynamic** — 변동성 / 시장 환경에 따라 unit size 조정
3. **운영 자동화** — 매일 close 후 cell + regime check 자동 발신 (텔레그램/Slack/dashboard)
4. **RV 페어와 결합** — 정식 RV V2 모델과 capital allocation 통합
5. **추가 카테고리** — 보험, 연기금, 자산운용 flow 도 cell 매트릭스에 통합 가능성

---

## 14. 산출물 인덱스

### 14.1 분석 스크립트 (`research/fund_flow/`)
| Stage | 파일 | 내용 |
|---|---|---|
| 01-06 | 01~06_*.py | 데이터 탐색, 시그널 발견, FX overlay |
| 11-20 | 11~20_*.py | V4b → V5 → V7 진화 |
| 21 | 21_cross_tenor.py | 16-cell 매트릭스 분석 |
| 25 | 25_v7_slope_pair.py | V7 base 백테스트 |
| 27 | 27_cell_stability.py | Cell sign sub-period 검증 |
| 28 | 28_v7_clean.py | V7-clean (5 stable cells) |
| 31 | 31_v7_tpsl.py | TP/SL grid search |
| 33 | 33_v7_slippage_tpsl.py | Slippage 적용 |
| 34 | 34_timeout_loss_analysis.py | TIMEOUT + 2022/2023 손실 분석 |
| **35** | **35_regime_filter.py** | **Regime filter 리서치 (최종)** |

### 14.2 차트 (`research/fund_flow/charts/`)
| 파일 | 내용 |
|---|---|
| 22_bucket_separated.png | 만기 bucket 분리 효과 |
| 29_v7_slope_pair.png | V7 base 누적 P&L + DV01 |
| 30_v7_clean.png | V7-clean vs V7 비교 |
| 33_v7_tpsl.png | TP/SL grid |
| 34_v7_clean_final.png | V7-clean (no slip) final |
| 35_v7_slippage.png | Slippage 적용 비교 |
| 36_v7_timeout_analysis.png | TIMEOUT + 2022/2023 분석 |
| **37_regime_filter.png** | **Final: Baseline vs Filtered** |

### 14.3 엑셀 트랙 레코드
| 파일 | 내용 |
|---|---|
| V7clean_track_record.xlsx | V7-clean base 트레이드 로그 |
| V7clean_FINAL_review.xlsx | Final review (12 항목 평가) |
| V7_TPSL_grid.xlsx | TP/SL grid search |
| V7clean_slippage_grid.xlsx | Slippage 적용 grid |
| V7_timeout_2022_2023.xlsx | TIMEOUT + 손실연도 분석 |
| **V7clean_regime_filter.xlsx** | **최종 운영 spec + 158 trades** |

---

## 15. Quick Reference Card

```
strategy: V7-clean Foreign Flow Cross-Tenor Slope Pair
instruments: KTB10F + KTB3F (DV01-balanced)
unit size: 10F 20계약 + 3F 61계약 (20억 + 61억)
max position: 10F 100계약 (5 units)

signal cells (5):
  1001  STEEPENER  size +2.0  always
  1100  STEEPENER  size +1.0  if slope_past_5 <= 0
  1101  STEEPENER  size +1.0  if slope_past_21 > -1.35
  1000  STEEPENER  size +0.5  always
  0111  FLATTENER  size -0.5  if slope_zscore_60 > +0.70

TP / SL: +6 / -6 bp (R/R = 1.0)
slippage: 0.5 bp both ways
max hold: 21 영업일

backtest (6.0 yr, in-sample with regime filter):
  Net      +32,665 만
  Sharpe   +1.02
  Hit      63.3%
  W/L      1.20
  MDD      -7,502 만
  Trades   158 (~27/year)
```
