# 채권 3팩터 RV 트레이딩 모니터

한국 국채(KTB) 상대가치 전략을 3개의 직교 팩터(**RV · MOM · CURVE**)로 분해해
실시간 signal, 포트폴리오 비중, 종목별 금리변동 분해를 제공하는 Streamlit 대시보드.

상세 명세: [factor_trading/bond_3factor_final_spec.md](factor_trading/bond_3factor_final_spec.md)

---

## 🌐 실행 모드

이 저장소는 **정적 snapshot (`data_cache/*.parquet`)** 을 기반으로 동작한다.
원시 데이터는 사내 DB에서 추출되며, 이 저장소에는 가공된 parquet 만 포함.

```bash
# Cloud 모드 (parquet 기반, DB 접속 불필요)
export USE_CACHED_DATA=1
streamlit run factor_trading/app.py
```

---

## 📂 저장소 구조

```
factor_trading/
├── app.py                        ★ Streamlit GUI entry
├── bond_3factor_final_spec.md    명세 확정본
├── data_loader_cached.py         parquet → DataLoader 인터페이스
├── main.py                       FactorPipeline orchestrator
├── monitor.py                    DailyMonitor (일일 신호 생성)
├── beta_estimator.py             2-factor RollingOLS
├── residual_builder.py           ε 계산 + horizon 누적
├── factors/
│   ├── rv_factor.py              RV (잔차 평균회귀, within-bucket Q5/Q1)
│   ├── mom_factor.py             MOM (raw_sign cum=63) + split_signals
│   └── curve_factor.py           CURVE (raw_sign slope=21 mean_rev)
└── portfolio/
    ├── single_factor.py          within-bucket quintile, 각 팩터 LS PnL
    ├── combiner.py               EW / RiskParity / Satellite overlay
    ├── dynamic_combiner.py       rank-based 동적 비중
    └── duration_neutral.py       BPV 비중 (CURVE 페어 실행용)

data_cache/                       ★ 정적 snapshot
├── dy_3y.parquet                 3Y 지표 dY
├── dy_10y.parquet                10Y 지표 dY
├── dy_panel.parquet              종목별 일간 dY
├── ytm_panel_bp.parquet          종목별 YTM
├── remain_panel.parquet          종목별 잔존만기
├── rollover_flag.parquet         지표 롤오버 플래그
├── instrument_meta.parquet       종목 메타 (이름, 카테고리)
└── meta.json                     as_of_date, 최신 benchmark code
```

---

## 🛰️ Streamlit GUI 탭

| 탭 | 내용 |
|---|---|
| 🎯 오늘 스냅샷 | MOM/CURVE signal, RV LONG/SHORT 후보, regime 경고 |
| 🔍 팩터별 상세 | MOM/CURVE/RV 시계열 · 누적 PnL · 포지션 음영 |
| 🧮 3팩터 분해표 | `dY_i = β_3·dY_3 + β_10·dY_10 + ε` 종목별 분해 |
| 📈 성과 & 합성 | 3팩터 + EW/RP 합성 누적 PnL, drawdown, Sharpe |
| ✅ 모델 검증 | 직교성 상관행렬, R² 분포, ε×signal 직교성 |
| 🔀 동적 비중 | rank-based 월 리밸런싱 (1w/2w/1m Sh 가중) |
| 🛰️ MOM_contra Satellite | event-driven 극단 반전 overlay |

---

## 🚀 Streamlit Cloud 배포

1. 이 저장소를 GitHub (private 권장) 에 push
2. [share.streamlit.io](https://share.streamlit.io) 에서 GitHub 연동
3. **New app** → repo 선택 → Main file path: `factor_trading/app.py`
4. **Advanced settings → Secrets** 에 환경변수:
   ```toml
   USE_CACHED_DATA = "1"
   STREAMLIT_APP_PASSWORD = "your_strong_password"
   ```
5. **Deploy** → `https://<app-name>.streamlit.app` 자동 발급

### 데이터 갱신
사내 작업자가 정해진 주기로:
```bash
python factor_trading/scripts/export_for_cloud.py   # DB → parquet
git add data_cache/
git commit -m "snapshot YYYY-MM-DD"
git push
```
→ Streamlit Cloud 자동 redeploy, 앱 최신 데이터 반영.

---

## 🔐 인증

`STREAMLIT_APP_PASSWORD` 환경변수 설정 시 로그인 게이트 자동 활성.
외부 공개 배포 시 필수.

---

## 📊 3팩터 확정 파라미터

| 팩터 | 정의 | IS Sharpe | OOS |
|---|---|---|---|
| **RV** | Q5 LONG / Q1 SHORT within 만기 3버킷 (horizon=1m, hold=21d) | +1.16 | — |
| **MOM** | `−sign(cum_dY_3Y_63d)` 3Y 현물 bang-bang | +0.87 | +0.09 ⚠ |
| **CURVE** | `−sign(cum_slope_21d)` mean-rev · 3Y/10Y duration-neutral 페어 | +1.49 | +1.18 |

3팩터 직교성: |corr|max = 0.256 (명세 기준 0.3 이하 통과)
모델 설명력: 2팩터 회귀 R² median = 0.87

자세한 실증 결과·regime 분석·OOS drawdown 해부는 명세서 §4~§8 참조.

---

## ⚖️ 면책

본 저장소는 investigate·research 용도. 실거래 주문 연결·자동화는 별도 OMS 통합 필요.
표시되는 신호·성과는 과거 데이터 기반이며 미래 수익을 보장하지 않는다.
