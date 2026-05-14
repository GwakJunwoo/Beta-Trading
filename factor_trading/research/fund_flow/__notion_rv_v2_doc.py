"""Create Notion page for active RV V2 strategy in 문서 허브 database."""
import os, sys, urllib.request, json, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

TOK = os.environ.get("NOTION")
DB_ID = "360f823b-5dd0-80d8-a143-f42674ee8490"

assert TOK, "NOTION token missing"

def api(method, path, body=None):
    url = f"https://api.notion.com/v1{path}"
    headers = {
        "Authorization": f"Bearer {TOK}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode()[:1000])
        raise


def h1(text):
    return {"object":"block","type":"heading_1","heading_1":{"rich_text":[{"type":"text","text":{"content":text}}]}}
def h2(text):
    return {"object":"block","type":"heading_2","heading_2":{"rich_text":[{"type":"text","text":{"content":text}}]}}
def h3(text):
    return {"object":"block","type":"heading_3","heading_3":{"rich_text":[{"type":"text","text":{"content":text}}]}}
def para(text, code=False):
    return {"object":"block","type":"paragraph","paragraph":{"rich_text":[{"type":"text","text":{"content":text}, "annotations":{"code":code}}]}}
def bullet(text, code=False):
    return {"object":"block","type":"bulleted_list_item","bulleted_list_item":{"rich_text":[{"type":"text","text":{"content":text}, "annotations":{"code":code}}]}}
def code_block(text, lang="plain text"):
    return {"object":"block","type":"code","code":{"rich_text":[{"type":"text","text":{"content":text}}],"language":lang}}
def divider():
    return {"object":"block","type":"divider","divider":{}}
def callout(text, emoji="💡"):
    return {"object":"block","type":"callout","callout":{
        "rich_text":[{"type":"text","text":{"content":text}}],
        "icon":{"type":"emoji","emoji":emoji}
    }}


TAGS = [
    # 모델
    "모델:RV",
    # 신호
    "신호:level_mode", "신호:diff_mode", "신호:OOS-shift1",
    # 컨셉
    "컨셉:상대가치", "컨셉:평균회귀", "컨셉:페어트레이드", "컨셉:BPV-neutral",
    # 리스크
    "리스크:델타", "리스크:듀레이션", "리스크:커브", "리스크:DV01",
    "리스크:regime", "리스크:look-ahead",
    # 자산
    "자산:국고채", "자산:비지표", "자산:지표",
    # 만기
    "만기:2-10y",
    # 상태
    "상태:운용중", "상태:확정", "상태:검증완료",
    # 구간
    "구간:2024-regime-shift", "구간:2025", "구간:2026",
    # 인사이트
    "인사이트:level>diff", "인사이트:OOS-shift1", "인사이트:2024-적자",
    # 산출
    "산출:백테스트", "산출:파이프라인", "산출:일일신호", "산출:대시보드",
]
CATEGORIES = ["전략 문서"]


children = [
    callout(
        "Level mode 2-factor regression 기반 국고채 페어 RV 트레이딩. "
        "ε spread ≥ 5bp 진입 / TP+3 / SL-3 / Hold ≤90d. "
        "운영중 (server/app/routers/rv_position.py).",
        emoji="🟢",
    ),

    h1("1. 알파 가설 & 모델"),
    para(
        "국고채 종목들 사이 cross-sectional fair value 관계. "
        "2-factor 회귀 잔차 ε (fair value gap) 가 mean revert (half-life 7.7일) 한다는 통계 패턴 이용."
    ),
    h3("회귀식 (Level Mode)"),
    code_block(
        "Y_i(t) = α_i + β_i · Y_3Y(t) + γ_i · slope(t) + ε_i(t)\n"
        "  slope = Y_10Y − Y_3Y  (bp)\n\n"
        "β_i : 3Y 평행이동 노출 (보통 ≈ 1)\n"
        "γ_i : slope 노출 (잔존 3Y ≈ 0, 10Y ≈ 1)\n"
        "ε_i : fair value gap (bp)\n"
        "  ε > 0 → bond cheap (LONG 후보)\n"
        "  ε < 0 → bond rich (SHORT 후보)"
    ),
    h3("지표 종목 처리"),
    bullet("3년지표: β=1, γ=0, ε ≡ 0"),
    bullet("10년지표: β=1, γ=1, ε ≡ 0"),
    bullet("(회귀 X1 / X2 자체이므로 잔차 0 강제)"),

    h1("2. 진입 룰 (V2 final)"),
    bullet("ε spread (LONG − SHORT) ≥ +5.0 bp"),
    bullet("페어 만기차 ≤ 1.5 Y"),
    bullet("잔존 2 ~ 13 Y"),
    bullet("Issue age ≤ 5 년 (양 다리)"),
    bullet("Face 사이징: DV01 매칭 (face_L = face_S × D_S / D_L)"),

    h1("3. 청산 룰 (셋 중 하나)"),
    bullet("🎯 TARGET: P&L bp ≥ +3.0 (on avg DV01)"),
    bullet("🛑 STOP: P&L bp ≤ -3.0 (R/R 1:1 대칭)"),
    bullet("⏰ TIME: 보유 ≥ 90 영업일"),
    para(""),
    para("V2 핵심 개선: trigger 기준이 raw spread bp 가 아닌 P&L bp on avg DV01 (DV01 mismatch 반영)."),

    h1("4. 사이즈"),
    bullet("비지표 종목: 100억 단위"),
    bullet("지표 종목: 10억 단위"),
    bullet("SHORT base + LONG DV01 매칭 후 반올림"),

    h1("5. P&L 분해"),
    code_block(
        "P&L_total = − D_i · N_i · ΔY_i  +  D_j · N_j · ΔY_j\n"
        "         = Delta_P&L   (β 곱 × ΔY_3Y)\n"
        "         + Curve_P&L   (γ 곱 × Δslope)\n"
        "         + Alpha_P&L   (Δε 잔차)\n\n"
        "Delta_DV01  = D_j·N_j·β_j − D_i·N_i·β_i   (만/bp_3Y)\n"
        "Curve_DV01  = D_j·N_j·γ_j − D_i·N_i·γ_i   (만/bp_slope)\n"
        "  양수 = steepener bet, 음수 = flattener bet"
    ),

    h1("6. 백테스트 성과 (V2 final)"),
    para("기간: 2023-08-22 ~ 2026-05-08 (≈ 2.7년, 659 영업일)"),
    para("엔진: pair_backtest_level_v2.py (look-ahead 제거, dynamic indicator, P&L bp trigger)"),
    h3("Final Rule (entry=5, target+3, stop-3, hold≤90d, issue≤5y)"),
    bullet("N (closed trades) : 51 (≈ 20/year)"),
    bullet("Total : +16,523 만"),
    bullet("Per_yr : +7,167 만/y (100억 base)"),
    bullet("Sharpe : +0.17"),
    bullet("Win rate : 51.0%"),
    bullet("Mean hold : 17.5 일"),
    h3("ε Mean Reversion 통계"),
    bullet("AR(1) β (평균) : 0.90"),
    bullet("Half-life : 7.7 일"),
    bullet("정상성 (β<1) : 100% (모든 종목 정상)"),
    bullet("Entry ≥ 5bp 일 때 30일 내 narrowing 100%"),
    h3("연도별"),
    bullet("2024 (4개월) : -4,600 만 (5 trades, win 40%)"),
    bullet("2025 : +15,936 만 (29 trades, win 48%)"),
    bullet("2026 (4월까지) : +5,186 만 (17 trades, win 59%)"),

    h1("7. 시스템 위치"),
    h3("운영 (fullstackjunior)"),
    bullet("server/app/routers/rv_position.py — 4 positions 실시간 ε/P&L API (/rv/positions)", code=True),
    bullet("server/app/routers/beta.py — β/γ rolling 회귀 함수", code=True),
    bullet("tools/rv-position/index.html — 대시보드 (P&L bp + Action 컬럼)", code=True),
    h3("백테스트 / 자동화 (Beta Trading/factor_trading/)"),
    bullet("scripts/pair_backtest_level_v2.py — V2 백테스트 엔진", code=True),
    bullet("scripts/pair_backtest_v2_charts.py — 차트 + 표", code=True),
    bullet("scripts/eps_mean_reversion.py — ε 수렴성 통계", code=True),
    bullet("scripts/daily_pair_signal.py — 매일 close 후 텔레그램 시그널", code=True),

    h1("8. 운영 의사결정 룰"),
    h3("진입 강도"),
    bullet("ε spread ≥ 5bp : 진입 (백테스트 영역)"),
    bullet("ε spread 3 ~ 5bp : 보류 (약 신호, V2 손실 영역)"),
    bullet("ε spread < 3bp : 진입 금지 (noise)"),
    h3("청산"),
    bullet("P&L bp ≥ +3 : 🎯 TARGET CLOSE"),
    bullet("P&L bp ≤ -3 : 🛑 STOP LOSS"),
    bullet("보유 ≥ 90일 : ⏰ TIME EXIT"),
    bullet("ε spread < 1.5bp & P&L 음수 : EARLY CLOSE 검토"),
    bullet("|delta + curve| > |alpha| : DV01·γ 헤지 추가 검토"),

    h1("9. 한계 & 리스크"),
    bullet("⚠️ Sharpe 0.17 — RV 단독 알파 한계 (분산 운용 시 portfolio sharpe 0.3~0.5 가능)"),
    bullet("⚠️ N 20/year — capital 비효율 (대부분 시간 idle)"),
    bullet("⚠️ 2024+ 환경에서 RV 약화 추세 (sample 작아 단정 어려움)"),
    bullet("⚠️ 차등 cost / funding cost 미반영 — 실제 운용 시 backtest 보다 1.5~2배 cost 가능"),
    bullet("⚠️ 2025 -14,748 만 큰 drawdown 발생 (V1 부록 B 기준)"),
    bullet("⚠️ 사용자 실 운용 약신호 진입 패턴 — V2 영역 (≥5bp) 만 사용 권장"),

    h1("10. V2 개선 사항 (V1 대비)"),
    bullet("β/γ 시점: t (look-ahead) → t-1 lagged (실시간 정보만)"),
    bullet("Indicator 처리: static (ever-indicator) → dynamic (per-date)"),
    bullet("Trigger 기준: raw spread bp → P&L bp on avg DV01"),
    bullet("Time stop 버그 fix"),

    h1("11. 관련 문서 (Notion)"),
    bullet("RV_MODEL_REPORT.md (fullstackjunior 폴더)"),
    bullet("RV_POSITION_ANALYSIS.md (4 positions 분석 + 부록 A/B/C 백테스트)"),
    bullet("FUND_FLOW_RESEARCH_2026_05_12.md (V7-clean curve pair, RV 와 결합 후보)"),

    divider(),
    callout(
        "Last updated: 2026-05-13\n"
        "Engine: server/app/routers/rv_position.py (level mode) + beta.py\n"
        "Backtest: factor_trading/scripts/pair_backtest_level_v2.py",
        emoji="📅",
    ),
]


# 페이지 생성
body = {
    "parent": {"database_id": DB_ID},
    "icon": {"type":"emoji","emoji":"📐"},
    "properties": {
        "문서 이름": {"title":[{"type":"text","text":{"content":"RV — Relative Value Pair Trading (V2 final, Level mode)"}}]},
        "태그": {"multi_select":[{"name":t} for t in TAGS]},
        "카테고리": {"multi_select":[{"name":c} for c in CATEGORIES]},
    },
    "children": children,
}

print("[create] page in 문서 허브 DB ...")
res = api("POST", "/pages", body)
print(f"  OK  id={res['id']}")
print(f"  url={res.get('url','')}")
print()
print("Tags applied:")
for t in TAGS:
    print(f"  - {t}")
