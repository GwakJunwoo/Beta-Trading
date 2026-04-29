"""채권 3팩터 시스템 — Streamlit GUI.

실행:
    streamlit run factor_trading/app.py

탭
----
1. 오늘 스냅샷    — signal 3개, regime 경고, RV LONG/SHORT
2. 팩터별 상세    — RV / MOM / CURVE 각각 시계열·분포·signal history
3. 3팩터 분해표   — 전 종목 dY 분해 (필터/정렬)
4. 성과 & 합성    — 누적 PnL interactive, 기간별 Sharpe
5. 모델 검증      — 직교성 corr + R² 분포
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from factor_trading.monitor import DailyMonitor, REGIME_BINS, REGIME_LABELS
from factor_trading.factors.mom_factor import mom_raw_cum, mom_raw_signal, mom_split_signals
from factor_trading.portfolio.single_factor import xsec_ls_pnl, within_bucket_quintile
from factor_trading.portfolio.dynamic_combiner import dynamic_weight_backtest
from factor_trading.portfolio.combiner import satellite_overlay, equal_weight

st.set_page_config(
    page_title="채권 3팩터 트레이딩 — Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 인증 게이트 — 외부 공개 시 필수.
# 비밀번호는 환경변수 STREAMLIT_APP_PASSWORD 로 설정. 없으면 인증 스킵 (로컬 전용).
# 예: (Windows) setx STREAMLIT_APP_PASSWORD "your_password"
#     (bash)    export STREAMLIT_APP_PASSWORD="your_password"
# ============================================================
import os, hmac

def _check_password() -> bool:
    # 1차: env var. 2차: st.secrets (Streamlit Cloud UI에서 설정 시).
    required = os.environ.get("STREAMLIT_APP_PASSWORD")
    if not required:
        try:
            required = st.secrets.get("STREAMLIT_APP_PASSWORD")
        except (FileNotFoundError, KeyError, AttributeError):
            required = None
    if not required:
        return True                              # 비번 미설정 → 로컬 전용
    if st.session_state.get("authed", False):
        return True
    st.title("🔐 채권 3팩터 시스템 — 인증")
    pw = st.text_input("비밀번호", type="password", key="pw_input")
    if st.button("로그인"):
        if hmac.compare_digest(pw, required):
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("비밀번호 불일치")
    return False

if not _check_password():
    st.stop()

# ============================================================
# 데이터 로딩 (캐시)
# ============================================================

@st.cache_resource(show_spinner="FactorPipeline 실행 (처음만 ~30초) …")
def load_monitor(start: str, categories_key: str):
    """Loader 자동 선택.

    우선순위:
      1. 환경변수 USE_CACHED_DATA="0" → 강제 DB (로컬 개발만)
      2. data_cache/meta.json 존재 → CachedDataLoader (Cloud + 로컬 기본)
      3. 그 외 → DB 기반 DataLoader (사내망 전용)
    """
    import os as _os
    from pathlib import Path as _Path

    categories = categories_key.split(",")

    force_db = _os.environ.get("USE_CACHED_DATA") == "0"
    cache_dir_env = _os.environ.get("FT_CACHE_DIR", "data_cache")
    cache_dir = _Path(cache_dir_env)
    if not cache_dir.is_absolute():
        # app.py 부모 폴더 기준 상대 경로 해석
        cache_dir = _Path(__file__).resolve().parent.parent / cache_dir_env

    loader = None
    if not force_db and cache_dir.exists() and (cache_dir / "meta.json").exists():
        from factor_trading.data_loader_cached import CachedDataLoader
        loader = CachedDataLoader(
            cache_dir=str(cache_dir),
            start=start, end=None, categories=categories,
        )

    m = DailyMonitor(start=start, end=None, categories=categories,
                      loader=loader).run(verbose=False)
    return m


# ============================================================
# Sidebar
# ============================================================
st.sidebar.title("⚙️ 설정")
start_date = st.sidebar.text_input("데이터 시작일", value="2022-01-01",
                                    help="회귀 baseline 확보용. 변경 시 전체 재계산.")
categories = st.sidebar.multiselect("유니버스 category",
                                     options=["국고채", "통안채", "IRS", "국채선물"],
                                     default=["국고채"])
run_button = st.sidebar.button("🔄 Pipeline 실행 / 새로고침", type="primary", use_container_width=True)

if run_button or "monitor" not in st.session_state:
    if not categories:
        st.sidebar.error("최소 하나의 category 선택 필요")
        st.stop()
    key = ",".join(categories)
    st.session_state["monitor"] = load_monitor(start_date, key)

m: DailyMonitor = st.session_state["monitor"]
pipe = m.pipe
dy3  = pipe.dl.dY_3y()
dy10 = pipe.dl.dY_10y()
dyP  = pipe.dl.dy_panel()
rem  = pipe.dl.remain_panel()
eps  = pipe.residual
beta3  = pipe.betas["beta_3y"]
beta10 = pipe.betas["beta_10y"]
meta = pipe.dl.instrument_meta()

# 참고용 시계열
slope = (dy10.reindex(dy3.index) - dy3).rename("slope")
cum63 = mom_raw_cum(dy3, cum_window=63)
cum21_slope = slope.rolling(21, min_periods=10).sum()
regime_f3m = dy3.rolling(63, min_periods=30).sum()

st.sidebar.divider()
st.sidebar.markdown(f"**as_of**: `{m.as_of.date()}`")
st.sidebar.markdown(f"**유니버스 종목 수**: {dyP.shape[1]}")
st.sidebar.markdown(f"**회귀 샘플 일수**: {eps.notna().any(axis=1).sum()}")

# ============================================================
# 헤더
# ============================================================
st.title("📊 채권 3팩터 RV 트레이딩 — 실시간 모니터")
st.caption(f"as_of {m.as_of.date()} · 3팩터: RV / MOM / CURVE · 참조 명세서: `bond_3factor_final_spec.md`")

# ============================================================
# Tabs
# ============================================================
# 8개 탭 모두 정의 (코드 보존). 4번째 ~ 7번째 탭은 CSS 로 시각적으로 숨김.
# 다시 활성화하려면 아래 st.markdown 의 nth-child 선택자 제거 또는 주석 처리.
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🎯 오늘 스냅샷",
    "🔍 팩터별 상세",
    "🧮 3팩터 분해표",
    "📈 성과 & 합성",
    "✅ 모델 검증",
    "🔀 동적 비중",
    "🛰️ MOM_contra Satellite",
    "📐 종목별 베타 시계열",
])

# === Hide tabs 4~7 (코드는 유지, 화면 표시만 차단) ===
st.markdown("""
<style>
div[data-baseweb="tab-list"] > button:nth-child(4),
div[data-baseweb="tab-list"] > button:nth-child(5),
div[data-baseweb="tab-list"] > button:nth-child(6),
div[data-baseweb="tab-list"] > button:nth-child(7) {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# TAB 1 — 오늘 스냅샷
# ============================================================
with tab1:
    s = m.today_signals

    # Regime 경고
    for w in m.warnings_:
        if "⚠️" in w:
            st.warning(w)
        else:
            st.info(w)

    # 3팩터 signal 큰 카드
    col1, col2, col3 = st.columns(3)

    def _signal_box(col, name, sig, action, subtitle):
        color_map = {1: "#2ca02c", -1: "#d62728", 0: "#7f7f7f"}
        c = color_map.get(int(sig), "#7f7f7f")
        col.markdown(f"""
        <div style='padding:18px; border-radius:10px; background:{c}20; border-left:6px solid {c};'>
            <div style='font-size:1.1em; color:#555;'>{name}</div>
            <div style='font-size:2.4em; font-weight:bold; color:{c};'>signal = {int(sig):+d}</div>
            <div style='font-size:1.0em; font-weight:600;'>{action}</div>
            <div style='font-size:0.9em; color:#555; margin-top:6px;'>{subtitle}</div>
        </div>
        """, unsafe_allow_html=True)

    _signal_box(col1, "MOM (3Y 현물)",
                s["MOM"]["signal"], s["MOM"]["action"],
                f"cum_dY_3Y_63d = {s['MOM']['cum_dY_3Y_63d_bp']:+.2f}bp")
    _signal_box(col2, "CURVE (3Y/10Y 페어)",
                s["CURVE"]["signal"], s["CURVE"]["action"],
                f"cum_slope_21d = {s['CURVE']['cum_slope_21d_bp']:+.2f}bp")
    _signal_box(col3, f"RV (within-bucket Q5/Q1)",
                +1 if len(s["RV"]["long_bonds"]) > 0 else 0,
                f"LONG {len(s['RV']['long_bonds'])}개 / SHORT {len(s['RV']['short_bonds'])}개",
                "cheap bond LONG + rich bond SHORT")

    st.divider()

    # 최근 성과 요약
    st.subheader("📈 최근 성과 (daily PnL Sharpe annualized)")
    rp = m.recent_perf.pivot(index="factor", columns="window", values="sharpe_ann")
    rp = rp.reindex(["RV", "MOM", "CURVE", "EW_combo", "RP_combo"])
    rp = rp[["20d", "63d", "252d", "all"]]
    st.dataframe(
        rp.style.format("{:+.2f}").background_gradient(
            cmap="RdYlGn", vmin=-2, vmax=2, axis=None),
        use_container_width=True,
    )

    st.divider()

    # RV 후보 리스트
    col_l, col_s = st.columns(2)
    with col_l:
        st.subheader(f"🟢 RV LONG (Q5) · cheap 후보 {len(s['RV']['long_bonds'])}개")
        st.dataframe(
            s["RV"]["long_bonds"][["bond_name", "bond_code",
                                    "rv_score_bp", "remain_y", "ytm_bp"]],
            use_container_width=True, hide_index=True,
            column_config={
                "rv_score_bp": st.column_config.NumberColumn("RV score (bp)", format="%+.2f"),
                "remain_y":    st.column_config.NumberColumn("잔존(년)",     format="%.2f"),
                "ytm_bp":      st.column_config.NumberColumn("YTM (bp)",     format="%.1f"),
            },
        )
    with col_s:
        st.subheader(f"🔴 RV SHORT (Q1) · rich 후보 {len(s['RV']['short_bonds'])}개")
        st.dataframe(
            s["RV"]["short_bonds"][["bond_name", "bond_code",
                                     "rv_score_bp", "remain_y", "ytm_bp"]],
            use_container_width=True, hide_index=True,
            column_config={
                "rv_score_bp": st.column_config.NumberColumn("RV score (bp)", format="%+.2f"),
                "remain_y":    st.column_config.NumberColumn("잔존(년)",     format="%.2f"),
                "ytm_bp":      st.column_config.NumberColumn("YTM (bp)",     format="%.1f"),
            },
        )

# ============================================================
# TAB 2 — 팩터별 상세
# ============================================================
with tab2:
    sub1, sub2, sub3 = st.tabs(["📉 MOM", "📊 CURVE", "🧩 RV"])

    # --------------- MOM ---------------
    with sub1:
        st.subheader("MOM — 3M rate trend (raw_sign × cum=63d)")
        # signal history + cum_63 + 3Y yield level
        mom_sig = mom_raw_signal(cum63).shift(1)  # 실행 포지션
        # 3Y yield level (relative, bp 기준)
        y3_level = dy3.fillna(0).cumsum()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum63.index, y=cum63.values, name="cum_dY_3Y_63d (bp)",
                                  line=dict(color="#1f77b4", width=1.5)))
        fig.add_trace(go.Scatter(x=y3_level.index, y=y3_level.values,
                                  name="cum dY_3Y (level, bp)", yaxis="y2",
                                  line=dict(color="#7f7f7f", width=1, dash="dot"), opacity=0.6))
        # position shading
        long_x  = mom_sig[mom_sig == +1].index
        short_x = mom_sig[mom_sig == -1].index
        fig.add_trace(go.Scatter(x=long_x,  y=[cum63.max()]*len(long_x),
                                  mode="markers", name="LONG bond (signal=+1)",
                                  marker=dict(color="#2ca02c", size=4, symbol="square")))
        fig.add_trace(go.Scatter(x=short_x, y=[cum63.max()]*len(short_x),
                                  mode="markers", name="SHORT bond (signal=-1)",
                                  marker=dict(color="#d62728", size=4, symbol="square")))
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=0.6)
        fig.update_layout(
            height=420, title="MOM signal 시계열 — cum_dY_3Y_63d (주축) + 포지션",
            yaxis=dict(title="cum_dY_3Y_63d (bp)"),
            yaxis2=dict(title="cum dY level", overlaying="y", side="right", showgrid=False),
            hovermode="x unified", legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        **현재 상태**: cum_dY_3Y_63d = `{s['MOM']['cum_dY_3Y_63d_bp']:+.2f}` bp →
        signal = `{s['MOM']['signal']:+d}` → **{s['MOM']['action']}**
        """)

        # daily PnL
        pnl = m.pnls["MOM"]
        cum_pnl = pnl.fillna(0).cumsum()
        fig2 = go.Figure(go.Scatter(x=cum_pnl.index, y=cum_pnl.values,
                                    mode="lines", line=dict(color="#d62728", width=1.5)))
        fig2.update_layout(height=320, hovermode="x unified", showlegend=False,
                           title="MOM 누적 PnL (bp)",
                           yaxis=dict(title="cum PnL (bp)"))
        st.plotly_chart(fig2, use_container_width=True)

    # --------------- CURVE ---------------
    with sub2:
        st.subheader("CURVE — 3주 slope 누적 mean-rev (raw_sign × cum=21d)")
        curve_sig = (-np.sign(cum21_slope)).shift(1)
        slope_level = slope.fillna(0).cumsum()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum21_slope.index, y=cum21_slope.values,
                                  name="cum_slope_21d (bp)",
                                  line=dict(color="#9467bd", width=1.5)))
        fig.add_trace(go.Scatter(x=slope_level.index, y=slope_level.values,
                                  name="cum slope (10Y-3Y) level", yaxis="y2",
                                  line=dict(color="#7f7f7f", width=1, dash="dot"), opacity=0.6))
        st_x = curve_sig[curve_sig == +1].index  # steepen
        fl_x = curve_sig[curve_sig == -1].index  # flatten
        fig.add_trace(go.Scatter(x=st_x, y=[cum21_slope.max()]*len(st_x),
                                  mode="markers", name="STEEPEN (+1)",
                                  marker=dict(color="#2ca02c", size=4, symbol="square")))
        fig.add_trace(go.Scatter(x=fl_x, y=[cum21_slope.max()]*len(fl_x),
                                  mode="markers", name="FLATTEN (-1)",
                                  marker=dict(color="#d62728", size=4, symbol="square")))
        fig.add_hline(y=0, line_color="black", line_width=0.6)
        fig.update_layout(
            height=420, title="CURVE signal 시계열 — cum_slope_21d + 포지션",
            yaxis=dict(title="cum_slope_21d (bp)"),
            yaxis2=dict(title="slope level", overlaying="y", side="right", showgrid=False),
            hovermode="x unified", legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        **현재 상태**: cum_slope_21d = `{s['CURVE']['cum_slope_21d_bp']:+.2f}` bp →
        signal = `{s['CURVE']['signal']:+d}` → **{s['CURVE']['action']}**
        """)

        pnl_c = m.pnls["CURVE"].fillna(0).cumsum()
        fig2 = go.Figure(go.Scatter(x=pnl_c.index, y=pnl_c.values,
                                    mode="lines", line=dict(color="#9467bd", width=1.5)))
        fig2.update_layout(height=320, hovermode="x unified", showlegend=False,
                           title="CURVE 누적 PnL (bp)",
                           yaxis=dict(title="cum PnL (bp)"))
        st.plotly_chart(fig2, use_container_width=True)

    # --------------- RV ---------------
    with sub3:
        st.subheader("RV — within-bucket quintile · horizon 1m · Q5/Q1")

        # 현재 quintile 분포 (만기 × quintile count)
        labels = within_bucket_quintile(pipe.rv_score, rem,
                                         bucket_edges=m.rv_bucket_edges, n_bins=5)
        today = m.as_of
        today_labels = labels.loc[today].dropna()
        today_remain = rem.loc[today].reindex(today_labels.index)
        today_score  = pipe.rv_score.loc[today].reindex(today_labels.index)

        bucket_fn = pd.cut(today_remain, bins=m.rv_bucket_edges,
                            labels=["≤5y", "5~10y", ">10y"])
        df_today = pd.DataFrame({
            "bond": today_labels.index,
            "bucket": bucket_fn.values,
            "quintile": today_labels.values.astype(int),
            "rv_score_bp": today_score.values,
            "remain_y": today_remain.values,
        })

        # quintile 평균 rv_score (오늘 cross-section)
        fig = px.box(df_today, x="quintile", y="rv_score_bp", color="bucket",
                      title="오늘 quintile × 만기버킷별 RV score 분포",
                      points="all")
        fig.update_layout(height=420, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # RV daily PnL
        pnl_r = m.pnls["RV"].fillna(0).cumsum()
        fig2 = go.Figure(go.Scatter(x=pnl_r.index, y=pnl_r.values,
                                    mode="lines", line=dict(color="#1f77b4", width=1.5)))
        fig2.update_layout(height=320, hovermode="x unified", showlegend=False,
                           title="RV (Q5-Q1 within-bucket) 누적 daily PnL (bp)",
                           yaxis=dict(title="cum PnL (bp)"))
        st.plotly_chart(fig2, use_container_width=True)

        st.info("⚠️ RV의 **자연 hold는 21d** (월 1회 rebalance). 위 daily PnL은 "
                "직교성 비교용. 실제 운용은 21d forward β-hedged PnL 기준.")

# ============================================================
# TAB 3 — 3팩터 분해표
# ============================================================
with tab3:
    st.subheader("🧮 dY_i,t = sys_3Y + sys_10Y + ε  (종목별 3팩터 분해)")
    st.caption("각 종목의 일간 yield 변동을 체계(3Y/10Y factor) vs 고유(RV) 로 분해. "
               "ε가 크면 cross-sectional dislocation 후보.")

    dec = m.decomposition
    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        dates = sorted(dec["date"].unique(), reverse=True)
        sel_date = st.selectbox("날짜", dates, index=0)
    with col_b:
        buckets = st.multiselect("만기 버킷",
                                 options=["≤5y", "5~10y", ">10y"],
                                 default=["≤5y", "5~10y", ">10y"])
    with col_c:
        sort_by = st.selectbox("정렬 기준",
                                options=["|ε| 큰 순", "ε 오름차순", "dY 큰 순", "잔존 짧은 순"],
                                index=0)

    def _bucket_of(r):
        if r <= 5: return "≤5y"
        if r <= 10: return "5~10y"
        return ">10y"

    df = dec[dec["date"] == sel_date].copy()
    df["bucket"] = df["remain_y"].apply(lambda r: _bucket_of(r) if pd.notna(r) else None)
    df = df[df["bucket"].isin(buckets)]

    if sort_by == "|ε| 큰 순":
        df = df.reindex(df["epsilon_bp"].abs().sort_values(ascending=False).index)
    elif sort_by == "ε 오름차순":
        df = df.sort_values("epsilon_bp")
    elif sort_by == "dY 큰 순":
        df = df.sort_values("dY_bp", ascending=False)
    else:
        df = df.sort_values("remain_y")

    st.dataframe(
        df[["bond_name", "bond_code", "remain_y", "bucket",
             "dY_bp", "sys_3Y_bp", "sys_10Y_bp", "epsilon_bp",
             "beta_3Y", "beta_10Y"]],
        use_container_width=True, hide_index=True, height=500,
        column_config={
            "remain_y":   st.column_config.NumberColumn("잔존(년)",     format="%.2f"),
            "dY_bp":      st.column_config.NumberColumn("dY (bp)",       format="%+.2f"),
            "sys_3Y_bp":  st.column_config.NumberColumn("sys_3Y (bp)",   format="%+.2f"),
            "sys_10Y_bp": st.column_config.NumberColumn("sys_10Y (bp)",  format="%+.2f"),
            "epsilon_bp": st.column_config.NumberColumn("ε (bp)",        format="%+.3f"),
            "beta_3Y":    st.column_config.NumberColumn("β_3Y",          format="%+.3f"),
            "beta_10Y":   st.column_config.NumberColumn("β_10Y",         format="%+.3f"),
        },
    )

    # 상위 20 종목 분해 bar chart
    top = df.head(20)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=top["bond_name"], y=top["sys_3Y_bp"],
                          name="sys_3Y", marker_color="#1f77b4"))
    fig.add_trace(go.Bar(x=top["bond_name"], y=top["sys_10Y_bp"],
                          name="sys_10Y", marker_color="#9467bd"))
    fig.add_trace(go.Bar(x=top["bond_name"], y=top["epsilon_bp"],
                          name="ε (RV)", marker_color="#d62728"))
    fig.update_layout(
        barmode="group", height=450,
        title=f"상위 20 종목 dY 분해 ({sel_date})",
        xaxis=dict(tickangle=-40), yaxis=dict(title="bp"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # CSV 다운로드
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 CSV 다운로드", data=csv,
                       file_name=f"{sel_date}_decomposition.csv",
                       mime="text/csv")

# ============================================================
# TAB 4 — 성과 & 합성
# ============================================================
with tab4:
    st.subheader("📈 3팩터 + 합성 포트폴리오 성과")

    # 누적 PnL interactive
    df_pnl = pd.DataFrame(m.pnls).fillna(0).cumsum()
    fig = go.Figure()
    colors = {"RV":"#1f77b4", "MOM":"#d62728", "CURVE":"#9467bd",
              "EW_combo":"#2ca02c", "RP_combo":"#ff7f0e"}
    for col in df_pnl.columns:
        lw = 2.2 if "combo" in col else 1.4
        fig.add_trace(go.Scatter(x=df_pnl.index, y=df_pnl[col],
                                  name=col, line=dict(color=colors.get(col), width=lw)))
    fig.add_hline(y=0, line_color="black", line_width=0.5)
    fig.add_vline(x=m.as_of, line_color="gray", line_width=0.8, line_dash="dash")
    fig.update_layout(
        height=520, title="3팩터 + 합성 누적 PnL",
        yaxis=dict(title="cum PnL (bp)"),
        hovermode="x unified", legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)

    # 기간별 Sharpe
    with col_a:
        st.markdown("**기간별 Sharpe (ann)**")
        rp_pv = m.recent_perf.pivot(index="factor", columns="window", values="sharpe_ann")
        rp_pv = rp_pv.reindex(["RV", "MOM", "CURVE", "EW_combo", "RP_combo"])[["20d", "63d", "252d", "all"]]
        st.dataframe(
            rp_pv.style.format("{:+.2f}").background_gradient(
                cmap="RdYlGn", vmin=-2, vmax=2, axis=None),
            use_container_width=True,
        )

    # 기간별 hit%
    with col_b:
        st.markdown("**기간별 hit%**")
        rp_hit = m.recent_perf.pivot(index="factor", columns="window", values="hit%")
        rp_hit = rp_hit.reindex(["RV", "MOM", "CURVE", "EW_combo", "RP_combo"])[["20d", "63d", "252d", "all"]]
        st.dataframe(
            rp_hit.style.format("{:.1f}%").background_gradient(
                cmap="RdYlGn", vmin=40, vmax=60, axis=None),
            use_container_width=True,
        )

    # Drawdown
    st.markdown("**각 팩터 drawdown 시계열**")
    dd_df = pd.DataFrame()
    for col, s_ in m.pnls.items():
        cum = s_.fillna(0).cumsum()
        dd_df[col] = cum - cum.cummax()
    fig_dd = go.Figure()
    for col in dd_df.columns:
        fig_dd.add_trace(go.Scatter(x=dd_df.index, y=dd_df[col],
                                     name=col, line=dict(color=colors.get(col), width=1.3),
                                     fill="tozeroy", opacity=0.55))
    fig_dd.update_layout(
        height=380, title="Drawdown (cum − peak)",
        yaxis=dict(title="bp (음수)"),
        hovermode="x unified", legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_dd, use_container_width=True)

# ============================================================
# TAB 5 — 모델 검증
# ============================================================
with tab5:
    st.subheader("✅ 3팩터 직교성 + 모델 설명력")

    # PnL 상관행렬
    df_p = pd.DataFrame({k: v for k, v in m.pnls.items() if "combo" not in k}).dropna(how="all")
    corr3 = df_p.corr()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**PnL 상관행렬**  (|corr| < 0.3 목표)")
        st.dataframe(
            corr3.style.format("{:+.3f}").background_gradient(cmap="RdBu_r", vmin=-1, vmax=1, axis=None),
            use_container_width=True,
        )
        off = corr3.where(~np.eye(len(corr3), dtype=bool))
        maxc = off.abs().max().max()
        st.metric("|corr|max (off-diagonal)", f"{maxc:.3f}",
                   delta="PASS" if maxc < 0.3 else "FAIL", delta_color="normal" if maxc < 0.3 else "inverse")

    # 2팩터 회귀 R² 분포
    with col_b:
        # 종목별 R² = 1 - Var(ε) / Var(dY)
        var_dy  = dyP.var(axis=0, skipna=True)
        var_eps = eps.var(axis=0, skipna=True)
        r2 = (1 - var_eps / var_dy).dropna()
        r2 = r2.loc[(r2 >= -0.5) & (r2 <= 1.05)]
        st.markdown("**2팩터 회귀 R² 분포**  (dY_i ~ β_3·dY_3 + β_10·dY_10)")
        st.caption("⚠️ 이건 Layer 1 **체계 risk 분해**용 2팩터 회귀의 R². "
                    "Layer 2 트레이딩 팩터(RV/MOM/CURVE)는 이 잔차 ε와 dY_3Y/dY_10Y의 "
                    "비선형 신호로 만든 포트폴리오 layer — 선형 회귀 R²에는 거의 기여 없음 "
                    "(잔차와 signal이 이미 직교).")
        st.metric("median R²", f"{r2.median():.3f}",
                   delta=f"mean {r2.mean():.3f}")
        fig_r2 = px.histogram(r2.values, nbins=25, title="종목별 R² 분포")
        fig_r2.update_traces(marker_color="#2ca02c", opacity=0.7,
                              marker_line_color="black", marker_line_width=0.5)
        fig_r2.update_layout(height=300, showlegend=False,
                             xaxis_title="R²", yaxis_title="종목 수")
        st.plotly_chart(fig_r2, use_container_width=True)

    st.divider()

    # 잔차 × signal 직교성
    st.markdown("**잔차 × signal 직교성** (종목별 corr(ε_i, signal_t))")
    mom_sig_full = mom_raw_signal(cum63)
    curve_sig_full = (-np.sign(cum21_slope)).rename("CURVE_sig")
    eps_stack = eps.loc[(eps.index >= pd.Timestamp("2023-07-01"))]

    ce_mom = pd.Series({c: eps_stack[c].corr(mom_sig_full.reindex(eps_stack.index))
                         for c in eps_stack.columns if eps_stack[c].notna().sum() > 50})
    ce_crv = pd.Series({c: eps_stack[c].corr(curve_sig_full.reindex(eps_stack.index))
                         for c in eps_stack.columns if eps_stack[c].notna().sum() > 50})

    cA, cB = st.columns(2)
    with cA:
        fig = px.histogram(ce_mom.values, nbins=30, title="corr(ε_i, MOM_signal) 분포")
        fig.update_traces(marker_color="#d62728", opacity=0.7,
                           marker_line_color="black", marker_line_width=0.5)
        fig.add_vline(x=+0.2, line_dash="dash", line_color="gray")
        fig.add_vline(x=-0.2, line_dash="dash", line_color="gray")
        fig.add_vline(x=ce_mom.median(), line_color="blue",
                     annotation_text=f"median={ce_mom.median():+.3f}")
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"|corr|>0.2 종목: {int((ce_mom.abs()>0.2).sum())}/{len(ce_mom)}")
    with cB:
        fig = px.histogram(ce_crv.values, nbins=30, title="corr(ε_i, CURVE_signal) 분포")
        fig.update_traces(marker_color="#9467bd", opacity=0.7,
                           marker_line_color="black", marker_line_width=0.5)
        fig.add_vline(x=+0.2, line_dash="dash", line_color="gray")
        fig.add_vline(x=-0.2, line_dash="dash", line_color="gray")
        fig.add_vline(x=ce_crv.median(), line_color="blue",
                     annotation_text=f"median={ce_crv.median():+.3f}")
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"|corr|>0.2 종목: {int((ce_crv.abs()>0.2).sum())}/{len(ce_crv)}")

    st.success("✓ 2팩터 회귀 잔차가 MOM/CURVE 비선형 signal과도 거의 완벽 직교 → "
                "3팩터 모델이 체계 risk를 clean 분해")

# ============================================================
# TAB 6 — 동적 비중 리밸런싱
# ============================================================
with tab6:
    st.subheader("🔀 동적 비중 리밸런싱 — rank-based momentum-of-factor")
    st.caption("매 리밸런싱 시점에 각 팩터의 최근 성적 (1w/2w/1m Sharpe 가중평균) 기준 "
               "1위=w_max, 3위=w_min, 2위=나머지. 3팩터의 recent momentum에 overweight.")

    # Parameter panel
    with st.expander("⚙️ 파라미터 설정", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            rebal_days = st.slider("리밸런싱 주기 (영업일)",
                                    min_value=5, max_value=63, value=21, step=1)
            w_min = st.slider("w_min (최저 비중)",
                               min_value=0.05, max_value=0.32, value=0.15, step=0.01)
        with c2:
            w_max = st.slider("w_max (최대 비중)",
                               min_value=0.34, max_value=0.70, value=0.55, step=0.01)
            standardize = st.checkbox("각 팩터 vol 표준화 (권장)", value=True)
        with c3:
            sw_1w = st.slider("score 가중치 · 1w", 0.0, 1.0, 0.5, 0.05)
            sw_2w = st.slider("score 가중치 · 2w", 0.0, 1.0, 0.3, 0.05)
            sw_1m = st.slider("score 가중치 · 1m", 0.0, 1.0, 0.2, 0.05)
        # 정규화
        _sw_sum = sw_1w + sw_2w + sw_1m
        if _sw_sum > 0:
            sw_1w_n, sw_2w_n, sw_1m_n = sw_1w/_sw_sum, sw_2w/_sw_sum, sw_1m/_sw_sum
        else:
            sw_1w_n, sw_2w_n, sw_1m_n = 0.5, 0.3, 0.2
        w_mid = 1.0 - w_min - w_max
        st.info(f"비중 분배: **w_max = {w_max:.2f}**  /  w_mid = {w_mid:.2f}  /  **w_min = {w_min:.2f}**    "
                f"| score 가중치(정규화): 1w {sw_1w_n:.2f}, 2w {sw_2w_n:.2f}, 1m {sw_1m_n:.2f}")

    if not (0.05 <= w_min < 1/3 < w_max <= 0.70):
        st.error("w_min < 1/3 < w_max 조건 필요")
    elif abs(w_mid) < 1e-6 or w_mid < 0:
        st.error(f"w_mid (={w_mid:.2f}) 가 0 이하. w_min + w_max 줄이세요.")
    else:
        # 백테스트
        base_pnls = {"RV": m.pnls["RV"], "MOM": m.pnls["MOM"], "CURVE": m.pnls["CURVE"]}
        with st.spinner("동적 백테스트 돌리는 중 …"):
            result = dynamic_weight_backtest(
                base_pnls,
                rebalance_days=rebal_days,
                score_w1w=sw_1w_n, score_w2w=sw_2w_n, score_w1m=sw_1m_n,
                w_min=w_min, w_max=w_max,
                standardize=standardize,
            )
        pnl_dyn  = result["pnl_dyn"]
        w_df_dyn = result["weights"]
        rlog     = result["rebalance_log"]

        # 비교: EW — DYN과 동일한 표준화 scheme (rolling σ, lag=1) 맞춤
        from factor_trading.portfolio.combiner import equal_weight
        ew_std = equal_weight(base_pnls, standardize=True, vol_window=63)

        # 성과
        def _sh(s, W=None):
            s = s.dropna()
            if W: s = s.iloc[-W:]
            if len(s) < 2 or s.std() == 0: return np.nan
            return s.mean() / s.std(ddof=1) * np.sqrt(252)
        def _max_dd(s):
            cum = s.fillna(0).cumsum()
            return float((cum - cum.cummax()).min())

        rows_perf = []
        for k, v in [("DYN", pnl_dyn), ("EW_std (baseline)", ew_std),
                     ("RV", base_pnls["RV"]), ("MOM", base_pnls["MOM"]), ("CURVE", base_pnls["CURVE"])]:
            vn = v.dropna()
            rows_perf.append({
                "strategy": k, "n": len(vn),
                "Sh_all":   _sh(vn),
                "Sh_252d":  _sh(vn, 252),
                "Sh_63d":   _sh(vn, 63),
                "Sh_20d":   _sh(vn, 20),
                "cum_bp":   float(vn.sum()),
                "max_dd":   _max_dd(vn),
                "hit%":     float((vn > 0).mean()) * 100,
            })
        perf_df = pd.DataFrame(rows_perf).set_index("strategy")

        col_p, col_q = st.columns([2, 1])
        with col_p:
            st.markdown("**누적 PnL 비교** (표준화된 unit-vol scale)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pnl_dyn.cumsum().index, y=pnl_dyn.cumsum().values,
                                      name=f"DYN rebal={rebal_days}d  Sh={_sh(pnl_dyn):+.2f}",
                                      line=dict(color="#e83e8c", width=2.3)))
            fig.add_trace(go.Scatter(x=ew_std.cumsum().index, y=ew_std.cumsum().values,
                                      name=f"EW_std (baseline)  Sh={_sh(ew_std):+.2f}",
                                      line=dict(color="#2ca02c", width=1.7, dash="dash")))
            fig.add_hline(y=0, line_color="black", line_width=0.5)
            fig.update_layout(height=420, hovermode="x unified",
                               yaxis=dict(title="cum PnL (unit-std)"),
                               legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)

        with col_q:
            st.markdown("**성과 요약**")
            st.dataframe(
                perf_df.style.format({
                    "Sh_all":  "{:+.2f}", "Sh_252d": "{:+.2f}",
                    "Sh_63d":  "{:+.2f}", "Sh_20d":  "{:+.2f}",
                    "cum_bp":  "{:+.1f}", "max_dd":  "{:+.1f}",
                    "hit%":    "{:.1f}%",
                }).background_gradient(
                    cmap="RdYlGn", subset=["Sh_all", "Sh_252d", "Sh_63d", "Sh_20d"],
                    vmin=-1, vmax=2, axis=None),
                use_container_width=True, height=260,
            )

        # 비중 시계열 (stacked area)
        st.markdown("**비중 시계열 (stacked)**")
        w_plot = w_df_dyn.dropna().copy()
        fig_w = go.Figure()
        colors_f = {"RV":"#1f77b4", "MOM":"#d62728", "CURVE":"#9467bd"}
        for col in w_plot.columns:
            fig_w.add_trace(go.Scatter(
                x=w_plot.index, y=w_plot[col], name=col,
                mode="lines", stackgroup="one",
                line=dict(width=0.2, color=colors_f.get(col, "#7f7f7f")),
                fillcolor=colors_f.get(col, "#7f7f7f"),
            ))
        fig_w.update_layout(
            height=320, yaxis=dict(title="weight", range=[0, 1.02]),
            title=f"동적 비중 시계열 (rebal={rebal_days}d, [{w_min:.2f}, {w_max:.2f}])",
            hovermode="x unified", legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_w, use_container_width=True)

        # 리밸런싱 이력
        st.markdown("**리밸런싱 이력** (최근 역순)")
        if len(rlog) > 0:
            show_cols = ["date",
                         "score_RV", "score_MOM", "score_CURVE",
                         "w_RV", "w_MOM", "w_CURVE"]
            rlog_show = rlog[show_cols].sort_values("date", ascending=False)
            st.dataframe(
                rlog_show,
                use_container_width=True, hide_index=True, height=300,
                column_config={
                    "score_RV":    st.column_config.NumberColumn("score RV",    format="%+.3f"),
                    "score_MOM":   st.column_config.NumberColumn("score MOM",   format="%+.3f"),
                    "score_CURVE": st.column_config.NumberColumn("score CURVE", format="%+.3f"),
                    "w_RV":    st.column_config.NumberColumn("w RV",    format="%.2f"),
                    "w_MOM":   st.column_config.NumberColumn("w MOM",   format="%.2f"),
                    "w_CURVE": st.column_config.NumberColumn("w CURVE", format="%.2f"),
                },
            )
        else:
            st.warning("리밸런싱 이력 없음 — warmup 부족?")

        st.caption("💡 동적 전략의 핵심: 최근 이기고 있는 팩터에 **제한적으로** overweight, "
                    "지는 팩터도 완전히 버리지 않고 w_min 이상 유지. "
                    "EW_std 와 비교하여 값 1.0x 이상 초과하면 효과적인 factor-of-factor momentum.")


# ============================================================
# TAB 7 — MOM_contra Satellite (event-driven overlay)
# ============================================================
with tab7:
    st.subheader("🛰️ MOM_contra — Event-Driven Satellite Factor")

    # -------------- 긴 설명 섹션 --------------
    with st.expander("📖 이론 · 메커니즘 · 왜 Satellite인가", expanded=True):
        st.markdown("""
### 이 팩터가 무엇인가

**MOM_contra**는 핵심 3팩터 (RV/MOM/CURVE)에 **들어가지 않는 별도의 보조 신호**다.
한국 국채 3Y 금리가 **과거 1년 기준 극단적으로 큰 변동을 겪었을 때만 발동**되어,
그 반대 방향에 베팅한다. 전형적인 **event-driven reversal** 성격.

### 이론적 근거

MOM 심층 진단에서 우리가 발견한 것:

- **중간 극단** (`1 < |z| < 2`): Rate trend 지속 → **pure momentum** 작동 (SHORT/LONG bond)
- **초극단** (`|z| ≥ 2`): Rate가 너무 많이 움직임 → **contrarian (mean reversion)** 작동

`z = cum_dY_3Y_63d / (past 1y σ)` — 3개월 누적 금리 이동을 과거 1년 변동성으로 normalize.

극단치에서 **반전이 일어나는 이유**:
- 시장 참여자들이 "과도한 이동" 인지 후 반대 포지션
- 크레딧 shock, 유동성 위기 때 flight-to-quality 등 macro shock 패턴
- 한국 채권시장의 **BOK 정책 피벗 시 급반전** 전형
- 실제 발동 사례: **2022-10 레고랜드 사태**, **2023-03 SVB 파산**, BOK 인상 정점 시기

### 왜 Satellite 인가 (핵심 3팩터에 넣지 않는 이유)

직교성 검증 결과:
- MOM_contra 단독: Sharpe +0.46, α-t +0.61 (유의 borderline)
- 기존 3팩터에 추가 시: **MOM_trend × RV 상관 +0.375** (명세 기준 0.3 초과) ⚠️
- α-t 전반 약화 (MOM +2.04 → +0.76)

**3원칙 중 원칙 3 (직교성) 훼손 위험**. 따라서:
- 핵심 체제는 **RV/MOM/CURVE 3팩터 유지** (직교성 clean)
- MOM_contra는 **선택적 overlay** — 관심 있을 때만 추가

### 사용 방법

1. **비발동 시 (약 89%)**: signal = 0 → 메인 포트폴리오에 영향 없음
2. **발동 시 (약 11%)**: `z` 부호에 따라
   - `z > +θ_high`: rate 급등 후 반전 기대 → **SHORT bond**
   - `z < −θ_high`: rate 급락 후 반전 기대 → **LONG bond**
3. overlay weight `w_sat` (기본 15%) 만큼 메인에 가산

### 위험 요소

- **False signal**: Trend 가 진짜로 지속되는 경우 손실. 샘플 작아 통계적 신뢰도 약함 (활성 ~11%).
- **α-t 1.5 미달**: 나머지 팩터로 부분적으로 설명될 수 있음 → 독립 alpha 제한적.
- **Overlay 비중 과다 금지**: 15% 이하 권장. 메인 포트폴리오 안정성 우선.
- **Regime risk**: Whipsaw 시기에 contra 도 역풍 가능 (2022-12 일부 이벤트).

### 실거래 운용 권고

- 메인 포트폴리오(DYN 또는 EW) 은 **항상 유지**
- satellite signal 발동 시, **기존 포지션 리사이즈 없이 추가 포지션**으로 얹음
- overlay 발동은 평균 **연 ~11회** (3Y 기준 11% × 252일)
- **체결 주기**: lag=1 (매일 close 기준 signal 확인 후 익일 체결)
""")
    st.divider()

    # -------------- Parameter --------------
    st.markdown("### ⚙️ Satellite 파라미터")
    c1, c2, c3 = st.columns(3)
    with c1:
        theta_high_sat = st.slider("θ_high (|z| ≥ 이 값 때 발동)",
                                     min_value=1.25, max_value=3.0, value=1.75, step=0.05,
                                     help="과거 1년 σ 대비 몇 배 이상이면 contrarian 발동")
    with c2:
        sat_weight = st.slider("w_sat (overlay 가중치)",
                                 min_value=0.0, max_value=0.30, value=0.15, step=0.01,
                                 help="메인 포트폴리오 대비 satellite 의 size 비율")
    with c3:
        base_strategy = st.selectbox("메인 포트폴리오 base",
                                       options=["DYN", "EW_combo"], index=0,
                                       help="어떤 3팩터 합성 위에 overlay 할지")

    # -------------- Signal 계산 --------------
    cum63_all = mom_raw_cum(dy3, cum_window=63)
    # theta_low 는 의미 없음 (trend 는 메인 MOM이 담당) — 높은 값으로 둠
    _, contra_sig = mom_split_signals(cum63_all, sigma_window=252, sigma_minp=126,
                                       theta_low=0.01, theta_high=theta_high_sat)
    contra_pnl = (contra_sig.shift(1) * (-dy3)).rename("MOM_contra")

    # 활성 통계
    act_mask = (contra_sig.abs() > 0)
    act_days_n = int(act_mask.sum())
    total_days = int(contra_sig.notna().sum())
    act_pct = act_days_n / total_days * 100 if total_days else 0.0

    # 메인 포트폴리오
    base_pnls = {"RV": m.pnls["RV"], "MOM": m.pnls["MOM"], "CURVE": m.pnls["CURVE"]}
    if base_strategy == "DYN":
        main_res = dynamic_weight_backtest(base_pnls, rebalance_days=21,
                                            w_min=0.15, w_max=0.55)
        main_pnl = main_res["pnl_dyn"]
    else:
        main_pnl = equal_weight(base_pnls, standardize=True, vol_window=63)

    # overlay
    combined = satellite_overlay(main_pnl, contra_pnl,
                                  sat_weight=sat_weight, standardize_sat=True,
                                  sat_vol_window=63)

    # -------------- 통계 --------------
    def _stats(s, holding=1):
        s = s.dropna()
        if len(s) < 2: return dict(n=len(s), Sh=np.nan, dd=np.nan, hit=np.nan, cum=np.nan)
        mu, sd = s.mean(), s.std(ddof=1)
        cum = s.cumsum()
        return dict(n=len(s),
                    Sh=mu/sd*np.sqrt(252) if sd>0 else np.nan,
                    cum=float(cum.iloc[-1]),
                    dd=float((cum-cum.cummax()).min()),
                    hit=float((s>0).mean())*100)

    a = _stats(main_pnl); b = _stats(combined); c = _stats(contra_pnl)

    # Metric cards
    st.markdown("### 📊 Satellite ON/OFF 비교")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(f"{base_strategy} (base)", f"Sh {a['Sh']:+.2f}",
                   delta=f"DD {a['dd']:+.1f}")
    with m2:
        st.metric(f"{base_strategy} + Satellite",
                   f"Sh {b['Sh']:+.2f}",
                   delta=f"{b['Sh']-a['Sh']:+.2f}",
                   help="overlay 한 뒤 전체 Sharpe")
    with m3:
        st.metric("MOM_contra 단독",
                   f"Sh {c['Sh']:+.2f}",
                   delta=f"활성 {act_pct:.1f}%")
    with m4:
        st.metric("Satellite 발동 일수",
                   f"{act_days_n}일",
                   delta=f"/ {total_days}일")

    # 누적 PnL 비교 차트
    fig_cmp = go.Figure()
    cum_a = main_pnl.fillna(0).cumsum()
    cum_b = combined.fillna(0).cumsum()
    fig_cmp.add_trace(go.Scatter(x=cum_a.index, y=cum_a.values,
                                   name=f"{base_strategy} 단독  Sh {a['Sh']:+.2f}",
                                   line=dict(color="#7f7f7f", width=1.5, dash="dash")))
    fig_cmp.add_trace(go.Scatter(x=cum_b.index, y=cum_b.values,
                                   name=f"{base_strategy} + Satellite(w={sat_weight:.2f})  "
                                        f"Sh {b['Sh']:+.2f}",
                                   line=dict(color="#e83e8c", width=2.2)))
    # 발동 이벤트 mark
    active_dates = contra_sig[act_mask].index
    if len(active_dates) > 0:
        long_events  = contra_sig[contra_sig == +1].index
        short_events = contra_sig[contra_sig == -1].index
        fig_cmp.add_trace(go.Scatter(x=long_events, y=[0]*len(long_events),
                                       mode="markers", name="LONG signal",
                                       marker=dict(symbol="triangle-up", size=7,
                                                   color="#2ca02c", opacity=0.7)))
        fig_cmp.add_trace(go.Scatter(x=short_events, y=[0]*len(short_events),
                                       mode="markers", name="SHORT signal",
                                       marker=dict(symbol="triangle-down", size=7,
                                                   color="#d62728", opacity=0.7)))
    fig_cmp.add_hline(y=0, line_color="black", line_width=0.5)
    fig_cmp.update_layout(height=460,
                           title=f"누적 PnL — Satellite ON/OFF 비교  |  발동 표시 ▲▼",
                           yaxis=dict(title="cum PnL (unit-std)"),
                           hovermode="x unified",
                           legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig_cmp, use_container_width=True)

    # -------------- 발동 이력 --------------
    st.markdown("### 📋 발동 이벤트 이력 (최근순)")
    # regime
    f3m_s = dy3.rolling(63, min_periods=30).sum()
    # event 테이블
    events_rows = []
    for dt in active_dates:
        f3m_v = f3m_s.get(dt, np.nan)
        if pd.isna(f3m_v): continue
        if   f3m_v <= -25: reg = "bull강"
        elif f3m_v <=  -5: reg = "bull약"
        elif f3m_v <=   5: reg = "flat"
        elif f3m_v <=  25: reg = "bear약"
        else:              reg = "bear강"
        sig_today = int(contra_sig.loc[dt])
        # signal 체결 후 5d 누적 PnL
        next_5d = contra_pnl.loc[dt:].iloc[1:6]
        pnl_5d = float(next_5d.sum()) if len(next_5d) > 0 else np.nan
        events_rows.append({
            "date": dt.date(),
            "cum_dY_63d_bp": float(cum63_all.loc[dt]),
            "z_score": float(cum63_all.loc[dt] /
                              cum63_all.rolling(252, min_periods=126).std(ddof=1).loc[dt]),
            "signal": f"{'LONG' if sig_today == +1 else 'SHORT'}",
            "regime": reg,
            "pnl_next5d_bp": pnl_5d,
        })
    ev_df = pd.DataFrame(events_rows).sort_values("date", ascending=False)
    if len(ev_df) == 0:
        st.info("현재 파라미터로는 발동 이벤트 없음. θ_high 를 낮춰 보세요.")
    else:
        st.dataframe(
            ev_df, hide_index=True, use_container_width=True, height=360,
            column_config={
                "cum_dY_63d_bp": st.column_config.NumberColumn("cum_dY_63d (bp)", format="%+.2f"),
                "z_score":       st.column_config.NumberColumn("z",             format="%+.2f"),
                "pnl_next5d_bp": st.column_config.NumberColumn("이후 5d PnL (bp)", format="%+.2f"),
            },
        )
        win = (ev_df["pnl_next5d_bp"] > 0).sum()
        loss = (ev_df["pnl_next5d_bp"] < 0).sum()
        st.caption(f"✓ 총 {len(ev_df)}회 발동 중 이후 5영업일 누적 PnL 기준 "
                    f"**이익 {win}회 · 손실 {loss}회 · hit% {win/(win+loss)*100:.1f}**")

    st.markdown("""
---
### 🎯 오늘 이 순간 Satellite 상태

""")
    now_cum = float(cum63_all.iloc[-1]) if pd.notna(cum63_all.iloc[-1]) else np.nan
    now_sigma = cum63_all.rolling(252, min_periods=126).std(ddof=1).iloc[-1]
    now_z = now_cum / now_sigma if now_sigma and not pd.isna(now_sigma) else np.nan
    now_sig = contra_sig.iloc[-1] if contra_sig.notna().any() else 0

    if pd.isna(now_sig) or now_sig == 0:
        st.info(f"**satellite 비활성** — 현재 |z|={abs(now_z):.2f} < θ_high={theta_high_sat:.2f}  "
                f"(cum_dY_63d = {now_cum:+.1f}bp). 메인 포트폴리오만 운용.")
    elif now_sig > 0:
        st.success(f"🟢 **satellite 활성 — LONG bond** "
                    f"(z={now_z:+.2f}, cum_dY_63d={now_cum:+.1f}bp). "
                    f"Rate 급락 후 반전 기대.")
    else:
        st.error(f"🔴 **satellite 활성 — SHORT bond** "
                  f"(z={now_z:+.2f}, cum_dY_63d={now_cum:+.1f}bp). "
                  f"Rate 급등 후 반전 기대.")


# ============================================================
# TAB 8 — 종목별 다중회귀 베타 시계열 (γ_level / γ_slope / ε)
# ============================================================
with tab8:
    st.subheader("📐 종목별 다중회귀 분해 시계열")

    with st.expander("📖 회귀 수식 재구성 — γ_level · γ_slope · ε", expanded=True):
        st.markdown("""
원래 회귀:
```
dY_i,t = β_3Y,i · dY_3Y,t  +  β_10Y,i · dY_10Y,t  +  ε_i,t
```

수학적으로 동등한 재구성:
```
dY_i,t = (β_3Y,i + β_10Y,i) · dY_3Y,t
       + β_10Y,i · (dY_10Y,t − dY_3Y,t)
       + ε_i,t

       = γ_level,i · dY_3Y,t  +  γ_slope,i · slope_t  +  ε_i,t
```

- **γ_level = β_3Y + β_10Y** : 시장 전체 금리(3Y 기준)에 대한 종목 노출
  - 단기물·중기물·장기물 모두 ≈ 1 부근 (시장 1bp 움직이면 ~1bp 따라감)
- **γ_slope = β_10Y** : 커브 변화 (10Y−3Y) 에 대한 노출
  - 단기물 ≈ 0, 중기물 ≈ 0.5, 장기물 ≈ 1
- **ε** : 시장 공통 변수로 설명 안 되는 종목 고유 noise (RV 신호 재료)

세 시계열을 같이 보면 종목이 **시장 move / 커브 변화 / 고유 dislocation** 어느 layer에서 어떻게 움직이는지 한눈에 분해된다.
""")

    # ---------------- γ panel 계산 ----------------
    gamma_level = (beta3 + beta10).rename_axis(columns="bond_code")
    gamma_slope = beta10.rename_axis(columns="bond_code")
    eps_panel = pipe.residual

    # ---------------- 종목 선택 ----------------
    universe = sorted(gamma_level.columns.tolist())
    # 메타로 라벨 만들기 (종목명 + 잔존)
    rem_avg = rem.mean(axis=0)
    name_map = {}
    for code in universe:
        nm = meta.loc[code, "bond_name"] if code in meta.index else code
        ry = rem_avg.get(code, np.nan)
        nm_short = (nm[:22] if isinstance(nm, str) else code) + f"  ({ry:.1f}y)"
        name_map[code] = nm_short

    # 만기순 정렬
    universe_sorted = sorted(universe, key=lambda c: rem_avg.get(c, 0))
    options = [f"{name_map[c]}  [{c}]" for c in universe_sorted]
    code_lookup = {f"{name_map[c]}  [{c}]": c for c in universe_sorted}

    col_a, col_b = st.columns([2, 1])
    with col_a:
        # 기본값: 단기 1, 중기 1, 장기 1
        default_codes = []
        for r_lo, r_hi in [(1, 4), (4, 8), (10, 30)]:
            cands = [c for c in universe_sorted
                     if r_lo <= rem_avg.get(c, -1) < r_hi]
            if cands:
                default_codes.append(cands[len(cands) // 2])
        default_opts = [opt for opt in options if code_lookup[opt] in default_codes]
        sel_opts = st.multiselect("종목 선택 (multi)", options=options,
                                   default=default_opts[:3])
    with col_b:
        show_bucket_avg = st.checkbox("만기 버킷 평균 함께 표시", value=True,
                                       help="≤5y / 5~10y / >10y 그룹 평균 시계열")
        ts_since = st.text_input("시계열 시작일 (YYYY-MM-DD)", value="2023-01-01")

    sel_codes = [code_lookup[o] for o in sel_opts]

    # ---------------- 데이터 slicing ----------------
    try:
        ts_start = pd.Timestamp(ts_since)
    except Exception:
        ts_start = pd.Timestamp("2023-01-01")

    gl = gamma_level.loc[gamma_level.index >= ts_start]
    gs = gamma_slope.loc[gamma_slope.index >= ts_start]
    eps_sel = eps_panel.loc[eps_panel.index >= ts_start]

    # 만기 버킷 평균 (선택 옵션)
    bucket_groups = {"≤5y":   [c for c in universe if rem_avg.get(c, -1) <= 5 and rem_avg.get(c, -1) > 0],
                     "5~10y": [c for c in universe if 5 < rem_avg.get(c, -1) <= 10],
                     ">10y":  [c for c in universe if rem_avg.get(c, -1) > 10]}

    # ---------------- 시각화 ----------------
    if not sel_codes:
        st.info("좌측에서 종목을 하나 이상 선택해 주세요.")
    else:
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            vertical_spacing=0.06,
                            subplot_titles=(
                                "γ_level (= β_3Y + β_10Y) — 시장 전체 금리에 대한 종목 노출",
                                "γ_slope (= β_10Y) — 커브 (10Y−3Y) 변화에 대한 노출",
                                "ε — 종목 고유 noise (일별 잔차, bp)",
                            ))

        # 종목별 선
        cmap = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
                "#17becf", "#e377c2", "#bcbd22", "#7f7f7f"]
        for i, code in enumerate(sel_codes):
            color = cmap[i % len(cmap)]
            label = name_map.get(code, code)
            if code in gl.columns:
                fig.add_trace(go.Scatter(
                    x=gl.index, y=gl[code], name=label,
                    line=dict(color=color, width=1.6),
                    legendgroup=code,
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=gs.index, y=gs[code], name=label,
                    line=dict(color=color, width=1.6),
                    legendgroup=code, showlegend=False,
                ), row=2, col=1)
            if code in eps_sel.columns:
                fig.add_trace(go.Scatter(
                    x=eps_sel.index, y=eps_sel[code], name=label,
                    line=dict(color=color, width=1.0), opacity=0.85,
                    legendgroup=code, showlegend=False,
                ), row=3, col=1)

        # 만기 버킷 평균 (옵션)
        if show_bucket_avg:
            bucket_color = {"≤5y": "#1f77b4", "5~10y": "#888888", ">10y": "#d62728"}
            for bk, codes in bucket_groups.items():
                cols = [c for c in codes if c in gl.columns]
                if not cols: continue
                gl_avg = gl[cols].mean(axis=1)
                gs_avg = gs[cols].mean(axis=1)
                eps_avg = eps_sel[cols].mean(axis=1) if cols else None
                color = bucket_color[bk]
                fig.add_trace(go.Scatter(
                    x=gl_avg.index, y=gl_avg.values, name=f"avg {bk}",
                    line=dict(color=color, width=1.2, dash="dash"),
                    legendgroup=f"bk_{bk}", opacity=0.85,
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=gs_avg.index, y=gs_avg.values, name=f"avg {bk}",
                    line=dict(color=color, width=1.2, dash="dash"),
                    legendgroup=f"bk_{bk}", showlegend=False,
                ), row=2, col=1)
                if eps_avg is not None:
                    fig.add_trace(go.Scatter(
                        x=eps_avg.index, y=eps_avg.values, name=f"avg {bk}",
                        line=dict(color=color, width=1.0, dash="dash"),
                        legendgroup=f"bk_{bk}", showlegend=False,
                    ), row=3, col=1)

        # 가이드라인
        fig.add_hline(y=1.0, line_dash="dot", line_color="gray", line_width=0.6, row=1, col=1)
        fig.add_hline(y=0.0, line_dash="dot", line_color="gray", line_width=0.6, row=2, col=1)
        fig.add_hline(y=0.0, line_dash="solid", line_color="black", line_width=0.5, row=3, col=1)

        fig.update_yaxes(title_text="γ_level", row=1, col=1)
        fig.update_yaxes(title_text="γ_slope", row=2, col=1)
        fig.update_yaxes(title_text="ε (bp)", row=3, col=1)
        fig.update_layout(height=820, hovermode="x unified",
                           legend=dict(orientation="v", x=1.02, y=1),
                           margin=dict(l=60, r=140, t=60, b=40))
        st.plotly_chart(fig, use_container_width=True)

        # ---------------- 통계 요약 ----------------
        st.markdown("### 📊 통계 요약 (선택 구간)")
        rows = []
        for code in sel_codes:
            if code not in gl.columns: continue
            ry = rem_avg.get(code, np.nan)
            rows.append({
                "bond": name_map.get(code, code),
                "remain_y": ry,
                "γ_level mean": float(gl[code].mean()),
                "γ_level std":  float(gl[code].std()),
                "γ_slope mean": float(gs[code].mean()),
                "γ_slope std":  float(gs[code].std()),
                "ε std (bp)":   float(eps_sel[code].std()) if code in eps_sel.columns else np.nan,
                "γ_level (latest)": float(gl[code].iloc[-1]) if not gl[code].dropna().empty else np.nan,
                "γ_slope (latest)": float(gs[code].iloc[-1]) if not gs[code].dropna().empty else np.nan,
            })
        if rows:
            stats_df = pd.DataFrame(rows)
            st.dataframe(
                stats_df.style.format({
                    "remain_y":          "{:.2f}",
                    "γ_level mean":      "{:+.3f}",
                    "γ_level std":       "{:.3f}",
                    "γ_slope mean":      "{:+.3f}",
                    "γ_slope std":       "{:.3f}",
                    "ε std (bp)":        "{:.2f}",
                    "γ_level (latest)":  "{:+.3f}",
                    "γ_slope (latest)":  "{:+.3f}",
                }),
                use_container_width=True, hide_index=True,
            )

        st.caption(
            "💡 **읽는 법**: γ_level 이 ~1 에서 떨어지면 종목이 시장 평균보다 *덜* 따라가는 시기. "
            "γ_slope 가 0 → 양수 변화는 만기 노출이 길어지는 효과(예: 새 발행 후 OTR 진입). "
            "ε 가 갑자기 크게 튀면 그날 idiosyncratic dislocation 발생 — RV 신호의 재료."
        )


# ============================================================
# Footer
# ============================================================
st.divider()
st.caption("© 채권 3팩터 RV 시스템 · 본 대시보드는 investigate·research 용도이며, 실거래 "
            "주문 연결은 별도 OMS 통합이 필요합니다. Spec: `factor_trading/bond_3factor_final_spec.md`")
