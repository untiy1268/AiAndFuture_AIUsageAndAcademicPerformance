# ============================================================
# AI 성적 예측 웹앱 (Streamlit + Plotly)
# 목적: 학생의 AI 사용 및 학습 데이터를 기반으로
#       AI 사용 후 성적(grades_after_ai)을 예측한다
# 사용 모델: Linear Regression, Random Forest Regressor
# 데이터: students_ai_usage.csv (100행)
# 변경사항: matplotlib/seaborn → Plotly (동적 인터랙티브 그래프)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import os

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# 0. CSV 실제 값 기반 고정 매핑 (LabelEncoder 대신 사용)
# ============================================================
CATEGORY_MAPS = {
    "education_level": {"school": 0, "college": 1},
    "uses_ai":         {"No": 0,     "Yes": 1},
    "ai_tools_used":   {"None": 0, "ChatGPT": 1, "Copilot": 2, "Gemini": 3},
    "purpose_of_ai":   {"None": 0, "Coding": 1,  "Homework": 2, "Research": 3},
}

# ============================================================
# 1. 모델 관리 딕셔너리
# ============================================================
MODEL_REGISTRY = {
    "Linear Regression": {
        "model_class": LinearRegression,
        "model_file":  "model_linear.pkl",
        "params":      {},
        "color":       "#4C9BE8",   # 모델별 색상 (Plotly 그래프 일관성)
    },
    "Random Forest": {
        "model_class": RandomForestRegressor,
        "model_file":  "model_rf.pkl",
        "params":      {"n_estimators": 100, "random_state": 42},
        "color":       "#E8714C",
    },
}

# ============================================================
# 2. 유틸리티 함수
# ============================================================

@st.cache_data
def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["ai_tools_used"] = df["ai_tools_used"].fillna("None")
    df["purpose_of_ai"] = df["purpose_of_ai"].fillna("None")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, mapping in CATEGORY_MAPS.items():
        df[col] = df[col].map(mapping)
        df[col] = df[col].fillna(0).astype(int)
    return df


def load_or_create_model(name: str):
    info = MODEL_REGISTRY[name]
    if os.path.exists(info["model_file"]):
        model = joblib.load(info["model_file"])
        st.sidebar.success(f"✅ {name}: 저장된 모델 불러옴")
    else:
        model = info["model_class"](**info["params"])
    return model


def train_and_save(model, X_train, y_train, filepath: str):
    model.fit(X_train, y_train)
    joblib.dump(model, filepath)
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    r2    = r2_score(y_test, preds)
    return {"RMSE": round(rmse, 4), "R² Score": round(r2, 4), "predictions": preds}


# ============================================================
# 3. Plotly 그래프 생성 함수
# ============================================================

def plot_actual_vs_predicted(results: dict, y_test) -> go.Figure:
    """
    실제값 vs 예측값 Scatter Plot (Plotly)
    - 마우스 호버 시 실제/예측 점수 표시
    - Perfect Fit 기준선 포함
    - 각 모델 색상 구분
    """
    fig = go.Figure()

    all_vals = list(y_test)
    for mname, res in results.items():
        all_vals.extend(res["predictions"].tolist())
    lo, hi = min(all_vals) - 1, max(all_vals) + 1

    # 모델별 Scatter
    for mname, res in results.items():
        color = MODEL_REGISTRY[mname]["color"]
        fig.add_trace(go.Scatter(
            x=list(y_test),
            y=list(res["predictions"]),
            mode="markers",
            name=mname,
            marker=dict(color=color, size=8, opacity=0.7,
                        line=dict(width=0.5, color="white")),
            hovertemplate=(
                f"<b>{mname}</b><br>"
                "실제: %{x:.1f}점<br>"
                "예측: %{y:.1f}점<br>"
                "<extra></extra>"
            ),
        ))

    # Perfect Fit 기준선
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines",
        name="Perfect Fit",
        line=dict(color="gray", dash="dash", width=1.5),
        hoverinfo="skip",
    ))

    fig.update_layout(
        title=dict(text="실제 성적 vs 예측 성적", font=dict(size=15)),
        xaxis_title="실제 grades_after_ai",
        yaxis_title="예측 grades_after_ai",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
    )
    return fig


def plot_feature_importance(rf_model, feature_cols: list) -> go.Figure:
    """
    Random Forest Feature Importance 수평 바 차트 (Plotly)
    - 호버 시 중요도 % 표시
    - 색상 그라데이션 (중요도 순)
    """
    importances = rf_model.feature_importances_
    feat_df = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=True)
    )

    fig = go.Figure(go.Bar(
        x=feat_df["importance"],
        y=feat_df["feature"],
        orientation="h",
        marker=dict(
            color=feat_df["importance"],
            colorscale="Blues",
            showscale=False,
            line=dict(width=0),
        ),
        hovertemplate="<b>%{y}</b><br>중요도: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Feature Importance (Random Forest)", font=dict(size=15)),
        xaxis_title="Importance",
        yaxis_title="",
        height=400,
        margin=dict(l=40, r=20, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
    )
    return fig


def plot_residuals(results: dict, y_test) -> go.Figure:
    """
    잔차(Residual) 분포 히스토그램 (Plotly) — 추가 동적 시각화
    - 예측 오차의 분포를 모델별로 오버레이하여 비교
    """
    fig = go.Figure()

    for mname, res in results.items():
        residuals = res["predictions"] - np.array(y_test)
        color = MODEL_REGISTRY[mname]["color"]
        fig.add_trace(go.Histogram(
            x=residuals,
            name=mname,
            opacity=0.6,
            marker_color=color,
            nbinsx=15,
            hovertemplate="잔차: %{x:.2f}<br>빈도: %{y}<extra></extra>",
        ))

    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1.5)

    fig.update_layout(
        title=dict(text="잔차(Residual) 분포", font=dict(size=15)),
        xaxis_title="예측값 - 실제값",
        yaxis_title="빈도",
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=40, r=20, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
    )
    return fig


def plot_performance_bar(results: dict) -> go.Figure:
    """
    모델별 RMSE / R² 비교 이중 축 바 차트 (Plotly) — 추가 동적 시각화
    """
    models = list(results.keys())
    rmses  = [results[m]["RMSE"]     for m in models]
    r2s    = [results[m]["R² Score"] for m in models]
    colors = [MODEL_REGISTRY[m]["color"] for m in models]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(name="RMSE", x=models, y=rmses,
               marker_color=colors, opacity=0.85,
               hovertemplate="<b>%{x}</b><br>RMSE: %{y:.4f}<extra></extra>"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(name="R² Score", x=models, y=r2s,
                   mode="markers+lines",
                   marker=dict(size=12, symbol="diamond",
                               color=colors, line=dict(width=1.5, color="white")),
                   line=dict(dash="dot", width=2, color="gray"),
                   hovertemplate="<b>%{x}</b><br>R²: %{y:.4f}<extra></extra>"),
        secondary_y=True,
    )

    fig.update_layout(
        title=dict(text="모델 성능 비교 (RMSE & R²)", font=dict(size=15)),
        height=360,
        margin=dict(l=40, r=60, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
    )
    fig.update_yaxes(title_text="RMSE (낮을수록 좋음)", secondary_y=False,
                     showgrid=True, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(title_text="R² Score (높을수록 좋음)", secondary_y=True,
                     showgrid=False, range=[0, 1.1])
    return fig


# ============================================================
# 4. Streamlit 앱 메인
# ============================================================

def main():
    st.set_page_config(page_title="AI 성적 예측기", page_icon="🎓", layout="wide")

    st.title("🎓 AI 사용 후 성적 예측 웹앱")
    st.markdown(
        "학생의 학습 정보와 AI 사용 패턴을 입력하면 "
        "**AI 사용 후 성적(grades_after_ai)** 을 예측합니다. "
        "모든 그래프는 **Plotly** 기반으로 동적 인터랙션을 지원합니다."
    )
    st.divider()

    # ── 데이터 로드 ───────────────────────────────────────────
    CSV_PATH = "students_ai_usage.csv"
    if not os.path.exists(CSV_PATH):
        st.error(f"❌ '{CSV_PATH}' 파일을 찾을 수 없습니다. 앱과 같은 폴더에 파일을 넣어주세요.")
        st.stop()

    df_raw = load_data(CSV_PATH)

    # ── 피처 / 타겟 정의 ─────────────────────────────────────
    FEATURE_COLS = [
        "age", "education_level", "study_hours_per_day", "uses_ai",
        "ai_tools_used", "purpose_of_ai", "grades_before_ai", "daily_screen_time_hours",
    ]
    TARGET_COL = "grades_after_ai"

    df_processed = preprocess(df_raw[FEATURE_COLS + [TARGET_COL]])
    X = df_processed[FEATURE_COLS]
    y = df_processed[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── 모델 학습 또는 로드 ───────────────────────────────────
    trained_models = {}
    results        = {}

    for model_name, info in MODEL_REGISTRY.items():
        model = load_or_create_model(model_name)
        if not os.path.exists(info["model_file"]):
            model = train_and_save(model, X_train, y_train, info["model_file"])
            st.sidebar.info(f"🔄 {model_name}: 새로 학습 후 저장됨")
        trained_models[model_name] = model
        results[model_name]        = evaluate_model(model, X_test, y_test)

    # ============================================================
    # 섹션 1: 모델 성능 비교
    # ============================================================
    st.header("📊 모델 성능 비교")

    df_cmp = pd.DataFrame({
        "모델":                    list(results.keys()),
        "RMSE (낮을수록 좋음)":    [results[m]["RMSE"]     for m in results],
        "R² Score (높을수록 좋음)":[results[m]["R² Score"] for m in results],
    }).set_index("모델")
    st.dataframe(df_cmp, use_container_width=True)

    # 성능 비교 바 차트 (Plotly)
    st.plotly_chart(plot_performance_bar(results), use_container_width=True)

    # ============================================================
    # 섹션 2: 시각화 (모두 Plotly)
    # ============================================================
    st.header("📈 시각화")

    col1, col2 = st.columns(2)

    with col1:
        # (a) 실제값 vs 예측값
        st.plotly_chart(
            plot_actual_vs_predicted(results, y_test),
            use_container_width=True,
        )

    with col2:
        # (b) Feature Importance
        rf_model = trained_models.get("Random Forest")
        if rf_model and hasattr(rf_model, "feature_importances_"):
            st.plotly_chart(
                plot_feature_importance(rf_model, FEATURE_COLS),
                use_container_width=True,
            )
        else:
            st.warning("Random Forest 모델이 없거나 feature_importances_ 를 계산할 수 없습니다.")

    # (c) 잔차 분포 — 추가 동적 그래프
    st.plotly_chart(plot_residuals(results, y_test), use_container_width=True)

    # ============================================================
    # 섹션 3: 사용자 입력 → 성적 예측
    # ============================================================
    st.divider()
    st.header("🔮 성적 예측하기")
    st.markdown("아래에 학생 정보를 입력한 뒤 **Predict** 버튼을 눌러 예측 성적을 확인하세요.")

    inp1, inp2 = st.columns(2)

    with inp1:
        age = st.slider("나이 (age)", min_value=14, max_value=19, value=17, step=1)
        education_level = st.selectbox(
            "교육 수준 (education_level)", options=["school", "college"]
        )
        study_hours = st.slider(
            "하루 공부 시간 (study_hours_per_day)",
            min_value=1.0, max_value=5.0, value=3.0, step=0.1
        )
        uses_ai = st.selectbox("AI 사용 여부 (uses_ai)", options=["Yes", "No"])

    with inp2:
        ai_disabled = (uses_ai == "No")
        ai_tools_used = st.selectbox(
            "사용 AI 도구 (ai_tools_used)",
            options=["ChatGPT", "Copilot", "Gemini"],
            disabled=ai_disabled,
            help="AI를 사용하지 않으면 선택이 비활성화됩니다."
        )
        purpose_of_ai = st.selectbox(
            "AI 사용 목적 (purpose_of_ai)",
            options=["Coding", "Homework", "Research"],
            disabled=ai_disabled,
            help="AI를 사용하지 않으면 선택이 비활성화됩니다."
        )
        grades_before = st.slider(
            "AI 사용 전 성적 (grades_before_ai)",
            min_value=55, max_value=75, value=65, step=1
        )
        screen_time = st.slider(
            "하루 스크린 타임 (daily_screen_time_hours)",
            min_value=2, max_value=7, value=4, step=1
        )

    if st.button("🚀 Predict", type="primary", use_container_width=True):
        actual_ai_tool    = "None" if uses_ai == "No" else ai_tools_used
        actual_ai_purpose = "None" if uses_ai == "No" else purpose_of_ai

        user_input = pd.DataFrame([{
            "age":                     age,
            "education_level":         education_level,
            "study_hours_per_day":     study_hours,
            "uses_ai":                 uses_ai,
            "ai_tools_used":           actual_ai_tool,
            "purpose_of_ai":           actual_ai_purpose,
            "grades_before_ai":        grades_before,
            "daily_screen_time_hours": screen_time,
        }])

        user_processed = preprocess(user_input)

        st.divider()
        st.subheader("🎯 예측 결과")

        # 예측값 수집
        pred_values = {}
        for mname, model in trained_models.items():
            pred = float(np.clip(model.predict(user_processed[FEATURE_COLS])[0], 0, 100))
            pred_values[mname] = pred

        # st.metric 카드
        res_cols = st.columns(len(trained_models))
        for col, (mname, pred) in zip(res_cols, pred_values.items()):
            with col:
                st.metric(
                    label=f"📌 {mname}",
                    value=f"{pred:.1f} 점",
                    delta=f"{pred - grades_before:+.1f} 점 (AI 사용 전 대비)",
                )

        # ── 예측값 게이지 차트 (Plotly) ──────────────────────
        gauge_cols = st.columns(len(trained_models))
        for col, (mname, pred) in zip(gauge_cols, pred_values.items()):
            color = MODEL_REGISTRY[mname]["color"]
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pred,
                delta={"reference": grades_before, "valueformat": ".1f"},
                title={"text": mname, "font": {"size": 14}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar":  {"color": color},
                    "steps": [
                        {"range": [0,  60], "color": "rgba(255,100,100,0.15)"},
                        {"range": [60, 80], "color": "rgba(255,200,100,0.15)"},
                        {"range": [80,100], "color": "rgba(100,200,100,0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": "gray", "width": 2},
                        "thickness": 0.75,
                        "value": grades_before,
                    },
                },
                number={"suffix": "점", "valueformat": ".1f"},
            ))
            fig_gauge.update_layout(
                height=280,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            with col:
                st.plotly_chart(fig_gauge, use_container_width=True)

        st.success("✅ 예측이 완료되었습니다!")

    # ============================================================
    # 섹션 4: 데이터 미리보기
    # ============================================================
    with st.expander("📂 학습 데이터 미리보기 (상위 10행)"):
        st.dataframe(df_raw.head(10), use_container_width=True)
        st.caption(f"전체 데이터: {df_raw.shape[0]}행 × {df_raw.shape[1]}열")


if __name__ == "__main__":
    main()
