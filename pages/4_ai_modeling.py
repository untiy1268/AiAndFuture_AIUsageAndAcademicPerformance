import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. 페이지 설정 및 디자인 정의
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="학생 AI 활용 및 성적 예측 대시보드",
    page_icon="🎓",
    layout="wide"
)

# 글로벌 컬러 테마 정의
COLOR_MAIN = "#47B39C"      # 메인 테마 (Mint/Emerald)
COLOR_SECOND = "#1F77B4"    # 보조 색상 (Classic Blue)
COLOR_BASELINE = "#8C969E"  # 기준선 (Muted Grey)
COLOR_HIGHLIGHT = "#FF6B6B" # 하이라이트 (Warm Coral)

st.title("🎓 학생 AI 활용 양상 분석 및 성적 변화 예측 대시보드")
st.markdown("""
이 대시보드는 학생들의 AI 도구 활용 데이터셋을 분석하고, 머신러닝 모델을 통해 성적 향상 요인을 예측합니다.
Scikit-Learn의 정교한 **전처리 파이프라인(Pipeline)** 기법을 사용하여 데이터 변환과 예측 모델을 결합했습니다.
""")

# -----------------------------------------------------------------------------
# 2. 데이터 로드 및 결측치 처리 (정제된 데이터셋만 사용)
# -----------------------------------------------------------------------------
@st.cache_data
def load_dataset():
    if not os.path.exists("cleaned_student_ai_data.csv"):
        st.error("데이터 파일을 찾을 수 없습니다. 'cleaned_student_ai_data.csv' 파일 경로를 확인해주세요.")
        st.stop()

    df = pd.read_csv("cleaned_student_ai_data.csv")

    # 결측치 처리 및 완벽한 타입 캐스팅 안전장치
    # ⚠️ pandas 2.x 이상에서는 문자열 컬럼이 'object'가 아닌 'str'/'string' dtype으로
    # 표시될 수 있으므로, is_numeric_dtype으로 "숫자가 아니면 범주형"으로 판단한다.
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna("None").astype(str)

    return df

df_active_original = load_dataset()

# -----------------------------------------------------------------------------
# 3. 사이드바 제어 패널 및 상호작용 선택 상자 로직
# -----------------------------------------------------------------------------
st.sidebar.header("🎛️ 제어 및 입력 패널")

df_active = df_active_original.copy()
dataset_key = "cleaned"
feature_cols = ["age", "education_level", "study_hours_per_day", "ai_tools_used", "purpose_of_ai", "grades_before_ai", "daily_screen_time_hours"]
theme_color = COLOR_MAIN
target_col = "grades_after_ai"

st.sidebar.markdown("---")
st.sidebar.subheader("🔮 개인 맞춤형 성적 예측 입력")

# 학생 특성 기본 입력
age = st.sidebar.number_input("나이 (age)", min_value=10, max_value=30, value=17)
education_level = st.sidebar.selectbox("교육 수준 (education_level)", options=sorted(list(df_active["education_level"].unique())))
study_hours = st.sidebar.slider("일평균 자습 시간 (study_hours_per_day)", 0.0, 12.0, 3.0, 0.5)
grades_before = st.sidebar.slider("AI 사용 전 성적 (grades_before_ai)", 0, 100, 65, 1)
screen_time = st.sidebar.slider("일평균 스크린 타임 (daily_screen_time_hours)", 0, 24, 4, 1)

# ★ 선택 상자 관련 고도화 상호작용 로직 적용 ★
uses_ai_sim = st.sidebar.selectbox("AI 도구 보유 여부", options=["예 (도구 선택)", "아니오 (사용 안 함)"])
if uses_ai_sim == "아니오 (사용 안 함)":
    st.sidebar.info("ℹ️ AI 도구를 사용하지 않는 학생으로 자동 매핑됩니다.")
    actual_ai_tool = "None"
    actual_ai_purpose = "None"
else:
    ai_tools_options = [opt for opt in df_active["ai_tools_used"].unique() if opt != "None"]
    purpose_options = [opt for opt in df_active["purpose_of_ai"].unique() if opt != "None"]

    actual_ai_tool = st.sidebar.selectbox("사용 AI 도구 (ai_tools_used)", options=ai_tools_options if ai_tools_options else ["ChatGPT"])
    actual_ai_purpose = st.sidebar.selectbox("AI 사용 목적 (purpose_of_ai)", options=purpose_options if purpose_options else ["Homework"])

# -----------------------------------------------------------------------------
# 4. 머신러닝 파이프라인 학습 및 평가 로직
# -----------------------------------------------------------------------------
MODEL_REGISTRY = {
    "Linear Regression": {
        "model_class": LinearRegression,
        "params": {}
    },
    "Random Forest": {
        "model_class": RandomForestRegressor,
        "params": {"n_estimators": 100, "random_state": 42}
    }
}

@st.cache_resource(show_spinner="모델을 학습하는 중입니다...")
def train_and_cache_pipelines(dataset_key: str, _df, feature_cols, target_col):
    # 1. 안전한 연산을 위해 독립된 데이터프레임 복사본 생성
    _df = _df.copy()

    # 2. 카테고리형 변수와 수치형 변수 자동 식별
    # ⚠️ pandas 2.x 이상에서는 문자열 컬럼의 dtype이 'object'가 아니라 'str'/'string'으로
    # 표시될 수 있어 dtype == 'object' 비교만으로는 범주형 컬럼을 놓칠 수 있다.
    # is_numeric_dtype으로 "숫자가 아니면 범주형"으로 판단해야 pandas 버전에 관계없이 안전하다.
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(_df[c])]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    # 범주형 컬럼 내에 숨어있을 수 있는 숫자형/Float 요소를 완전히 str로 직렬화
    for c in cat_cols:
        _df[c] = _df[c].astype(str).fillna("None")

    # 수치형 변수 내부의 텍스트 오염 예방
    for c in num_cols:
        _df[c] = pd.to_numeric(_df[c], errors='coerce').fillna(0)

    X = _df[feature_cols]
    y = pd.to_numeric(_df[target_col], errors='coerce').fillna(0)

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 수동 매핑을 대체하는 깔끔한 ColumnTransformer 파이프라인 구성
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
            ('num', 'passthrough', num_cols)
        ]
    )

    trained_pipelines = {}
    results = {}

    # 메인 예측 모델 파이프라인 학습 루프
    # 💡 st.cache_resource가 코드/데이터 변경 시 캐시를 자동으로 무효화해주므로,
    # 예전처럼 디스크에 joblib 파일로 저장해두고 재사용하지 않는다.
    # (스키마가 바뀐 뒤에도 예전 파일을 그대로 불러오는 게 실시간 예측 오류의 원인이었다.)
    for model_name, info in MODEL_REGISTRY.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', info["model_class"](**info["params"]))
        ])
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        trained_pipelines[model_name] = pipeline
        results[model_name] = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "preds": preds}

    return trained_pipelines, results, y_test, cat_cols, num_cols

trained_models, evaluation_results, y_test_actual, cat_cols, num_cols = train_and_cache_pipelines(
    dataset_key, df_active, feature_cols, target_col
)

# -----------------------------------------------------------------------------
# 5. 시각화 그래프 강화 및 대시보드 화면 구성
# -----------------------------------------------------------------------------
tabs = st.tabs(["📊 데이터 통계 및 탐색", "📈 모델 성능 및 분석", "🔮 실시간 맞춤 성적 예측"])

# --- Tab 1: 데이터 탐색 ---
with tabs[0]:
    st.subheader("📋 정제된 학생 AI 활용 데이터 기본 정보")

    col1, col2, col3 = st.columns(3)
    col1.metric("총 학생 수 (행 개수)", f"{len(df_active)} 명")
    col2.metric("사용된 예측 피처 수", f"{len(feature_cols)} 개")
    col3.metric("AI 도입 전 평균 성적", f"{df_active['grades_before_ai'].mean():.1f} 점")

    st.dataframe(df_active.head(10), width='stretch')

    st.subheader("📊 주요 변수별 분포 시각화")
    c1, c2 = st.columns(2)

    with c1:
        fig_hist = px.histogram(
            df_active, x="grades_after_ai", color="education_level", barmode="overlay",
            title="📈 AI 사용 후 성적(Target) 분포 현황",
            labels={"grades_after_ai": "성적 (점)", "count": "학생 수"},
            color_discrete_sequence=[COLOR_SECOND, COLOR_HIGHLIGHT],
            template="plotly_white"
        )
        fig_hist.update_layout(margin=dict(l=40, r=40, t=50, b=40))
        st.plotly_chart(fig_hist, width='stretch')

    with c2:
        fig_box = px.box(
            df_active, x="ai_tools_used", y="grades_after_ai", color="ai_tools_used",
            title="🎯 사용 AI 도구별 성적 분포 편차",
            labels={"ai_tools_used": "AI 도구", "grades_after_ai": "AI 활용 후 성적"},
            template="plotly_white"
        )
        fig_box.update_layout(showlegend=False, margin=dict(l=40, r=40, t=50, b=40))
        st.plotly_chart(fig_box, width='stretch')

# --- Tab 2: 모델 성능 및 분석 ---
with tabs[1]:
    st.subheader("📈 모델 성능 비교 및 심층 분석")

    st.markdown("#### 📊 모델별 성능 비교")

    metric_rows = []
    for model_name, res in evaluation_results.items():
        metric_rows.append({"모델": model_name, "MSE": res["MSE"], "RMSE": res["RMSE"], "MAE": res["MAE"], "R2": res["R2"]})
    metric_df = pd.DataFrame(metric_rows)

    metric_cols = st.columns(len(metric_df))
    for mcol, (_, row) in zip(metric_cols, metric_df.iterrows()):
        with mcol:
            st.metric(row["모델"], f"R² {row['R2']:.3f}", f"RMSE {row['RMSE']:.2f}")

    st.dataframe(
        metric_df.style.format({"MSE": "{:.2f}", "RMSE": "{:.2f}", "MAE": "{:.2f}", "R2": "{:.3f}"}),
        width='stretch'
    )

    st.caption("MSE·RMSE·MAE는 낮을수록, R²는 1에 가까울수록 예측이 정확하다는 의미입니다.")

    m1, m2 = st.columns(2)
    with m1:
        fig_mse = px.bar(
            metric_df, x="모델", y="MSE", color="모델",
            title="📉 모델별 MSE 비교 (낮을수록 좋음)",
            color_discrete_sequence=[COLOR_SECOND, COLOR_MAIN],
            template="plotly_white"
        )
        fig_mse.update_layout(showlegend=False, margin=dict(l=40, r=40, t=50, b=40))
        st.plotly_chart(fig_mse, width='stretch')
    with m2:
        fig_mae = px.bar(
            metric_df, x="모델", y="MAE", color="모델",
            title="📉 모델별 MAE 비교 (낮을수록 좋음)",
            color_discrete_sequence=[COLOR_SECOND, COLOR_MAIN],
            template="plotly_white"
        )
        fig_mae.update_layout(showlegend=False, margin=dict(l=40, r=40, t=50, b=40))
        st.plotly_chart(fig_mae, width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        fig_rmse = px.bar(
            metric_df, x="모델", y="RMSE", color="모델",
            title="📉 모델별 RMSE 비교 (낮을수록 좋음)",
            color_discrete_sequence=[COLOR_SECOND, COLOR_MAIN],
            template="plotly_white"
        )
        fig_rmse.update_layout(showlegend=False, margin=dict(l=40, r=40, t=50, b=40))
        st.plotly_chart(fig_rmse, width='stretch')
    with c2:
        fig_r2 = px.bar(
            metric_df, x="모델", y="R2", color="모델",
            title="📈 모델별 R² 비교 (높을수록 좋음)",
            color_discrete_sequence=[COLOR_SECOND, COLOR_MAIN],
            template="plotly_white"
        )
        fig_r2.update_layout(showlegend=False, margin=dict(l=40, r=40, t=50, b=40))
        st.plotly_chart(fig_r2, width='stretch')

    st.markdown("---")
    st.markdown("#### 🎯 실제값 vs 예측값 비교")

    selected_model_for_analysis = st.selectbox(
        "분석할 모델 선택", options=list(evaluation_results.keys()), key="analysis_model_select"
    )

    preds = evaluation_results[selected_model_for_analysis]["preds"]

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=y_test_actual, y=preds, mode="markers",
        marker=dict(color=theme_color, size=8, opacity=0.6),
        name="예측 결과"
    ))
    min_val = float(min(y_test_actual.min(), preds.min()))
    max_val = float(max(y_test_actual.max(), preds.max()))
    fig_scatter.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val], mode="lines",
        line=dict(color=COLOR_HIGHLIGHT, dash="dash"), name="완벽한 예측선"
    ))
    fig_scatter.update_layout(
        title=f"🔍 {selected_model_for_analysis}: 실제 성적 vs 예측 성적",
        xaxis_title="실제 성적", yaxis_title="예측 성적",
        template="plotly_white", margin=dict(l=40, r=40, t=50, b=40)
    )
    st.plotly_chart(fig_scatter, width='stretch')

    st.markdown("---")
    st.markdown("#### 🧬 피처 중요도 / 회귀 계수 분석")

    feature_names_ordered = cat_cols + num_cols
    pipeline_for_analysis = trained_models[selected_model_for_analysis]
    regressor = pipeline_for_analysis.named_steps["regressor"]

    if hasattr(regressor, "feature_importances_"):
        importance_df = pd.DataFrame({
            "피처": feature_names_ordered,
            "중요도": regressor.feature_importances_
        }).sort_values("중요도", ascending=True)
        fig_importance = px.bar(
            importance_df, x="중요도", y="피처", orientation="h",
            title=f"🌲 {selected_model_for_analysis} 피처 중요도",
            color="중요도", color_continuous_scale=[COLOR_BASELINE, theme_color],
            template="plotly_white"
        )
        fig_importance.update_layout(margin=dict(l=40, r=40, t=50, b=40))
        st.plotly_chart(fig_importance, width='stretch')
    elif hasattr(regressor, "coef_"):
        coef_df = pd.DataFrame({
            "피처": feature_names_ordered,
            "계수": regressor.coef_
        }).sort_values("계수", ascending=True)
        fig_coef = px.bar(
            coef_df, x="계수", y="피처", orientation="h",
            title=f"📐 {selected_model_for_analysis} 회귀 계수",
            color="계수", color_continuous_scale=[COLOR_HIGHLIGHT, theme_color],
            template="plotly_white"
        )
        fig_coef.update_layout(margin=dict(l=40, r=40, t=50, b=40))
        st.plotly_chart(fig_coef, width='stretch')
    else:
        st.info("이 모델은 피처 중요도/계수 정보를 제공하지 않습니다.")

# --- Tab 3: 실시간 맞춤 성적 예측 ---
with tabs[2]:
    st.subheader("🔮 입력하신 정보를 바탕으로 한 성적 예측 결과")
    st.markdown("좌측 사이드바에서 입력한 학생 정보를 바탕으로, 학습된 머신러닝 모델이 AI 도구 활용 후 예상 성적을 예측합니다.")

    model_choice = st.radio(
        "예측에 사용할 모델 선택", options=list(trained_models.keys()), horizontal=True
    )

    input_df = pd.DataFrame([{
        "age": age,
        "education_level": education_level,
        "study_hours_per_day": study_hours,
        "ai_tools_used": actual_ai_tool,
        "purpose_of_ai": actual_ai_purpose,
        "grades_before_ai": grades_before,
        "daily_screen_time_hours": screen_time
    }])[feature_cols]

    # 학습 시점과 동일한 dtype으로 맞춰 예측 오류를 예방 (범주형→문자열, 수치형→숫자)
    for c in cat_cols:
        input_df[c] = input_df[c].astype(str)
    for c in num_cols:
        input_df[c] = pd.to_numeric(input_df[c], errors='coerce').fillna(0)

    selected_pipeline = trained_models[model_choice]
    predicted_grade = selected_pipeline.predict(input_df)[0]
    predicted_grade_clipped = float(np.clip(predicted_grade, 0, 100))
    delta = predicted_grade_clipped - grades_before

    pcol1, pcol2, pcol3 = st.columns(3)
    pcol1.metric("AI 사용 전 입력 성적", f"{grades_before} 점")
    pcol2.metric("🔮 예측된 AI 사용 후 성적", f"{predicted_grade_clipped:.1f} 점", f"{delta:+.1f} 점")
    pcol3.metric("사용 모델", model_choice)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=predicted_grade_clipped,
        delta={"reference": grades_before, "increasing": {"color": COLOR_MAIN}, "decreasing": {"color": COLOR_HIGHLIGHT}},
        title={"text": "예측 성적 (100점 만점)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": theme_color},
            "steps": [
                {"range": [0, 60], "color": "#F0F2F5"},
                {"range": [60, 80], "color": "#E1E8ED"},
                {"range": [80, 100], "color": "#D0DDE5"}
            ],
            "threshold": {"line": {"color": COLOR_HIGHLIGHT, "width": 4}, "value": grades_before}
        }
    ))
    fig_gauge.update_layout(margin=dict(l=40, r=40, t=60, b=20))
    st.plotly_chart(fig_gauge, width='stretch')

    st.markdown("---")
    st.markdown("#### 📝 입력된 학생 정보 요약")
    st.dataframe(input_df, width='stretch')

    with st.expander("ℹ️ 예측 신뢰도 참고 정보"):
        res = evaluation_results[model_choice]
        st.write(
            f"이 예측에 사용된 **{model_choice}** 모델의 테스트 데이터 성능은 "
            f"MSE **{res['MSE']:.2f}**, RMSE **{res['RMSE']:.2f}**, MAE **{res['MAE']:.2f}**, R² **{res['R2']:.3f}** 입니다."
        )
        st.caption("MSE·RMSE·MAE는 0에 가까울수록, R²는 1에 가까울수록 예측 정확도가 높다는 의미입니다.")
