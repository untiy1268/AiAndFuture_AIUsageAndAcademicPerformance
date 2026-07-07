import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
# 1. 페이지 설정 및 디자인 정의
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="학생 AI 활용 및 성적 예측 대시보드",
    page_icon="🎓",
    layout="wide"
)

# 글로벌 컬러 테마 정의
COLOR_PRE = "#1F77B4"       # 전처리 전 (Classic Blue)
COLOR_POST = "#47B39C"      # 전처리 후 (Mint/Emerald)
COLOR_BASELINE = "#8C969E"  # 기준선 (Muted Grey)
COLOR_HIGHLIGHT = "#FF6B6B" # 하이라이트 (Warm Coral)

st.title("🎓 학생 AI 활용 양상 분석 및 성적 변화 예측 대시보드")
st.markdown("""
이 대시보드는 학생들의 AI 도구 활용 데이터셋을 분석하고, 머신러닝 모델을 통해 성적 향상 요인을 예측합니다.
Scikit-Learn의 정교한 **전처리 파이프라인(Pipeline)** 기법을 사용하여 데이터 변환과 예측 모델을 결합했습니다.
""")

# -----------------------------------------------------------------------------
# 2. 데이터 로드 및 결측치 처리
# -----------------------------------------------------------------------------
@st.cache_data
def load_datasets():
    if os.path.exists("students_ai_usage.csv"):
        df_raw = pd.read_csv("students_ai_usage.csv")
    elif os.path.exists("cleaned_student_ai_data.csv"):
        df_raw = pd.read_csv("cleaned_student_ai_data.csv")
        if "uses_ai" not in df_raw.columns:
            df_raw["uses_ai"] = df_raw["ai_tools_used"].apply(lambda x: "No" if pd.isna(x) or x == "None" else "Yes")
    else:
        st.error("데이터 파일을 찾을 수 없습니다. 'cleaned_student_ai_data.csv' 파일 경로를 확인해주세요.")
        st.stop()
        
    if os.path.exists("cleaned_student_ai_data.csv"):
        df_after = pd.read_csv("cleaned_student_ai_data.csv")
    else:
        df_after = df_raw.copy()

    # ✨ [수정] 결측치 처리 및 완벽한 타입 캐스팅 안전장치 추가
    for df in [df_raw, df_after]:
        for col in df.columns:
            if df[col].dtype == 'object':
                # 결측치는 'None' 문자열로 바꾸고, 전체 데이터를 str 타입으로 강제 통일
                df[col] = df[col].fillna("None").astype(str)
            else:
                df[col] = df[col].fillna(0)
                
    return df_raw, df_after

df_raw_original, df_after_original = load_datasets()

# -----------------------------------------------------------------------------
# 3. 사이드바 제어 패널 및 상호작용 선택 상자 로직
# -----------------------------------------------------------------------------
st.sidebar.header("🎛️ 제어 및 입력 패널")

dataset_option = st.sidebar.radio(
    "📊 분석에 사용할 데이터셋 선택",
    options=["전처리 전 원본 데이터 (Dataset 1)", "전처리 후 정제 데이터 (Dataset 2)"]
)

# 선택된 데이터셋에 따른 타겟 및 피처 정의
if dataset_option == "전처리 전 원본 데이터 (Dataset 1)":
    df_active = df_raw_original.copy()
    dataset_key = "raw"
    RAW_FEATURE_COLS = ["age", "education_level", "study_hours_per_day", "uses_ai", "ai_tools_used", "purpose_of_ai", "grades_before_ai", "daily_screen_time_hours"]
    feature_cols = RAW_FEATURE_COLS
    theme_color = COLOR_PRE
else:
    df_active = df_after_original.copy()
    dataset_key = "after"
    AFTER_FEATURE_COLS = ["age", "education_level", "study_hours_per_day", "ai_tools_used", "purpose_of_ai", "grades_before_ai", "daily_screen_time_hours"]
    feature_cols = AFTER_FEATURE_COLS
    theme_color = COLOR_POST

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
if "uses_ai" in feature_cols:
    uses_ai = st.sidebar.selectbox("AI 사용 여부 (uses_ai)", options=["Yes", "No"])
    
    if uses_ai == "No":
        st.sidebar.info("ℹ️ AI를 사용하지 않으므로 아래 도구 및 목적이 자동으로 'None'으로 제한됩니다.")
        actual_ai_tool = "None"
        actual_ai_purpose = "None"
    else:
        # AI를 사용하는 경우에만 선택 상자 활성화 및 옵션 노출
        ai_tools_options = [opt for opt in df_active["ai_tools_used"].unique() if opt != "None"]
        purpose_options = [opt for opt in df_active["purpose_of_ai"].unique() if opt != "None"]
        
        actual_ai_tool = st.sidebar.selectbox("사용 AI 도구 (ai_tools_used)", options=ai_tools_options if ai_tools_options else ["ChatGPT"])
        actual_ai_purpose = st.sidebar.selectbox("AI 사용 목적 (purpose_of_ai)", options=purpose_options if purpose_options else ["Homework"])
else:
    # Dataset 2(전처리 후)는 uses_ai 컬럼이 없으므로 직접 AI 도구 상태 선택 조율
    uses_ai_sim = st.sidebar.selectbox("AI 도구 보유 여부", options=["예 (도구 선택)", "아니오 (사용 안 함)"])
    if uses_ai_sim == "아니오 (사용 안 함)":
        st.sidebar.info("ℹ️ AI 도구를 사용하지 않는 학생 스나이퍼로 자동 매핑됩니다.")
        actual_ai_tool = "None"
        actual_ai_purpose = "None"
        uses_ai = "No"
    else:
        ai_tools_options = [opt for opt in df_active["ai_tools_used"].unique() if opt != "None"]
        purpose_options = [opt for opt in df_active["purpose_of_ai"].unique() if opt != "None"]
        
        actual_ai_tool = st.sidebar.selectbox("사용 AI 도구 (ai_tools_used)", options=ai_tools_options if ai_tools_options else ["ChatGPT"])
        actual_ai_purpose = st.sidebar.selectbox("AI 사용 목적 (purpose_of_ai)", options=purpose_options if purpose_options else ["Homework"])
        uses_ai = "Yes"

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

def get_model_filename(dk: str, mn: str):
    clean_name = mn.lower().replace(" ", "_")
    return f"pipeline_{dk}_{clean_name}.joblib"

def train_and_cache_pipelines(dataset_key: str, _df, feature_cols, target_col):
    # 1. 안전한 연산을 위해 독립된 데이터프레임 복사본 생성
    _df = _df.copy()
    
    # 2. 카테고리형(object) 변수와 수치형 변수 자동 식별
    cat_cols = [c for c in feature_cols if _df[c].dtype == 'object']
    num_cols = [c for c in feature_cols if c not in cat_cols]
    
    # 🔥 [핵심 수정] 범주형 컬럼 내에 숨어있을 수 있는 숫자형/Float 요소를 완전히 str로 직렬화
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
    
    # 1. 🎯 Baseline 모델 (기준선 수평 타겟용 레퍼런스 모델)
    baseline_model = DummyRegressor(strategy='mean')
    baseline_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', baseline_model)])
    
    # 이제 타입이 완벽히 통일되어 에러 없이 정상 fit 됩니다!
    baseline_pipeline.fit(X_train, y_train)
    b_preds = baseline_pipeline.predict(X_test)
    baseline_metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_test, b_preds)),
        "R2": r2_score(y_test, b_preds)
    }
    
    # 2. 메인 예측 모델 파이프라인 학습 루프 (이하 동일)
    for model_name, info in MODEL_REGISTRY.items():
        filepath = get_model_filename(dataset_key, model_name)
        
        if os.path.exists(filepath):
            pipeline = joblib.load(filepath)
        else:
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', info["model_class"](**info["params"]))
            ])
            pipeline.fit(X_train, y_train)
            joblib.dump(pipeline, filepath)
            
        preds = pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        trained_pipelines[model_name] = pipeline
        results[model_name] = {"RMSE": rmse, "R2": r2, "preds": preds}
        
    return trained_pipelines, results, y_test, cat_cols, num_cols, baseline_metrics

trained_models, evaluation_results, y_test_actual, cat_cols, num_cols, baseline_metrics = train_and_cache_pipelines(
    dataset_key, df_active, feature_cols, target_col
)

# -----------------------------------------------------------------------------
# 5. 시각화 그래프 강화 및 대시보드 화면 구성
# -----------------------------------------------------------------------------
tabs = st.tabs(["📊 데이터 통계 및 탐색", "📈 모델 성능 및 분석", "🔮 실시간 맞춤 성적 예측"])

# --- Tab 1: 데이터 탐색 ---
with tabs[0]:
    st.subheader(f"📋 선택된 데이터 기본 정보 ({dataset_option})")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("총 학생 수 (행 개수)", f"{len(df_active)} 명")
    col2.metric("사용된 예측 피처 수", f"{len(feature_cols)} 개")
    col3.metric("AI 도입 전 평균 성적", f"{df_active['grades_before_ai'].mean():.1f} 점")
    
    st.dataframe(df_active.head(10), use_container_width=True)
    
    st.subheader("📊 주요 변수별 분포 시각화")
    c1, c2 = st.columns(2)
    
    with c1:
        fig_hist = px.histogram(
            df_active, x="grades_after_ai", color="education_level", barmode="overlay",
            title="📈 AI 사용 후 성적(Target) 분포 현황",
            labels={"grades_after_ai": "성적 (점)", "count": "학생 수"},
            color_discrete_sequence=[COLOR_PRE, COLOR_HIGHLIGHT],
            template="plotly_white"
        )
        fig_hist.update_layout(margin=dict(l=40, r=40, t=50, b=40))
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with c2:
        fig_box = px.box(
            df_active, x="ai_tools_used", y="grades_after_ai", color="ai_tools_used",
            title="🎯 사용 AI 도구별 성적 분포 편차",
            labels={"ai_tools_used": "AI 도구", "grades_after_ai": "AI 활용 후 성적"},
            template="plotly_white"
        )
        fig_box.update_layout(showlegend=False, margin=dict(l=40, r=40, t=50, b=40))
