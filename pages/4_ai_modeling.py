import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. 오직 요청하신 단일 데이터셋만 깔끔하게 로드 및 타입 고정
@st.cache_data
def load_clean_data():
    file_path = "cleaned_student_ai_data.csv"
    if not os.path.exists(file_path):
        st.error(f"'{file_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        st.stop()
        
    df = pd.read_csv(file_path)
    
    # 범주형 컬럼 안전하게 문자열로 고정
    cat_cols = ['education_level', 'ai_tools_used', 'purpose_of_ai']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("None").astype(str)
            
    # 수치형 컬럼 안전하게 숫자 타입으로 고정
    num_cols = ['age', 'study_hours_per_day', 'grades_before_ai', 'daily_screen_time_hours', 'grades_after_ai']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    return df

# 2. 불필요한 외부 파일 저장/복잡한 루프 없이 파이프라인 학습 및 결과 반환
def train_clean_models(df):
    feature_cols = ['age', 'education_level', 'study_hours_per_day', 'ai_tools_used', 'purpose_of_ai', 'grades_before_ai', 'daily_screen_time_hours']
    target_col = 'grades_after_ai'
    
    X = df[feature_cols]
    y = df[target_col]
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    cat_cols = ['education_level', 'ai_tools_used', 'purpose_of_ai']
    num_cols = ['age', 'study_hours_per_day', 'grades_before_ai', 'daily_screen_time_hours']
    
    # 인코더 및 전처리 구성
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
            ('num', 'passthrough', num_cols)
        ]
    )
    
    # 모델 저장고 정의
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    trained_pipelines = {}
    evaluation_results = {}
    
    # 순수 파이프라인 학습 및 성능 계산
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        pipeline.fit(X_train, y_train)
        
        preds = pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        trained_pipelines[name] = pipeline
        evaluation_results[name] = {"RMSE": rmse, "R2": r2}
        
    return trained_pipelines, evaluation_results, cat_cols, num_cols

# --- UI 레이아웃 실행 영역 ---
st.title("🎓 학생 AI 사용 및 학업 성적 예측 모델")

# 데이터 및 모델 로드
df = load_clean_data()
trained_models, evaluation_results, cat_cols, num_cols = train_clean_models(df)

# 대시보드 탭 분할
tab1, tab2 = st.tabs(["📊 모델 성능 분석", "🔮 실시간 맞춤 성적 예측"])

# [탭 1] 모델 성능 분석 시각화 작동
with tab1:
    st.subheader("모델별 성능 평가지표")
    perf_list = []
    for name, metrics in evaluation_results.items():
        perf_list.append({
            "Model": name,
            "RMSE (낮을수록 좋음)": f"{metrics['RMSE']:.4f}",
            "R² Score (1에 가까울수록 좋음)": f"{metrics['R2']:.4f}"
        })
    st.table(pd.DataFrame(perf_list))

# [탭 2] 실시간 맞춤 성적 예측 기능 작동
with tab2:
    st.subheader("개인 특성 입력 기반 성적 예측")
    selected_model_name = st.selectbox("예측에 사용할 인공지능 모델 선택", list(trained_models.keys()))
    active_pipeline = trained_models[selected_model_name]
    
    # 데이터셋 구조와 100% 일치하는 입력 가이드 폼
    age_input = st.number_input("나이 (Age)", min_value=10, max_value=30, value=18)
    edu_input = st.selectbox("학습 단계 (Education Level)", ["school", "college"])
    study_input = st.number_input("하루 평균 자습 시간 (Study Hours)", min_value=0.0, max_value=24.0, value=3.0)
    ai_tool_input = st.selectbox("주로 사용하는 AI 툴 (AI Tool)", ["None", "ChatGPT", "Gemini", "Copilot"])
    purpose_input = st.selectbox("AI 활용 목적 (Purpose)", ["None", "Homework", "Research", "Coding"])
    grades_before_input = st.number_input("AI 도입 전 기존 성적 점수 (Grades Before)", min_value=0.0, max_value=100.0, value=60.0)
    screen_input = st.number_input("하루 스크린 타임 시간 (Screen Time)", min_value=0.0, max_value=24.0, value=4.0)
    
    if st.button("실시간 맞춤 성적 예측하기"):
        # 입력 데이터프레임 빌드
        input_data = pd.DataFrame([{
            "age": age_input,
            "education_level": edu_input,
            "study_hours_per_day": study_input,
            "ai_tools_used": ai_tool_input,
            "purpose_of_ai": purpose_input,
            "grades_before_ai": grades_before_input,
            "daily_screen_time_hours": screen_input
        }])
        
        # 예측 직전 타입 불일치 오류 완전 방어
        for c in cat_cols:
            input_data[c] = input_data[c].astype(str)
        for c in num_cols:
            input_data[c] = pd.to_numeric(input_data[c], errors='coerce').fillna(0)
            
        # 예측 수행 및 출력
        prediction = active_pipeline.predict(input_data)
        st.success(f"🎯 해당 조건의 학생 예상 성적 점수(Grades After AI): **{prediction[0]:.2f}점**")
