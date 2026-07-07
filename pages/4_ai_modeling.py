import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

# 데이터 경로 및 타겟 설정
DATA_PATH = "cleaned_student_ai_data.csv"
TARGET_COL = "grades_after_ai"

st.set_page_config(page_title="AI 성적 예측기", layout="wide")
st.title("🎓 AI 사용 후 성적 예측 프로그램")

if not os.path.exists(DATA_PATH):
    st.error(f"'{DATA_PATH}' 파일이 필요합니다. 앱과 같은 폴더에 위치해 있는지 확인해주세요.")
    st.stop()

# 1. 데이터 로드 및 결측치 처리
df = pd.read_csv(DATA_PATH)
df["ai_tools_used"] = df["ai_tools_used"].fillna("None")
df["purpose_of_ai"] = df["purpose_of_ai"].fillna("None")

# 범주형 데이터 인코딩
df_model = df.copy()
for col in ["education_level", "ai_tools_used", "purpose_of_ai"]:
    df_model[col] = df_model[col].astype("category").cat.codes

# 2. 데이터 분할 및 모델 학습
X = df_model.drop(columns=[TARGET_COL])
y = df_model[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# 테스트 데이터 예측 및 평가지표 계산
lr_preds = lr_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
lr_mse = mean_squared_error(y_test, lr_preds)
lr_mae = mean_absolute_error(y_test, lr_preds)
lr_r2 = r2_score(y_test, lr_preds)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_mse = mean_squared_error(y_test, rf_preds)
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_r2 = r2_score(y_test, rf_preds)


# ==========================================
# 3. 시각화 및 성능 비교 기능 (접힌 드롭다운 형태)
# ==========================================
st.write("---")
st.subheader("📊 모델 성능 비교 및 시각화")

with st.expander("모델 성능 지표 확인 (RMSE, MSE, MAE, R²)", expanded=False):
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("**Linear Regression (선형 회귀)**")
        st.metric(label="RMSE (낮을수록 좋음)", value=f"{lr_rmse:.4f}")
        st.metric(label="MSE (낮을수록 좋음)", value=f"{lr_mse:.4f}")
        st.metric(label="MAE (낮을수록 좋음)", value=f"{lr_mae:.4f}")
        st.metric(label="R² Score (높을수록 좋음)", value=f"{lr_r2:.4f}")
    with col_m2:
        st.markdown("**Random Forest (랜덤 포레스트)**")
        st.metric(label="RMSE (낮을수록 좋음)", value=f"{rf_rmse:.4f}")
        st.metric(label="MSE (낮을수록 좋음)", value=f"{rf_mse:.4f}")
        st.metric(label="MAE (낮을수록 좋음)", value=f"{rf_mae:.4f}")
        st.metric(label="R² Score (높을수록 좋음)", value=f"{rf_r2:.4f}")

with st.expander("실제 vs 예측 성적 산점도 비교", expanded=False):
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=y_test, y=lr_preds, mode='markers', name='Linear Regression', marker=dict(color='#1F5FA8', opacity=0.7)))
    fig1.add_trace(go.Scatter(x=y_test, y=rf_preds, mode='markers', name='Random Forest', marker=dict(color='#E8714C', opacity=0.7)))

    lo = min(y_test.min(), lr_preds.min(), rf_preds.min()) - 1
    hi = max(y_test.max(), lr_preds.max(), rf_preds.max()) + 1
    fig1.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode='lines', name='Perfect Fit', line=dict(color='gray', dash='dash')))

    fig1.update_layout(title="실제 vs 예측 성적 비교", xaxis_title="실제 성적", yaxis_title="예측 성적", height=400)
    st.plotly_chart(fig1, use_container_width=True)

with st.expander("각 모델별 변수 중요도 막대 그래프", expanded=False):
    col_fig1, col_fig2 = st.columns(2)
    with col_fig1:
        lr_coef = lr_model.coef_
        feat_lr = pd.DataFrame({"feature": X.columns, "coefficient": lr_coef})
        feat_lr["abs_coefficient"] = feat_lr["coefficient"].abs()
        feat_lr = feat_lr.sort_values("abs_coefficient", ascending=True)
        
        fig2 = go.Figure(go.Bar(x=feat_lr["coefficient"], y=feat_lr["feature"], orientation="h", marker=dict(color='#1F5FA8')))
        fig2.update_layout(title="Linear Regression 변수 중요도 (계수)", xaxis_title="Coefficient", yaxis_title="", height=380, margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(fig2, use_container_width=True)

    with col_fig2:
        importances = rf_model.feature_importances_
        feat_rf = pd.DataFrame({"feature": X.columns, "importance": importances}).sort_values("importance", ascending=True)
        
        fig3 = go.Figure(go.Bar(x=feat_rf["importance"], y=feat_rf["feature"], orientation="h", marker=dict(color='#E8714C')))
        fig3.update_layout(title="Random Forest 변수 중요도", xaxis_title="Importance", yaxis_title="", height=380, margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(fig3, use_container_width=True)


# ==========================================
# 4. 사용자 입력 및 예측 UI
# ==========================================
st.write("---")
st.subheader("🔮 새로운 학생 데이터 입력 및 성적 예측")

# 버튼 클릭 후에도 창이 열려 있도록 세션 상태 사용
if "predict_clicked" not in st.session_state:
    st.session_state.predict_clicked = False

with st.expander("데이터 입력 및 예측 실행 패널", expanded=st.session_state.predict_clicked):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("나이 (age)", min_value=14, max_value=19, value=17)
        education_level = st.selectbox("교육 수준 (education_level)", options=["school", "college"])
        study_hours = st.slider("하루 공부 시간 (study_hours_per_day)", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
        ai_tools_used = st.selectbox("사용 AI 도구 (ai_tools_used)", options=["None", "ChatGPT", "Copilot", "Gemini"])

    with col2:
        purpose_of_ai = st.selectbox("AI 사용 목적 (purpose_of_ai)", options=["None", "Coding", "Homework", "Research"])
        grades_before = st.slider("AI 사용 전 성적 (grades_before_ai)", min_value=55, max_value=75, value=65)
        screen_time = st.slider("하루 스크린 타임 (daily_screen_time_hours)", min_value=2, max_value=7, value=4)

    if st.button("성적 예측 실행", type="primary"):
        st.session_state.predict_clicked = True
        
        # 입력 데이터 매핑
        input_data = pd.DataFrame([{
            "age": age,
            "education_level": education_level,
            "study_hours_per_day": study_hours,
            "ai_tools_used": ai_tools_used,
            "purpose_of_ai": purpose_of_ai,
            "grades_before_ai": grades_before,
            "daily_screen_time_hours": screen_time
        }])
        
        for col in ["education_level", "ai_tools_used", "purpose_of_ai"]:
            input_data[col] = pd.Categorical(input_data[col], categories=df[col].unique()).codes
        
        # 예측
        lr_pred = lr_model.predict(input_data)[0]
        rf_pred = rf_model.predict(input_data)[0]
        
        st.write("---")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(label="Linear Regression 예측 결과", value=f"{lr_pred:.2f} 점")
        with res_col2:
            st.metric(label="Random Forest 예측 결과", value=f"{rf_pred:.2f} 점")
