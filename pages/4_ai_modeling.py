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
# 3. 시각화 및 성능 비교 기능
# ==========================================
st.subheader("📊 모델 성능 비교 및 시각화")
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

st.write("---")
# 실제 vs 예측 성적 Plotly 그래프
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=y_test, y=lr_preds, mode='markers', name='Linear Regression', marker=dict(color='#1F5FA8', opacity=0.7)))
fig1.add_trace(go.Scatter(x=y_test, y=rf_preds, mode='markers', name='Random Forest', marker=dict(color='#E8714C', opacity=0.7)))

# Perfect Fit 라인 생성
lo = min(y_test.min(), lr_preds.min(), rf_preds.min()) - 1
hi = max(y_test.max(), lr_preds.max(), rf_preds.max()) + 1
fig1.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode='lines', name='Perfect Fit', line=dict(color='gray', dash='dash')))

fig1.update_layout(title="실제 vs 예측 성적 비교", xaxis_title="실제 성적", yaxis_title="예측 성적", height=400)
st.plotly_chart(fig1, use_container_width
