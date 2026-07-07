# ============================================================
# AI 성적 예측 웹앱 (Streamlit + Plotly)
# 목적: 학생의 AI 사용 및 학습 데이터를 기반으로 AI 사용 후 성적(grades_after_ai)을 예측한다.
# 사용 모델: Linear Regression, Random Forest Regressor
# 데이터: cleaned_student_ai_data.csv (전처리 완료 데이터)
# 특징: 원-핫 인코딩(One-Hot Encoding) 및 스케일링을 적용하여 예측 정확도를 높임
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

TARGET_COL = "grades_after_ai"
DATA_PATH = "cleaned_student_ai_data.csv"

# ============================================================
# 1. 데이터 로드 및 전처리 파이프라인
# ============================================================
@st.cache_data
def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    # 결측치(NaN)를 문자열 'None'으로 치환 (ai_tools_used, purpose_of_ai)
    if "ai_tools_used" in df.columns:
        df["ai_tools_used"] = df["ai_tools_used"].fillna("None")
    if "purpose_of_ai" in df.columns:
        df["purpose_of_ai"] = df["purpose_of_ai"].fillna("None")
    return df

def preprocess_data(df: pd.DataFrame, is_train=True, encoder_cols=None, scaler=None):
    """원-핫 인코딩 및 스케일링 수행"""
    df_proc = df.copy()
    
    # 1. 범주형 변수에 대해 원-핫 인코딩 (One-Hot Encoding) 수행
    df_proc = pd.get_dummies(
        df_proc, 
        columns=["education_level", "ai_tools_used", "purpose_of_ai"],
        drop_first=False
    )
    
    # 학습 시 더미 변수 목록과 스케일러를 생성하여 반환 (테스트 시 활용)
    if is_train:
        encoder_cols = [c for c in df_proc.columns if c != TARGET_COL]
        # bool 형태를 int(0, 1)로 변환
        for c in encoder_cols:
            if df_proc[c].dtype == bool:
                df_proc[c] = df_proc[c].astype(int)
                
        # 스케일링 적용
        scaler = StandardScaler()
        df_proc[encoder_cols] = scaler.fit_transform(df_proc[encoder_cols])
        return df_proc, encoder_cols, scaler
        
    else:
        # 예측(추론)용 데이터는 학습 시 없던 컬럼을 0으로 채워 추가
        for col in encoder_cols:
            if col not in df_proc.columns:
                df_proc[col] = 0
        df_proc = df_proc[encoder_cols]
        
        # bool 형태를 int(0, 1)로 변환
        for c in encoder_cols:
            if df_proc[c].dtype == bool:
                df_proc[c] = df_proc[c].astype(int)
                
        # 스케일링 적용
        df_proc[encoder_cols] = scaler.transform(df_proc[encoder_cols])
        return df_proc

# ============================================================
# 2. 모델 학습 및 평가
# ============================================================
def train_models(X_train, y_train):
    lr_model = LinearRegression()
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    lr_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    
    return {"Linear Regression": lr_model, "Random Forest": rf_model}

def evaluate_model(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return {"RMSE": round(rmse, 4), "R² Score": round(r2, 4), "predictions": preds}


# ============================================================
# 3. Plotly 그래프 생성 함수
# ============================================================
def plot_actual_vs_predicted(results: dict, y_test: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure()

    all_vals = list(y_test)
    for res in results.values():
        all_vals.extend(res["predictions"].tolist())
    lo, hi = min(all_vals) - 1, max(all_vals) + 1

    colors = {"Linear Regression": "#1F5FA8", "Random Forest": "#E8714C"}

    for mname, res in results.items():
        fig.add_trace(go.Scatter(
            x=y_test,
            y=res["predictions"],
            mode="markers",
            name=mname,
            marker=dict(size=8, opacity=0.7, color=colors.get(mname), line=dict(width=0.5, color="white")),
            hovertemplate=(f"<b>{mname}</b><br>실제: %{{x:.1f}}점<br>예측: %{{y:.1f}}점<extra></extra>"),
        ))

    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect Fit",
        line=dict(color="gray", dash="dash", width=1.5), hoverinfo="skip",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis_title="실제 성적 (grades_after_ai)",
        yaxis_title="예측 성적 (grades_after_ai)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380, margin=dict(l=40, r=20, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
    )
    return fig

def plot_feature_importance(rf_model, feature_cols: list) -> go.Figure:
    importances = rf_model.feature_importances_
    feat_df = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=True)
    )

    fig = go.Figure(go.Bar(
        x=feat_df["importance"], y=feat_df["feature"], orientation="h",
        marker=dict(color=feat_df["importance"], colorscale="Oranges", showscale=False),
        hovertemplate="<b>%{y}</b><br>중요도: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Random Forest 변수 중요도 (Feature Importance)", font=dict(size=15)),
        xaxis_title="Importance", yaxis_title="",
        height=380, margin=dict(l=40, r=20, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
    )
    return fig


# ============================================================
# 4. Streamlit 앱 메인
# ============================================================
def main():
    st.set_page_config(page_title="AI 성적 예측기", page_icon="🎓", layout="wide")

    # 테이블 헤더 폰트 사이즈 조정
    st.markdown(
        """<style>
        div[data-testid="stDataFrame"] thead tr th { font-size: 16px !important; font-weight: 600 !important; }
        </style>""", unsafe_allow_html=True
    )

    st.title("🎓 AI 사용 후 성적 예측 (Cleaned Data)")
    st.markdown(
        "정제된 데이터(`cleaned_student_ai_data.csv`)를 활용하여 **선형 회귀**와 **랜덤 포레스트** "
        "모델의 성능을 비교하고 새로운 학생의 성적을 예측합니다."
    )
    st.divider()

    if not os.path.exists(DATA_PATH):
        st.error(f"❌ '{DATA_PATH}' 파일을 찾을 수 없습니다. 앱과 같은 폴더에 파일을 넣어주세요.")
        st.stop()

    # 데이터 로드
    df_raw = load_data(DATA_PATH)
    
    # 훈련/테스트 데이터 분리 (Target 분리)
    X_raw = df_raw.drop(columns=[TARGET_COL])
    y = df_raw[TARGET_COL]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42
    )

    # 전처리 적용 (인코딩 & 스케일링)
    X_train_processed, encoder_cols, scaler = preprocess_data(X_train_raw, is_train=True)
    X_test_processed = preprocess_data(X_test_raw, is_train=False, encoder_cols=encoder_cols, scaler=scaler)

    # 모델 학습
    models = train_models(X_train_processed, y_train)
    results = {
        mname: evaluate_model(model, X_test_processed, y_test)
        for mname, model in models.items()
    }

    # ============================================================
    # 섹션 1: 모델 성능 및 데이터 시각화
    # ============================================================
    with st.expander("📊 모델 성능 비교 및 시각화", expanded=True):
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**Linear Regression (선형 회귀)**")
            st.metric(label="RMSE (낮을수록 좋음)", value=f"{results['Linear Regression']['RMSE']:.4f}")
            st.metric(label="R² Score (높을수록 좋음)", value=f"{results['Linear Regression']['R² Score']:.4f}")
        with col_m2:
            st.markdown("**Random Forest (랜덤 포레스트)**")
            st.metric(label="RMSE (낮을수록 좋음)", value=f"{results['Random Forest']['RMSE']:.4f}")
            st.metric(label="R² Score (높을수록 좋음)", value=f"{results['Random Forest']['R² Score']:.4f}")

        st.divider()
        col_fig1, col_fig2 = st.columns(2)
        with col_fig1:
            st.plotly_chart(plot_actual_vs_predicted(results, y_test.values, "실제 vs 예측 성적"), use_container_width=True)
        with col_fig2:
            st.plotly_chart(plot_feature_importance(models["Random Forest"], encoder_cols), use_container_width=True)

    # ============================================================
    # 섹션 2: 새로운 학생 성적 예측하기
    # ============================================================
    st.divider()
    with st.expander("🔮 새로운 학생 성적 예측하기", expanded=True):
        st.markdown("학생 정보를 입력하고 **Predict** 버튼을 눌러 모델별 예측 결과를 확인하세요.")

        inp1, inp2 = st.columns(2)
        with inp1:
            age = st.slider("나이 (age)", min_value=14, max_value=19, value=17, step=1)
            education_level = st.selectbox("교육 수준 (education_level)", options=["school", "college"])
            study_hours = st.slider("하루 공부 시간 (study_hours_per_day)", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
            
            # uses_ai 컬럼이 사라졌으므로, 도구 선택 옵션에 'None'을 명시적으로 포함
            ai_tools_used = st.selectbox(
                "사용 AI 도구 (ai_tools_used)", 
                options=["None", "ChatGPT", "Copilot", "Gemini"]
            )
        
        with inp2:
            purpose_of_ai = st.selectbox(
                "AI 사용 목적 (purpose_of_ai)", 
                options=["None", "Coding", "Homework", "Research"]
            )
            grades_before = st.slider("AI 사용 전 성적 (grades_before_ai)", min_value=55, max_value=75, value=65, step=1)
            screen_time = st.slider("하루 스크린 타임 (daily_screen_time_hours)", min_value=2, max_value=7, value=4, step=1)

        if st.button("🚀 Predict (성적 예측)", type="primary", use_container_width=True):
            # 입력받은 데이터를 DataFrame으로 변환
            user_raw = pd.DataFrame([{
                "age": age,
                "education_level": education_level,
                "study_hours_per_day": study_hours,
                "ai_tools_used": ai_tools_used,
                "purpose_of_ai": purpose_of_ai,
                "grades_before_ai": grades_before,
                "daily_screen_time_hours": screen_time,
            }])

            # 사용자 데이터 전처리 적용 (동일한 원-핫 인코딩 및 스케일링)
            user_processed = preprocess_data(
                user_raw, is_train=False, encoder_cols=encoder_cols, scaler=scaler
            )

            st.subheader("🎯 예측 결과")
            pred_cols = st.columns(2)
            
            for col, (mname, model) in zip(pred_cols, models.items()):
                # 점수는 0~100점 사이로 클리핑
                pred = float(np.clip(model.predict(user_processed)[0], 0, 100))
                with col:
                    st.metric(
                        label=f"📌 {mname}",
                        value=f"{pred:.1f} 점",
                        delta=f"{pred - grades_before:+.1f} 점 (AI 사용 전 대비)"
                    )
            st.success("✅ 예측이 완료되었습니다!")

    # ============================================================
    # 섹션 3: 데이터 미리보기
    # ============================================================
    with st.expander("📂 데이터 미리보기"):
        st.dataframe(df_raw.head(10), use_container_width=True)
        st.caption(f"전체 데이터 크기: {df_raw.shape[0]}행 × {df_raw.shape[1]}열")

if __name__ == "__main__":
    main()
