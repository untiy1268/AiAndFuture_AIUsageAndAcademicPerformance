
import streamlit as st

st.set_page_config(page_title="인공지능 모델링", page_icon="🧠")

st.title("🧠 4. 인공지능 모델링 및 평가")
st.markdown("---")

st.subheader("🤖 머신러닝 모델 선택")
model_type = st.selectbox(
    "학습시킬 모델을 선택하세요:",
    ("선형 회귀 (Linear Regression)", "로지스틱 회귀 (Logistic Regression)", "랜덤 포레스트 (Random Forest)", "XGBoost")
)

# 하이퍼파라미터 조절 슬라이더 예시
test_size = st.slider("테스트 데이터 비율 (Test Size):", 0.1, 0.5, 0.2)
st.write(f"⚙️ 설정된 테스틑 데이터 비율: **{int(test_size*100)}%**")

# 학습 버튼 구동 예시
if st.button("🚀 모델 학습 시작"):
    with st.spinner('모델을 학습하는 중입니다...'):
        import time
        time.sleep(2) # 학습 시뮬레이션
    st.success(f"🎉 {model_type} 모델 학습 완료!")
    
    # 평가 지표 예시 수치
    st.metric(label="Accuracy (정확도)", value="92.4%", delta="📊 양호")
