
import streamlit as st

st.set_page_config(page_title="데이터 전처리", page_icon="⚙️")

st.title("⚙️ 2. 데이터 전처리")
st.markdown("---")

st.subheader("🛠️ 데이터 정제 작업")
st.write("결측치(Null) 처리, 데이터 타입 변경, 중복 제거 등의 전처리 작업을 수행합니다.")

# 전처리 옵션 선택 인터페이스 예시
preprocessing_option = st.multiselect(
    '적용할 전처리 기법을 선택하세요:',
    ['결측치 제거 (Drop Na)', '평균값으로 결측치 채우기 (Imputation)', '원-핫 인코딩 (One-Hot Encoding)', '데이터 표준화 (Standard Scaling)']
)

if preprocessing_option:
    st.write(f"🔄 **선택된 전처리:** {', '.join(preprocessing_option)} 기능이 준비 중입니다.")
