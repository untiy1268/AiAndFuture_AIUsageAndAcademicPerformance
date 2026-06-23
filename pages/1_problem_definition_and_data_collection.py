
import streamlit as st
import pandas as pd

st.set_page_config(page_title="문제 정의 & 데이터 수집", page_icon="📊")

st.title("📋 1. 문제 정의 및 데이터 수집")
st.markdown("---")

st.subheader("💡 1.1 문제 정의")
st.write("해당 프로젝트가 해결하고자 하는 문제를 정의하고 목표를 설정합니다.")
st.caption("예시: 고객 이탈 데이터 분석을 통한 이탈률 감소 및 예측 모델 개발")

st.markdown("---")

st.subheader("📥 1.2 데이터 수집 (샘플 데이터 로드)")
# 사용자가 직접 CSV 파일을 업로드할 수 있는 컴포넌트
uploaded_file = st.file_uploader("분석할 CSV 파일을 업로드하세요.", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("데이터 로드 성공!")
    st.dataframe(df.head())
else:
    st.info("파일을 업로드하면 이곳에 데이터 프리뷰가 표시됩니다.")
