
import streamlit as st
import pandas as pd

st.set_page_config(page_title="문제 정의 & 데이터 수집", page_icon="📊")

st.title("📋 1. 문제 정의 및 데이터 수집")
st.markdown("---")

st.subheader("💡 1.1 문제 정의")
st.write("최근 학생들이 AI를 사용하여 학습하는 사례가 늘어나고 있다.
고교생 2명중 1명은 AI를 사용하여 학습하고있고, 대학생의 86%가 AI를 사용해 학습하고
있는 가운데 과연 AI를 이용한 학습이 학생의 성적 증진에 도움이 되는지가 화두에 오르고있다.")

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
