import streamlit as st
import pandas as pd

# 1. 페이지 설정
st.set_page_config(page_title="문제 정의 & 데이터 수집", page_icon="📊", layout="wide")

# 2. 메인 타이틀
st.title("📋 1. 문제 정의 및 데이터 수집")
st.markdown("---")

# 3. 문제 정의 섹션
st.subheader("💡 1.1 문제 정의")
st.markdown("""
최근 학생들이 AI를 사용하여 학습하는 사례가 늘어나고 있습니다.  
고교생 2명 중 1명이 AI를 활용하고 있으며, 대학생의 경우 무려 **86%**가 AI를 사용해 학습하고 있는 가운데,  
**"과연 AI를 이용한 학습이 학생의 성적 향상에 실질적인 도움이 되는가?"**에 대한 데이터 기반의 검증이 필요한 시점입니다.

* **목표:** AI 학습 플랫폼 이용 시간 및 자기주도 학습 데이터와 최종 성적 간의 상관관계를 분석합니다.
""")

st.markdown("---")

# 4. 데이터 수집 섹션 (요청하신 대로 제목을 수정했습니다)
st.subheader("📥 1.2 데이터 수집")

# 사용자가 직접 CSV 파일을 업로드할 수 있는 컴포넌트
uploaded_file = st.file_uploader("분석할 CSV 파일을 업로드하세요.", type=["csv"])

if uploaded_file is not None:
    # 한글 깨짐 방지를 위해 encoding='utf-8-sig' 추가
    df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
    st.success("🎉 데이터 로드 성공!")
    
    # 데이터 프리뷰
    st.markdown("### 📊 데이터 프리뷰 (상위 5개 행)")
    st.dataframe(df.head())
else:
    st.info("💡 파일을 업로드하면 이곳에 데이터 프리뷰가 표시됩니다.")
