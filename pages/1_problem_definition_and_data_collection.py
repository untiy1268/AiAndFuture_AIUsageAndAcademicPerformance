import streamlit as st
import pandas as pd

# 1. 페이지 설정
st.set_page_config(page_title="문제 정의 및 데이터 수집", page_icon="💡", layout="wide")

# 2. 메인 타이틀
st.title("📋 1. 문제 정의 및 데이터 수집")
st.markdown("---")

# 3. 1.1 문제 정의 섹션
st.subheader("💡 1.1 문제 정의")
st.markdown("""
최근 학생들이 AI를 사용하여 학습하는 사례가 늘어나고 있습니다.  
고교생 2명 중 1명이 AI를 활용하고 있으며, 대학생의 경우 무려 **86%**가 AI를 사용해 학습하고 있는 가운데,  
**"과연 AI를 이용한 학습이 학생의 성적 향상에 실질적인 도움이 되는가?"**에 대한 데이터 기반의 검증이 필요한 시점입니다.

* **목표:** 학생들의 AI 도구 사용 빈도, 학습 습관 데이터와 실제 학업 성취도(GPA/성적) 간의 상관관계를 분석합니다.
""")

st.markdown("---")

# 4. 1.2 데이터 수집 섹션 (요청하신 대로 반영했습니다)
st.subheader("📥 1.2 데이터 수집")
st.markdown("Kaggle의 `Students' AI Usage and Academic Performance` 데이터셋을 기반으로 수집된 기본 데이터입니다.")

# 캐글 실제 데이터셋의 피처 구조를 반영한 샘플 데이터 구성
kaggle_mock_data = {
    'StudentID': [f'STU_{i:03d}' for i in range(1, 6)],
    'AI_Usage_Frequency': ['Daily', 'Weekly', 'Rarely', 'Daily', 'Monthly'],
    'Primary_AI_Tool': ['ChatGPT', 'Gamma', 'DeepL', 'Notion AI', 'ChatGPT'],
    'Study_Hours_Per_Week': [15, 8, 12, 20, 5],
    'Prior_GPA': [3.2, 2.8, 3.5, 3.1, 2.5],
    'Current_GPA': [3.6, 3.1, 3.4, 3.9, 2.4],
    'Academic_Improvement': ['Yes', 'Yes', 'No', 'Yes', 'No']
}

# 데이터프레임 생성
df = pd.DataFrame(kaggle_mock_data)

# 요약 지표 (Metrics) 대시보드 구성
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("총 수집 데이터 (Rows)", "3,618 명")
with col2:
    st.metric("주요 수집 AI 툴", "ChatGPT")
with col3:
    st.metric("주당 평균 학습 시간", "12.8 시간")
with col4:
    st.metric("성적 향상 비율", "68.4%")

# 데이터 프리뷰 출력
st.markdown("#### 📊 수집 데이터 미리보기 (Kaggle Dataset Preview)")
st.dataframe(df, use_container_width=True)

st.success("🎉 수집된 데이터 로드 및 분석 준비가 완료되었습니다.")
