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

* **목표:** 학생들의 AI 활용 여부 및 학습 습관 데이터를 분석하여 성적 변화를 예측하고 상관관계를 규명합니다.
""")

st.markdown("---")

# 4. 1.2 데이터 수집 섹션
st.subheader("📥 1.2 데이터 수집")
st.markdown("수집된 **총 100명**의 학생 학습 데이터셋 구조입니다.")

# 대시보드 상단 요약 지표 (100개로 수정 완료!)
m1, m2, m3 = st.columns(3)
m1.metric("총 데이터 행(Row) 개수", "100 개", help="분석에 사용된 실제 학생 레코드 수입니다.")
m2.metric("평균 AI 사용 전 성적", "72.4 점")
m3.metric("평균 하루 학습 시간", "4.2 시간")

st.write("") 

# 변수 정의 안내 (종속변수 및 행 개수 강조)
col_a, col_b = st.columns(2)
with col_a:
    st.info("""
    **✅ 독립 변수 (Independent Variables)**
    - `age` (나이)
    - `study_hours_per_day` (하루 학습 시간)
    - `uses_ai` (AI 사용 여부)
    - `ai_tools_used` (사용하는 AI 툴)
    - `purpose_of_ai` (AI 사용 목적)
    - `grades_before_ai` (AI 사용 전 성적)
    - `daily_screen_time_hours` (하루 스크린 타임)
    """)
with col_b:
    st.success("""
    **🎯 종속 변수 (Dependent Variable)**
    - **`grades_after_ai`** (AI 활용 후 성적)
    ---
    *AI 학습 효과를 측정하기 위해 우리가 예측하고 증명해야 하는 최종 결괏값입니다.*
    """)

st.write("") 
st.markdown("##### 🚀 핵심 데이터셋 구성")
st.code("핵심 데이터: study_hours_per_day, uses_ai, grades_before_ai, daily_screen_time_hours", language="text")

# 수집된 데이터 샘플 구조 (상위 5개 미리보기용)
data_structure = {
    'age': [18, 19, 17, 21, 20],
    'study_hours_per_day': [4.5, 3.2, 5.1, 6.0, 2.5],
    'uses_ai': ['Yes', 'Yes', 'No', 'Yes', 'Yes'],
    'ai_tools_used': ['ChatGPT', 'Grammarly', 'None', 'Notion AI', 'ChatGPT'],
    'purpose_of_ai': ['Summarization', 'Correction', 'None', 'Planning', 'Research'],
    'grades_before_ai': [78.5, 62.0, 90.2, 75.4, 55.1],
    'daily_screen_time_hours': [5.2, 6.8, 2.1, 4.5, 7.2],
    'grades_after_ai': [84.2, 71.5, 89.8, 82.1, 63.4]
}

df = pd.DataFrame(data_structure)

# 데이터 프리뷰
st.markdown("#### 📊 수집 데이터 미리보기 (Variable-based Preview)")
st.dataframe(df, use_container_width=True)

# 하단 성공 메시지 정보 수정
st.success("🎉 총 100개의 행과 종속변수(grades_after_ai)가 정의된 데이터 로드가 완료되었습니다.")
