import streamlit as st
import pandas as pd
import os

# 1. 페이지 설정
st.set_page_config(page_title="문제 정의 및 데이터 확인", page_icon="💡", layout="wide")

# 2. 메인 타이틀
st.title("📋 1. 문제 정의 및 데이터 수집")
st.markdown("---")

# 3. 문제 정의 상세 내용
st.subheader("💡 1.1 문제 정의 및 데이터 확인")
st.markdown("""
최근 학생들이 AI를 사용하여 학습하는 사례가 늘어나고 있습니다.  
고교생 2명 중 1명이 AI를 활용하고 있으며, 대학생의 경우 무려 **86%**가 AI를 사용해 학습하고 있는 가운데,  
**"과연 AI를 이용한 학습이 학생의 성적 향상에 실질적인 도움이 되는가?"**에 대한 데이터 기반의 검증이 필요한 시점입니다.

* **목표:** AI 학습 플랫폼 이용 시간 및 자기주도 학습 데이터와 최종 성적 간의 상관관계를 분석합니다.
""")

# 기본 파일 정의
csv_filename = "students_ai_usage.csv"

st.markdown("---")
st.markdown(f"### 📥 분석 데이터 확인 (`{csv_filename}`)")

# 파일이 현재 폴더에 존재하는지 확인 후 자동 로드
if os.path.exists(csv_filename):
    try:
        # 한글 및 유효 데이터 로드 (utf-8-sig로 인코딩 유연성 확보)
        df = pd.read_csv(csv_filename, encoding='utf-8-sig')
        
        st.success(f"🎉 기본 데이터 파일 `{csv_filename}`을 성공적으로 로드했습니다!")
        
        # 데이터의 기본 정보 요약 출력
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 학생 수 (행)", f"{df.shape[0]}명")
        with col2:
            st.metric("수집된 변수 수 (열)", f"{df.shape[1]}개")
        with col3:
            st.metric("평균 AI 사용 시간", f"{df.iloc[:, 1].mean():.1f}시간")

        # 데이터 프리뷰 출력
        st.markdown("#### 📊 데이터 미리보기 (상위 5개 행)")
        st.dataframe(df.head())
        
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
else:
    st.warning(f"⚠️ `{csv_filename}` 파일이 현재 디렉토리에 없습니다. 프로젝트 폴더에 파일을 배치해주세요.")
