
import streamlit as st

# 페이지 기본 설정
st.set_page_config(
    page_title="데이터 분석 & AI 모델링 프로젝트",
    page_icon="🤖",
    layout="wide"
)

# 메인 화면 타이틀
st.title("🚀 데이터 분석 및 AI 모델링 웹 애플리케이션")
st.subheader("Streamlit을 활용한 머신러닝 파이프라인 구현")

st.markdown("---")

# 프로젝트 개요 구역
st.markdown("""
### 📌 프로젝트 소개
이 웹 애플리케이션은 데이터 수집부터 전처리, 시각화, 그리고 인공지능 모델링까지의 전체 프로세스를 시각적으로 확인하고 제어할 수 있도록 제작되었습니다.

**왼쪽 사이드바의 메뉴를 클릭하여 각 단계별 프로세스를 확인해보세요!**
""")

# 간단한 안내 카드 구성
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.info("**1. 문제 정의 & 수집**\n\n프로젝트 목표 설정 및 데이터 로드")
with col2:
    st.success("**2. 데이터 전처리**\n\n결측치 처리, 이상치 제거 및 피처 엔지니어링")
with col3:
    st.warning("**3. 데이터 시각화**\n\nEDA를 통한 데이터 인사이트 발굴")
with col4:
    st.error("**4. AI 모델링**\n\n머신러닝/딥러닝 모델 학습 및 예측 결과 확인")
