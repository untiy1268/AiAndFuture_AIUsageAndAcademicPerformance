
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="데이터 시각화", page_icon="📈")

st.title("📈 3. 데이터 시각화 (EDA)")
st.markdown("---")

st.subheader("📊 차트 생성 및 분석")

# 샘플 데이터 생성 (시각화 테스트용)
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['Feature A', 'Feature B', 'Feature C']
)

# 스트림릿 내장 차트 활용 예시
st.write("✏️ **기본 라인 차트**")
st.line_chart(chart_data)

st.write("✏️ **기본 바 차트**")
st.bar_chart(chart_data)
