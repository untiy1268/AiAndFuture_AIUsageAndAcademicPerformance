import os
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# 페이지 설정
st.set_page_config(page_title="데이터 전처리", page_icon="⚙️", layout="wide")

st.title("⚙️ 2. 데이터 전처리 (고정 파이프라인)")
st.markdown("---")

# 1. 파일 업로드 및 데이터 로드 인터페이스
# 안내 문구를 라벨로 깔끔하게 처리하고, 내부 자동 생성 문구(200MB...)는 간결하게 유지되도록 구성
uploaded_file = st.file_uploader("📂 전처리할 CSV 파일을 선택하여 업로드하세요.", type=["csv"])

df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif os.path.exists("students_ai_usage.csv"):
    df = pd.read_csv("students_ai_usage.csv")
    st.info("💡 업로드된 파일이 없어, 기존 제공된 `students_ai_usage.csv` 데이터를 기본으로 사용합니다.")

# 데이터가 성공적으로 로드된 경우에만 이후 로직 실행
if df is not None:
    # 학생 데이터셋의 고유 연속형 수치 컬럼 (이상치 탐지 및 박스플롯용)
    NUMERIC_TARGET_COLS = ['age', 'study_hours_per_day', 'grades_before_ai', 'grades_after_ai', 'daily_screen_time_hours']

    # ----------------------------------------------------
    # 데이터 전처리 연산 프로세스 (백엔드 고정 처리)
    # ----------------------------------------------------
    
    # [데이터 1] 순수 원본 데이터 복사
    original_df = df.copy()
    
    # [데이터 2] 특정 전처리 적용 (AI 사용 여부 삭제 & 비어있는 AI 도구명 'None' 대체)
    processed_base_df = original_df.copy()
    
    # 1. AI 사용 여부(uses_ai) 컬럼 삭제 (존재할 경우)
    if 'uses_ai' in processed_base_df.columns:
        processed_base_df = processed_base_df.drop(columns=['uses_ai'])
        
    # 2. 사용한 AI(ai_tools_used) 컬럼이 비어있으면 'None'으로 대체
    if 'ai_tools_used' in processed_base_df.columns:
        processed_base_df['ai_tools_used'] = processed_base_df['ai_tools_used'].fillna('None')
    
    # 수치형 결측치도 안전하게 처리하기 위해 평균값 대체 추가
    for col in processed_base_df.columns:
        if processed_base_df[col].dtype in ['int64', 'float64']:
            processed_base_df[col] = processed_base_df[col].fillna(processed_base_df[col].mean())
        else:
            processed_base_df[col] = processed_base_df[col].fillna('None')

    # [데이터 3] 결측치/변수 정제 상태에서 이상치(행 제거) 추가 변환
    outlier_removed_df = processed_base_df.copy()
    outlier_info = []
    
    mask = pd.Series(True, index=outlier_removed_df.index)
    for col in [c for c in NUMERIC_TARGET_COLS if c in outlier_removed_df.columns]:
        q1 = outlier_removed_df[col].quantile(0.25)
        q3 = outlier_removed_df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        col_mask = outlier_removed_df[col].between(lower_bound, upper_bound)
        n_outliers = int((~col_mask).sum())
        
        outlier_info.append({
            "컬럼명": col,
            "Q1": round(q1, 3),
            "Q3": round(q3, 3),
            "IQR": round(iqr, 3),
            "하한 경계": round(lower_bound, 3),
            "상한 경계": round(upper_bound, 3),
            "제거된 이상치 수": n_outliers
        })
        mask &= col_mask
        
    outlier_removed_df = outlier_removed_df[mask]

    # ----------------------------------------------------
    # UI 화면 구성: 원본 및 전처리 데이터 각각 따로 보여주기
    # ----------------------------------------------------
    
    st.write("### 📊 단계별 데이터 정제 비교 모니터링")
    
    # 탭 메뉴 구성
    tab1, tab2, tab3 = st.tabs([
        "① 순수 원본 데이터 (Original)", 
        "② 결측치 처리 및 변수 정제 (Cleaned Base)", 
        "③ 이상치 제거 및 시각화 (Outlier Removed)"
    ])
    
    # Tab 1: 순수 원본 데이터 요약 및 미리보기
    with tab1:
        col_o1, col_o2 = st.columns([2, 1])
        with col_o1:
            st.write("##### 📋 원본 데이터 상위 10개 행")
            st.dataframe(original_df.head(10))
        with col_o2:
            st.write("##### 📋 데이터 요약")
            st.metric(label="전체 데이터 수", value=f"{original_df.shape[0]} 행")
            st.metric(label="변수(컬럼) 수", value=f"{original_df.shape[1]} 개")
            
            # 원본 결측치 세부 요약
            null_counts = original_df.isnull().sum()
            if null_counts.sum() > 0:
                st.warning(f"⚠️ 결측치 존재 컬럼:\n{null_counts[null_counts > 0]}")
            else:
                st.success("✅ 원본 데이터에 결측치가 없습니다.")

    # Tab 2: 특정 전처리 적용 데이터 미리보기 (uses_ai 삭제 / ai_tools_used 대체)
    with tab2:
        col_d1, col_d2 = st.columns([2, 1])
        with col_d1:
            st.write("##### 🧹 `uses_ai` 삭제 및 비어있는 AI 도구 `'None'` 대체 데이터 (상위 10개 행)")
            st.dataframe(processed_base_df.head(10))
        with col_d2:
            st.write("##### 📊 정제 결과 및 변경점")
            st.metric(
                label="최종 변수(열) 개수", 
                value=processed_base_df.shape[1], 
                delta=int(processed_base_df.shape[1] - original_df.shape[1])
            )
            st.markdown("""
            - **`uses_ai`** 변수가 데이터셋에서 제거되었습니다.
            - **`ai_tools_used`** 컬럼 내 비어있던 칸(NaN)이 전부 **`'None'`**으로 변경되었습니다.
            """)

    # Tab 3: 이상치까지 제거한 데이터 미리보기 + 박스플롯 그래프
    with tab3:
        st.write("##### 🚨 IQR 기준 이상치 정제 데이터 (상위 10개 행)")
        st.dataframe(outlier_removed_df.head(10))
        
        st.markdown("---")
        
        # 레이아웃 분할 (왼쪽: 수치 표, 오른쪽: 박스플롯)
        col_graph1, col_graph2 = st.columns([1, 1])
        
        with col_graph1:
            st.write("**📌 수치형 변수별 IQR 이상치 탐지 상세 내역**")
            st.dataframe(pd.DataFrame(outlier_info), use_container_width=True)
            
            # 변동 행 개수 메트릭
            st.metric(
                label="이상치 제거 후 최종 행 개수", 
                value=outlier_removed_df.shape[0], 
                delta=int(outlier_removed_df.shape[0] - original_df.shape[0])
            )
            
        with col_graph2:
            st.write("**📊 이상치 정제 후 데이터 박스플롯(Boxplot) 분포**")
            available_num_cols = [c for c in NUMERIC_TARGET_COLS if c in outlier_removed_df.columns]
            
            if available_num_cols:
                selected_box_col = st.selectbox("확인할 수치형 변수를 선택하세요:", available_num_cols)
                
                fig = px.box(
                    outlier_removed_df, 
                    y=selected_box_col, 
                    points="all", 
                    title=f"[{selected_box_col}] 이상치 처리 후 분포 상태",
                    color_discrete_sequence=['#FF4B4B']
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("시각화할 수치형 변수가 없습니다.")

    st.markdown("---")
    
    # ----------------------------------------------------
    # 최종 정제 데이터 다운로드 영역
    # ----------------------------------------------------
    st.write("### 📥 최종 정제 완료 데이터 다운로드")
    st.info("결측치 정제, 변수 제거, 이상치 행 제거가 모두 끝난 최종 데이터셋입니다.")
    
    csv_data = outlier_removed_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 전처리 완료 CSV 다운로드",
        data=csv_data,
        file_name="cleaned_student_ai_data.csv",
        mime="text/csv",
        use_container_width=True
    )
    
else:
    st.warning("데이터를 불러오지 못했습니다. `students_ai_usage.csv` 파일이 스크립트와 같은 경로에 있거나 웹 화면에 파일을 직접 업로드해주세요.")
