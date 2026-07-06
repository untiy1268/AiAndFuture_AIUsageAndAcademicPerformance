import os
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

# 페이지 설정
st.set_page_config(page_title="데이터 전처리", page_icon="⚙️", layout="wide")

st.title("⚙️ 2. 데이터 전처리")
st.markdown("---")

st.subheader("🛠️ 데이터 정제 작업")
st.write("결측치(Null) 처리, 범주형 인코딩, 데이터 표준화 등의 실제 전처리 작업을 수행합니다.")

# 1. 파일 업로드 및 데이터 로드 인터페이스
uploaded_file = st.file_uploader("전처리할 CSV 파일을 업로드하세요.", type=["csv"])

df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif os.path.exists("students_ai_usage.csv"):
    df = pd.read_csv("students_ai_usage.csv")
    st.info("💡 업로드된 파일이 없어, 기존 제공된 `students_ai_usage.csv` 데이터를 기본으로 사용합니다.")

# 데이터가 성공적으로 로드된 경우에만 이후 로직 실행
if df is not None:
    # 원본 데이터 정보 보여주기
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("### 📊 원본 데이터 미리보기")
        st.dataframe(df.head(10))
    with col2:
        st.write("### 📋 데이터 요약")
        st.write(f"- **전체 데이터 수:** {df.shape[0]} 행")
        st.write(f"- **변수(컬럼) 수:** {df.shape[1]} 개")
        
        # 결측치 확인
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            st.warning(f"⚠️ 결측치가 존재하는 컬럼이 있습니다:\n{null_counts[null_counts > 0]}")
        else:
            st.success("✅ 데이터에 결측치가 없습니다.")

    st.markdown("---")

    # 2. 전처리 옵션 선택 인터페이스
    preprocessing_option = st.multiselect(
        '적용할 전처리 기법을 선택하세요 (선택 순서대로 순차 적용됩니다):',
        ['결측치 제거 (Drop Na)', '평균값으로 결측치 채우기 (Imputation)', '원-핫 인코딩 (One-Hot Encoding)', '데이터 표준화 (Standard Scaling)']
    )

    if preprocessing_option:
        # 원본 데이터 복사하여 전처리 수행
        processed_df = df.copy()
        
        st.write("### 🔄 전처리 진행 과정")
        
        # [STEP 1] 결측치 제거
        if '결측치 제거 (Drop Na)' in preprocessing_option:
            before_rows = processed_df.shape[0]
            processed_df = processed_df.dropna()
            after_rows = processed_df.shape[0]
            st.success(f"🧹 **결측치 제거 완료:** 기존 {before_rows}행 ➡️ 처리 후 {after_rows}행 (AI 미사용자의 데이터가 유실될 수 있습니다.)")
            
        # [STEP 2] 결측치 채우기 (결측치 제거가 선택되지 않았거나, 남아있는 결측치 처리)
        if '평균값으로 결측치 채우기 (Imputation)' in preprocessing_option:
            for col in processed_df.columns:
                if processed_df[col].dtype in ['int64', 'float64']:
                    # 수치형 변수는 평균값으로 대체
                    mean_val = processed_df[col].mean()
                    processed_df[col] = processed_df[col].fillna(mean_val)
                else:
                    # 범주형 변수(ai_tools_used 등)는 AI 미사용을 뜻하므로 맥락상 'None'으로 대체
                    processed_df[col] = processed_df[col].fillna('None')
            st.success("✏️ **결측치 대체 완료:** 수치형 변수는 평균값으로, 범주형 변수는 'None' 문자열로 채웠습니다.")
            
        # [STEP 3] 원-핫 인코딩
        if '원-핫 인코딩 (One-Hot Encoding)' in preprocessing_option:
            cat_cols = processed_df.select_dtypes(include=['object']).columns.tolist()
            if cat_cols:
                # 데이터 확인 및 0과 1 숫자로 직관적인 변환을 위해 dtype=int 부여
                processed_df = pd.get_dummies(processed_df, columns=cat_cols, drop_first=True, dtype=int)
                st.success(f"🔢 **원-핫 인코딩 완료:** 범주형 변수 {cat_cols} 변환 완료 (컬럼 수가 늘어납니다.)")
            else:
                st.info("ℹ️ 인코딩할 범주형(텍스트) 변수가 데이터에 없습니다.")
                
        # [STEP 4] 데이터 표준화
        if '데이터 표준화 (Standard Scaling)' in preprocessing_option:
            # 학생 데이터셋의 고유 연속형 수치 컬럼 지정
            num_cols = ['age', 'study_hours_per_day', 'grades_before_ai', 'grades_after_ai', 'daily_screen_time_hours']
            # 인코딩 등으로 컬럼이 유실되거나 변한 경우를 대비해 존재하는 컬럼만 필터링
            num_cols = [col for col in num_cols if col in processed_df.columns]
            
            if num_cols:
                scaler = StandardScaler()
                processed_df[num_cols] = scaler.fit_transform(processed_df[num_cols])
                st.success(f"⚖️ **데이터 표준화 완료:** 주요 수치형 변수({num_cols})를 평균 0, 표준편차 1로 스케일링했습니다.")

        st.markdown("---")
        
        # 3. 전처리 완료 데이터 출력 및 다운로드
        st.write("### 📉 전처리 완료 데이터 확인")
        st.dataframe(processed_df.head(10))
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric(label="최종 행 개수", value=processed_df.shape[0], delta=int(processed_df.shape[0] - df.shape[0]))
        with col_res2:
            st.metric(label="최종 열 개수", value=processed_df.shape[1], delta=int(processed_df.shape[1] - df.shape[1]))
            
        # CSV 다운로드 기능
        csv_data = processed_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 전처리된 CSV 파일 다운로드",
            data=csv_data,
            file_name="preprocessed_student_data.csv",
            mime="text/csv"
        )
    else:
        st.info("💡 좌측 혹은 상단의 다중 선택 창에서 전처리 기법을 1개 이상 선택하면 실시간으로 연산이 수행됩니다.")
else:
    st.warning("데이터를 불러오지 못했습니다. `students_ai_usage.csv` 파일이 스크립트와 같은 경로에 있거나 웹 화면에 파일을 직접 업로드해주세요.")
