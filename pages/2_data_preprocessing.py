import os
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

# 페이지 설정
st.set_page_config(page_title="데이터 전처리", page_icon="⚙️", layout="wide")

st.title("⚙️ 2. 데이터 전처리 (고정 파이프라인)")
st.markdown("---")

st.subheader("🛠️ 데이터 정제 작업 결과")
st.write("업로드된 데이터로부터 결측치 제거 및 IQR 기준 이상치 제거를 순차적으로 수행한 독립된 데이터셋들을 비교합니다.")

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
    # 학생 데이터셋의 고유 연속형 수치 컬럼 (이상치 탐지에 사용)
    NUMERIC_TARGET_COLS = ['age', 'study_hours_per_day', 'grades_before_ai', 'grades_after_ai', 'daily_screen_time_hours']

    # ----------------------------------------------------
    # 데이터 전처리 연산 프로세스 (백엔드 고정 처리)
    # ----------------------------------------------------
    
    # [데이터 1] 순수 원본 데이터 복사
    original_df = df.copy()
    
    # [데이터 2] 결측치 제거 변환
    dropna_df = original_df.dropna()
    
    # [데이터 3] 결측치 제거 상태에서 이상치(행 제거) 추가 변환
    outlier_removed_df = dropna_df.copy()
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
    
    # 탭 메뉴를 구성하여 사용자가 각 상태별 데이터를 완전히 따로 볼 수 있게 구성
    tab1, tab2, tab3 = st.tabs([
        "① 순수 원본 데이터 (Original)", 
        "② 결측치 제거 데이터 (Drop NA)", 
        "③ 이상치 제거 데이터 (Outlier Removed)"
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
            null_sum = original_df.isnull().sum().sum()
            if null_sum > 0:
                st.warning(f"⚠️ 총 {null_sum}개의 결측치가 감지되었습니다.")
            else:
                st.success("✅ 데이터에 결측치가 없습니다.")

    # Tab 2: 결측치만 제거한 데이터 미리보기
    with tab2:
        col_d1, col_d2 = st.columns([2, 1])
        with col_d1:
            st.write("##### 🧹 결측치 행 제거 완료 데이터 (상위 10개 행)")
            st.dataframe(dropna_df.head(10))
        with col_d2:
            st.write("##### 📊 정제 결과 및 변경점")
            st.metric(
                label="최종 행 개수", 
                value=dropna_df.shape[0], 
                delta=int(dropna_df.shape[0] - original_df.shape[0])
            )
            st.caption("※ 결측치(NaN)가 하나라도 포함된 행이 목록에서 삭제되었습니다.")

    # Tab 3: 이상치까지 제거한 데이터 미리보기
    with tab3:
        col_out1, col_out2 = st.columns([2, 1])
        with col_out1:
            st.write("##### 🚨 IQR 기준 이상치 제거 완료 데이터 (상위 10개 행)")
            st.dataframe(outlier_removed_df.head(10))
            
            # 이상치 파단 기준 상세 표
            st.write("**📌 수치형 변수별 IQR 이상치 탐지 상세 내역**")
            st.dataframe(pd.DataFrame(outlier_info), use_container_width=True)
        with col_out2:
            st.write("##### 📊 정제 결과 및 변경점")
            st.metric(
                label="최종 행 개수", 
                value=outlier_removed_df.shape[0], 
                delta=int(outlier_removed_df.shape[0] - original_df.shape[0])
            )
            st.caption("※ 각 수치형 컬럼별로 [Q1 - 1.5*IQR] 미만이거나 [Q3 + 1.5*IQR] 초과인 극단치 행이 누적 제거되었습니다.")

    st.markdown("---")
    
    # ----------------------------------------------------
    # 최종 정제 데이터 다운로드 영역
    # ----------------------------------------------------
    st.write("### 📥 최종 정제 완료 데이터 다운로드")
    st.info("결측치와 이상치가 모두 깔끔하게 처리된 최종 데이터셋을 다운로드할 수 있습니다.")
    
    # 최종 완성 데이터 내보내기용 바이트 변환
    csv_data = outlier_removed_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 전처리(결측치+이상치 제거) 완료 CSV 다운로드",
        data=csv_data,
        file_name="cleaned_student_data.csv",
        mime="text/csv",
        use_container_width=True
    )
    
else:
    st.warning("데이터를 불러오지 못했습니다. `students_ai_usage.csv` 파일이 스크립트와 같은 경로에 있거나 웹 화면에 파일을 직접 업로드해주세요.")
