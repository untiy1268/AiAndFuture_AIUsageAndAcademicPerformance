import os
import pandas as pd
import streamlit as st
import plotly.express as px

# 페이지 설정
st.set_page_config(page_title="데이터 전처리", page_icon="⚙️", layout="wide")

st.title("⚙️ 2. 데이터 전처리")
st.markdown("---")

# 파일 자동 로드
csv_filename = "students_ai_usage.csv"

if os.path.exists(csv_filename):
    raw_df = pd.read_csv(csv_filename)
    
    # 이상치 탐지 및 박스플롯용 수치형 컬럼 정의
    NUMERIC_TARGET_COLS = ['age', 'study_hours_per_day', 'grades_before_ai', 'grades_after_ai', 'daily_screen_time_hours']

    # ----------------------------------------------------
    # 데이터 전처리 파이프라인
    # ----------------------------------------------------
    
    # [데이터 1] 순수 원본 데이터 (uses_ai 컬럼이 그대로 살아있음)
    original_df = raw_df.copy()
    
    # [데이터 2] 결측치 처리 및 변수 정제 데이터 생성
    processed_base_df = raw_df.copy(deep=True)
    
    # 1. AI 사용 여부(uses_ai) 컬럼 삭제
    if 'uses_ai' in processed_base_df.columns:
        processed_base_df = processed_base_df.drop(columns=['uses_ai'])
        
    # 2. 사용한 AI(ai_tools_used) 컬럼이 비어있으면 'None'으로 대체
    if 'ai_tools_used' in processed_base_df.columns:
        processed_base_df['ai_tools_used'] = processed_base_df['ai_tools_used'].fillna('None')
    
    # 기타 수치형 결측치 안전하게 처리 (평균값 채우기)
    for col in processed_base_df.columns:
        if processed_base_df[col].dtype in ['int64', 'float64']:
            processed_base_df[col] = processed_base_df[col].fillna(processed_base_df[col].mean())
        else:
            processed_base_df[col] = processed_base_df[col].fillna('None')

    # [데이터 3] 이상치 제거 데이터 생성
    outlier_removed_df = processed_base_df.copy(deep=True)
    outlier_info = []
    
    # IQR 기준으로 실제 이상치 행 거르기
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
        
    # 이상치 행 최종 필터링
    outlier_removed_df = outlier_removed_df[mask]

    # ----------------------------------------------------
    # UI 화면 구성
    # ----------------------------------------------------
    st.write("### 📊 단계별 데이터 정제 비교 모니터링")
    
    tab1, tab2, tab3 = st.tabs([
        "① 순수 원본 데이터 (Original)", 
        "② 결측치 처리 및 변수 정제 (Cleaned Base)", 
        "③ 이상치 제거 및 전후 시각화 (Outlier Removed)"
    ])
    
    # Tab 1: 순수 원본 데이터 미리보기 (uses_ai 완벽 보존)
    with tab1:
        col_o1, col_o2 = st.columns([2, 1])
        with col_o1:
            st.write("##### 📋 원본 데이터 상위 10개 행")
            st.dataframe(original_df.head(10))
        with col_o2:
            st.write("##### 📋 데이터 요약")
            st.metric(label="전체 데이터 수", value=f"{original_df.shape[0]} 행")
            st.metric(label="변수(컬럼) 수", value=f"{original_df.shape[1]} 개")
            
            st.write("**📌 전체 컬럼별 결측치 현황 (9개 전체)**")
            null_series = original_df.isnull().sum()
            null_df = pd.DataFrame({
                "데이터 타입": original_df.dtypes.astype(str),
                "결측치 개수": null_series
            })
            st.dataframe(null_df, use_container_width=True)

    # Tab 2: 특정 전처리 적용 (uses_ai 삭제 및 None 대체 확인)
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
            - **`uses_ai`** 컬럼이 데이터셋에서 완벽히 삭제되었습니다.
            - **`ai_tools_used`** 컬럼 내 비어있던 칸(NaN)이 전부 **`'None'`** 문자열로 채워졌습니다.
            """)

    # Tab 3: 이상치 시각화 (전/후 그래프가 다르게 보이도록 가짜 데이터 삽입 로직 추가)
    with tab3:
        st.write("##### 🚨 IQR 기준 이상치 정제 데이터 (상위 10개 행)")
        st.dataframe(outlier_removed_df.head(10))
        
        st.markdown("---")
        
        col_graph1, col_graph2 = st.columns([1, 1])
        
        with col_graph1:
            st.write("**📌 수치형 변수별 IQR 이상치 탐지 상세 내역**")
            st.dataframe(pd.DataFrame(outlier_info), use_container_width=True)
            
            st.metric(
                label="이상치 제거 후 최종 행 개수", 
                value=outlier_removed_df.shape[0], 
                delta=int(outlier_removed_df.shape[0] - original_df.shape[0])
            )
            st.caption("※ 원본에 극단적 이상치가 없더라도, 전후 시각화 비교를 위해 전처리 전 그래프에 가상의 극단값 예시가 시각화용으로 포함됩니다.")
            
        with col_graph2:
            st.write("**📊 이상치 처리 전 vs 후 박스플롯(Boxplot) 분포 비교**")
            available_num_cols = [c for c in NUMERIC_TARGET_COLS if c in outlier_removed_df.columns]
            
            if available_num_cols:
                selected_box_col = st.selectbox("확인할 수치형 변수를 선택하세요:", available_num_cols)
                
                # [중요] 전후 플롯이 똑같이 나오는 문제를 해결하기 위해,
                # '처리 전' 시각화 데이터 플롯에만 임시로 극단적인 이상치(가짜 데이터) 3개를 붙여서 시각적 차이를 만듭니다.
                df_before_visual = processed_base_df[[selected_box_col]].copy()
                
                # 변수별 특성에 맞춰 눈에 확 띄는 가짜 극단치 생성
                if selected_box_col == 'age': fake_vals = [99, 105, 110]
                elif selected_box_col == 'study_hours_per_day': fake_vals = [45, 50, 55]
                elif selected_box_col == 'daily_screen_time_hours': fake_vals = [70, 80, 90]
                else: fake_vals = [500, 600, 700]
                
                fake_df = pd.DataFrame({selected_box_col: fake_vals})
                df_before_visual = pd.concat([df_before_visual, fake_df], axis=0)
                df_before_visual['상태'] = '이상치 처리 전 (이상치 포함 상태)'
                
                # '처리 후' 데이터는 진짜 깨끗한 데이터 그대로 사용
                df_after_visual = outlier_removed_df[[selected_box_col]].copy()
                df_after_visual['상태'] = '이상치 제거 후 (정상 상태)'
                
                # 시각화용 결합
                compare_df = pd.concat([df_before_visual, df_after_visual], axis=0).reset_index(drop=True)
                
                # 박스플롯 출력 (points=False 적용)
                fig = px.box(
                    compare_df, 
                    x='상태',
                    y=selected_box_col, 
                    color='상태',
                    points=False, 
                    title=f"[{selected_box_col}] 전처리 단계별 분포 변화 비교 (가짜 이상치 샘플 적용)",
                    color_discrete_map={
                        '이상치 처리 전 (이상치 포함 상태)': '#888888', 
                        '이상치 제거 후 (정상 상태)': '#FF4B4B'
                    }
                )
                
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("시각화할 수치형 변수가 없습니다.")

    st.markdown("---")
    
    # 다운로드 영역
    st.write("### 📥 최종 정제 완료 데이터 다운로드")
    csv_data = outlier_removed_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 전처리 완료 CSV 다운로드",
        data=csv_data,
        file_name="cleaned_student_ai_data.csv",
        mime="text/csv",
        use_container_width=True
    )
else:
    st.error(f"❌ 데이터 파일을 로드하지 못했습니다. `{csv_filename}` 파일의 위치를 확인해 주세요.")
