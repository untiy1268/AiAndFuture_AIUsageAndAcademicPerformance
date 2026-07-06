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
            fill_info = []  # 어떤 컬럼에 어떤 값이 채워졌는지 기록
            for col in processed_df.columns:
                if processed_df[col].dtype in ['int64', 'float64']:
                    # 수치형 변수는 평균값으로 대체
                    mean_val = processed_df[col].mean()
                    n_filled = processed_df[col].isnull().sum()
                    if n_filled > 0:
                        fill_info.append({
                            "컬럼명": col,
                            "타입": "수치형",
                            "채운 값": round(mean_val, 3),
                            "채운 개수": int(n_filled)
                        })
                    processed_df[col] = processed_df[col].fillna(mean_val)
                else:
                    # 범주형 변수(ai_tools_used 등)는 AI 미사용을 뜻하므로 맥락상 'None'으로 대체
                    n_filled = processed_df[col].isnull().sum()
                    if n_filled > 0:
                        fill_info.append({
                            "컬럼명": col,
                            "타입": "범주형",
                            "채운 값": "None",
                            "채운 개수": int(n_filled)
                        })
                    processed_df[col] = processed_df[col].fillna('None')

            st.success("✏️ **결측치 대체 완료:** 수치형 변수는 평균값으로, 범주형 변수는 'None' 문자열로 채웠습니다.")

            # 실제로 어떤 값이 채워졌는지 표로 보여주기
            if fill_info:
                st.write("**📌 컬럼별 결측치 대체 상세 내역**")
                fill_info_df = pd.DataFrame(fill_info)
                st.dataframe(fill_info_df, use_container_width=True)
            else:
                st.info("ℹ️ 결측치가 있는 컬럼이 없어 대체된 값이 없습니다.")

        # [STEP 3] 원-핫 인코딩
        if '원-핫 인코딩 (One-Hot Encoding)' in preprocessing_option:
            cat_cols = processed_df.select_dtypes(include=['object']).columns.tolist()
            if cat_cols:
                # 인코딩 전, 각 컬럼의 고유값들을 미리 기록
                before_unique = {col: sorted(processed_df[col].dropna().unique().tolist()) for col in cat_cols}
                before_cols_snapshot = processed_df.columns.tolist()

                # 데이터 확인 및 0과 1 숫자로 직관적인 변환을 위해 dtype=int 부여
                processed_df = pd.get_dummies(processed_df, columns=cat_cols, drop_first=True, dtype=int)

                st.success(f"🔢 **원-핫 인코딩 완료:** 범주형 변수 {cat_cols} 변환 완료 (컬럼 수가 늘어납니다.)")

                # 어떤 컬럼이 어떤 값들로, 어떤 새 컬럼으로 바뀌었는지 상세 표시
                st.write("**📌 원-핫 인코딩 상세 내역**")
                encoding_detail = []
                new_cols = [c for c in processed_df.columns if c not in before_cols_snapshot]

                for col in cat_cols:
                    uniques = before_unique[col]
                    # drop_first=True이므로 첫 번째 값은 기준값(baseline)으로 드롭됨 (전부 0으로 표현)
                    baseline = uniques[0] if uniques else None
                    generated_cols = [c for c in new_cols if c.startswith(col + "_")]

                    encoding_detail.append({
                        "원본 컬럼": col,
                        "원본 고유값 목록": ", ".join(map(str, uniques)),
                        "기준값(삭제됨)": baseline,
                        "생성된 컬럼": ", ".join(generated_cols) if generated_cols else "(없음)"
                    })

                encoding_detail_df = pd.DataFrame(encoding_detail)
                st.dataframe(encoding_detail_df, use_container_width=True)

                with st.expander("💡 원-핫 인코딩 결과 해석 방법 보기"):
                    st.markdown("""
                    - 각 원본 컬럼의 고유값 중 **기준값(baseline)** 은 `drop_first=True` 옵션에 의해 별도 컬럼 없이 삭제됩니다.
                    - 생성된 컬럼들이 모두 **0**이면 → 해당 행은 **기준값**에 해당합니다.
                    - 생성된 컬럼 중 하나가 **1**이면 → 해당 행은 그 컬럼 이름에 해당하는 값입니다.
                    - 예: `education_level` 고유값이 `['고졸', '대졸', '대학원']`이라면, `고졸`이 기준값으로 삭제되고
                      `education_level_대졸`, `education_level_대학원` 두 컬럼이 생성됩니다.
                    """)
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
