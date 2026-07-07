# [수정 가이드] 코드 중간의 "# 원본 데이터 정보 보여주기" 부분을 아래처럼 변경해보세요!

st.write("### 📊 단계별 데이터 상태 모니터링")
tab1, tab2, tab3 = st.tabs(["순수 원본 데이터", "결측치 처리 후", "이상치 처리 후"])

with tab1:
    st.write("#### 📋 오리지널 업로드 데이터")
    st.dataframe(df.head(10))
    st.caption(f"행: {df.shape[0]}개 / 열: {df.shape[1]}개")

with tab2:
    st.write("#### 🧹 결측치 정제 상태 미리보기")
    # 결측치 처리 로직만 임시 반영해서 보여주기
    tmp_df = df.dropna() 
    st.dataframe(tmp_df.head(10))
    st.caption(f"행: {tmp_df.shape[0]}개 (결측치로 인해 {df.shape[0] - tmp_df.shape[0]}행 제거됨)")

with tab3:
    st.write("#### 🚨 IQR 기준 이상치 제거 상태 미리보기")
    # 수치형 변수에서 이상치를 제거한 임시 데이터 생성 후 출력
    tmp_outlier_df = df.dropna()
    mask = pd.Series(True, index=tmp_outlier_df.index)
    for col in [c for c in NUMERIC_TARGET_COLS if c in tmp_outlier_df.columns]:
        q1 = tmp_outlier_df[col].quantile(0.25)
        q3 = tmp_outlier_df[col].quantile(0.75)
        iqr = q3 - q1
        mask &= tmp_outlier_df[col].between(q1 - 1.5*iqr, q3 + 1.5*iqr)
    tmp_outlier_df = tmp_outlier_df[mask]
    
    st.dataframe(tmp_outlier_df.head(10))
    st.caption(f"행: {tmp_outlier_df.shape[0]}개 (이상치로 인해 추가 제거됨)")
