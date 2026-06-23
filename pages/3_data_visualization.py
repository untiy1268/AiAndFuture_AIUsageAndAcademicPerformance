# ════════════════════════════════════════════════════════════════
# 6. Box Plot — 교육 수준별 (오류 수정본)
# ════════════════════════════════════════════════════════════════
elif selected == "boxplot":
    edu_order  = ["school", "college"]
    edu_labels = {"school": "중·고등학교", "college": "대학교"}

    fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharey=True)
    
    for ax, edu in zip(axes, edu_order):
        # 해당 교육 수준의 데이터 필터링
        sub = df[df["education_level"] == edu]
        
        # 💡 해결책: 에러에 취약한 plt.boxplot 대신 빈 데이터도 안전하게 넘기는 sns.boxplot 활용
        sns.boxplot(
            data=sub,
            x="uses_ai",
            y="grade_change",
            order=["No", "Yes"],  # X축 순서 고정
            ax=ax,
            palette={"No": COLOR_NO, "Yes": COLOR_YES},  # 색상 매핑
            width=0.45,
            fliersize=4,   # 이상치(Outlier) 점 크기
            linewidth=1.5
        )
        
        # 박스 내부 투명도 살짝 적용 (기존 투명도 무드 유지)
        for patch in ax.patches:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, 0.75)) # 투명도 75%
            
        # 차트 세부 라벨 및 폰트 설정
        ax.set_title(edu_labels[edu], fontsize=13)
        ax.set_xlabel("AI 사용 여부", fontsize=11)
        ax.set_xticklabels(["AI 미사용", "AI 사용"])
        
        # 첫 번째 서브플롯에만 Y축 라벨 표시 (sharey=True 대응)
        if edu == "school":
            ax.set_ylabel("성적 향상 (점)", fontsize=11)
        else:
            ax.set_ylabel("") # 대학교 차트에서는 중복 라벨 제거

    fig.suptitle("교육 수준 × AI 사용 여부별 성적 향상 분포", fontsize=14, y=1.02)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # 💡 하단 안내 문구 출력 시 데이터가 없는 경우(NaN) 예외 처리 추가
    for edu in edu_order:
        sub_ai = df[(df["education_level"] == edu) & (df["uses_ai"] == "Yes")]
        if not sub_ai.empty:
            m = sub_ai["grade_change"].mean()
            st.info(f"**{edu_labels[edu]}** AI 사용 학생 평균 성적 향상: **+{m:.1f}점**")
        else:
            st.warning(f"**{edu_labels[edu]}** AI를 사용한 학생 데이터가 부족하여 평균을 계산할 수 없습니다.")
