import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

st.set_page_config(page_title="데이터 시각화", page_icon="📈")
st.title("📈 3. 데이터 시각화 (EDA)")
st.markdown("---")

# ── 데이터 로드 ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("students_ai_usage.csv")
    df["grade_change"] = df["grades_after_ai"] - df["grades_before_ai"]
    return df

df = load_data()

# ── 공통 색상 팔레트 ─────────────────────────────────────────────
COLOR_YES  = "#4C72B0"   # AI 사용
COLOR_NO   = "#DD8452"   # AI 미사용
TOOL_COLORS = {"ChatGPT": "#10A37F", "Copilot": "#7B61FF", "Gemini": "#EA4335"}
PURPOSE_COLORS = {"Research": "#4C72B0", "Homework": "#DD8452", "Coding": "#55A868"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 120,
})

st.subheader("📊 차트 생성 및 분석")

# ════════════════════════════════════════════════════════════════
# 1. AI 사용 여부 비율  (Donut Chart)
# ════════════════════════════════════════════════════════════════
st.markdown("### 1️⃣ AI 사용 여부 비율")

ai_counts = df["uses_ai"].value_counts()
fig1, ax1 = plt.subplots(figsize=(5, 5))
wedges, texts, autotexts = ax1.pie(
    ai_counts,
    labels=ai_counts.index,
    autopct="%1.1f%%",
    colors=[COLOR_YES, COLOR_NO],
    startangle=90,
    wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
    textprops=dict(fontsize=13),
)
for at in autotexts:
    at.set_fontsize(13)
    at.set_fontweight("bold")
ax1.set_title("AI 사용 여부 분포 (n=100)", fontsize=14, pad=14)
st.pyplot(fig1)
plt.close(fig1)

with st.expander("💡 분석 인사이트"):
    st.write(
        f"전체 학생 100명 중 AI를 사용하는 학생은 **{ai_counts['Yes']}명({ai_counts['Yes']}%)**, "
        f"미사용 학생은 **{ai_counts['No']}명({ai_counts['No']}%)** 입니다. "
        "약 40%의 학생이 이미 AI를 학습에 활용하고 있습니다."
    )

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# 2. AI 사용 전후 성적 비교  (Grouped Bar)
# ════════════════════════════════════════════════════════════════
st.markdown("### 2️⃣ AI 사용 전후 성적 비교")

groups = {
    "AI 사용 (n=40)": df[df["uses_ai"] == "Yes"],
    "AI 미사용 (n=60)": df[df["uses_ai"] == "No"],
}
labels   = list(groups.keys())
before   = [g["grades_before_ai"].mean() for g in groups.values()]
after    = [g["grades_after_ai"].mean()  for g in groups.values()]

x   = np.arange(len(labels))
w   = 0.35

fig2, ax2 = plt.subplots(figsize=(7, 5))
b1 = ax2.bar(x - w/2, before, w, label="AI 도입 전", color="#A8C4E0", edgecolor="white")
b2 = ax2.bar(x + w/2, after,  w, label="AI 도입 후", color=[COLOR_YES, COLOR_NO], edgecolor="white")

ax2.bar_label(b1, fmt="%.1f", padding=3, fontsize=11)
ax2.bar_label(b2, fmt="%.1f", padding=3, fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=12)
ax2.set_ylabel("평균 성적", fontsize=12)
ax2.set_title("AI 사용 전후 평균 성적 변화", fontsize=14, pad=10)
ax2.legend(fontsize=11)
ax2.set_ylim(0, 100)
st.pyplot(fig2)
plt.close(fig2)

ai_users = df[df["uses_ai"] == "Yes"]
avg_change = ai_users["grade_change"].mean()
with st.expander("💡 분석 인사이트"):
    st.write(
        f"AI를 사용한 학생들의 평균 성적은 **+{avg_change:.1f}점** 향상되었습니다. "
        "미사용 학생들의 성적은 변화가 없어, AI 활용이 학업 성취에 긍정적인 영향을 미치는 것으로 보입니다."
    )

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# 3. AI 도구별 평균 성적 향상  (Horizontal Bar)
# ════════════════════════════════════════════════════════════════
st.markdown("### 3️⃣ AI 도구별 평균 성적 향상")

tool_change = (
    df[df["uses_ai"] == "Yes"]
    .groupby("ai_tools_used")["grade_change"]
    .mean()
    .sort_values()
)

fig3, ax3 = plt.subplots(figsize=(7, 4))
colors3 = [TOOL_COLORS.get(t, "#999") for t in tool_change.index]
bars3 = ax3.barh(tool_change.index, tool_change.values, color=colors3, edgecolor="white", height=0.5)
ax3.bar_label(bars3, fmt="+%.1f점", padding=5, fontsize=11)
ax3.set_xlabel("평균 성적 향상 (점)", fontsize=12)
ax3.set_title("AI 도구별 평균 성적 향상 비교", fontsize=14, pad=10)
ax3.set_xlim(0, tool_change.max() * 1.3)
st.pyplot(fig3)
plt.close(fig3)

best_tool = tool_change.idxmax()
with st.expander("💡 분석 인사이트"):
    st.write(
        f"세 가지 AI 도구(ChatGPT, Copilot, Gemini) 중 **{best_tool}**를 사용한 학생들의 성적 향상이 "
        f"가장 컸습니다(+{tool_change.max():.1f}점). 각 도구마다 성적 향상폭에 차이가 있어, "
        "도구 선택이 학습 효과에 영향을 미칠 수 있습니다."
    )

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# 4. AI 활용 목적별 성적 향상  (Bar + 개별 점수 scatter)
# ════════════════════════════════════════════════════════════════
st.markdown("### 4️⃣ AI 활용 목적별 성적 향상")

purpose_df = df[df["uses_ai"] == "Yes"].copy()
purpose_change = purpose_df.groupby("purpose_of_ai")["grade_change"].mean().sort_values()

fig4, ax4 = plt.subplots(figsize=(7, 4.5))
colors4 = [PURPOSE_COLORS.get(p, "#999") for p in purpose_change.index]
bars4 = ax4.bar(purpose_change.index, purpose_change.values, color=colors4, edgecolor="white", width=0.5)

# 개별 데이터 포인트 overlay
for i, purpose in enumerate(purpose_change.index):
    vals = purpose_df[purpose_df["purpose_of_ai"] == purpose]["grade_change"]
    ax4.scatter(
        np.full(len(vals), i) + np.random.uniform(-0.12, 0.12, len(vals)),
        vals,
        color="black", alpha=0.4, s=20, zorder=3
    )

ax4.bar_label(bars4, fmt="+%.1f점", padding=4, fontsize=11)
ax4.set_ylabel("평균 성적 향상 (점)", fontsize=12)
ax4.set_title("AI 활용 목적별 평균 성적 향상", fontsize=14, pad=10)
ax4.set_ylim(0, purpose_change.max() * 1.3)
st.pyplot(fig4)
plt.close(fig4)

best_purpose = purpose_change.idxmax()
with st.expander("💡 분석 인사이트"):
    st.write(
        f"AI를 **{best_purpose}** 목적으로 활용한 학생들의 성적 향상이 가장 두드러졌습니다. "
        "막대 위 점들은 개별 학생 데이터로, 활용 목적 내에서도 개인차가 존재합니다."
    )

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# 5. 공부 시간 vs 성적 향상  (Scatter)
# ════════════════════════════════════════════════════════════════
st.markdown("### 5️⃣ 일일 공부 시간과 성적 향상의 관계")

ai_df  = df[df["uses_ai"] == "Yes"]
non_df = df[df["uses_ai"] == "No"]

fig5, ax5 = plt.subplots(figsize=(7, 5))
ax5.scatter(non_df["study_hours_per_day"], non_df["grade_change"],
            color=COLOR_NO,  alpha=0.6, s=60, label="AI 미사용", edgecolors="white")
ax5.scatter(ai_df["study_hours_per_day"],  ai_df["grade_change"],
            color=COLOR_YES, alpha=0.7, s=70, label="AI 사용",   edgecolors="white")

# 추세선 (AI 사용 그룹만)
m, b = np.polyfit(ai_df["study_hours_per_day"], ai_df["grade_change"], 1)
x_line = np.linspace(ai_df["study_hours_per_day"].min(), ai_df["study_hours_per_day"].max(), 100)
ax5.plot(x_line, m * x_line + b, color=COLOR_YES, linewidth=1.8, linestyle="--", alpha=0.8)

ax5.set_xlabel("일일 공부 시간 (시간)", fontsize=12)
ax5.set_ylabel("성적 향상 (점)", fontsize=12)
ax5.set_title("일일 공부 시간 vs 성적 향상 (AI 사용 여부별)", fontsize=14, pad=10)
ax5.legend(fontsize=11)
st.pyplot(fig5)
plt.close(fig5)

corr = ai_df[["study_hours_per_day", "grade_change"]].corr().iloc[0, 1]
with st.expander("💡 분석 인사이트"):
    st.write(
        f"AI 사용 학생 그룹에서 공부 시간과 성적 향상의 상관계수는 **{corr:.2f}**입니다. "
        "AI를 활용하더라도 충분한 공부 시간이 뒷받침될 때 더 큰 성적 향상을 기대할 수 있습니다."
    )

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# 6. 교육 수준별 성적 향상 분포  (Box Plot)
# ════════════════════════════════════════════════════════════════
st.markdown("### 6️⃣ 교육 수준 × AI 사용 여부별 성적 향상 분포")

edu_order  = ["school", "college"]
edu_labels = {"school": "중·고등학교", "college": "대학교"}

fig6, axes6 = plt.subplots(1, 2, figsize=(9, 5), sharey=True)
for ax, edu in zip(axes6, edu_order):
    sub = df[df["education_level"] == edu]
    data_yes = sub[sub["uses_ai"] == "Yes"]["grade_change"]
    data_no  = sub[sub["uses_ai"] == "No"]["grade_change"]

    bp = ax.boxplot(
        [data_no, data_yes],
        labels=["AI 미사용", "AI 사용"],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        widths=0.45,
    )
    bp["boxes"][0].set_facecolor(COLOR_NO  + "99")
    bp["boxes"][1].set_facecolor(COLOR_YES + "99")
    ax.set_title(f"{edu_labels[edu]}", fontsize=13)
    ax.set_ylabel("성적 향상 (점)" if edu == "school" else "", fontsize=11)

fig6.suptitle("교육 수준 × AI 사용 여부별 성적 향상 분포", fontsize=14, y=1.02)
plt.tight_layout()
st.pyplot(fig6)
plt.close(fig6)

with st.expander("💡 분석 인사이트"):
    for edu in edu_order:
        sub_yes = df[(df["education_level"] == edu) & (df["uses_ai"] == "Yes")]["grade_change"]
        st.write(
            f"**{edu_labels[edu]}** AI 사용 학생의 평균 성적 향상: "
            f"+{sub_yes.mean():.1f}점 (중앙값 +{sub_yes.median():.1f}점)"
        )

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# 7. 스크린 타임 vs 성적 향상  (Heatmap / Pivot)
# ════════════════════════════════════════════════════════════════
st.markdown("### 7️⃣ 스크린 타임과 공부 시간에 따른 성적 향상 히트맵")

ai_heat = df[df["uses_ai"] == "Yes"].copy()
ai_heat["screen_bin"] = pd.cut(
    ai_heat["daily_screen_time_hours"],
    bins=[1, 3, 5, 7],
    labels=["2~3시간", "4~5시간", "6~7시간"],
)
ai_heat["study_bin"] = pd.cut(
    ai_heat["study_hours_per_day"],
    bins=[0, 2, 3, 5],
    labels=["~2시간", "2~3시간", "3시간+"],
)

pivot = ai_heat.pivot_table(
    index="screen_bin", columns="study_bin",
    values="grade_change", aggfunc="mean"
)

fig7, ax7 = plt.subplots(figsize=(7, 4))
sns.heatmap(
    pivot,
    annot=True, fmt=".1f", cmap="YlOrRd",
    linewidths=0.5, linecolor="white",
    cbar_kws={"label": "평균 성적 향상 (점)"},
    ax=ax7,
)
ax7.set_xlabel("일일 공부 시간", fontsize=12)
ax7.set_ylabel("일일 스크린 타임", fontsize=12)
ax7.set_title("스크린 타임 × 공부 시간별 성적 향상 (AI 사용 학생)", fontsize=13, pad=10)
plt.tight_layout()
st.pyplot(fig7)
plt.close(fig7)

with st.expander("💡 분석 인사이트"):
    st.write(
        "히트맵에서 색이 진할수록 성적 향상이 큽니다. "
        "공부 시간이 길고 스크린 타임이 적절한 구간에서 AI 활용 효과가 극대화되는 경향을 확인할 수 있습니다."
    )

st.markdown("---")
st.caption("📌 데이터 출처: students_ai_usage.csv (n=100) | 시각화: Matplotlib · Seaborn")
