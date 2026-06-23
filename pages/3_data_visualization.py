import subprocess, sys, os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

st.set_page_config(page_title="데이터 시각화", page_icon="📈")

# ── 한글 폰트 자동 설치 (최초 1회) ──────────────────────────────
@st.cache_resource
def setup_korean_font():
    nanum = [f for f in fm.fontManager.ttflist if "Nanum" in f.name]
    if nanum:
        return nanum[0].name
    try:
        subprocess.run(["apt-get", "install", "-y", "fonts-nanum"],
                       capture_output=True, check=True)
        fm._load_fontmanager(try_read_cache=False)
        nanum = [f for f in fm.fontManager.ttflist if "Nanum" in f.name]
        if nanum:
            return nanum[0].name
    except Exception:
        pass
    return "DejaVu Sans"

font_name = setup_korean_font()
plt.rcParams.update({
    "font.family": font_name,
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 120,
})

st.title("📈 3. 데이터 시각화 (EDA)")
st.markdown("---")

# ── 데이터 로드 ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("students_ai_usage.csv")
    df["grade_change"] = df["grades_after_ai"] - df["grades_before_ai"]
    return df

df = load_data()

# ── 공통 색상 ────────────────────────────────────────────────────
COLOR_YES = "#4C72B0"
COLOR_NO  = "#DD8452"
TOOL_COLORS    = {"ChatGPT": "#10A37F", "Copilot": "#7B61FF", "Gemini": "#EA4335"}
PURPOSE_COLORS = {"Research": "#4C72B0", "Homework": "#DD8452", "Coding": "#55A868"}

# ── 사이드바 차트 선택 ───────────────────────────────────────────
st.sidebar.title("📊 차트 선택")
chart_options = {
    "1️⃣ AI 사용 여부 비율": "donut",
    "2️⃣ AI 사용 전후 성적 비교": "grouped_bar",
    "3️⃣ AI 도구별 평균 성적 향상": "tool_bar",
    "4️⃣ AI 활용 목적별 성적 향상": "purpose_bar",
    "5️⃣ 공부 시간 vs 성적 향상": "scatter",
    "6️⃣ 교육 수준별 성적 분포": "boxplot",
    "7️⃣ 스크린 타임 × 공부 시간 히트맵": "heatmap",
}
selected_label = st.sidebar.radio("보고 싶은 차트를 선택하세요", list(chart_options.keys()))
selected = chart_options[selected_label]

st.subheader(selected_label)

# ════════════════════════════════════════════════════════════════
# 1. Donut Chart — AI 사용 비율
# ════════════════════════════════════════════════════════════════
if selected == "donut":
    ai_counts = df["uses_ai"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(
        ai_counts,
        labels=["AI 사용", "AI 미사용"],
        autopct="%1.1f%%",
        colors=[COLOR_YES, COLOR_NO],
        startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=13),
    )
    for at in autotexts:
        at.set_fontsize(13)
        at.set_fontweight("bold")
    ax.set_title("AI 사용 여부 분포 (n=100)", fontsize=14, pad=14)
    st.pyplot(fig)
    plt.close(fig)
    st.info(
        f"전체 학생 100명 중 AI 사용 **{ai_counts['Yes']}명(40%)**, "
        f"미사용 **{ai_counts['No']}명(60%)** 입니다. "
        "약 40%의 학생이 이미 AI를 학습에 활용하고 있습니다."
    )

# ════════════════════════════════════════════════════════════════
# 2. Grouped Bar — 전후 성적 비교
# ════════════════════════════════════════════════════════════════
elif selected == "grouped_bar":
    groups = {"AI 사용 (n=40)": df[df["uses_ai"]=="Yes"], "AI 미사용 (n=60)": df[df["uses_ai"]=="No"]}
    labels = list(groups.keys())
    before = [g["grades_before_ai"].mean() for g in groups.values()]
    after  = [g["grades_after_ai"].mean()  for g in groups.values()]
    x, w   = np.arange(len(labels)), 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    b1 = ax.bar(x - w/2, before, w, label="AI 도입 전", color="#A8C4E0", edgecolor="white")
    b2 = ax.bar(x + w/2, after,  w, label="AI 도입 후", color=[COLOR_YES, COLOR_NO], edgecolor="white")
    ax.bar_label(b1, fmt="%.1f", padding=3, fontsize=11)
    ax.bar_label(b2, fmt="%.1f", padding=3, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("평균 성적", fontsize=12)
    ax.set_title("AI 사용 전후 평균 성적 변화", fontsize=14, pad=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 100)
    st.pyplot(fig)
    plt.close(fig)
    avg_change = df[df["uses_ai"]=="Yes"]["grade_change"].mean()
    st.info(f"AI를 사용한 학생들의 평균 성적은 **+{avg_change:.1f}점** 향상되었습니다. 미사용 학생 성적은 변화가 없었습니다.")

# ════════════════════════════════════════════════════════════════
# 3. Horizontal Bar — 도구별 향상
# ════════════════════════════════════════════════════════════════
elif selected == "tool_bar":
    tool_change = df[df["uses_ai"]=="Yes"].groupby("ai_tools_used")["grade_change"].mean().sort_values()
    colors3 = [TOOL_COLORS.get(t, "#999") for t in tool_change.index]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(tool_change.index, tool_change.values, color=colors3, edgecolor="white", height=0.5)
    ax.bar_label(bars, fmt="+%.1f점", padding=5, fontsize=11)
    ax.set_xlabel("평균 성적 향상 (점)", fontsize=12)
    ax.set_title("AI 도구별 평균 성적 향상 비교", fontsize=14, pad=10)
    ax.set_xlim(0, tool_change.max() * 1.3)
    st.pyplot(fig)
    plt.close(fig)
    best = tool_change.idxmax()
    st.info(f"**{best}**를 사용한 학생의 성적 향상이 가장 컸습니다 (+{tool_change.max():.1f}점).")

# ════════════════════════════════════════════════════════════════
# 4. Bar + Scatter overlay — 목적별 향상
# ════════════════════════════════════════════════════════════════
elif selected == "purpose_bar":
    purpose_df     = df[df["uses_ai"]=="Yes"].copy()
    purpose_change = purpose_df.groupby("purpose_of_ai")["grade_change"].mean().sort_values()
    colors4 = [PURPOSE_COLORS.get(p, "#999") for p in purpose_change.index]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(purpose_change.index, purpose_change.values, color=colors4, edgecolor="white", width=0.5)
    for i, purpose in enumerate(purpose_change.index):
        vals = purpose_df[purpose_df["purpose_of_ai"]==purpose]["grade_change"]
        ax.scatter(np.full(len(vals), i) + np.random.uniform(-0.12, 0.12, len(vals)),
                   vals, color="black", alpha=0.4, s=20, zorder=3)
    ax.bar_label(bars, fmt="+%.1f점", padding=4, fontsize=11)
    ax.set_ylabel("평균 성적 향상 (점)", fontsize=12)
    ax.set_title("AI 활용 목적별 평균 성적 향상", fontsize=14, pad=10)
    ax.set_ylim(0, purpose_change.max() * 1.3)
    st.pyplot(fig)
    plt.close(fig)
    best = purpose_change.idxmax()
    st.info(f"**{best}** 목적으로 AI를 활용한 학생의 성적 향상이 가장 컸습니다. 점들은 개별 학생 데이터입니다.")

# ════════════════════════════════════════════════════════════════
# 5. Scatter + 추세선
# ════════════════════════════════════════════════════════════════
elif selected == "scatter":
    ai_df  = df[df["uses_ai"]=="Yes"]
    non_df = df[df["uses_ai"]=="No"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(non_df["study_hours_per_day"], non_df["grade_change"],
               color=COLOR_NO,  alpha=0.6, s=60, label="AI 미사용", edgecolors="white")
    ax.scatter(ai_df["study_hours_per_day"],  ai_df["grade_change"],
               color=COLOR_YES, alpha=0.7, s=70, label="AI 사용",   edgecolors="white")
    m, b = np.polyfit(ai_df["study_hours_per_day"], ai_df["grade_change"], 1)
    x_line = np.linspace(ai_df["study_hours_per_day"].min(), ai_df["study_hours_per_day"].max(), 100)
    ax.plot(x_line, m*x_line+b, color=COLOR_YES, linewidth=1.8, linestyle="--", alpha=0.8)
    ax.set_xlabel("일일 공부 시간 (시간)", fontsize=12)
    ax.set_ylabel("성적 향상 (점)", fontsize=12)
    ax.set_title("일일 공부 시간 vs 성적 향상", fontsize=14, pad=10)
    ax.legend(fontsize=11)
    st.pyplot(fig)
    plt.close(fig)
    corr = ai_df[["study_hours_per_day","grade_change"]].corr().iloc[0,1]
    st.info(f"AI 사용 학생 그룹에서 공부 시간과 성적 향상의 상관계수는 **{corr:.2f}**입니다.")

# ════════════════════════════════════════════════════════════════
# 6. Box Plot — 교육 수준별
# ════════════════════════════════════════════════════════════════
elif selected == "boxplot":
    edu_order  = ["school", "college"]
    edu_labels = {"school": "중·고등학교", "college": "대학교"}

    fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharey=True)
    for ax, edu in zip(axes, edu_order):
        sub = df[df["education_level"]==edu]
        bp  = ax.boxplot(
            [sub[sub["uses_ai"]=="No"]["grade_change"], sub[sub["uses_ai"]=="Yes"]["grade_change"]],
            labels=["AI 미사용", "AI 사용"],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            widths=0.45,
        )
        bp["boxes"][0].set_facecolor(COLOR_NO  + "99")
        bp["boxes"][1].set_facecolor(COLOR_YES + "99")
        ax.set_title(edu_labels[edu], fontsize=13)
        if edu == "school":
            ax.set_ylabel("성적 향상 (점)", fontsize=11)
    fig.suptitle("교육 수준 × AI 사용 여부별 성적 향상 분포", fontsize=14, y=1.02)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    for edu in edu_order:
        m = df[(df["education_level"]==edu)&(df["uses_ai"]=="Yes")]["grade_change"].mean()
        st.info(f"**{edu_labels[edu]}** AI 사용 학생 평균 성적 향상: **+{m:.1f}점**")

# ════════════════════════════════════════════════════════════════
# 7. Heatmap — 스크린타임 × 공부시간
# ════════════════════════════════════════════════════════════════
elif selected == "heatmap":
    ai_heat = df[df["uses_ai"]=="Yes"].copy()
    ai_heat["screen_bin"] = pd.cut(ai_heat["daily_screen_time_hours"],
                                   bins=[1,3,5,7], labels=["2~3시간","4~5시간","6~7시간"])
    ai_heat["study_bin"]  = pd.cut(ai_heat["study_hours_per_day"],
                                   bins=[0,2,3,5], labels=["~2시간","2~3시간","3시간+"])
    pivot = ai_heat.pivot_table(index="screen_bin", columns="study_bin",
                                values="grade_change", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "평균 성적 향상 (점)"}, ax=ax)
    ax.set_xlabel("일일 공부 시간", fontsize=12)
    ax.set_ylabel("일일 스크린 타임", fontsize=12)
    ax.set_title("스크린 타임 × 공부 시간별 성적 향상\n(AI 사용 학생)", fontsize=13, pad=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.info("색이 진할수록 성적 향상이 큽니다. 공부 시간이 길수록 AI 활용 효과가 더 크게 나타납니다.")

st.markdown("---")
st.caption("📌 데이터 출처: students_ai_usage.csv (n=100) | 시각화: Matplotlib · Seaborn")
