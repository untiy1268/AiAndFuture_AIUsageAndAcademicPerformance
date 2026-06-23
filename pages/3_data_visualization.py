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
        f"전체 학생 100명 중 AI 사용 **{ai_counts.get('Yes', 0)}명(40%)**, "
        f"미사용 **{ai_counts.get('No', 0)}명(60%)** 입니다. "
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
# 3. Horizontal Bar — 도구별
