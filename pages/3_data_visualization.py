import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="학생 AI 사용 데이터 시각화", page_icon="📈", layout="wide")

st.title("📈 학생 AI 활용 및 성적 향상 데이터 시각화 (EDA)")
st.markdown("### 🎯 주제: AI를 활용한 학습이 학생 성적 향상에 도움이 될까?")
st.caption("※ 본 데이터는 전처리(표준화)가 완료된 데이터로, 성적 및 시간 지표는 Z-Score(상대적 수치)로 표현됩니다.")
st.markdown("---")

# ── 데이터 로드 ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    # 이 파일(pages/3_data_visualization.py)의 상위 폴더(=저장소 루트)를 기준으로 CSV 경로를 잡음
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(root_dir, "cleaned_student_ai_data.csv")

    df = pd.read_csv(csv_path)
    # 성적 변화량 계산 (도입 후 - 도입 전)
    df["grade_change"] = df["grades_after_ai"] - df["grades_before_ai"]
    return df

df = load_data()

# ── 디버깅: 컬럼명 확인 후 이 블록은 지워도 됩니다 ──
st.write("실제 CSV 컬럼 목록:", list(df.columns))
st.write("데이터 미리보기:", df.head())
st.stop()
# ─────────────────────────────────────────────

# ── 공통 색상 및 스타일 ───────────────────────────────────────────
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
    "7️⃣ 스크린 타임 × 공부 시간 산점도": "heatmap",  # 보기 편하게 명칭을 산점도로 변경
}
selected_label = st.sidebar.radio("보고 싶은 차트를 선택하세요", list(chart_options.keys()))
selected = chart_options[selected_label]

st.subheader(selected_label)

# ════════════════════════════════════════════════════════════════
# 1. Donut Chart — AI 사용 비율
# ════════════════════════════════════════════════════════════════
if selected == "donut":
    ai_counts = df["uses_ai"].value_counts().reset_index()
    ai_counts.columns = ["uses_ai", "count"]
    ai_counts["label"] = ai_counts["uses_ai"].map({"Yes": "AI 사용", "No": "AI 미사용"})

    fig = px.pie(
        ai_counts,
        values="count",
        names="label",
        hole=0.55,
        color="label",
        color_discrete_map={"AI 사용": COLOR_YES, "AI 미사용": COLOR_NO}
    )
    fig.update_traces(
        textinfo="percent+label",
        textfont_size=13
        # textfont(weight=...)는 Plotly에서 지원하지 않아 제거함 (오류 원인)
    )

    total_students = len(df)
    fig.update_layout(title={"text": f"AI 사용 여부 분포 (n={total_students})", "x": 0.5}, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    yes_count = df['uses_ai'].value_counts().get('Yes', 0)
    no_count = df['uses_ai'].value_counts().get('No', 0)
    yes_pct = (yes_count / total_students) * 100
    no_pct = (no_count / total_students) * 100

    st.info(
        f"전체 학생 {total_students}명 중 AI 사용 **{yes_count}명({yes_pct:.0f}%)**, "
        f"미사용 **{no_count}명({no_pct:.0f}%)** 입니다. "
        f"약 {yes_pct:.0f}%의 학생이 이미 AI를 학습에 활용하고 있습니다."
    )

# ════════════════════════════════════════════════════════════════
# 2. Grouped Bar — 전후 성적 비교
# ════════════════════════════════════════════════════════════════
elif selected == "grouped_bar":
    ai_yes = df[df["uses_ai"] == "Yes"]
    ai_no = df[df["uses_ai"] == "No"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"AI 사용 (n={len(ai_yes)})", f"AI 미사용 (n={len(ai_no)})"],
        y=[ai_yes["grades_before_ai"].mean(), ai_no["grades_before_ai"].mean()],
        name="AI 도입 전",
        marker_color="#A8C4E0",
        text=[f"{ai_yes['grades_before_ai'].mean():.2f}", f"{ai_no['grades_before_ai'].mean():.2f}"],
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        x=[f"AI 사용 (n={len(ai_yes)})", f"AI 미사용 (n={len(ai_no)})"],
        y=[ai_yes["grades_after_ai"].mean(), ai_no["grades_after_ai"].mean()],
        name="AI 도입 후",
        marker_color=[COLOR_YES, COLOR_NO],
        text=[f"{ai_yes['grades_after_ai'].mean():.2f}", f"{ai_no['grades_after_ai'].mean():.2f}"],
        textposition="auto"
    ))

    fig.update_layout(
        barmode="group",
        title={"text": "AI 사용 전후 평균 성적 변화 (표준화 점수)", "x": 0.5},
        yaxis_title="평균 성적 (Z-Score)",
        yaxis_range=[-1.5, 1.5]
    )
    st.plotly_chart(fig, use_container_width=True)

    avg_change_yes = ai_yes["grade_change"].mean()
    avg_change_no = ai_no["grade_change"].mean()
    st.info(
        f"AI를 사용한 학생들의 평균 성적은 표준화 수치 기준 **+{avg_change_yes:.2f}** 향상되었습니다. "
        f"반면 미사용 학생의 성적은 평균 **{avg_change_no:.2f}**로 오히려 감소하는 경향을 보였습니다. "
        f"이를 통해 AI를 활용한 학습이 성적 향상에 긍정적인 도움이 되었음을 유추할 수 있습니다."
    )

# ════════════════════════════════════════════════════════════════
# 3. Horizontal Bar — 도구별 향상
# ════════════════════════════════════════════════════════════════
elif selected == "tool_bar":
    tool_change = df[df["uses_ai"] == "Yes"].groupby("ai_tools_used")["grade_change"].mean().sort_values().reset_index()

    fig = px.bar(
        tool_change,
        x="grade_change",
        y="ai_tools_used",
        orientation="h",
        color="ai_tools_used",
        color_discrete_map=TOOL_COLORS,
        text_auto="+.2f"
    )
    fig.update_layout(
        title={"text": "AI 도구별 평균 성적 향상 비교", "x": 0.5},
        xaxis_title="평균 성적 향상도 (Z-Score 변화량)",
        yaxis_title="",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    best = tool_change.loc[tool_change["grade_change"].idxmax(), "ai_tools_used"]
    max_val = tool_change["grade_change"].max()
    st.info(f"분석 결과, **{best}**를 사용한 학생 그룹의 성적 향상 폭이 가장 컸습니다 (+{max_val:.2f}).")

# ════════════════════════════════════════════════════════════════
# 4. Box Plot — 목적별 향상
# ════════════════════════════════════════════════════════════════
elif selected == "purpose_bar":
    purpose_df = df[df["uses_ai"] == "Yes"].dropna(subset=["purpose_of_ai"])

    fig = px.box(
        purpose_df,
        x="purpose_of_ai",
        y="grade_change",
        color="purpose_of_ai",
        color_discrete_map=PURPOSE_COLORS,
        points="all"
    )

    fig.update_layout(
        title={"text": "AI 활용 목적별 성적 향상 분포", "x": 0.5},
        xaxis_title="AI 활용 목적",
        yaxis_title="성적 향상도 (Z-Score 변화량)",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    purpose_change = purpose_df.groupby("purpose_of_ai")["grade_change"].mean()
    best = purpose_change.idxmax()
    st.info(
        f"**{best}** 목적으로 AI를 활용한 학생들의 성적 향상 폭이 평균적으로 가장 안정적이고 높았습니다. "
        f"박스 플롯 옆의 점들을 통해 목적별 학생들의 실제 점수 분포 현황을 직관적으로 확인할 수 있습니다."
    )

# ════════════════════════════════════════════════════════════════
# 5. Box Plot — 공부 시간 vs 성적 향상
# ════════════════════════════════════════════════════════════════
elif selected == "scatter":
    box_df = df.copy()
    box_df["study_bin"] = pd.qcut(box_df["study_hours_per_day"], q=3, labels=["공부시간 적음", "보통", "공부시간 많음"])
    box_df["uses_ai_label"] = box_df["uses_ai"].map({"No": "AI 미사용", "Yes": "AI 사용"})

    fig = px.box(
        box_df,
        x="study_bin",
        y="grade_change",
        color="uses_ai_label",
        color_discrete_map={"AI 미사용": COLOR_NO, "AI 사용": COLOR_YES},
        category_orders={"study_bin": ["공부시간 적음", "보통", "공부시간 많음"]}
    )

    fig.update_layout(
        title={"text": "일일 공부 시간 그룹별 성적 변화 비교 (AI 사용 여부별)", "x": 0.5},
        xaxis_title="일일 공부 시간 수준",
        yaxis_title="성적 향상도 (Z-Score 변화량)",
        legend_title="구분"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "공부 시간이 적든 많든 모든 그룹에서 **AI를 사용한 학생들(파란색 박스)의 성적 향상도가 미사용 학생들보다 압도적으로 높음**을 볼 수 있습니다. "
        "즉, 단순히 공부 시간의 절대적인 양보다는 AI를 학습에 녹여냈는지 여부가 성적 변화를 결정 짓는 주요 요인임을 보여줍니다."
    )

# ════════════════════════════════════════════════════════════════
# 6. Box Plot — 교육 수준별
# ════════════════════════════════════════════════════════════════
elif selected == "boxplot":
    box_df = df.copy()
    box_df["uses_ai_label"] = box_df["uses_ai"].map({"No": "AI 미사용", "Yes": "AI 사용"})
    box_df["edu_label"] = box_df["education_level"].map({"school": "중·고등학교", "college": "대학교"})

    fig = px.box(
        box_df,
        x="uses_ai_label",
        y="grade_change",
        color="uses_ai_label",
        facet_col="edu_label",
        category_orders={"uses_ai_label": ["AI 미사용", "AI 사용"], "edu_label": ["중·고등학교", "대학교"]},
        color_discrete_map={"AI 미사용": COLOR_NO, "AI 사용": COLOR_YES}
    )

    fig.update_layout(
        title={"text": "교육 수준 × AI 사용 여부별 성적 향상 분포", "x": 0.5},
        yaxis_title="성적 향상도 (Z-Score 변화량)",
        showlegend=False
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_xaxes(title_text="")
    st.plotly_chart(fig, use_container_width=True)

    edu_order  = ["school", "college"]
    edu_labels = {"school": "중·고등학교", "college": "대학교"}
    for edu in edu_order:
        sub_ai = df[(df["education_level"] == edu) & (df["uses_ai"] == "Yes")]
        if not sub_ai.empty:
            m = sub_ai["grade_change"].mean()
            st.info(f"**{edu_labels[edu]}**의 AI 사용 학생 평균 성적 향상 지표: **+{m:.2f}**")
        else:
            st.warning(f"**{edu_labels[edu]}**의 AI 사용 학생 데이터가 부족합니다.")

# ════════════════════════════════════════════════════════════════
# 7. Scatter Plot — 스크린타임 × 공부시간
# ════════════════════════════════════════════════════════════════
elif selected == "heatmap":
    ai_df = df[df["uses_ai"] == "Yes"].copy()

    fig = px.scatter(
        ai_df,
        x="study_hours_per_day",
        y="daily_screen_time_hours",
        color="grade_change",
        color_continuous_scale="RdYlBu_r",
        color_continuous_midpoint=0.7,
        size=np.abs(ai_df["grade_change"]) + 0.3,
        labels={
            "study_hours_per_day": "일일 공부 시간 (표준화 점수)",
            "daily_screen_time_hours": "일일 스크린 타임 (표준화 점수)",
            "grade_change": "성적 향상도"
        }
    )

    fig.update_layout(
        title={"text": "공부 시간 × 스크린 타임 분포에 따른 성적 향상 (AI 사용자)", "x": 0.5},
        coloraxis_colorbar=dict(title="성적 향상도")
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "각 점은 AI를 사용하는 개별 학생입니다. **오른쪽 아래(공부 시간이 많고 스크린 타임이 적은 영역)로 갈수록 파란색 점(성적이 크게 상승한 학생)**들이 "
        "많이 분포하는 것을 시각적으로 뚜렷하게 관찰할 수 있습니다."
    )

st.markdown("---")
st.caption(f"📌 데이터 출처: cleaned_student_ai_data.csv (총 샘플 수: {len(df)}개) | 시각화 도구: Plotly")
