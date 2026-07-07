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
    # 업로드된 전처리 완료 파일명 지정
    df = pd.read_csv("preprocessed_sau_data.csv")
    # 성적 변화량 계산 (도입 후 - 도입 전)
    df["grade_change"] = df["grades_after_ai"] - df["grades_before_ai"]
    return df

df = load_data()

# ── 공통 색상 및 스타일 ───────────────────────────────────────────
COLOR_YES = "#4C72B0"
COLOR_NO  = "#DD8452"
TOOL_COLORS    = {"ChatGPT": "#10A37F", "Copilot": "#7B61FF", "Gemini": "#EA4335"}
PURPOSE_COLORS = {"Research": "#4C72B0", "Homework": "#DD8452", "Coding": "#55A868"}
PURPOSE_KR = {"Research": "자료조사", "Homework": "과제", "Coding": "코딩"}

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
        textfont_size=13,
        textfont=dict(weight="bold")
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
        yaxis_range=[-1.5, 1.5]  # 표준화 스케일에 맞게 조정
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
    tool_change = df[df["uses_ai"]=="Yes"].groupby("ai_tools_used")["grade_change"].mean().sort_values().reset_index()

    fig = px.bar(
        tool_change,
        x="grade_change",
        y="ai_tools_used",
        orientation="h",
        color="ai_tools_used",
        color_discrete_map=TOOL_COLORS,
        text_auto="+%.2f"
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
# 4. 목적별 성적 향상 (단순 막대그래프 - 한글 라벨, 지터 산점도 제거)
# ════════════════════════════════════════════════════════════════
elif selected == "purpose_bar":
    purpose_df = df[df["uses_ai"] == "Yes"].copy()
    purpose_df["purpose_kr"] = purpose_df["purpose_of_ai"].map(PURPOSE_KR)
    purpose_change = purpose_df.groupby("purpose_kr")["grade_change"].mean().sort_values().reset_index()

    color_map_kr = {PURPOSE_KR[k]: v for k, v in PURPOSE_COLORS.items()}

    fig = px.bar(
        purpose_change,
        x="purpose_kr",
        y="grade_change",
        color="purpose_kr",
        color_discrete_map=color_map_kr,
        text=[f"+{v:.2f}" for v in purpose_change["grade_change"]],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        title={"text": "AI 활용 목적별 평균 성적 향상", "x": 0.5},
        xaxis_title="AI 활용 목적",
        yaxis_title="평균 성적 향상도 (Z-Score 변화량)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    best = purpose_change.loc[purpose_change["grade_change"].idxmax(), "purpose_kr"]
    max_val = purpose_change["grade_change"].max()
    st.info(f"**{best}** 목적으로 AI를 활용한 학생들의 성적 향상 폭이 가장 컸습니다 (+{max_val:.2f}).")

# ════════════════════════════════════════════════════════════════
# 5. 공부 시간대별 성적 향상 (구간 막대그래프 - 상관계수/산점도 제거)
# ════════════════════════════════════════════════════════════════
elif selected == "scatter":
    ai_df = df[df["uses_ai"] == "Yes"].copy()

    # 공부 시간을 3단계로 구간화하여 직관적으로 비교
    ai_df["study_group"] = pd.qcut(
        ai_df["study_hours_per_day"], q=3, labels=["공부시간 적음", "보통", "많음"]
    )
    grouped = ai_df.groupby("study_group", observed=True)["grade_change"].mean().reset_index()

    fig = px.bar(
        grouped,
        x="study_group",
        y="grade_change",
        color="study_group",
        color_discrete_sequence=["#A8C4E0", "#4C72B0", "#2C4870"],
        text=[f"+{v:.2f}" for v in grouped["grade_change"]],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        title={"text": "공부 시간대별 평균 성적 향상 (AI 사용 학생 대상)", "x": 0.5},
        xaxis_title="일일 공부 시간대",
        yaxis_title="평균 성적 향상도 (Z-Score 변화량)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "공부 시간을 '적음/보통/많음' 세 그룹으로 나누어 비교한 결과입니다. "
        "그룹 간 차이가 크지 않다면, 단순히 공부 시간을 늘리는 것보다 "
        "**AI를 얼마나 효율적으로 활용하는지가 성적 향상에 더 중요한 요인**일 수 있음을 시사합니다."
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
# 7. 히트맵 — 스크린타임 × 공부시간 (라벨/설명 보강)
# ════════════════════════════════════════════════════════════════
elif selected == "heatmap":
    ai_heat = df[df["uses_ai"]=="Yes"].copy()

    ai_heat["screen_bin"] = pd.qcut(
        ai_heat["daily_screen_time_hours"], q=3,
        labels=["스크린타임 낮음", "스크린타임 보통", "스크린타임 높음"]
    )
    ai_heat["study_bin"] = pd.qcut(
        ai_heat["study_hours_per_day"], q=3,
        labels=["공부시간 적음", "공부시간 보통", "공부시간 많음"]
    )

    pivot = ai_heat.pivot_table(index="screen_bin", columns="study_bin",
                                values="grade_change", aggfunc="mean", observed=True)

    fig = px.imshow(
        pivot,
        text_auto=".2f",
        color_continuous_scale="YlOrRd",
        labels=dict(x="일일 공부 시간대", y="일일 스크린 타임대", color="평균 성적 향상도"),
        aspect="auto",
    )
    fig.update_layout(
        title={
            "text": "스크린 타임 × 공부 시간별 성적 향상<br><sup>(색이 진할수록 성적 향상 폭이 큼 / AI 사용 학생 대상)</sup>",
            "x": 0.5
        }
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "칸 안의 숫자는 해당 조합(예: 스크린타임 낮음 + 공부시간 많음)에 속한 학생들의 "
        "평균 성적 향상도입니다. 색이 진할수록 향상 폭이 크다는 뜻이며, "
        "일반적으로 스크린 타임은 적고 공부 시간은 확보된 구간에서 성적 향상 효과가 큰 경향을 보입니다."
    )

st.markdown("---")
st.caption(f"📌 데이터 출처: preprocessed_sau_data.csv (총 샘플 수: {len(df)}개) | 시각화 도구: Plotly")
