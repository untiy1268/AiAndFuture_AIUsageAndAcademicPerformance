import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="학생 AI 사용 데이터 시각화", page_icon="📈", layout="wide")

st.title("📈 학생 AI 활용 및 성적 향상 데이터 시각화 (EDA)")
st.markdown("### 🎯 주제: AI를 활용한 학습이 학생 성적 향상에 도움이 될까?")
st.caption("※ 본 데이터는 학생 100명의 실제 성적(0~100점) 및 학습 습관 원자료를 기반으로 합니다.")

# ── 설명 박스(st.info/st.warning) 글자 크기 확대를 위한 CSS ──────────
st.markdown("""
<style>
div[data-testid="stAlertContainer"] p, div[data-testid="stAlert"] p {
    font-size: 1.15rem !important;
    line-height: 1.6 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("---")

# ── 데이터 로드 ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    # 원본 데이터 파일명 지정 (uses_ai 컬럼이 필요하므로 전처리 전 원본을 사용)
    df = pd.read_csv("students_ai_usage.csv")
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
    "3️⃣ AI 도구별 평균 성적 향상": "lollipop",
    "4️⃣ AI 활용 목적별 성적 향상": "purpose_box",
    "5️⃣ 공부 시간 vs 성적 향상": "study_bar",
    "6️⃣ 교육 수준별 성적 분포": "edu_bar",
    "7️⃣ 상관관계 히트맵": "heatmap",
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
        title={"text": "AI 사용 전후 평균 성적 변화 (원점수)", "x": 0.5},
        yaxis_title="평균 성적 (100점 만점)",
    )
    st.plotly_chart(fig, use_container_width=True)

    avg_change_yes = ai_yes["grade_change"].mean()
    avg_change_no = ai_no["grade_change"].mean()
    st.info(
        f"AI를 사용한 학생들의 평균 성적은 **+{avg_change_yes:.2f}점** 향상되었습니다. "
        f"반면 미사용 학생의 성적은 평균 **{avg_change_no:+.2f}점**으로 거의 변화가 없었습니다. "
        f"이를 통해 AI를 활용한 학습이 성적 향상에 긍정적인 도움이 되었음을 유추할 수 있습니다."
    )

# ════════════════════════════════════════════════════════════════
# 3. Lollipop Chart — 도구별 향상
# ════════════════════════════════════════════════════════════════
elif selected == "lollipop":
    tool_change = df[df["uses_ai"]=="Yes"].groupby("ai_tools_used")["grade_change"].agg(
        mean="mean", std="std", count="count"
    ).sort_values("mean").reset_index()

    fig = go.Figure()
    for _, row in tool_change.iterrows():
        fig.add_trace(go.Scatter(
            x=[0, row["mean"]], y=[row["ai_tools_used"], row["ai_tools_used"]],
            mode="lines", line=dict(color=TOOL_COLORS[row["ai_tools_used"]], width=3),
            showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=tool_change["mean"], y=tool_change["ai_tools_used"],
        mode="markers+text",
        marker=dict(size=18, color=[TOOL_COLORS[t] for t in tool_change["ai_tools_used"]]),
        text=[f"+{v:.2f}" for v in tool_change["mean"]],
        textposition="middle right",
        showlegend=False,
    ))
    fig.update_layout(
        title={"text": "AI 도구별 평균 성적 향상 비교 (롤리팝 차트)", "x": 0.5},
        xaxis_title="평균 성적 향상도 (점)",
        yaxis_title="",
        xaxis=dict(range=[0, tool_change["mean"].max() * 1.35]),
    )
    st.plotly_chart(fig, use_container_width=True)

    best = tool_change.loc[tool_change["mean"].idxmax(), "ai_tools_used"]
    max_val = tool_change["mean"].max()
    gap = tool_change["mean"].max() - tool_change["mean"].min()
    st.info(
        f"분석 결과, **{best}**를 사용한 학생 그룹의 성적 향상 폭이 가장 컸습니다 (+{max_val:.2f}). "
        f"다만 도구 간 평균 차이는 최대 {gap:.2f}점으로, 그룹별 표준편차(약 3~4점)에 비해 크지 않아 "
        "도구 선택보다는 '사용 여부 자체'가 더 결정적인 요인으로 보입니다."
    )

# ════════════════════════════════════════════════════════════════
# 4. Box Plot — 목적별 성적 향상 (분포까지 함께 표시)
#    ※ 평균만 막대로 보여주면 목적별 개인차(편차)가 가려집니다. 박스플롯 +
#      개별 점(strip)을 쓰면 그룹 간 차이가 실제로 유의미한 수준인지,
#      아니면 그룹 내 편차에 묻히는 수준인지까지 함께 판단할 수 있습니다.
# ════════════════════════════════════════════════════════════════
elif selected == "purpose_box":
    purpose_df = df[df["uses_ai"] == "Yes"].copy()
    purpose_df["purpose_kr"] = purpose_df["purpose_of_ai"].map(PURPOSE_KR)

    order = purpose_df.groupby("purpose_kr")["grade_change"].mean().sort_values().index.tolist()
    color_map_kr = {PURPOSE_KR[k]: v for k, v in PURPOSE_COLORS.items()}

    fig = px.box(
        purpose_df,
        x="purpose_kr",
        y="grade_change",
        color="purpose_kr",
        category_orders={"purpose_kr": order},
        color_discrete_map=color_map_kr,
        points="all",
    )
    fig.update_traces(marker=dict(opacity=0.6, size=6))
    fig.update_layout(
        title={"text": "AI 활용 목적별 성적 향상 분포 (박스플롯)", "x": 0.5},
        xaxis_title="AI 활용 목적",
        yaxis_title="성적 향상도 (점)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    means = purpose_df.groupby("purpose_kr")["grade_change"].mean()
    best = means.idxmax()
    st.info(
        f"평균만 보면 **{best}** 목적 그룹의 향상 폭이 가장 크지만(+{means.max():.2f}), "
        "박스와 점들이 서로 크게 겹쳐 있어 목적 간 차이가 뚜렷하다고 보기는 어렵습니다. "
        "즉 '무엇을 위해' AI를 썼는지보다, 앞서 본 것처럼 'AI를 썼는지 여부'가 더 중요한 요인으로 보입니다."
    )

# ════════════════════════════════════════════════════════════════
# 5. 공부 시간대별 성적 향상 (구간 막대그래프)
# ════════════════════════════════════════════════════════════════
elif selected == "study_bar":
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
        yaxis_title="평균 성적 향상도 (점)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    corr_val = ai_df["study_hours_per_day"].corr(ai_df["grade_change"])
    st.info(
        "공부 시간을 '적음/보통/많음' 세 그룹으로 나누어 비교한 결과입니다. "
        f"참고로 전체 데이터의 상관계수는 r = {corr_val:.2f}로 0에 가까워, "
        "그룹 간 차이가 크지 않다면 단순히 공부 시간을 늘리는 것보다 "
        "**AI를 얼마나 효율적으로 활용하는지가 성적 향상에 더 중요한 요인**일 수 있음을 시사합니다."
    )

# ════════════════════════════════════════════════════════════════
# 6. Grouped Bar — 교육 수준별 평균 성적 향상
#    ※ AI 미사용 그룹은 grade_change가 전원 정확히 0점으로 값의 편차가
#      전혀 없습니다. 분산이 없는 데이터는 박스/바이올린플롯보다
#      막대그래프(오차막대 포함)로 표현하는 것이 더 명확합니다.
# ════════════════════════════════════════════════════════════════
elif selected == "edu_bar":
    box_df = df.copy()
    box_df["uses_ai_label"] = box_df["uses_ai"].map({"No": "AI 미사용", "Yes": "AI 사용"})
    box_df["edu_label"] = box_df["education_level"].map({"school": "중·고등학교", "college": "대학교"})

    summary = (
        box_df.groupby(["edu_label", "uses_ai_label"], observed=True)["grade_change"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    summary["std"] = summary["std"].fillna(0)
    summary["label"] = summary["mean"].map(lambda v: f"{v:+.2f}")

    fig = px.bar(
        summary,
        x="edu_label",
        y="mean",
        color="uses_ai_label",
        barmode="group",
        error_y="std",
        text="label",
        category_orders={"uses_ai_label": ["AI 미사용", "AI 사용"], "edu_label": ["중·고등학교", "대학교"]},
        color_discrete_map={"AI 미사용": COLOR_NO, "AI 사용": COLOR_YES},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        title={"text": "교육 수준 × AI 사용 여부별 평균 성적 향상 비교", "x": 0.5},
        xaxis_title="",
        yaxis_title="평균 성적 향상도 (점, 오차막대=표준편차)",
        legend_title_text="",
    )
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

    st.info(
        "💡 **그래프 해석 안내**: 막대의 값은 'AI 도입 후 점수 − 도입 전 점수'를 그대로 나타낸 "
        "원점수 변화량(점)의 평균입니다. AI를 사용하지 않은 학생들은 특별한 학습 개입이 없었기 때문에 "
        "도입 전후 성적이 그대로 유지되어 변화량이 정확히 0점이며, 이는 AI 사용 "
        "그룹과 비교했을 때 '개선 효과가 없었다'는 점을 보여주는 정상적인 기준선(baseline)입니다."
    )

# ════════════════════════════════════════════════════════════════
# 7. 히트맵 — 상관관계 히트맵 (-1~1 범위로 고정)
#    ※ 기존에는 구간별 '평균 성적 향상 점수'(최대 13~15점)를 색상으로
#      표시해 값이 1을 훌쩍 넘어갔습니다. 히트맵에서 일반적으로 기대하는
#      -1~1 범위를 지키도록 상관계수(correlation) 기반 히트맵으로 변경했습니다.
# ════════════════════════════════════════════════════════════════
elif selected == "heatmap":
    corr_df = df.copy()
    # AI 사용 여부를 0/1 숫자로 변환해 다른 변수들과 함께 상관관계를 볼 수 있게 함
    corr_df["ai_used_num"] = (corr_df["uses_ai"] == "Yes").astype(int)

    corr_cols = {
        "ai_used_num": "AI 사용 여부",
        "study_hours_per_day": "공부 시간",
        "daily_screen_time_hours": "스크린 타임",
        "grade_change": "성적 향상도",
    }
    corr_matrix = corr_df[list(corr_cols.keys())].rename(columns=corr_cols).corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        labels=dict(color="상관계수"),
        aspect="auto",
    )
    fig.update_layout(
        title={
            "text": "AI 사용 여부 · 공부 시간 · 스크린 타임 · 성적 향상도 간 상관관계<br><sup>(값의 범위: -1 ~ 1 / 전체 학생 대상)</sup>",
            "x": 0.5
        }
    )
    st.plotly_chart(fig, use_container_width=True)

    ai_corr = corr_matrix.loc["AI 사용 여부", "성적 향상도"]
    study_corr = corr_matrix.loc["공부 시간", "성적 향상도"]
    screen_corr = corr_matrix.loc["스크린 타임", "성적 향상도"]
    st.info(
        f"**'AI 사용 여부'와 '성적 향상도'의 상관계수는 +{ai_corr:.2f}로 매우 강한 양의 관계**를 보입니다. "
        f"반면 공부 시간(**{study_corr:+.2f}**)과 스크린 타임(**{screen_corr:+.2f}**)은 성적 향상도와 "
        "거의 관계가 없습니다. 즉, 이 데이터에서는 얼마나 오래 공부하거나 화면을 보느냐보다 "
        "**AI를 사용했는지 여부 자체가 성적 향상과 가장 밀접하게 연관된 요인**으로 나타납니다. "
        "상관계수는 -1(강한 음의 관계) ~ +1(강한 양의 관계) 사이의 값을 가지며, 0에 가까울수록 "
        "두 변수 간 선형적 관계가 약하다는 뜻입니다."
    )

st.markdown("---")
st.caption(f"📌 데이터 출처: students_ai_usage.csv (총 샘플 수: {len(df)}개) | 시각화 도구: Plotly")
