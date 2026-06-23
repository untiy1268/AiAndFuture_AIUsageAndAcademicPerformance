import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="데이터 시각화 (Plotly)", page_icon="📈", layout="wide")

st.title("📈 3. 데이터 시각화 (EDA) — Plotly 버전")
st.markdown("---")

# ── 데이터 로드 ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("students_ai_usage.csv")
    df["grade_change"] = df["grades_after_ai"] - df["grades_before_ai"]
    return df

df = load_data()

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
    "7️⃣ 스크린 타임 × 공부 시간 히트맵": "heatmap",
}
selected_label = st.sidebar.radio("보고 싶은 차트를 선택하세요", list(chart_options.keys()))
selected = chart_options[selected_label]

st.subheader(selected_label)

# ════════════════════════════════════════════════════════════════
# 1. Donut Chart — AI 사용 비율 (★오류 수정 완료)
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
    # 💡 폰트 굵기 문법 수정: textfont=dict(weight="bold")로 변경
    fig.update_traces(
        textinfo="percent+label", 
        textfont_size=13, 
        textfont=dict(weight="bold")
    )
    fig.update_layout(title={"text": "AI 사용 여부 분포 (n=100)", "x": 0.5}, showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(
        f"전체 학생 100명 중 AI 사용 **{df['uses_ai'].value_counts().get('Yes', 0)}명(40%)**, "
        f"미사용 **{df['uses_ai'].value_counts().get('No', 0)}명(60%)** 입니다. "
        "약 40%의 학생이 이미 AI를 학습에 활용하고 있습니다."
    )

# ════════════════════════════════════════════════════════════════
# 2. Grouped Bar — 전후 성적 비교
# ════════════════════════════════════════════════════════════════
elif selected == "grouped_bar":
    ai_yes = df[df["uses_ai"] == "Yes"]
    ai_no = df[df["uses_ai"] == "No"]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["AI 사용 (n=40)", "AI 미사용 (n=60)"],
        y=[ai_yes["grades_before_ai"].mean(), ai_no["grades_before_ai"].mean()],
        name="AI 도입 전",
        marker_color="#A8C4E0",
        text=[f"{ai_yes['grades_before_ai'].mean():.1f}", f"{ai_no['grades_before_ai'].mean():.1f}"],
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        x=["AI 사용 (n=40)", "AI 미사용 (n=60)"],
        y=[ai_yes["grades_after_ai"].mean(), ai_no["grades_after_ai"].mean()],
        name="AI 도입 후",
        marker_color=[COLOR_YES, COLOR_NO],
        text=[f"{ai_yes['grades_after_ai'].mean():.1f}", f"{ai_no['grades_after_ai'].mean():.1f}"],
        textposition="auto"
    ))
    
    fig.update_layout(
        barmode="group", 
        title={"text": "AI 사용 전후 평균 성적 변화", "x": 0.5},
        yaxis_title="평균 성적", 
        yaxis_range=[0, 100]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    avg_change = ai_yes["grade_change"].mean()
    st.info(f"AI를 사용한 학생들의 평균 성적은 **+{avg_change:.1f}점** 향상되었습니다. 미사용 학생 성적은 변화가 없었습니다.")

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
        text_auto="+%.1f점"
    )
    fig.update_layout(
        title={"text": "AI 도구별 평균 성적 향상 비교", "x": 0.5},
        xaxis_title="평균 성적 향상 (점)", 
        yaxis_title="", 
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    best = tool_change.loc[tool_change["grade_change"].idxmax(), "ai_tools_used"]
    st.info(f"**{best}**를 사용한 학생의 성적 향상이 가장 컸습니다 (+{tool_change['grade_change'].max():.1f}점).")

# ════════════════════════════════════════════════════════════════
# 4. Bar + Scatter overlay — 목적별 향상
# ════════════════════════════════════════════════════════════════
elif selected == "purpose_bar":
    purpose_df = df[df["uses_ai"]=="Yes"].copy()
    purpose_change = purpose_df.groupby("purpose_of_ai")["grade_change"].mean().sort_values().reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=purpose_change["purpose_of_ai"],
        y=purpose_change["grade_change"],
        marker_color=[PURPOSE_COLORS.get(p, "#999") for p in purpose_change["purpose_of_ai"]],
        text=[f"+{v:.1f}점" for v in purpose_change["grade_change"]],
        textposition="outside",
        showlegend=False
    ))
    
    for i, purpose in enumerate(purpose_change["purpose_of_ai"]):
        vals = purpose_df[purpose_df["purpose_of_ai"] == purpose]["grade_change"]
        jitter = np.random.uniform(-0.12, 0.12, len(vals))
        fig.add_trace(go.Scatter(
            x=np.full(len(vals), i) + jitter,
            y=vals,
            mode="markers",
            marker=dict(color="black", opacity=0.4, size=6),
            showlegend=False,
            hoverinfo="y"
        ))
        
    fig.update_layout(
        title={"text": "AI 활용 목적별 평균 성적 향상", "x": 0.5},
        yaxis_title="평균 성적 향상 (점)",
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(purpose_change))),
            ticktext=purpose_change["purpose_of_ai"]
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    best = purpose_change.loc[purpose_change["grade_change"].idxmax(), "purpose_of_ai"]
    st.info(f"**{best}** 목적으로 AI를 활용한 학생의 성적 향상이 가장 컸습니다. 점들은 개별 학생 데이터입니다.")

# ════════════════════════════════════════════════════════════════
# 5. Scatter + 추세선
# ════════════════════════════════════════════════════════════════
elif selected == "scatter":
    ai_df  = df[df["uses_ai"]=="Yes"]
    non_df = df[df["uses_ai"]=="No"]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=non_df["study_hours_per_day"], y=non_df["grade_change"],
        mode="markers", name="AI 미사용",
        marker=dict(color=COLOR_NO, size=8, opacity=0.6, line=dict(width=1, color="white"))
    ))
    fig.add_trace(go.Scatter(
        x=ai_df["study_hours_per_day"], y=ai_df["grade_change"],
        mode="markers", name="AI 사용",
        marker=dict(color=COLOR_YES, size=10, opacity=0.7, line=dict(width=1, color="white"))
    ))
    
    if len(ai_df) > 1:
        m, b = np.polyfit(ai_df["study_hours_per_day"], ai_df["grade_change"], 1)
        x_line = np.linspace(ai_df["study_hours_per_day"].min(), ai_df["study_hours_per_day"].max(), 100)
        fig.add_trace(go.Scatter(
            x=x_line, y=m*x_line+b,
            mode="lines", name="AI 사용 추세선",
            line=dict(color=COLOR_YES, width=2, dash="dash")
        ))
        
    fig.update_layout(
        title={"text": "일일 공부 시간 vs 성적 향상", "x": 0.5},
        xaxis_title="일일 공부 시간 (시간)", 
        yaxis_title="성적 향상 (점)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    corr = ai_df[["study_hours_per_day","grade_change"]].corr().iloc[0,1]
    st.info(f"AI 사용 학생 그룹에서 공부 시간과 성적 향상의 상관계수는 **{corr:.2f}**입니다.")

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
        yaxis_title="성적 향상 (점)", 
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
            st.info(f"**{edu_labels[edu]}** AI 사용 학생 평균 성적 향상: **+{m:.1f}점**")
        else:
            st.warning(f"**{edu_labels[edu]}** AI를 사용한 학생 데이터가 부족합니다.")

# ════════════════════════════════════════════════════════════════
# 7. Heatmap — 스크린타임 × 공부시간
# ════════════════════════════════════════════════════════════════
elif selected == "heatmap":
    ai_heat = df[df["uses_ai"]=="Yes"].copy()
    
    ai_heat["screen_bin"] = pd.cut(ai_heat["daily_screen_time_hours"],
                                   bins=[0, 3, 5, 24], labels=["3시간 이하", "4~5시간", "6시간 이상"])
    ai_heat["study_bin"]  = pd.cut(ai_heat["study_hours_per_day"],
                                   bins=[0, 2, 4, 24], labels=["2시간 이하", "3~4시간", "5시간 이상"])
    
    pivot = ai_heat.pivot_table(index="screen_bin", columns="study_bin",
                                values="grade_change", aggfunc="mean")
    
    fig = px.imshow(
        pivot, 
        text_auto=".1f", 
        color_continuous_scale="YlOrRd",
        labels=dict(x="일일 공부 시간", y="일일 스크린 타임", color="평균 성적 향상")
    )
    fig.update_layout(title={"text": "스크린 타임 × 공부 시간별 성적 향상<br><sup>(AI 사용 학생)</sup>", "x": 0.5})
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("색이 진할수록 성적 향상이 큽니다. 공부 시간이 길수록 AI 활용 효과가 더 크게 나타납니다.")

st.markdown("---")
st.caption("📌 데이터 출처: students_ai_usage.csv (n=100) | 시각화: Plotly")
