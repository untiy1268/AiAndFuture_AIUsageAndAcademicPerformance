# ============================================================
# AI 성적 예측 웹앱 (Streamlit + Plotly)
# 목적: 학생의 AI 사용 및 학습 데이터를 기반으로
#       AI 사용 후 성적(grades_after_ai)을 예측한다
# 사용 모델: Linear Regression, Random Forest Regressor
# 데이터: students_ai_usage.csv       (전처리 전 원본 데이터, 100행)
#         preprocessed_sau_data.csv   (전처리 후 원-핫 인코딩 데이터, 100행)
# 특징: 두 데이터셋으로 각각 모델을 학습하여 성능을 비교한다
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import os

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

TARGET_COL = "grades_after_ai"

# ============================================================
# 0. 데이터셋 정의 (전처리 전 / 전처리 후)
# ============================================================
DATASETS = {
    "전처리 전": {
        "path":  "students_ai_usage.csv",
        "color": "#4C9BE8",
    },
    "전처리 후": {
        "path":  "preprocessed_sau_data.csv",
        "color": "#E8714C",
    },
}

# 전처리 전 데이터는 문자열 카테고리를 포함하므로,
# 모델 학습이 가능하도록 최소한의 순서형(ordinal) 인코딩만 적용한다.
# (전처리 후 데이터는 이미 원-핫 인코딩이 되어 있어 그대로 사용)
RAW_CATEGORY_MAPS = {
    "education_level": {"school": 0, "college": 1},
    "uses_ai":         {"No": 0,     "Yes": 1},
    "ai_tools_used":   {"None": 0, "ChatGPT": 1, "Copilot": 2, "Gemini": 3},
    "purpose_of_ai":   {"None": 0, "Coding": 1,  "Homework": 2, "Research": 3},
}

RAW_FEATURE_COLS = [
    "age", "education_level", "study_hours_per_day", "uses_ai",
    "ai_tools_used", "purpose_of_ai", "grades_before_ai", "daily_screen_time_hours",
]

# ============================================================
# 1. 모델 관리 딕셔너리
# ============================================================
MODEL_REGISTRY = {
    "Linear Regression": {
        "model_class": LinearRegression,
        "params":      {},
    },
    "Random Forest": {
        "model_class": RandomForestRegressor,
        "params":      {"n_estimators": 100, "random_state": 42},
    },
}

MODEL_COLORS = {
    "Linear Regression": {"전처리 전": "#4C9BE8", "전처리 후": "#1F5FA8"},
    "Random Forest":      {"전처리 전": "#F2A65A", "전처리 후": "#E8714C"},
}


# ============================================================
# 2. 데이터 로드 & 인코딩 함수
# ============================================================

@st.cache_data
def load_raw_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["ai_tools_used"] = df["ai_tools_used"].fillna("None")
    df["purpose_of_ai"] = df["purpose_of_ai"].fillna("None")
    return df


@st.cache_data
def load_processed_csv(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def encode_raw_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    """전처리 전 데이터를 학습 가능하도록 최소 순서형 인코딩"""
    df = df.copy()
    for col, mapping in RAW_CATEGORY_MAPS.items():
        df[col] = df[col].map(mapping).fillna(0).astype(int)
    return df


def encode_user_input_after(user_raw: pd.DataFrame, after_feature_cols: list) -> pd.DataFrame:
    """전처리 후(원-핫) 모델에 넣기 위해 사용자 입력을 동일한 방식으로 인코딩"""
    dummies = pd.get_dummies(
        user_raw,
        columns=["education_level", "uses_ai", "ai_tools_used", "purpose_of_ai"],
    )
    return dummies.reindex(columns=after_feature_cols, fill_value=0)


def get_model_filename(dataset_key: str, model_name: str) -> str:
    safe_dataset = dataset_key.replace(" ", "")
    safe_model   = model_name.replace(" ", "_")
    return f"model_{safe_dataset}_{safe_model}.pkl"


def load_or_create_model(dataset_key: str, model_name: str):
    info = MODEL_REGISTRY[model_name]
    filepath = get_model_filename(dataset_key, model_name)
    if os.path.exists(filepath):
        model = joblib.load(filepath)
    else:
        model = info["model_class"](**info["params"])
    return model


def train_and_save(model, X_train, y_train, filepath: str):
    model.fit(X_train, y_train)
    joblib.dump(model, filepath)
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    r2    = r2_score(y_test, preds)
    return {"RMSE": round(rmse, 4), "R² Score": round(r2, 4), "predictions": preds}


# ============================================================
# 3. Plotly 그래프 생성 함수
# ============================================================

def plot_actual_vs_predicted(results_for_dataset: dict, y_test, title: str) -> go.Figure:
    fig = go.Figure()

    all_vals = list(y_test)
    for res in results_for_dataset.values():
        all_vals.extend(res["predictions"].tolist())
    lo, hi = min(all_vals) - 1, max(all_vals) + 1

    for mname, res in results_for_dataset.items():
        fig.add_trace(go.Scatter(
            x=list(y_test),
            y=list(res["predictions"]),
            mode="markers",
            name=mname,
            marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color="white")),
            hovertemplate=(
                f"<b>{mname}</b><br>실제: %{{x:.1f}}점<br>예측: %{{y:.1f}}점<extra></extra>"
            ),
        ))

    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect Fit",
        line=dict(color="gray", dash="dash", width=1.5), hoverinfo="skip",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis_title="실제 grades_after_ai",
        yaxis_title="예측 grades_after_ai",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
    )
    return fig


def plot_feature_importance(rf_model, feature_cols: list, title: str) -> go.Figure:
    importances = rf_model.feature_importances_
    feat_df = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=True)
    )

    fig = go.Figure(go.Bar(
        x=feat_df["importance"], y=feat_df["feature"], orientation="h",
        marker=dict(color=feat_df["importance"], colorscale="Blues", showscale=False),
        hovertemplate="<b>%{y}</b><br>중요도: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis_title="Importance",
        yaxis_title="",
        height=380,
        margin=dict(l=40, r=20, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
    )
    return fig


def plot_performance_comparison(all_results: dict) -> go.Figure:
    """
    전처리 전 vs 전처리 후 데이터로 학습한 모델들의 RMSE / R² 를
    한 화면에서 비교하는 그룹형 바 차트
    """
    rows = []
    for dataset_key, model_results in all_results.items():
        for mname, res in model_results.items():
            rows.append({
                "데이터셋": dataset_key,
                "모델":    mname,
                "RMSE":    res["RMSE"],
                "R² Score": res["R² Score"],
            })
    df_cmp = pd.DataFrame(rows)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("RMSE (낮을수록 좋음)", "R² Score (높을수록 좋음)"))

    for dataset_key in DATASETS:
        sub = df_cmp[df_cmp["데이터셋"] == dataset_key]
        color = DATASETS[dataset_key]["color"]
        fig.add_trace(
            go.Bar(x=sub["모델"], y=sub["RMSE"], name=f"{dataset_key} · RMSE",
                   marker_color=color, legendgroup=dataset_key,
                   hovertemplate="<b>%{x}</b><br>RMSE: %{y:.4f}<extra></extra>"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Bar(x=sub["모델"], y=sub["R² Score"], name=f"{dataset_key} · R²",
                   marker_color=color, legendgroup=dataset_key, showlegend=False,
                   hovertemplate="<b>%{x}</b><br>R²: %{y:.4f}<extra></extra>"),
            row=1, col=2,
        )

    fig.update_layout(
        title=dict(text="전처리 전 vs 전처리 후: 모델 성능 비교", font=dict(size=16)),
        barmode="group",
        height=420,
        margin=dict(l=40, r=20, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ============================================================
# 4. Streamlit 앱 메인
# ============================================================

def main():
    st.set_page_config(page_title="AI 성적 예측기", page_icon="🎓", layout="wide")

    st.title("🎓 AI 사용 후 성적 예측 웹앱")
    st.markdown(
        "**전처리 전 데이터(students_ai_usage.csv)** 와 "
        "**전처리 후 데이터(preprocessed_sau_data.csv)** 로 각각 모델을 학습하여 "
        "성능 차이를 비교합니다."
    )
    st.divider()

    # ── 데이터 로드 ───────────────────────────────────────────
    raw_path       = DATASETS["전처리 전"]["path"]
    processed_path = DATASETS["전처리 후"]["path"]

    for p in (raw_path, processed_path):
        if not os.path.exists(p):
            st.error(f"❌ '{p}' 파일을 찾을 수 없습니다. 앱과 같은 폴더에 파일을 넣어주세요.")
            st.stop()

    df_raw_original  = load_raw_csv(raw_path)          # 전처리 전 원본 (문자열 카테고리 포함)
    df_raw_encoded   = encode_raw_ordinal(df_raw_original)  # 학습용 최소 인코딩
    df_processed     = load_processed_csv(processed_path)   # 전처리 후 (원-핫)

    AFTER_FEATURE_COLS = [c for c in df_processed.columns if c != TARGET_COL]

    FEATURE_COLS_BY_DATASET = {
        "전처리 전": RAW_FEATURE_COLS,
        "전처리 후": AFTER_FEATURE_COLS,
    }
    DATA_BY_DATASET = {
        "전처리 전": df_raw_encoded,
        "전처리 후": df_processed,
    }

    # ── 모델 학습/평가 (데이터셋 × 모델 조합) ───────────────────
    trained_models = {"전처리 전": {}, "전처리 후": {}}
    all_results    = {"전처리 전": {}, "전처리 후": {}}
    split_cache    = {}  # 데이터셋별 X_test, y_test 저장

    for dataset_key, feature_cols in FEATURE_COLS_BY_DATASET.items():
        df = DATA_BY_DATASET[dataset_key]
        X = df[feature_cols]
        y = df[TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        split_cache[dataset_key] = (X_test, y_test)

        for model_name, info in MODEL_REGISTRY.items():
            filepath = get_model_filename(dataset_key, model_name)
            model = load_or_create_model(dataset_key, model_name)
            if not os.path.exists(filepath):
                model = train_and_save(model, X_train, y_train, filepath)
            trained_models[dataset_key][model_name] = model
            all_results[dataset_key][model_name] = evaluate_model(model, X_test, y_test)

    # ============================================================
    # 섹션 1: 전처리 전 vs 전처리 후 모델 성능 비교
    # ============================================================
    st.header("📊 전처리 전 vs 전처리 후: 모델 성능 비교")

    rows = []
    for dataset_key, model_results in all_results.items():
        for mname, res in model_results.items():
            rows.append({
                "데이터셋":  dataset_key,
                "모델":     mname,
                "특성 개수": len(FEATURE_COLS_BY_DATASET[dataset_key]),
                "RMSE":     res["RMSE"],
                "R² Score": res["R² Score"],
            })
    df_cmp_table = pd.DataFrame(rows).set_index(["데이터셋", "모델"])
    st.dataframe(df_cmp_table, use_container_width=True)

    st.plotly_chart(plot_performance_comparison(all_results), use_container_width=True)

    st.caption(
        "💡 두 데이터셋 모두 동일한 100개 행(같은 학생, 같은 순서)에서 "
        "동일한 `random_state=42`로 분할했기 때문에, 테스트셋에 포함된 학생이 "
        "동일하여 공정하게 비교할 수 있습니다. "
        "'전처리 전'은 카테고리를 숫자로만 바꾼 순서형 인코딩, "
        "'전처리 후'는 원-핫 인코딩을 사용합니다."
    )

    # ============================================================
    # 섹션 2: 데이터셋별 상세 시각화 (나란히 비교)
    # ============================================================
    st.header("📈 데이터셋별 상세 시각화")

    for dataset_key in DATASETS:
        st.subheader(f"🔹 {dataset_key} 데이터 기반 모델")
        X_test, y_test = split_cache[dataset_key]
        feature_cols = FEATURE_COLS_BY_DATASET[dataset_key]

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                plot_actual_vs_predicted(
                    all_results[dataset_key], y_test,
                    title=f"실제 vs 예측 ({dataset_key})",
                ),
                use_container_width=True,
            )
        with col2:
            rf_model = trained_models[dataset_key].get("Random Forest")
            if rf_model and hasattr(rf_model, "feature_importances_"):
                st.plotly_chart(
                    plot_feature_importance(
                        rf_model, feature_cols,
                        title=f"Feature Importance ({dataset_key})",
                    ),
                    use_container_width=True,
                )

    # ============================================================
    # 섹션 3: 사용자 입력 → 두 파이프라인 동시 예측 비교
    # ============================================================
    st.divider()
    st.header("🔮 성적 예측하기 (전처리 전 / 후 모델 동시 비교)")
    st.markdown("아래에 학생 정보를 입력한 뒤 **Predict** 버튼을 눌러 두 파이프라인의 예측 결과를 비교하세요.")

    inp1, inp2 = st.columns(2)

    with inp1:
        age = st.slider("나이 (age)", min_value=14, max_value=19, value=17, step=1)
        education_level = st.selectbox(
            "교육 수준 (education_level)", options=["school", "college"]
        )
        study_hours = st.slider(
            "하루 공부 시간 (study_hours_per_day)",
            min_value=1.0, max_value=5.0, value=3.0, step=0.1
        )
        uses_ai = st.selectbox("AI 사용 여부 (uses_ai)", options=["Yes", "No"])

    with inp2:
        ai_disabled = (uses_ai == "No")
        ai_tools_used = st.selectbox(
            "사용 AI 도구 (ai_tools_used)",
            options=["ChatGPT", "Copilot", "Gemini"],
            disabled=ai_disabled,
            help="AI를 사용하지 않으면 선택이 비활성화됩니다."
        )
        purpose_of_ai = st.selectbox(
            "AI 사용 목적 (purpose_of_ai)",
            options=["Coding", "Homework", "Research"],
            disabled=ai_disabled,
            help="AI를 사용하지 않으면 선택이 비활성화됩니다."
        )
        grades_before = st.slider(
            "AI 사용 전 성적 (grades_before_ai)",
            min_value=55, max_value=75, value=65, step=1
        )
        screen_time = st.slider(
            "하루 스크린 타임 (daily_screen_time_hours)",
            min_value=2, max_value=7, value=4, step=1
        )

    if st.button("🚀 Predict", type="primary", use_container_width=True):
        actual_ai_tool    = "None" if uses_ai == "No" else ai_tools_used
        actual_ai_purpose = "None" if uses_ai == "No" else purpose_of_ai

        user_raw = pd.DataFrame([{
            "age":                     age,
            "education_level":         education_level,
            "study_hours_per_day":     study_hours,
            "uses_ai":                 uses_ai,
            "ai_tools_used":           actual_ai_tool,
            "purpose_of_ai":           actual_ai_purpose,
            "grades_before_ai":        grades_before,
            "daily_screen_time_hours": screen_time,
        }])

        user_before = encode_raw_ordinal(user_raw)[RAW_FEATURE_COLS]
        user_after  = encode_user_input_after(user_raw, AFTER_FEATURE_COLS)

        user_by_dataset = {"전처리 전": user_before, "전처리 후": user_after}

        st.divider()
        st.subheader("🎯 예측 결과 비교")

        # 예측값 수집: pred_values[데이터셋][모델] = 값
        pred_values = {"전처리 전": {}, "전처리 후": {}}
        for dataset_key, models in trained_models.items():
            for mname, model in models.items():
                pred = float(np.clip(
                    model.predict(user_by_dataset[dataset_key])[0], 0, 100
                ))
                pred_values[dataset_key][mname] = pred

        # st.metric 카드: 데이터셋별로 구분해서 표시
        for dataset_key in DATASETS:
            st.markdown(f"**{dataset_key} 모델**")
            m_cols = st.columns(len(MODEL_REGISTRY))
            for col, (mname, pred) in zip(m_cols, pred_values[dataset_key].items()):
                with col:
                    st.metric(
                        label=f"📌 {mname}",
                        value=f"{pred:.1f} 점",
                        delta=f"{pred - grades_before:+.1f} 점 (AI 사용 전 대비)",
                    )

        # ── 예측값 비교 바 차트 (Plotly) ──────────────────────
        fig_pred_cmp = go.Figure()
        for dataset_key in DATASETS:
            fig_pred_cmp.add_trace(go.Bar(
                name=dataset_key,
                x=list(pred_values[dataset_key].keys()),
                y=list(pred_values[dataset_key].values()),
                marker_color=DATASETS[dataset_key]["color"],
                hovertemplate="<b>%{x}</b><br>예측: %{y:.1f}점<extra></extra>",
            ))
        fig_pred_cmp.add_hline(
            y=grades_before, line_dash="dash", line_color="gray",
            annotation_text="AI 사용 전 성적", annotation_position="top left",
        )
        fig_pred_cmp.update_layout(
            title=dict(text="모델·데이터셋별 예측 성적 비교", font=dict(size=15)),
            yaxis_title="예측 grades_after_ai",
            barmode="group",
            height=380,
            margin=dict(l=40, r=20, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_pred_cmp, use_container_width=True)

        st.success("✅ 예측이 완료되었습니다!")

    # ============================================================
    # 섹션 4: 데이터 미리보기
    # ============================================================
    with st.expander("📂 데이터 미리보기"):
        st.markdown("**전처리 전 데이터 (students_ai_usage.csv)**")
        st.dataframe(df_raw_original.head(10), use_container_width=True)
        st.caption(f"{df_raw_original.shape[0]}행 × {df_raw_original.shape[1]}열")

        st.markdown("**전처리 후 데이터 (preprocessed_sau_data.csv)**")
        st.dataframe(df_processed.head(10), use_container_width=True)
        st.caption(f"{df_processed.shape[0]}행 × {df_processed.shape[1]}열")


if __name__ == "__main__":
    main()
