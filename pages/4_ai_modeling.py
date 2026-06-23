# ============================================================
# AI 성적 예측 웹앱 (Streamlit)
# 목적: 학생의 AI 사용 및 학습 데이터를 기반으로
#       AI 사용 후 성적(grades_after_ai)을 예측한다
# 사용 모델: Linear Regression, Random Forest Regressor
# 데이터: students_ai_usage.csv (100행)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import joblib
import os

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ── matplotlib 기본 폰트 설정 ─────────────────────────────────
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# 0. CSV 실제 값 기반 고정 매핑 (LabelEncoder 대신 사용)
#
#    LabelEncoder 는 학습 데이터에서 본 값만 인식하므로,
#    UI 선택값과 조금이라도 다르면 "unseen label" 오류가 난다.
#    → 아래처럼 딕셔너리로 고정하면 학습/예측 때 항상 동일한 숫자를 사용한다.
#
#    CSV 실제 고유값 확인 결과:
#      education_level : 'college', 'school'
#      uses_ai         : 'No', 'Yes'
#      ai_tools_used   : 'ChatGPT', 'Copilot', 'Gemini'  (uses_ai=No 이면 NaN → "None")
#      purpose_of_ai   : 'Coding', 'Homework', 'Research' (uses_ai=No 이면 NaN → "None")
# ============================================================
CATEGORY_MAPS = {
    "education_level": {"school": 0, "college": 1},
    "uses_ai":         {"No": 0,     "Yes": 1},
    # NaN(AI 미사용) 행은 "None" 으로 채운 뒤 0 으로 인코딩
    "ai_tools_used":   {"None": 0, "ChatGPT": 1, "Copilot": 2, "Gemini": 3},
    "purpose_of_ai":   {"None": 0, "Coding": 1,  "Homework": 2, "Research": 3},
}

# ============================================================
# 1. 모델 관리 딕셔너리 (확장성 높은 구조)
#    새 모델을 추가할 때 이 딕셔너리에만 항목을 추가하면
#    학습·평가·예측·시각화가 자동으로 확장된다.
# ============================================================
MODEL_REGISTRY = {
    "Linear Regression": {
        "model_class": LinearRegression,
        "model_file":  "model_linear.pkl",   # 파일이 있으면 자동으로 불러옴
        "params":      {},
    },
    "Random Forest": {
        "model_class": RandomForestRegressor,
        "model_file":  "model_rf.pkl",
        "params":      {"n_estimators": 100, "random_state": 42},
    },
    # ── 새 모델 추가 예시 ──────────────────────────────────────
    # "Gradient Boosting": {
    #     "model_class": GradientBoostingRegressor,
    #     "model_file":  "model_gb.pkl",
    #     "params":      {"n_estimators": 100, "random_state": 42},
    # },
}

# ============================================================
# 2. 유틸리티 함수
# ============================================================

@st.cache_data  # 같은 파일을 반복 로드하지 않도록 캐싱
def load_data(filepath: str) -> pd.DataFrame:
    """
    CSV 파일을 읽어 반환한다.
    uses_ai=No 인 행의 ai_tools_used, purpose_of_ai 는 NaN 이므로
    "None" 문자열로 채워 인코딩 오류를 방지한다.
    """
    df = pd.read_csv(filepath)
    df["ai_tools_used"] = df["ai_tools_used"].fillna("None")
    df["purpose_of_ai"] = df["purpose_of_ai"].fillna("None")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    CATEGORY_MAPS 에 정의된 고정 딕셔너리로 범주형 컬럼을 숫자로 변환한다.
    - 고정 매핑이므로 학습/예측 때 항상 동일한 숫자를 보장한다.
    - "unseen label" 오류가 발생하지 않는다.
    """
    df = df.copy()
    for col, mapping in CATEGORY_MAPS.items():
        df[col] = df[col].map(mapping)
        # 매핑에 없는 값이 있으면 NaN → 0 으로 대체 (안전장치)
        df[col] = df[col].fillna(0).astype(int)
    return df


def load_or_create_model(name: str):
    """
    joblib 파일이 존재하면 불러오고, 없으면 새 인스턴스를 생성한다.
    추후 학습된 .pkl 파일을 같은 폴더에 넣으면 자동으로 사용된다.
    """
    info = MODEL_REGISTRY[name]
    if os.path.exists(info["model_file"]):
        model = joblib.load(info["model_file"])
        st.sidebar.success(f"✅ {name}: 저장된 모델 불러옴")
    else:
        model = info["model_class"](**info["params"])
    return model


def train_and_save(model, X_train, y_train, filepath: str):
    """모델을 학습하고 joblib 으로 저장한다."""
    model.fit(X_train, y_train)
    joblib.dump(model, filepath)
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """
    RMSE 와 R² Score 를 계산해 반환한다.
    RMSE : 예측 오차 크기 (낮을수록 좋음)
    R²   : 설명력          (1에 가까울수록 좋음)
    """
    preds = model.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    r2    = r2_score(y_test, preds)
    return {"RMSE": round(rmse, 4), "R² Score": round(r2, 4), "predictions": preds}


# ============================================================
# 3. Streamlit 앱 메인
# ============================================================

def main():
    st.set_page_config(page_title="AI 성적 예측기", page_icon="🎓", layout="wide")

    st.title("🎓 AI 사용 후 성적 예측 웹앱")
    st.markdown(
        "학생의 학습 정보와 AI 사용 패턴을 입력하면 "
        "**AI 사용 후 성적(grades_after_ai)** 을 예측합니다."
    )
    st.divider()

    # ── 데이터 로드 ───────────────────────────────────────────
    # CSV 파일이 같은 폴더에 없으면 경고 후 종료
    CSV_PATH = "students_ai_usage.csv"
    if not os.path.exists(CSV_PATH):
        st.error(f"❌ '{CSV_PATH}' 파일을 찾을 수 없습니다. 앱과 같은 폴더에 파일을 넣어주세요.")
        st.stop()

    df_raw = load_data(CSV_PATH)

    # ── 피처 / 타겟 정의 ─────────────────────────────────────
    FEATURE_COLS = [
        "age", "education_level", "study_hours_per_day", "uses_ai",
        "ai_tools_used", "purpose_of_ai", "grades_before_ai", "daily_screen_time_hours",
    ]
    TARGET_COL = "grades_after_ai"

    # 전처리 (고정 매핑 적용)
    df_processed = preprocess(df_raw[FEATURE_COLS + [TARGET_COL]])
    X = df_processed[FEATURE_COLS]
    y = df_processed[TARGET_COL]

    # 학습 / 테스트 분리 (8:2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── 모델 학습 또는 로드 ───────────────────────────────────
    trained_models = {}
    results        = {}

    for model_name, info in MODEL_REGISTRY.items():
        model = load_or_create_model(model_name)
        if not os.path.exists(info["model_file"]):
            model = train_and_save(model, X_train, y_train, info["model_file"])
            st.sidebar.info(f"🔄 {model_name}: 새로 학습 후 저장됨")
        trained_models[model_name] = model
        results[model_name]        = evaluate_model(model, X_test, y_test)

    # ============================================================
    # 섹션 1: 모델 성능 비교 표
    # ============================================================
    st.header("📊 모델 성능 비교")
    df_cmp = pd.DataFrame({
        "모델":                    list(results.keys()),
        "RMSE (낮을수록 좋음)":    [results[m]["RMSE"]     for m in results],
        "R² Score (높을수록 좋음)":[results[m]["R² Score"] for m in results],
    }).set_index("모델")
    st.dataframe(df_cmp, use_container_width=True)

    # ============================================================
    # 섹션 2: 시각화
    # ============================================================
    st.header("📈 시각화")
    col1, col2 = st.columns(2)

    # ── (a) 실제값 vs 예측값 Scatter Plot ────────────────────
    with col1:
        st.subheader("실제 성적 vs 예측 성적")
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        colors = ["steelblue", "tomato"]
        for i, (mname, res) in enumerate(results.items()):
            ax1.scatter(y_test, res["predictions"],
                        alpha=0.6, label=mname,
                        color=colors[i % len(colors)], s=30)
        lo = min(y_test.min(), min(r["predictions"].min() for r in results.values()))
        hi = max(y_test.max(), max(r["predictions"].max() for r in results.values()))
        ax1.plot([lo, hi], [lo, hi], "k--", lw=1, label="Perfect Fit")
        ax1.set_xlabel("Actual grades_after_ai")
        ax1.set_ylabel("Predicted grades_after_ai")
        ax1.set_title("Actual vs Predicted")
        ax1.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    # ── (b) Random Forest Feature Importance ─────────────────
    with col2:
        st.subheader("Feature Importance (Random Forest)")
        rf_model = trained_models.get("Random Forest")
        if rf_model and hasattr(rf_model, "feature_importances_"):
            feat_s = pd.Series(rf_model.feature_importances_,
                               index=FEATURE_COLS).sort_values(ascending=True)
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            feat_s.plot(kind="barh", ax=ax2,
                        color=sns.color_palette("Blues_d", len(feat_s)))
            ax2.set_xlabel("Importance")
            ax2.set_title("Feature Importance")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.warning("Random Forest 모델이 없거나 feature_importances_ 를 계산할 수 없습니다.")

    # ============================================================
    # 섹션 3: 사용자 입력 → 성적 예측
    # ============================================================
    st.divider()
    st.header("🔮 성적 예측하기")
    st.markdown("아래에 학생 정보를 입력한 뒤 **Predict** 버튼을 눌러 예측 성적을 확인하세요.")

    inp1, inp2 = st.columns(2)

    with inp1:
        # CSV 실제 범위: age 14~19
        age = st.slider("나이 (age)", min_value=14, max_value=19, value=17, step=1)

        # CSV 실제 값: 'school', 'college'
        education_level = st.selectbox(
            "교육 수준 (education_level)", options=["school", "college"]
        )

        # CSV 실제 범위: 1.0 ~ 5.0
        study_hours = st.slider(
            "하루 공부 시간 (study_hours_per_day)",
            min_value=1.0, max_value=5.0, value=3.0, step=0.1
        )

        uses_ai = st.selectbox("AI 사용 여부 (uses_ai)", options=["Yes", "No"])

    with inp2:
        # uses_ai=No 이면 AI 도구/목적 선택 비활성화
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

        # CSV 실제 범위: 55~75
        grades_before = st.slider(
            "AI 사용 전 성적 (grades_before_ai)",
            min_value=55, max_value=75, value=65, step=1
        )

        # CSV 실제 범위: 2~7
        screen_time = st.slider(
            "하루 스크린 타임 (daily_screen_time_hours)",
            min_value=2, max_value=7, value=4, step=1
        )

    # ── Predict 버튼 ─────────────────────────────────────────
    if st.button("🚀 Predict", type="primary", use_container_width=True):

        # uses_ai=No 이면 ai_tools_used / purpose_of_ai 는 "None" 으로 처리
        actual_ai_tool    = "None" if uses_ai == "No" else ai_tools_used
        actual_ai_purpose = "None" if uses_ai == "No" else purpose_of_ai

        # 사용자 입력 → DataFrame
        user_input = pd.DataFrame([{
            "age":                   age,
            "education_level":       education_level,
            "study_hours_per_day":   study_hours,
            "uses_ai":               uses_ai,
            "ai_tools_used":         actual_ai_tool,
            "purpose_of_ai":         actual_ai_purpose,
            "grades_before_ai":      grades_before,
            "daily_screen_time_hours": screen_time,
        }])

        # 고정 매핑으로 인코딩 (학습 때와 동일한 변환)
        user_processed = preprocess(user_input)

        st.divider()
        st.subheader("🎯 예측 결과")

        res_cols = st.columns(len(trained_models))
        for col, (mname, model) in zip(res_cols, trained_models.items()):
            pred = float(np.clip(model.predict(user_processed[FEATURE_COLS])[0], 0, 100))
            with col:
                st.metric(
                    label=f"📌 {mname}",
                    value=f"{pred:.1f} 점",
                    delta=f"{pred - grades_before:+.1f} 점 (AI 사용 전 대비)",
                )

        st.success("✅ 예측이 완료되었습니다!")

    # ============================================================
    # 섹션 4: 데이터 미리보기 (접이식)
    # ============================================================
    with st.expander("📂 학습 데이터 미리보기 (상위 10행)"):
        st.dataframe(df_raw.head(10), use_container_width=True)
        st.caption(f"전체 데이터: {df_raw.shape[0]}행 × {df_raw.shape[1]}열")


# ── 엔트리 포인트 ────────────────────────────────────────────
if __name__ == "__main__":
    main()
