# ============================================================
# AI 성적 예측 웹앱 (Streamlit)
# 목적: 학생의 AI 사용 및 학습 데이터를 기반으로
#       AI 사용 후 성적(grades_after_ai)을 예측한다
# 사용 모델: Linear Regression, Random Forest Regressor
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
from sklearn.preprocessing import LabelEncoder

# ── 한글 폰트 설정 (matplotlib 에서 한글 깨짐 방지) ──────────────────
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 모델 관리 딕셔너리 (확장성 높은 구조)
#    - 나중에 새 모델을 추가할 때 이 딕셔너리에만 추가하면 된다
# ============================================================
MODEL_REGISTRY = {
    "Linear Regression": {
        "model_class": LinearRegression,   # 사용할 sklearn 클래스
        "model_file": "model_linear.pkl",  # joblib 으로 저장된 파일 경로 (공란 → 미구현)
        "params": {},                       # 모델 하이퍼파라미터
    },
    "Random Forest": {
        "model_class": RandomForestRegressor,
        "model_file": "model_rf.pkl",      # 공란 (아직 파일 없음)
        "params": {"n_estimators": 100, "random_state": 42},
    },
    # ── 새 모델 추가 예시 (주석 해제 후 사용) ──
    # "Gradient Boosting": {
    #     "model_class": GradientBoostingRegressor,
    #     "model_file": "model_gb.pkl",
    #     "params": {"n_estimators": 100, "random_state": 42},
    # },
}

# ============================================================
# 2. 유틸리티 함수
# ============================================================

def load_or_create_model(name: str, registry: dict):
    """
    joblib 파일이 존재하면 파일에서 모델을 불러오고,
    없으면 registry 정보로 새 모델 인스턴스를 생성한다.
    → 추후 학습된 모델 파일(.pkl)을 같은 폴더에 넣으면 자동으로 사용된다.
    """
    info = registry[name]
    filepath = info["model_file"]

    if os.path.exists(filepath):
        # 저장된 모델 파일이 있으면 불러오기
        model = joblib.load(filepath)
        st.sidebar.success(f"✅ {name}: 모델 불러옴")
    else:
        # 파일이 없으면 새 인스턴스 생성 (아직 학습 전 상태)
        model = info["model_class"](**info["params"])
    return model


def save_model(model, filepath: str):
    """
    학습된 모델을 joblib 으로 저장한다.
    """
    joblib.dump(model, filepath)


def generate_sample_data(n: int = 300) -> pd.DataFrame:
    """
    실제 데이터 파일이 없을 때 사용할 샘플 데이터를 생성한다.
    실제 서비스 시에는 이 함수 대신 CSV 파일을 로드하면 된다.
    """
    np.random.seed(42)
    education_options = ["High School", "Undergraduate", "Graduate"]
    ai_tools_options = ["ChatGPT", "Copilot", "Gemini", "Other"]
    purpose_options = ["Homework", "Research", "Practice", "Entertainment"]
    uses_ai_options = ["Yes", "No"]

    data = {
        "age": np.random.randint(15, 30, n),
        "education_level": np.random.choice(education_options, n),
        "study_hours_per_day": np.round(np.random.uniform(0.5, 10, n), 1),
        "uses_ai": np.random.choice(uses_ai_options, n),
        "ai_tools_used": np.random.choice(ai_tools_options, n),
        "purpose_of_ai": np.random.choice(purpose_options, n),
        "grades_before_ai": np.round(np.random.uniform(40, 100, n), 1),
        "daily_screen_time_hours": np.round(np.random.uniform(1, 12, n), 1),
    }
    df = pd.DataFrame(data)

    # 타겟 변수: 기존 성적 + 학습시간 효과 + AI 사용 효과 + 노이즈
    ai_boost = np.where(df["uses_ai"] == "Yes", np.random.uniform(2, 8, n), 0)
    df["grades_after_ai"] = (
        df["grades_before_ai"] * 0.7
        + df["study_hours_per_day"] * 2.5
        + ai_boost
        - df["daily_screen_time_hours"] * 0.8
        + np.random.normal(0, 3, n)
    ).clip(0, 100).round(1)

    return df


def preprocess(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    """
    범주형(문자열) 컬럼을 숫자로 변환한다 (Label Encoding).
    fit=True  → 새 인코더를 학습하고 반환
    fit=False → 기존 인코더를 사용해 변환만 수행
    """
    cat_cols = ["education_level", "uses_ai", "ai_tools_used", "purpose_of_ai"]
    df = df.copy()

    if fit:
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
    else:
        for col in cat_cols:
            le = encoders[col]
            df[col] = le.transform(df[col])

    return df, encoders


def evaluate_model(model, X_test, y_test) -> dict:
    """
    모델의 RMSE 와 R² Score 를 계산해 딕셔너리로 반환한다.
    RMSE : 예측 오차의 크기 (낮을수록 좋음)
    R²   : 모델이 데이터를 얼마나 잘 설명하는지 (1에 가까울수록 좋음)
    """
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return {"RMSE": round(rmse, 4), "R² Score": round(r2, 4), "predictions": preds}


# ============================================================
# 3. Streamlit 앱 메인
# ============================================================

def main():
    # ── 페이지 기본 설정 ──────────────────────────────────────
    st.set_page_config(
        page_title="AI 성적 예측기",
        page_icon="🎓",
        layout="wide",
    )

    st.title("🎓 AI 사용 후 성적 예측 웹앱")
    st.markdown(
        "학생의 학습 정보와 AI 사용 패턴을 입력하면 **AI 사용 후 성적(grades_after_ai)** 을 예측합니다."
    )
    st.divider()

    # ── 사이드바: 데이터 및 모델 불러오기 ───────────────────────
    #st.sidebar.header("⚙️ 설정")
    """st.sidebar.info(
        "모델 파일(.pkl)이 같은 폴더에 있으면 자동으로 불러옵니다.\n"
        "파일이 없으면 샘플 데이터로 새로 학습합니다."
    )"""

    # 샘플 데이터 생성 (실제 CSV 파일로 교체 가능)
    df_raw = generate_sample_data(n=300)

    # 데이터 전처리 (학습용)
    feature_cols = [
        "age", "education_level", "study_hours_per_day", "uses_ai",
        "ai_tools_used", "purpose_of_ai", "grades_before_ai", "daily_screen_time_hours",
    ]
    target_col = "grades_after_ai"

    df_processed, encoders = preprocess(df_raw[feature_cols + [target_col]])
    X = df_processed[feature_cols]
    y = df_processed[target_col]

    # 학습 / 테스트 분리 (8:2 비율)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── 모델 불러오기 또는 새로 학습 ────────────────────────────
    trained_models = {}
    results = {}

    for model_name, info in MODEL_REGISTRY.items():
        model = load_or_create_model(model_name, MODEL_REGISTRY)

        # 저장된 파일이 없으면 학습 후 저장
        if not os.path.exists(info["model_file"]):
            model.fit(X_train, y_train)
            save_model(model, info["model_file"])
            st.sidebar.info(f"🔄 {model_name}: 새로 학습 후 저장됨")

        trained_models[model_name] = model
        results[model_name] = evaluate_model(model, X_test, y_test)

    # ============================================================
    # 섹션 1: 모델 성능 비교 표
    # ============================================================
    st.header("📊 모델 성능 비교")

    comparison_data = {
        "모델": list(results.keys()),
        "RMSE (낮을수록 좋음)": [results[m]["RMSE"] for m in results],
        "R² Score (높을수록 좋음)": [results[m]["R² Score"] for m in results],
    }
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison.set_index("모델"), use_container_width=True)

    # ============================================================
    # 섹션 2: 시각화 - Scatter Plot & Feature Importance
    # ============================================================
    st.header("📈 시각화")

    col1, col2 = st.columns(2)

    # ── (a) 실제값 vs 예측값 Scatter Plot ────────────────────────
    with col1:
        st.subheader("실제 성적 vs 예측 성적")

        fig1, ax1 = plt.subplots(figsize=(5, 4))
        colors = ["steelblue", "tomato"]

        for i, (model_name, res) in enumerate(results.items()):
            ax1.scatter(
                y_test, res["predictions"],
                alpha=0.5, label=model_name,
                color=colors[i % len(colors)], s=25
            )

        # 완벽한 예측을 나타내는 대각선
        min_val = min(y_test.min(), min(r["predictions"].min() for r in results.values()))
        max_val = max(y_test.max(), max(r["predictions"].max() for r in results.values()))
        ax1.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1, label="Perfect Fit")

        ax1.set_xlabel("Actual grades_after_ai")
        ax1.set_ylabel("Predicted grades_after_ai")
        ax1.set_title("Actual vs Predicted")
        ax1.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    # ── (b) Random Forest Feature Importance ─────────────────────
    with col2:
        st.subheader("Feature Importance (Random Forest)")

        rf_model = trained_models.get("Random Forest")
        if rf_model and hasattr(rf_model, "feature_importances_"):
            importances = rf_model.feature_importances_
            feat_series = pd.Series(importances, index=feature_cols).sort_values(ascending=True)

            fig2, ax2 = plt.subplots(figsize=(5, 4))
            feat_series.plot(
                kind="barh", ax=ax2,
                color=sns.color_palette("Blues_d", len(feat_series))
            )
            ax2.set_xlabel("Importance")
            ax2.set_title("Feature Importance")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.warning("Random Forest 모델이 없거나 feature importance 를 계산할 수 없습니다.")

    # ============================================================
    # 섹션 3: 사용자 입력 → 성적 예측
    # ============================================================
    st.divider()
    st.header("🔮 성적 예측하기")
    st.markdown("아래에 학생 정보를 입력한 뒤 **Predict** 버튼을 눌러 예측 성적을 확인하세요.")

    # ── 입력 위젯 배치 (2열 구성) ────────────────────────────────
    inp_col1, inp_col2 = st.columns(2)

    with inp_col1:
        age = st.slider(
            "나이 (age)", min_value=10, max_value=40, value=18, step=1
        )
        education_level = st.selectbox(
            "교육 수준 (education_level)",
            options=["High School", "Undergraduate", "Graduate"],
        )
        study_hours = st.slider(
            "하루 공부 시간 (study_hours_per_day)", min_value=0.0, max_value=12.0,
            value=3.0, step=0.5
        )
        uses_ai = st.selectbox(
            "AI 사용 여부 (uses_ai)", options=["Yes", "No"]
        )

    with inp_col2:
        ai_tools_used = st.selectbox(
            "사용 AI 도구 (ai_tools_used)",
            options=["ChatGPT", "Copilot", "Gemini", "Other"],
        )
        purpose_of_ai = st.selectbox(
            "AI 사용 목적 (purpose_of_ai)",
            options=["Homework", "Research", "Practice", "Entertainment"],
        )
        grades_before = st.slider(
            "AI 사용 전 성적 (grades_before_ai)", min_value=0.0, max_value=100.0,
            value=70.0, step=0.5
        )
        screen_time = st.slider(
            "하루 스크린 타임 (daily_screen_time_hours)", min_value=0.0, max_value=16.0,
            value=4.0, step=0.5
        )

    # ── 예측 버튼 ────────────────────────────────────────────────
    if st.button("🚀 Predict", type="primary", use_container_width=True):
        # 사용자 입력을 DataFrame 으로 만들기
        user_input = pd.DataFrame([{
            "age": age,
            "education_level": education_level,
            "study_hours_per_day": study_hours,
            "uses_ai": uses_ai,
            "ai_tools_used": ai_tools_used,
            "purpose_of_ai": purpose_of_ai,
            "grades_before_ai": grades_before,
            "daily_screen_time_hours": screen_time,
        }])

        # 학습 때와 동일한 인코더로 변환 (fit=False)
        user_processed, _ = preprocess(user_input, encoders=encoders, fit=False)

        st.divider()
        st.subheader("🎯 예측 결과")

        # 각 모델의 예측값을 큰 텍스트로 표시
        res_cols = st.columns(len(trained_models))
        for col, (model_name, model) in zip(res_cols, trained_models.items()):
            pred = model.predict(user_processed[feature_cols])[0]
            pred = float(np.clip(pred, 0, 100))  # 성적 범위 0~100으로 제한
            with col:
                st.metric(
                    label=f"📌 {model_name}",
                    value=f"{pred:.1f} 점",
                    delta=f"{pred - grades_before:+.1f} 점 (AI 사용 전 대비)",
                )

        st.success("✅ 예측이 완료되었습니다!")

    # ============================================================
    # 섹션 4: 데이터 미리보기 (접이식)
    # ============================================================
    with st.expander("📂 샘플 학습 데이터 미리보기 (상위 10행)"):
        st.dataframe(df_raw.head(10), use_container_width=True)
        st.caption(f"전체 데이터 크기: {df_raw.shape[0]}행 × {df_raw.shape[1]}열")


# ── 엔트리 포인트 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
