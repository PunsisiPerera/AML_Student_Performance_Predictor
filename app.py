from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "ebm_model.joblib"
IMPORTANCE_PATH = ROOT / "outputs" / "feature_importance.csv"

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")
st.title("🎓 Student Performance Prediction")
st.write("An educational demonstration using an Explainable Boosting Machine (EBM).")
st.warning("This is not a definitive judgement of a student. Use predictions only to explore possible support needs, never to punish or deny opportunities.")

if not MODEL_PATH.exists():
    st.error("Trained model not found. In the project folder, run: `python src/train_model.py`")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model, threshold, feature_columns = bundle["model"], bundle["threshold"], bundle["feature_columns"]

st.sidebar.header("Student information")
st.sidebar.caption("Enter study and academic-history information. Fields are based on the UCI dataset.")
values = {}
categorical_options = {
    "guardian": ["mother", "father", "other"],
    "schoolsup": ["yes", "no"],
    "famsup": ["yes", "no"],
    "paid": ["yes", "no"],
    "activities": ["yes", "no"],
    "nursery": ["yes", "no"],
    "higher": ["yes", "no"],
    "internet": ["yes", "no"],
    "romantic": ["yes", "no"],
    "Mjob": ["teacher", "health", "services", "at_home", "other"],
    "Fjob": ["teacher", "health", "services", "at_home", "other"],
    "reason": ["home", "reputation", "course", "other"],
    "famsize": ["LE3", "GT3"],
    "Pstatus": ["T", "A"]
}
numeric_defaults = {
    "Medu": (0, 4, 2), "Fedu": (0, 4, 2), "traveltime": (1, 4, 2),
    "studytime": (1, 4, 2), "failures": (0, 4, 0), "famrel": (1, 5, 4),
    "freetime": (1, 5, 3), "goout": (1, 5, 3), "Dalc": (1, 5, 1),
    "Walc": (1, 5, 2), "health": (1, 5, 4), "absences": (0, 30, 3)
}
for col in feature_columns:
    if col in categorical_options:
        values[col] = st.sidebar.selectbox(col, categorical_options[col])
    elif col in numeric_defaults:
        low, high, default = numeric_defaults[col]
        values[col] = st.sidebar.slider(col, min_value=low, max_value=high, value=default)
    else:
        # For an unexpected source column, use a safe common value inferred from model training schema.
        values[col] = 0

if st.button("Predict performance", type="primary"):
    row = pd.DataFrame([{c: values[c] for c in feature_columns}], columns=feature_columns)
    prob_pass = float(model.predict_proba(row)[0, 1])
    predicted = int(prob_pass >= threshold)
    left, right = st.columns(2)
    with left:
        st.subheader("Prediction")
        if predicted:
            st.success("Model prediction: likely to pass")
        else:
            st.error("Model prediction: at risk of not passing")
        st.metric("Estimated pass probability", f"{prob_pass:.1%}")
        st.progress(min(max(prob_pass, 0.0), 1.0))
        st.caption(f"Decision threshold selected using validation data: {threshold:.2f}")
    with right:
        st.subheader("How to interpret this")
        st.write("This estimate reflects patterns in the historical dataset, not certainty about an individual.")
        st.write("A teacher or student-support team should review the wider context before taking action.")
    st.subheader("Entered information")
    st.dataframe(row, use_container_width=True)
    st.info("This interface displays global feature importance below. It does not claim that a feature caused the prediction.")

st.divider()
st.subheader("What influences the model overall?")
if IMPORTANCE_PATH.exists():
    imp = pd.read_csv(IMPORTANCE_PATH).head(12).sort_values("importance")
    if not imp.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(imp["feature"], imp["importance"])
        ax.set_xlabel("EBM global importance")
        ax.set_ylabel("Feature")
        st.pyplot(fig)
        st.caption("Global importance describes model reliance across the test/training task; it is not proof of causation.")
else:
    st.info("Run the training script to generate feature-importance results.")

with st.expander("Project details and limitations"):
    st.markdown("""
    - **Algorithm:** Explainable Boosting Machine (EBM), not included in the supplied lecture-covered list.
    - **Target:** final mathematics grade G3 ≥ 10/20 is labelled pass; otherwise at risk.
    - **Leakage prevention:** G1, G2, and G3 are not predictors.
    - **Limitations:** dataset comes from Portuguese secondary schools, may be outdated, and may not generalize to Sri Lankan students.
    - **Ethics:** predictions should support early help, not be used as labels or to restrict educational opportunities.
    """)
