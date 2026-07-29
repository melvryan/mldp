import joblib
import streamlit as st
import numpy as np
import pandas as pd
import base64

st.set_page_config(page_title="Insurance Charges Prediction", layout="centered")

## Load trained model and feature names
model = joblib.load("gradient_boosting.pkl")
feature_names = joblib.load("feature_names.pkl")

sexes = ['female', 'male']
smokers = ['no', 'yes']

## ---------- Styling ----------
def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

img = get_base64("backgroundpng_files/28eab19c-1753-44e1-b372-508609c58b34.png")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image:
            linear-gradient(rgba(4,20,20,0.55), rgba(4,20,20,0.75)),
            url("data:image/png;base64,{img}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    .card {{
        background-color: rgba(10, 30, 30, 0.72);
        padding: 22px 26px;
        border-radius: 14px;
        margin-bottom: 18px;
        border: 1px solid rgba(255,255,255,0.08);
    }}

    h1, h2, h3, p, label, span {{
        color: #F2F2F0 !important;
    }}

    .result-box {{
        background-color: rgba(10, 30, 30, 0.85);
        border-left: 4px solid #D4A94B;
        padding: 20px 24px;
        border-radius: 10px;
        margin-top: 10px;
    }}

    .stButton > button {{
        background-color: #D4A94B;
        color: #10201F;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 8px 22px;
    }}

    .stButton > button:hover {{
        background-color: #E6BC5D;
        color: #10201F;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: rgba(10, 30, 30, 0.55);
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        color: #F2F2F0;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: rgba(10, 30, 30, 0.9);
        border-bottom: 2px solid #D4A94B;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

## ---------- Top tab navigation ----------
tab_predict, tab_about, tab_insights = st.tabs(["Predict", "About", "Model Insights"])

## ============================================================
## TAB: PREDICT
## ============================================================
with tab_predict:

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.title("Insurance Charges Prediction")
    st.write("Predict your medical insurance charges using a trained Machine Learning model.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        age_selected = st.number_input("Age", min_value=18, max_value=100, value=30)
        height_selected = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
        sex_selected = st.selectbox("Sex", sexes)
    with col2:
        weight_selected = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0)
        children_selected = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        smoker_selected = st.selectbox("Smoker", smokers)

    height_m = height_selected / 100
    bmi = weight_selected / (height_m ** 2)
    st.write(f"Calculated BMI: **{bmi:.2f}**")

    btn_col, _ = st.columns([1, 3])
    with btn_col:
        predict_clicked = st.button("Predict")

    st.markdown('</div>', unsafe_allow_html=True)

    if predict_clicked:

        try:
            df_input = pd.DataFrame({
                'age': [age_selected],
                'bmi': [bmi],
                'children': [children_selected],
                'sex': [sex_selected],
                'smoker': [smoker_selected]
            })

            df_input["sex"] = pd.Categorical(df_input["sex"], categories=sexes)
            df_input["smoker"] = pd.Categorical(df_input["smoker"], categories=smokers)

            df_input = pd.get_dummies(df_input, columns=['sex', 'smoker'], drop_first=True)
            df_input = df_input.reindex(columns=feature_names, fill_value=0)

            prediction = model.predict(df_input)[0]

            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.subheader("Predicted Insurance Charges")
            st.markdown(f"### ${prediction:,.2f}")
            st.markdown('</div>', unsafe_allow_html=True)

            reference = pd.DataFrame({
                "Group": ["Typical Non-Smoker", "Typical Smoker", "Your Prediction"],
                "Charges": [8500, 32000, prediction]
            })

            st.bar_chart(reference.set_index("Group"))

        except Exception as e:
            st.error("Something went wrong while generating the prediction. Please check your inputs and try again.")

## ============================================================
## TAB: ABOUT
## ============================================================
with tab_about:

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.title("About This Tool")
    st.write(
        """
        This app estimates medical insurance charges using a Gradient Boosting
        model trained on historical policyholder data.

        **Inputs used by the model:**
        - Age
        - Sex
        - BMI (calculated from height and weight)
        - Number of children
        - Smoker status

        These features were selected after testing feature importance. The region
        variables were removed as they contributed almost no predictive value.
        Sex was initially considered for removal too, but testing showed that
        keeping it slightly improved both MAE and R² compared to dropping it,
        so it was retained in the final model.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

## ============================================================
## TAB: MODEL INSIGHTS
## ============================================================
with tab_insights:

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.title("Model Insights")
    st.write("Feature importance from the trained Gradient Boosting model:")

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=True)

    st.bar_chart(importance.set_index("Feature"))
    st.markdown('</div>', unsafe_allow_html=True)