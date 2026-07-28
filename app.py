import streamlit as st
import pandas as pd
import joblib

model = joblib.load("gradient_boosting.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("Insurance Charges Prediction")

st.write(
    "Predict your medical insurance charges using a trained Machine Learning model."
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

height = st.number_input(
    "Height (cm)",
    min_value=100.0,
    max_value=250.0,
    value=170.0
)

weight = st.number_input(
    "Weight (kg)",
    min_value=20.0,
    max_value=300.0,
    value=70.0
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0
)

smoker = st.selectbox(
    "Smoker",
    ["no", "yes"]
)

# Calculate BMI
height_m = height / 100
bmi = weight / (height_m ** 2)

st.write(f"Calculated BMI: **{bmi:.2f}**")

input_df = pd.DataFrame({
    "age": [age],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker]
})

# Tell pandas all smoker categories
input_df["smoker"] = pd.Categorical(
    input_df["smoker"],
    categories=["no", "yes"]
)

input_df = pd.get_dummies(
    input_df,
    columns=["smoker"],
    drop_first=True
)

input_df = input_df.reindex(
    columns=feature_names,
    fill_value=0
)

if st.button("Predict Insurance Charges"):

    prediction = model.predict(input_df)[0]

    st.subheader("Predicted Insurance Charges")

    st.success(f"Predicted Insurance Charges: ${prediction:,.2f}")