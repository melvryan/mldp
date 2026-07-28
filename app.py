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

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

smoker = st.selectbox(
    "Smoker",
    ["yes", "no"]
)

region = st.selectbox(
    "Region",
    [
        "northeast",
        "northwest",
        "southeast",
        "southwest"
    ]
)

# Calculate BMI
height_m = height / 100
bmi = weight / (height_m ** 2)

input_df = pd.DataFrame({
    "age": [age],
    "bmi": [bmi],
    "children": [children],
    "sex": [sex],
    "smoker": [smoker],
    "region": [region]
})

# Force pandas to know about all possible categories,
# so get_dummies creates the correct columns even with a single row
input_df["sex"] = pd.Categorical(input_df["sex"], categories=["female", "male"])
input_df["smoker"] = pd.Categorical(input_df["smoker"], categories=["no", "yes"])
input_df["region"] = pd.Categorical(
    input_df["region"],
    categories=["northeast", "northwest", "southeast", "southwest"]
)

input_df = pd.get_dummies(
    input_df,
    columns=["sex", "smoker", "region"],
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