import joblib
import streamlit as st
import numpy as np
import pandas as pd

## Load trained model and feature names
model = joblib.load("gradient_boosting.pkl")
feature_names = joblib.load("feature_names.pkl")

## Streamlit app
st.title("Insurance Charges Prediction")

st.write(
    "Predict your medical insurance charges using a trained Machine Learning model."
)

## Define the input options
sexes = ['male', 'female']
smokers = ['yes', 'no']
regions = ['northeast', 'northwest', 'southeast', 'southwest']

## User inputs
age_selected = st.number_input("Age", min_value=18, max_value=100, value=30)
height_selected = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
weight_selected = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0)
children_selected = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
sex_selected = st.selectbox("Sex", sexes)
smoker_selected = st.selectbox("Smoker", smokers)
region_selected = st.selectbox("Region", regions)

## Predict button
if st.button("Predict Insurance Charges"):

    ## Calculate BMI
    height_m = height_selected / 100
    bmi = weight_selected / (height_m ** 2)

    ## Create dict for input features
    input_data = {
        'age': age_selected,
        'bmi': bmi,
        'children': children_selected,
        'sex': sex_selected,
        'smoker': smoker_selected,
        'region': region_selected
    }

    ## Convert input data to a DataFrame
    df_input = pd.DataFrame({
        'age': [age_selected],
        'bmi': [bmi],
        'children': [children_selected],
        'sex': [sex_selected],
        'smoker': [smoker_selected],
        'region': [region_selected]
    })

    ## Force pandas to know about all possible categories,
    ## so get_dummies creates the correct columns even with a single row
    df_input["sex"] = pd.Categorical(df_input["sex"], categories=sexes)
    df_input["smoker"] = pd.Categorical(df_input["smoker"], categories=smokers)
    df_input["region"] = pd.Categorical(df_input["region"], categories=regions)

    ## One-hot encoding
    df_input = pd.get_dummies(
        df_input,
        columns=['sex', 'smoker', 'region'],
        drop_first=True
    )

    df_input = df_input.reindex(columns=feature_names, fill_value=0)

    ## Predict
    prediction = model.predict(df_input)[0]

    st.subheader("Predicted Insurance Charges")
    st.success(f"Predicted Insurance Charges: ${prediction:,.2f}")

## Page design