import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Patient Medication Adherence Calculator", layout="wide")

st.sidebar.markdown("""
# System Instructions

## About This System
This is a Patient Medication Adherence Calculator based on Random Forest algorithm that predicts treatment outcomes by analyzing patient health indicators.

## Prediction Results
The system predicts treatment outcome categories (0-2):
- Category 0: Poor Treatment Effect
- Category 1: Moderate Treatment Effect
- Category 2: Good Treatment Effect

## How to Use
1. Fill in personal health information on the main screen
2. Click the Predict button to generate predictions
3. View prediction results and feature importance analysis

## Important Notes
- Please ensure accurate personal information
- All fields are required
- Enter numbers for numerical fields
- Select from options for other fields
""")

st.title("Patient Medication Adherence Calculator")

@st.cache_resource
def load_model_package():
    try:
        model_package = joblib.load('rfc.pkl')
        return model_package
    except Exception as e:
        st.error(f"Error loading model package: {str(e)}")
        return None

model_package = load_model_package()
if model_package is None:
    st.stop()

model = model_package['model']
scaler = model_package['scaler']
encoder = model_package['encoder'] 
continuous_cols = model_package['continuous_cols']
categorical_cols = model_package['categorical_cols']

st.header("Please input patient's clinical indicators:")

# Using text_input to ensure empty initial state
def number_or_none(label, key, placeholder="Enter value"):
    raw = st.text_input(label, value="", key=key, placeholder=placeholder)
    if raw.strip() == "":
        return None
    try:
        return float(raw)
    except ValueError:
        st.error(f"Invalid value for {label}. Please enter a valid number.")
        return None

# Inputs
age = number_or_none('Age', 'age')
height = number_or_none('Height (cm)', 'height')
weight = number_or_none('Weight (kg)', 'weight')
diseases = number_or_none('Number of Comorbid Diseases', 'diseases')
medications = number_or_none('Number of Concomitant Medications', 'medications')
gender = st.selectbox('Gender', ['Female', 'Male'], index=None, placeholder="Select Gender")
location = st.selectbox('Treatment Type', ['Outpatient', 'Inpatient'], index=None, placeholder="Select Treatment Type")
education = st.selectbox('Education Level', ['Primary School or Below', 'High School', 'College or Above'], index=None, placeholder="Select Education Level")
stroke = st.selectbox('History of Stroke', ['No', 'Yes'], index=None, placeholder="Select Stroke History")

predict_button = st.button('Predict')

if predict_button:
    gender_map = {'Female': 0, 'Male': 1}
    location_map = {'Outpatient': 0, 'Inpatient': 1}
    education_map = {'Primary School or Below': 0, 'High School': 1, 'College or Above': 2}
    stroke_map = {'No': 0, 'Yes': 1}

    data = {
        'age': age,
        'height': height,
        'weight': weight,
        'number_of_comorbid_diseases': diseases,
        'number_of_concomitant_medications': medications,
        'gender': gender_map[gender],
        'location': location_map[location],
        'educational_attainment': education_map[education],
        'history_of_stroke': stroke_map[stroke]
    }
    df = pd.DataFrame(data, index=[0])

    df_continuous = df[continuous_cols]
    df_categorical = df[categorical_cols]

    df_categorical_encoded = encoder.transform(df_categorical)

    df_combined = np.concatenate([df_continuous, df_categorical_encoded], axis=1)

    df_preprocessed = scaler.transform(df_combined)

    prediction = model.predict(df_preprocessed)
    prediction_proba = model.predict_proba(df_preprocessed)
    
    st.subheader("Prediction Result")
    result_map = {0: "Poor Treatment Effect", 1: "Moderate Treatment Effect", 2: "Good Treatment Effect"}
    st.write(f"Predicted Category: {result_map[prediction[0]]}")
    
    st.subheader("Prediction Probability")
    prob_df = pd.DataFrame(prediction_proba, 
                          columns=['Category 0 (Poor)', 'Category 1 (Moderate)', 'Category 2 (Good)'])
    st.write(prob_df)

    st.header("Feature Importance Analysis")
    st.write("The charts below show how each factor influences the prediction:")

    @st.cache_resource
    def get_shap_explainer():
        explainer = shap.TreeExplainer(model)
        return explainer

    explainer = get_shap_explainer()
    shap_values = explainer.shap_values(df_preprocessed)

    st.subheader("Feature Importance Ranking")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    feature_names = (continuous_cols.tolist() + 
                    [f"{col}_{val}" for col, vals in 
                     zip(categorical_cols, encoder.categories_) 
                     for val in vals])
    
    shap.summary_plot(shap_values, 
                     df_preprocessed,
                     feature_names=feature_names,
                     plot_type="bar",
                     show=False)
    st.pyplot(fig)
    plt.clf()
