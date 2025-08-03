import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load model
with open("../disease_models/diabetes_model.pkl", "rb") as f:
    model = joblib.load(f)

# Create dummy data matching feature names
# Replace feature names with real ones if you know them
data = pd.DataFrame({
    "Pregnancies": [6, 1, 3],
    "Glucose": [148, 85, 183],
    "BloodPressure": [72, 66, 64],
    "SkinThickness": [35, 29, 0],
    "Insulin": [0, 0, 0],
    "BMI": [33.6, 26.6, 23.3],
    "DiabetesPedigreeFunction": [0.627, 0.351, 0.672],
    "Age": [50, 31, 32]
})

# SHAP explainer
explainer = shap.Explainer(model, data)
shap_values = explainer(data)

# Summary plot
shap.summary_plot(shap_values, data)
