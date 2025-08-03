import numpy as np
import pandas as pd
import joblib

# Load the model
model = joblib.load('diabetes_model.pkl')

# Define feature names (same as used in training)
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Example input
input_data = [2, 120, 70, 25, 85, 32.0, 0.55, 33]

# Convert to DataFrame
input_df = pd.DataFrame([input_data], columns=feature_names)

# Predict
prediction = model.predict(input_df)

# Result
if prediction[0] == 1:
    print("⚠️ The person is likely to have diabetes.")
else:
    print("✅ The person is not likely to have diabetes.")
