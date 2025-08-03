import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib

# Load your trained model
model = joblib.load("diabetes_model.pkl")

# Create main window
root = tk.Tk()
root.title("Diabetes Prediction")
root.geometry("450x650")
root.configure(bg="#f8f9fa")

# Field labels with units
labels = [
    "Pregnancies (count)",
    "Glucose (mg/dL)",
    "Blood Pressure (mm Hg)",
    "Skin Thickness (mm)",
    "Insulin (mu U/ml)",
    "BMI (kg/m²)",
    "Diabetes Pedigree Function",
    "Age (years)"
]

entries = []

# Header
header = tk.Label(root, text="🩺 Diabetes Prediction System", font=("Helvetica", 18, "bold"), bg="#f8f9fa", fg="#2c3e50")
header.pack(pady=20)

# Form layout
form_frame = tk.Frame(root, bg="#f8f9fa")
form_frame.pack()

for label_text in labels:
    row = tk.Frame(form_frame, bg="#f8f9fa")
    row.pack(pady=8, fill='x')

    label = tk.Label(row, text=label_text, font=("Arial", 12), width=25, anchor='w', bg="#f8f9fa")
    label.pack(side=tk.LEFT, padx=10)

    entry = tk.Entry(row, font=("Arial", 12), width=20)
    entry.pack(side=tk.RIGHT, padx=10)
    entries.append(entry)

# Prediction function
def predict_diabetes():
    try:
        inputs = []
        for entry in entries:
            value = entry.get().strip()
            if value == "":
                messagebox.showerror("Input Error", "Please fill all the fields.")
                return
            float_val = float(value)
            if float_val < 0 or float_val > 1000:  # basic sanity check
                messagebox.showerror("Input Error", f"Invalid value: {float_val}. Please enter valid numbers.")
                return
            inputs.append(float_val)

        input_array = np.array([inputs])
        prediction = model.predict(input_array)

        result = "🟢 No Diabetes Detected" if prediction[0] == 0 else "🔴 Diabetes Detected!"
        messagebox.showinfo("Prediction Result", result)

    except ValueError:
        messagebox.showerror("Input Error", "Please enter only numeric values.")

# Reset function
def reset_fields():
    for entry in entries:
        entry.delete(0, tk.END)

# Buttons
button_frame = tk.Frame(root, bg="#f8f9fa")
button_frame.pack(pady=20)

predict_btn = tk.Button(button_frame, text="Predict", font=("Arial", 12, "bold"), command=predict_diabetes,
                        bg="#2980b9", fg="white", width=15)
predict_btn.pack(side=tk.LEFT, padx=10)

reset_btn = tk.Button(button_frame, text="Reset", font=("Arial", 12, "bold"), command=reset_fields,
                      bg="#7f8c8d", fg="white", width=15)
reset_btn.pack(side=tk.RIGHT, padx=10)

# Start GUI loop
root.mainloop()
