import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load model, features, accuracy
# -----------------------------
with open("disease_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("features.pkl", "rb") as f:
    feature_names = pickle.load(f)

with open("accuracy.pkl", "rb") as f:
    accuracy = pickle.load(f)

st.title("🩺 Disease Prediction Chatbot")

# Show model accuracy
st.info(f"✅ Model Accuracy: {accuracy*100:.2f}% on testing data")

# -----------------------------
# Prepare medical guide for all diseases
# -----------------------------
# Existing info for some diseases
predefined_info = {
    "Dengue": {
        "precautions": ["Drink plenty of fluids", "Avoid mosquito bites", "Rest as much as possible"],
        "medications": ["Paracetamol for fever (avoid aspirin/ibuprofen)", "ORS"]
    },
    "Malaria": {
        "precautions": ["Use mosquito nets", "Avoid stagnant water", "Wear protective clothing"],
        "medications": ["Antimalarial drugs (as prescribed)", "Paracetamol"]
    },
    "Typhoid": {
        "precautions": ["Drink boiled water", "Wash hands", "Eat well-cooked food"],
        "medications": ["Antibiotics (as prescribed)", "ORS"]
    },
    "Fungal infection": {
        "precautions": ["Keep affected area dry", "Avoid sharing personal items", "Wear loose clothes"],
        "medications": ["Topical antifungal creams", "Antifungal tablets if severe"]
    }
}

# Automatically add placeholder info for diseases not listed
disease_info = {}
for disease in model.classes_:
    if disease in predefined_info:
        disease_info[disease] = predefined_info[disease]
    else:
        disease_info[disease] = {
            "precautions": ["Consult a doctor for proper guidance."],
            "medications": ["Consult a doctor before taking any medication."]
        }

# -----------------------------
# User input
# -----------------------------
selected_symptoms = st.multiselect("Select your symptoms:", feature_names)

# Create input vector
input_vector = [1 if feature in selected_symptoms else 0 for feature in feature_names]
input_data = np.array(input_vector).reshape(1, -1)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Disease"):
    try:
        probabilities = model.predict_proba(input_data)[0]
        best_index = np.argmax(probabilities)
        best_disease = model.classes_[best_index]
        best_prob = probabilities[best_index] * 100

        st.success(f"🩺 You are most likely suffering from **{best_disease}** ({best_prob:.2f}% confidence)")

        # Show precautions & medications
        st.subheader("📝 Recommended Precautions")
        for item in disease_info[best_disease]["precautions"]:
            st.write(f"- {item}")

        st.subheader("💊 Suggested Medications")
        for item in disease_info[best_disease]["medications"]:
            st.write(f"- {item}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
