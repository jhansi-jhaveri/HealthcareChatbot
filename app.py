import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load model
with open("disease_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load feature list
with open("features.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Load accuracy
with open("accuracy.pkl", "rb") as f:
    accuracy = pickle.load(f)

st.title("🩺 Disease Prediction Chatbot")

# Show model accuracy
st.info(f"✅ Model Accuracy: {accuracy*100:.2f}% on testing data")

# Dictionary for known disease info
disease_info = {
    "Dengue": {
        "precautions": [
            "Drink plenty of fluids",
            "Avoid mosquito bites",
            "Rest as much as possible"
        ],
        "medications": [
            "Paracetamol for fever (avoid aspirin/ibuprofen)",
            "ORS (Oral Rehydration Solution)"
        ]
    },
    "Malaria": {
        "precautions": [
            "Use mosquito nets and repellents",
            "Wear protective clothing",
            "Avoid stagnant water near home"
        ],
        "medications": [
            "Antimalarial drugs (as prescribed)",
            "Paracetamol for fever"
        ]
    },
    "Typhoid": {
        "precautions": [
            "Drink boiled/filtered water",
            "Maintain good hand hygiene",
            "Eat well-cooked food only"
        ],
        "medications": [
            "Antibiotics (doctor prescribed)",
            "ORS and fluids to prevent dehydration"
        ]
    },
    "Fungal infection": {
        "precautions": [
            "Keep the affected area dry and clean",
            "Avoid sharing personal items",
            "Wear loose cotton clothes"
        ],
        "medications": [
            "Topical antifungal creams",
            "Antifungal tablets (if severe)"
        ]
    },
}

# Automatically add generic info for all other diseases
for disease in model.classes_:
    if disease not in disease_info:
        disease_info[disease] = {
            "precautions": ["Consult a doctor for precautions"],
            "medications": ["Consult a doctor for medications"]
        }

# Multi-select symptoms
selected_symptoms = st.multiselect("Select your symptoms:", feature_names)

# Create input vector
input_vector = [1 if feature in selected_symptoms else 0 for feature in feature_names]
input_data = np.array(input_vector).reshape(1, -1)

if st.button("Predict Disease"):
    try:
        # Get prediction probabilities
        probabilities = model.predict_proba(input_data)[0]

        # Get top prediction
        best_index = np.argmax(probabilities)
        best_disease = model.classes_[best_index]
        best_prob = probabilities[best_index] * 100

        # Show main result
        st.success(f"🩺 You are most likely suffering from **{best_disease}** ({best_prob:.2f}% confidence)")

        # Show precautions & medications (always available now)
        st.subheader("📝 Recommended Precautions")
        for item in disease_info[best_disease]["precautions"]:
            st.write(f"- {item}")

        st.subheader("💊 Suggested Medications")
        for item in disease_info[best_disease]["medications"]:
            st.write(f"- {item}")

        # Optional pie chart for top 3 predictions
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_diseases = [model.classes_[i] for i in top_indices]
        top_probs = [probabilities[i] for i in top_indices]

        fig, ax = plt.subplots()
        ax.pie(top_probs, labels=top_diseases, autopct="%1.2f%%", startangle=90)
        ax.set_title("Top Predictions")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error during prediction: {e}")

