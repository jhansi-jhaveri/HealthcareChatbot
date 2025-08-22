import streamlit as st
import pickle
import numpy as np

# Cache model loading so it loads once per session
@st.cache_resource
def load_model():
    with open("disease_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_features():
    with open("features.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_accuracy():
    with open("accuracy.pkl", "rb") as f:
        return pickle.load(f)

# Load resources
model = load_model()
feature_names = load_features()
accuracy = load_accuracy()

st.title("🩺 Healthcare Disease Prediction Chatbot")
st.info(f"✅ Model Accuracy: {accuracy*100:.2f}% on testing data")

# Dictionary for precautions & medications
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
    }
    # Add more diseases as needed
}

# Multi-select symptoms
selected_symptoms = st.multiselect("Select your symptoms:", feature_names)

if st.button("Predict Disease"):
    try:
        # Create input vector
        input_vector = [1 if feature in selected_symptoms else 0 for feature in feature_names]
        input_data = np.array(input_vector).reshape(1, -1)

        # Prediction
        probabilities = model.predict_proba(input_data)[0]
        best_index = np.argmax(probabilities)
        best_disease = model.classes_[best_index]
        best_prob = probabilities[best_index] * 100

        st.success(f"🩺 You are most likely suffering from **{best_disease}** ({best_prob:.2f}% confidence)")

        # Show precautions & medications
        if best_disease in disease_info:
            st.subheader("📝 Recommended Precautions")
            for item in disease_info[best_disease]["precautions"]:
                st.write(f"- {item}")

            st.subheader("💊 Suggested Medications")
            for item in disease_info[best_disease]["medications"]:
                st.write(f"- {item}")
        else:
            st.warning("⚠️ No medical guide available for this disease yet.")

        # Optional: Pie chart for top 3 predictions
        import matplotlib.pyplot as plt  # Lazy import for speed
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_diseases = [model.classes_[i] for i in top_indices]
        top_probs = [probabilities[i] for i in top_indices]

        fig, ax = plt.subplots()
        ax.pie(top_probs, labels=top_diseases, autopct="%1.2f%%", startangle=90)
        ax.set_title("Top Predictions")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
