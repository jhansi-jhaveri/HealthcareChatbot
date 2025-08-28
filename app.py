import pickle
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Disease Prediction Chatbot", page_icon="🩺", layout="wide")

# =========================
# Load saved model & data
# =========================
with open("disease_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("features.pkl", "rb") as f:
    feature_names = pickle.load(f)

with open("accuracy.pkl", "rb") as f:
    accuracy = pickle.load(f)  # e.g., 0.91 for 91%

# =========================
# Disease info (editable)
# =========================
BASE_INFO = {
    "Dengue": {
        "precautions": [
            "Drink plenty of fluids",
            "Avoid mosquito bites",
            "Rest as much as possible",
        ],
        "medications": [
            "Paracetamol for fever (avoid aspirin/ibuprofen)",
            "ORS (Oral Rehydration Solution)",
        ],
    },
    "Malaria": {
        "precautions": [
            "Use mosquito nets and repellents",
            "Wear protective clothing",
            "Avoid stagnant water near home",
        ],
        "medications": [
            "Antimalarial drugs (as prescribed)",
            "Paracetamol for fever",
        ],
    },
    "Typhoid": {
        "precautions": [
            "Drink boiled/filtered water",
            "Maintain good hand hygiene",
            "Eat well-cooked food only",
        ],
        "medications": [
            "Antibiotics (doctor prescribed)",
            "ORS and fluids to prevent dehydration",
        ],
    },
    "Fungal infection": {
        "precautions": [
            "Keep the affected area dry and clean",
            "Avoid sharing personal items",
            "Wear loose cotton clothes",
        ],
        "medications": [
            "Topical antifungal creams",
            "Antifungal tablets (if severe)",
        ],
    },
}

GENERIC_INFO = {
    "precautions": ["Consult a doctor for proper precautions."],
    "medications": ["Consult a doctor for appropriate medications."],
}

def norm(s: str) -> str:
    return s.strip().lower().replace("_", " ").replace("-", " ")

# Build a case/underscore-insensitive lookup
INFO_MAP = {norm(k): v for k, v in BASE_INFO.items()}

# Add generic entries for any other classes in the model
for cls in getattr(model, "classes_", []):
    if norm(cls) not in INFO_MAP:
        INFO_MAP[norm(cls)] = GENERIC_INFO

# =========================
# UI
# =========================
st.title("🩺 Disease Prediction Chatbot")
st.write("This tool suggests possible diseases from your selected symptoms. This is **not** medical advice.")

st.sidebar.header("User Input Features")
st.sidebar.write("Select symptoms you are experiencing:")

user_bits = []
for feat in feature_names:
    user_bits.append(1 if st.sidebar.checkbox(feat, False) else 0)

# =========================
# Predict
# =========================
if st.button("Predict"):
    try:
        # Build DataFrame with names
        input_df = pd.DataFrame([user_bits], columns=feature_names)

        # Align to model's training columns (prevents sklearn warning)
        expected = getattr(model, "feature_names_in_", feature_names)
        input_df = input_df.reindex(columns=expected, fill_value=0)

        # Predict + probabilities
        probs = model.predict_proba(input_df)[0]
        classes = list(model.classes_)
        top_idx = np.argsort(probs)[-3:][::-1]
        top_labels = [classes[i] for i in top_idx]
        top_probs = [float(probs[i] * 100) for i in top_idx]

        # Main result
        best_label = top_labels[0]
        best_prob = top_probs[0]
        st.success("✅ Prediction")
        st.write(f"Based on your symptoms, the most likely disease is: **{best_label}** ({best_prob:.2f}% confidence)")
        st.write(f"📊 Model Accuracy: **{accuracy*100:.2f}%**")

        # Show top-3 table
        st.subheader("📈 Top 3 predictions")
        st.table(pd.DataFrame({"Disease": top_labels, "Probability (%)": [round(p, 2) for p in top_probs]}))

        # Precautions & Medications (robust lookup)
        key = norm(best_label)
        info = INFO_MAP.get(key, GENERIC_INFO)

        st.subheader("📝 Recommended Precautions")
        for item in info["precautions"]:
            st.write(f"- {item}")

        st.subheader("💊 Suggested Medications")
        for item in info["medications"]:
            st.write(f"- {item}")

        st.caption("⚠️ For diagnosis/treatment, please consult a licensed medical professional.")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
