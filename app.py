import pickle
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Healthcare Chatbot for Disease Prediction", page_icon="🩺", layout="wide")

# --- Load model & artifacts ---
with open("disease_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("features.pkl", "rb") as f:
    feature_names = pickle.load(f)
with open("accuracy.pkl", "rb") as f:
    accuracy = pickle.load(f)

# --- Friendly name map ---
NAME_MAP = {
    "Dimorphic hemmorhoids(piles)": "Piles",
    "Urinary tract infection": "UTI (Bladder Infection)",
    "Alcoholic hepatitis": "Liver Inflammation (Alcohol-related)",
    "Arthritis": "Joint Pain / Arthritis",
}

# --- Basic info for common diseases ---
INFO_MAP = {
    "Piles": {
        "precautions": ["Eat high-fiber foods", "Drink plenty of water", "Avoid prolonged sitting"],
        "medications": ["Topical creams", "Pain relievers (doctor prescribed)"],
    },
    "UTI (Bladder Infection)": {
        "precautions": ["Drink more water", "Maintain hygiene", "Don’t hold urine"],
        "medications": ["Antibiotics (doctor prescribed)", "Pain relievers if needed"],
    },
    "Liver Inflammation (Alcohol-related)": {
        "precautions": ["Avoid alcohol", "Eat a healthy diet", "Regular checkups"],
        "medications": ["Doctor-prescribed liver medicines"],
    },
    "Joint Pain / Arthritis": {
        "precautions": ["Gentle exercise", "Healthy weight", "Hot/cold compress"],
        "medications": ["Pain relievers", "Anti-inflammatory drugs (doctor prescribed)"],
    },
}

# --- Sidebar (User Input) ---
st.sidebar.header("🧾 Input Symptoms")
st.sidebar.write("Select one or more symptoms you are experiencing:")

user_symptoms = st.sidebar.multiselect("Symptoms:", feature_names)
user_bits = [1 if feat in user_symptoms else 0 for feat in feature_names]

# --- Main Title ---
st.title("🩺 Healthcare Chatbot for Disease Prediction")
st.caption("This tool suggests possible diseases from your selected symptoms. **Note:** This is *not* medical advice.")

# --- Prediction Logic ---
if st.button("🔮 Predict"):
    try:
        input_df = pd.DataFrame([user_bits], columns=feature_names)

        # Reorder columns for model compatibility
        expected = getattr(model, "feature_names_in_", feature_names)
        input_df = input_df.reindex(columns=expected, fill_value=0)

        # Predict
        probs = model.predict_proba(input_df)[0]
        classes = list(model.classes_)
        top_idx = np.argsort(probs)[-3:][::-1]
        top_labels = [classes[i] for i in top_idx]
        top_probs = [float(probs[i] * 100) for i in top_idx]

        # Main prediction
        best_label = top_labels[0]
        display_label = NAME_MAP.get(best_label, best_label)

        # --- Results Section ---
        st.success("✅ Prediction Complete")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Most Likely Disease", display_label)
            st.metric("Confidence", f"{top_probs[0]:.2f}%")

        with col2:
            st.metric("Model Accuracy", f"{accuracy*100:.2f}%")
            st.write("🔎 Evaluated on training/validation dataset.")

        # Top 3 table
        st.subheader("📈 Top 3 Predictions")
        st.table(pd.DataFrame({
            "Disease": [NAME_MAP.get(lbl, lbl) for lbl in top_labels],
            "Probability (%)": [round(p, 2) for p in top_probs]
        }))

        # Info
        info = INFO_MAP.get(display_label, {"precautions": ["Consult a doctor"], "medications": ["Consult a doctor"]})
        st.subheader("📝 Recommended Precautions")
        for item in info["precautions"]:
            st.write(f"- {item}")
        st.subheader("💊 Suggested Medications")
        for item in info["medications"]:
            st.write(f"- {item}")

        # --- Disclaimer ---
        st.warning("⚠️ This tool is for educational purposes only. For diagnosis/treatment, please consult a licensed medical professional.")
        st.caption("📌 Privacy: This app does not store or share your input data.")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")