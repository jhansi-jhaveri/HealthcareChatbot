import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------------
# Load your dataset
# -------------------------------
# Example: Replace with your actual dataset
# Suppose your dataset has columns: symptoms + 'Disease'
data = pd.read_csv("disease_dataset.csv")

# Separate features and target
feature_names = [col for col in data.columns if col != "Disease"]
X = data[feature_names]
y = data["Disease"]

# -------------------------------
# Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train Random Forest
# -------------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# Evaluate Model
# -------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"✅ Model trained with accuracy: {accuracy:.2f}%")

# -------------------------------
# Save Model + Metadata
# -------------------------------
with open("disease_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("features.pkl", "wb") as f:
    pickle.dump(feature_names, f)

with open("accuracy.pkl", "wb") as f:
    pickle.dump(accuracy, f)

print("✅ Model, features, and accuracy saved!")
