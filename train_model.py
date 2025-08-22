import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "Data", "archive", "Training.csv")
MODEL_PATH = os.path.join(BASE_DIR, "disease_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "features.pkl")
ACCURACY_PATH = os.path.join(BASE_DIR, "accuracy.pkl")

# Load dataset
data = pd.read_csv(DATA_PATH)
data = data.loc[:, ~data.columns.str.contains("^Unnamed")]

X = data.iloc[:, :-1]  # features
y = data.iloc[:, -1]   # target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"✅ Model trained! Test accuracy: {accuracy*100:.2f}%")

# Save model and metadata
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
with open(FEATURES_PATH, "wb") as f:
    pickle.dump(list(X.columns), f)
with open(ACCURACY_PATH, "wb") as f:
    pickle.dump(accuracy, f)
