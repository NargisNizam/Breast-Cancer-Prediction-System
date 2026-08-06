import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("breast_cancer_dataset.csv")

# Features and Target
X = df.drop("target", axis=1)
y = df["target"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print("=" * 40)
print(f"Model Accuracy : {accuracy*100:.2f}%")
print("=" * 40)

# Save Model
joblib.dump(model, "model.pkl")

print("model.pkl saved successfully.")