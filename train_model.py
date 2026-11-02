
import pandas as pd
import json
import joblib

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Generate sample student result data
X, y = make_classification(
    n_samples=500,
    n_features=8,
    n_informative=5,
    n_redundant=1,
    random_state=42
)

# Save dataset
df = pd.DataFrame(X, columns=[
    "attendance", "study_hours", "assignment",
    "internal_marks", "previous_score", "activity",
    "participation", "practice"
])
df["result"] = y
df.to_csv("student_results.csv", index=False)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Save model
joblib.dump(model, "student_result_model.pkl")

# Save metrics
metrics = {"accuracy": float(accuracy)}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Model trained successfully!")
print("Accuracy:", accuracy)
print("All ML artifacts generated successfully!")
