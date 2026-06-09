import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import mlflow
import mlflow.sklearn

# =========================
# Load Dataset
# =========================
df = pd.read_csv("telco_preprocessed.csv")

X = df.drop("Churn", axis=1)
y = df["Churn"]

# =========================
# Split Data
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# MLflow
# =========================]

mlflow.sklearn.autolog()

# =========================
# Training
# =========================
with mlflow.start_run(run_name="RandomForest-Autolog"):

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Training selesai")
    print(f"Accuracy: {acc:.4f}")
