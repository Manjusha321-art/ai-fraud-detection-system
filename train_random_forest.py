import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


# 1. Load dataset

df = pd.read_csv(
    "data/creditcard.csv"
)

print("Dataset loaded successfully")

print(
    f"Dataset shape: {df.shape}"
)


# 2. Separate features and target

X = df.drop(
    "Class",
    axis=1
)

y = df["Class"]


# 3. Scale the Amount feature

scaler = StandardScaler()

X["Amount"] = scaler.fit_transform(
    X[["Amount"]]
)


# 4. Save the scaler

joblib.dump(
    scaler,
    "amount_scaler.pkl"
)

print(
    "Scaler saved successfully"
)


# 5. Split dataset

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)


print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples: {len(X_test)}"
)


# 6. Create Random Forest model

model = RandomForestClassifier(

    n_estimators=100,

    max_depth=12,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1

)


# 7. Train model

print(
    "\nTraining Random Forest..."
)

model.fit(

    X_train,

    y_train

)


print(
    "Model training completed"
)


# 8. Save trained model

joblib.dump(

    model,

    "fraud_model.pkl"

)

print(
    "Model saved successfully"
)


# 9. Get probability predictions

y_probability = model.predict_proba(

    X_test

)[:, 1]


# 10. Use threshold 0.3

threshold = 0.3

y_pred = (

    y_probability >= threshold

).astype(int)


# 11. Classification report

print(
    "\n--- Classification Report ---"
)

print(

    classification_report(

        y_test,

        y_pred,

        zero_division=0

    )

)


# 12. Confusion matrix

print(
    "\n--- Confusion Matrix ---"
)

print(

    confusion_matrix(

        y_test,

        y_pred

    )

)


# 13. ROC-AUC

print(
    "\n--- ROC-AUC Score ---"
)

print(

    roc_auc_score(

        y_test,

        y_probability

    )

)


# 14. PR-AUC

print(
    "\n--- PR-AUC Score ---"
)

print(

    average_precision_score(

        y_test,

        y_probability

    )

)