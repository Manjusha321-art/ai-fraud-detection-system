import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# 1. Load dataset
df = pd.read_csv("data/creditcard.csv")

print("Dataset loaded successfully")
print(f"Dataset shape: {df.shape}")


# 2. Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]


# 3. Scale the Amount feature
scaler = StandardScaler()

X["Amount"] = scaler.fit_transform(
    X[["Amount"]]
)


# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# 5. Create the model
model = LogisticRegression(
     max_iter=3000,
     solver="lbfgs"
)


# 6. Train the model
print("\nTraining model...")

model.fit(
    X_train,
    y_train
)
print("Model training completed")

# Save the trained model

joblib.dump(

    model,

    "fraud_model.pkl"

)

print(
    "\nModel saved successfully"
)



# 7. Make predictions
y_pred = model.predict(
    X_test
)


# 8. Get probability predictions
y_probability = model.predict_proba(
    X_test
)[:, 1]


# 9. Evaluate model
# 9. Threshold tuning

print("\n--- Threshold Tuning ---")

thresholds = [
    0.5,
    0.3,
    0.2,
    0.1
]

for threshold in thresholds:

    y_threshold_pred = (
        y_probability >= threshold
    ).astype(int)

    print(
        f"\n===== Threshold: {threshold} ====="
    )

    print(
        classification_report(
            y_test,
            y_threshold_pred,
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_threshold_pred
        )
    )


print("\n--- ROC-AUC Score ---")

print(
    roc_auc_score(
        y_test,
        y_probability
    )
)


print("\n--- Confusion Matrix ---")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


print("\n--- ROC-AUC Score ---")

print(
    roc_auc_score(
        y_test,
        y_probability
    )
)
