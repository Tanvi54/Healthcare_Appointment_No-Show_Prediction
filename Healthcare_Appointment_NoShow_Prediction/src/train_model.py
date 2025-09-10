import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import os
import pickle
from sklearn.metrics import classification_report, accuracy_score

def train_and_predict():
    print("📂 Loading processed data...")
    data = pd.read_csv("data/processed/noshow_clean_for_powerbi.csv")

    # -----------------------
    # ✅ Feature Engineering for Dates
    # -----------------------
    data["ScheduledDay"] = pd.to_datetime(data["ScheduledDay"], errors="coerce")
    data["AppointmentDay"] = pd.to_datetime(data["AppointmentDay"], errors="coerce")

    # Extract waiting days
    data["Waiting_Days"] = (data["AppointmentDay"] - data["ScheduledDay"]).dt.days

    # Extract weekday info
    data["ScheduledDay_Weekday"] = data["ScheduledDay"].dt.dayofweek
    data["AppointmentDay_Weekday"] = data["AppointmentDay"].dt.dayofweek

    # Drop original datetime columns
    data = data.drop(columns=["ScheduledDay", "AppointmentDay"])

    # Separate features and target
    X = data.drop("No-show", axis=1)
    y = data["No-show"]

    # Encode categorical variables for ML but keep original labels for later
    label_encoders = {}
    X_encoded = X.copy()

    for col in X.columns:
        if X[col].dtype == "object":  # categorical
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col])
            label_encoders[col] = le

    # Save encoders
    joblib.dump(label_encoders, "data/model/label_encoders.pkl")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    # Handle imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Train Decision Tree
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train_res, y_train_res)

    # Save model and features
    joblib.dump(clf, "data/model/decision_tree_model.pkl")
    joblib.dump(X_encoded.columns.tolist(), "data/model/model_features.pkl")

    # Predictions
    y_pred = clf.predict(X_test)

    # ✅ Evaluate model
    print("\n📊 Model Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))



    # -----------------------
    # ✅ Create Power BI file with original labels
    # -----------------------
    X_test_original = X.loc[X_test.index].copy()  # keep original (text) values
    X_test_original["Predicted_NoShow"] = y_pred
    X_test_original["Actual_NoShow"] = y_test.values

    # Replace 0/1 with labels
    X_test_original["Predicted_NoShow"] = X_test_original["Predicted_NoShow"].map({0: "Show", 1: "NoShow"})
    X_test_original["Actual_NoShow"] = X_test_original["Actual_NoShow"].map({0: "Show", 1: "NoShow"})

    X_test_original.to_csv("data/processed/test_with_predictions.csv", index=False)
    print("✅ Training complete! Predictions saved in data/processed/test_with_predictions.csv")

if __name__ == "__main__":
    train_and_predict()
