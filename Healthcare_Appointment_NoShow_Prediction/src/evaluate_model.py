import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

model = joblib.load("data/model/decision_tree_model.pkl")
features = joblib.load("data/model/model_features.pkl")

def evaluate(test_file):
    df = pd.read_csv(test_file)
    X = df.drop("No_show", axis=1)
    y = df["No_show"]

    X = X.reindex(columns=features, fill_value=0)

    y_pred = model.predict(X)
    print(classification_report(y, y_pred))
    print(confusion_matrix(y, y_pred))

if __name__ == "__main__":
    evaluate("data/processed/appointments_cleaned.csv")
