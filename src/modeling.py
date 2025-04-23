import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib

def train_model():
    X_train = pd.read_csv("output/X_train.csv")
    y_train = pd.read_csv("output/y_train.csv").values.ravel()
    X_test = pd.read_csv("output/X_test.csv")
    y_test = pd.read_csv("output/y_test.csv").values.ravel()

    #experimented with various algo, and went ahead with RF
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    print(f"Test AUC: {auc:.4f}")

    joblib.dump(model, "output/rf_model.joblib")
    pd.Series(probs, name="churn_prob").to_csv("output/predictions.csv", index=False)

if __name__ == "__main__":
    train_model()
    print("Model training completed and saved to output/rf_model.joblib")


