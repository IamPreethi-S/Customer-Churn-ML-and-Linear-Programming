import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import joblib
from sklearn.model_selection import train_test_split

def train_model():
    # Load preprocessed data
    X = pd.read_csv("output/processed_data.csv")
    y = pd.read_csv("output/target.csv").values.ravel()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # y_train.value_counts()
    # print(y_train.value_counts())
    # Train XGBoost
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=100,
        max_depth=3
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    print(f"Test AUC: {auc:.4f}")
    
    # Save model and predictions
    joblib.dump(model, "output/xgboost_model.joblib")
    pd.Series(preds, name="churn_prob").to_csv("output/predictions.csv", index=False)

if __name__ == "__main__":
    train_model()