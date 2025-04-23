import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

def preprocess_data():
    df = pd.read_csv("data/data.csv")
    df = df.dropna(subset=["Churn"])
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    cat_features = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", 
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", 
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", 
        "Contract", "PaperlessBilling", "PaymentMethod"
    ]
    num_features = ["tenure", "MonthlyCharges", "TotalCharges"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first"), cat_features)
        ]
    )

    X = preprocessor.fit_transform(df.drop(columns=["Churn", "customerID"]))
    y = df["Churn"]

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, df["customerID"], test_size=0.2, random_state=42
    )

    os.makedirs("output", exist_ok=True)
    joblib.dump(preprocessor, "output/preprocessor.joblib")
    pd.DataFrame(X_train).to_csv("output/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv("output/X_test.csv", index=False)
    y_train.to_csv("output/y_train.csv", index=False)
    y_test.to_csv("output/y_test.csv", index=False)
    id_test.to_csv("output/customer_ids.csv", index=False)

if __name__ == "__main__":
    preprocess_data()
    print("Preprocessing completed. Data saved to output directory.")
