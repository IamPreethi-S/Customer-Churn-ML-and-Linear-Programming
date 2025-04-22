import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

def preprocess_data():
    # Load data
    train_df = pd.read_csv("data/train.csv")
    
    train_df = train_df.dropna(subset=["Churn"])
    train_df["Churn"] = train_df["Churn"].astype(int)
    
    # Define features
    cat_features = ["Gender", "Subscription Type", "Contract Length"]
    num_features = ["Age", "Tenure", "Usage Frequency", 
                   "Support Calls", "Payment Delay", "Total Spend", "Last Interaction"]
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first"), cat_features)
        ])
    
    # Fit and transform
    X = preprocessor.fit_transform(train_df.drop("Churn", axis=1))
    y = train_df["Churn"]
    
    # Save preprocessor and split data
    os.makedirs("output", exist_ok=True)
    joblib.dump(preprocessor, "output/preprocessor.joblib")
    pd.DataFrame(X).to_csv("output/processed_data.csv", index=False)
    y.to_csv("output/target.csv", index=False)

if __name__ == "__main__":
    preprocess_data()