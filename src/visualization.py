import matplotlib.pyplot as plt
import pandas as pd

def plot_results():
    # Load data
    df = pd.read_csv("output/risk_scores.csv")
    
    # Calculate metrics
    original_loss = df["churn_risk"].sum()
    mitigated_loss = df[df["target_customer"] == 0]["churn_risk"].sum()
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.bar(["Original Churn Risk", "Post-Intervention Risk"], 
           [original_loss, mitigated_loss], color=["red", "green"])
    plt.ylabel("Total Churn Risk Score")
    plt.title("Churn Risk Reduction through Optimization")
    plt.savefig("output/risk_reduction.png")
    plt.close()

if __name__ == "__main__":
    plot_results()