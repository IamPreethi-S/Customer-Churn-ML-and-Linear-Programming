import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, PULP_CBC_CMD

def simulate_budget_curve(
    df_path="output/risk_scores.csv",
    avg_revenue=500,
    cost_per_customer=100,
    budget_range=list(range(0, 20000, 5000)),
    output_path="output/revenue_retained_vs_budget.png"
):
    # Load churn data and compute retained value
    df = pd.read_csv(df_path)
    df["retained_value"] = df["churn_prob"] * avg_revenue

    retained_values = []

    for budget in budget_range:
        prob = LpProblem("ChurnRetentionMaxRevenue", LpMaximize)
        customers = list(df.index)
        x = LpVariable.dicts("Target", customers, cat="Binary")

        prob += lpSum(x[i] * df.loc[i, "retained_value"] for i in customers)
        prob += lpSum(x[i] * cost_per_customer for i in customers) <= budget

        prob.solve(PULP_CBC_CMD(msg=False))

        retained = sum(
            x[i].value() * df.loc[i, "retained_value"]
            for i in customers
            if x[i].value() is not None
        )
        retained_values.append(retained)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(budget_range, retained_values, marker='o', linewidth=2, color='royalblue')
    plt.title("📈 Revenue Retained vs Retention Budget")
    plt.xlabel("Retention Budget ($)")
    plt.ylabel("Expected Revenue Retained ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print("Budget curve simulation completed")
    return pd.DataFrame({
        "Budget": budget_range,
        "Revenue_Retained": retained_values
    })

# Optional direct run
if __name__ == "__main__":
    simulate_budget_curve()
    print("Budget curve simulation complete and saved to output directory.")