from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpStatus, PULP_CBC_CMD
import pandas as pd
import numpy as np
import os

def optimize_retention():
    # Load customer IDs and predicted churn probabilities
    ids = pd.read_csv("output/customer_ids.csv")
    probs = pd.read_csv("output/predictions.csv")["churn_prob"]

    # Business parameters
    avg_revenue = 500
    cost_per_intervention = 100
    budget = 20000

    # Combine into a working dataframe
    df = pd.DataFrame({
        "customerID": ids["customerID"],
        "churn_prob": probs
    })

    # Calculate retained revenue potential = P(churn) * revenue
    df["retained_value"] = df["churn_prob"] * avg_revenue

    # Define LP maximization problem
    prob = LpProblem("ChurnRetentionOptimization", LpMaximize)

    customers = list(df.index)
    x = LpVariable.dicts("Target", customers, cat="Binary")

    # Objective: Maximize retained revenue
    prob += lpSum(x[i] * df.loc[i, "retained_value"] for i in customers)

    # Constraint: Total intervention cost must stay within budget
    prob += lpSum(x[i] * cost_per_intervention for i in customers) <= budget

    # Solve the LP problem
    solver = PULP_CBC_CMD(timeLimit=300, gapRel=0.02, gapAbs=100, threads=8, msg=True)
    prob.solve(solver)

    # Check feasibility
    if LpStatus[prob.status] not in ["Optimal", "Feasible"]:
        print("No optimal solution found.")
        return

    # Mark customers for retention
    df["target_customer"] = [int(x[i].value()) for i in customers]

    # Save results
    os.makedirs("output", exist_ok=True)
    df.to_csv("output/risk_scores.csv", index=False)

    total_cost = df["target_customer"].sum() * cost_per_intervention
    total_retained_value = df[df["target_customer"] == 1]["retained_value"].sum()

    print(f"\n{'STATUS':~^40}")
    print(f"{LpStatus[prob.status]} solution found")
    print(f"Total cost: ${total_cost:,.2f} / ${budget:,}")
    print(f"Total expected revenue retained: ${total_retained_value:,.2f}")

if __name__ == "__main__":
    optimize_retention()
