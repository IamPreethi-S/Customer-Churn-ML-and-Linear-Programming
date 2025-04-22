from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatus, PULP_CBC_CMD
import pandas as pd
import numpy as np

def optimize_retention():
    # Load predictions and data
    df = pd.read_csv("data/test.csv")
    probs = pd.read_csv("output/predictions.csv")["churn_prob"]
    
    # Clean data
    df = df.dropna(subset=["CustomerID"])
    probs = probs.fillna(probs.mean())  # Handle NaN in predictions
    
    # Validate business parameters
    avg_revenue = 500       # Verify with finance team
    cost_per_intervention = 150  # Verify with marketing team
    budget = 10000          # Verify with budget constraints
    
    # Create problem
    prob = LpProblem("ChurnLossReduction", LpMinimize)
    
    # Decision variables
    customers = list(df.index)
    x = LpVariable.dicts("Target", customers, cat="Binary")
    
    # Objective: Minimize potential churn loss
    prob += lpSum((1 - x[i]) * probs[i] * avg_revenue for i in customers)
    
    # Constraints
    prob += lpSum(x[i] * cost_per_intervention for i in customers) <= budget
    
    # Configure solver for real-world performance
    solver = PULP_CBC_CMD(
    timeLimit=300,      # 5-minute timeout
    gapRel=0.02,        # 2% optimality gap
    gapAbs=100,         # Absolute gap tolerance
    threads=8,          # Use all CPU cores
    msg=True
)
    
    # Solve and validate
    prob.solve(solver)
    
    if LpStatus[prob.status] not in ["Optimal", "Feasible"]:
        print("⚠️ Warning: No provably optimal solution found!")
        print("Recommendations:")
        print("- Increase budget/time limit")
        print("- Check for NaN in predictions")
        print("- Verify cost/revenue parameters")
        return  # Exit before saving invalid results

    # Save validated results
    df["target_customer"] = [1 if x[i].value() == 1 else 0 for i in customers]
    df["churn_risk"] = np.round(probs, 4)
    
    # Sanity check results
    total_cost = df["target_customer"].sum() * cost_per_intervention
    assert total_cost <= budget * 1.01, "Budget constraint violated"
    
    df[["CustomerID", "churn_risk", "target_customer"]].to_csv(
        "output/risk_scores.csv", index=False
    )
    
    # Business-friendly reporting
    print(f"\n{' STATUS ':~^40}")
    print(f"🔍 {LpStatus[prob.status]} solution found")
    print(f"💵 Total budget used: ${total_cost:,} / ${budget:,}")