# # import matplotlib.pyplot as plt
# # import pandas as pd

# # def plot_results():
# #     # Load data
# #     df = pd.read_csv("output/risk_scores.csv")
    
# #     # Calculate metrics
# #     original_loss = df["churn_risk"].sum()
# #     mitigated_loss = df[df["target_customer"] == 0]["churn_risk"].sum()
    
# #     # Create plot
# #     plt.figure(figsize=(10, 6))
# #     plt.bar(["Original Churn Risk", "Post-Intervention Risk"], 
# #            [original_loss, mitigated_loss], color=["red", "green"])
# #     plt.ylabel("Total Churn Risk Score")
# #     plt.title("Churn Risk Reduction through Optimization")
# #     plt.savefig("output/risk_reduction.png")
# #     plt.close()

# # if __name__ == "__main__":
# #     plot_results()


# #updated code
# import matplotlib.pyplot as plt
# import pandas as pd

# def plot_results():
#     # Load LP optimization output
#     df = pd.read_csv("output/risk_scores.csv")

#     # Use the correct column name: 'churn_prob'
#     total_churn_risk = df["churn_prob"].sum()
#     risk_after_intervention = df[df["target_customer"] == 0]["churn_prob"].sum()
#     reduction = total_churn_risk - risk_after_intervention

#     # Plot
#     plt.figure(figsize=(10, 6))
#     bars = plt.bar(
#         ["Original Churn Risk", "After Retention Efforts"],
#         [total_churn_risk, risk_after_intervention],
#         color=["tomato", "mediumseagreen"]
#     )
#     plt.title("Churn Risk Reduction through LP Optimization")
#     plt.ylabel("Total Expected Churn Risk")
#     plt.ylim(0, total_churn_risk * 1.1)
#     plt.bar_label(bars, fmt="%.2f")
#     plt.grid(axis="y", linestyle="--", alpha=0.7)

#     # Save to file
#     plt.tight_layout()
#     plt.savefig("output/risk_reduction.png")
#     plt.close()

#     # Print insights
#     print("\nVisualization complete:")
#     print(f"Churn risk reduced by: {reduction:.2f} points ({(reduction / total_churn_risk) * 100:.2f}%)")

# if __name__ == "__main__":
#     plot_results()

import matplotlib.pyplot as plt
import pandas as pd

def plot_results(avg_revenue=500):
    # Load optimized targeting results
    df = pd.read_csv("output/risk_scores.csv")

    # Calculate expected retained revenue
    df["retained_value"] = df["churn_prob"] * avg_revenue

    # Total revenue at risk
    total_revenue_at_risk = df["retained_value"].sum()

    # Revenue retained by targeting
    retained_revenue = df[df["target_customer"] == 1]["retained_value"].sum()

    # Revenue still at risk (non-targeted)
    revenue_at_risk = df[df["target_customer"] == 0]["retained_value"].sum()

    # Plot revenue breakdown
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        ["Total Revenue at Risk", "Revenue Retained", "Revenue Still at Risk"],
        [total_revenue_at_risk, retained_revenue, revenue_at_risk],
        color=["gray", "mediumseagreen", "tomato"]
    )
    plt.title("Revenue Retention via Optimized Targeting")
    plt.ylabel("Revenue ($)")
    plt.ylim(0, total_revenue_at_risk * 1.1)
    plt.bar_label(bars, fmt="%.2f")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("output/revenue_retention.png")
    plt.close()

    # Print insights
    print("\n Revenue Retention Visualization Complete:")
    print(f" Total revenue at risk      : ${total_revenue_at_risk:,.2f}")
    print(f" Revenue retained           : ${retained_revenue:,.2f}")
    print(f" Revenue still at risk     : ${revenue_at_risk:,.2f}")

if __name__ == "__main__":
    plot_results()
