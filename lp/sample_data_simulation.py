import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary, PULP_CBC_CMD

# Simulated synthetic customer churn data and use LP to solve the optimization problem

np.random.seed(42)
n_customers = 1000

df = pd.DataFrame({
    'tenure': np.random.randint(1, 60, n_customers),
    'monthly_bill': np.random.uniform(50, 200, n_customers),
    'num_service_calls': np.random.randint(0, 5, n_customers),
    'contract_type': np.random.choice([0, 1], size=n_customers),
    'churn': np.random.choice([0, 1], size=n_customers, p=[0.7, 0.3])
})

df['churn'] = df['churn'].astype(int) 
df['revenue'] = df['tenure'] * df['monthly_bill']
df['offer_cost'] = np.random.uniform(10, 50, n_customers)

# Train a churn prediction model
X = df[['tenure', 'monthly_bill', 'num_service_calls', 'contract_type']]
y = df['churn']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Predict churn probabilities
df['churn_probability'] = model.predict_proba(X_scaled)[:, 1]

# LP Optimization for retention targeting, $10,000 total budget
budget = 10000  

lp_model = LpProblem(name="maximize-retained-revenue", sense=LpMaximize)
x = {i: LpVariable(name=f"x_{i}", cat=LpBinary) for i in df.index}

# Objective - Maximize retained revenue
lp_model += lpSum(x[i] * (1 - df.loc[i, 'churn_probability']) * df.loc[i, 'revenue'] for i in df.index)

# Constraint - budget
lp_model += lpSum(x[i] * df.loc[i, 'offer_cost'] for i in df.index) <= budget

# Solve LP
lp_model.solve(PULP_CBC_CMD(msg=0))


df['selected_for_offer'] = [x[i].value() for i in df.index]
selected_customers = df[df['selected_for_offer'] == 1]


print("Total Retained Revenue: ${:.2f}".format(sum((1 - selected_customers['churn_probability']) * selected_customers['revenue'])))

print("Total Offer Cost Used: ${:.2f}".format(selected_customers['offer_cost'].sum()))

# list of selected customers for offer
print("\nTop Selected Customers:\n", selected_customers[['tenure', 'monthly_bill', 'revenue', 'churn_probability', 'offer_cost']].head())
