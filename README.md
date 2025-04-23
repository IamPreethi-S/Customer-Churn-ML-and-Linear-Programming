# Customer-Churn-ML-and-Linear Programming

- This is an end-to-end project that combines machine learning and optimization techniques to identify and retain high-risk churn customers while operating under budget constraints. Leveraging customer behavior data, it predicts churn risk and applies mathematical optimization to prioritize retention efforts that maximize business value

- This project was created as part of a hands-on learning series to explore how machine learning and optimization can be combined to solve real-world customer retention problems under business constraints

---

## Overview
- This project combines Machine Learning (ML) and Linear Programming (LP) to predict and reduce customer churn using the Telco Customer Churn dataset.
- Used Machine Learning to predict churn probabilities
- Used Linear Programming to optimize which customers to retain under a fixed budget
- Goal: Maximize retained revenue while staying within cost constraints (budget)

---

## ML Model

- Input: Preprocessed customer data
- Output: Probability of churn (between 0 and 1)
- Evaluation Metric: ROC AUC Score (to identify how effectively the model distinguishes "churn" vs "no churn")

---

## LP Optimization

- Decision: Which customers to offer retention efforts to
- Objective: Maximize expected **retained revenue** = `churn_prob × revenue` for targeted customers
- Constraints: Stay within total retention **budget** (defined by business)
- Tool: [`PuLP`](https://coin-or.github.io/pulp/) linear programming solver 

---

### Dataset
Source: Telco Customer Churn - Kaggle - https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data
