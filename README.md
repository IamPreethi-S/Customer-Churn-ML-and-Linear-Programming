# Customer-Churn-ML-and-Linear Programming

- This is an end-to-end project that combines machine learning and optimization techniques to identify and retain high-risk churn customers while operating under budget constraints. Leveraging customer behavior data, it predicts churn risk and applies mathematical optimization to prioritize retention efforts that maximize business value.

- This project was created as part of a hands-on learning series to explore how Linear programming (optimization) is used to solve real-world business problems.

- This project combines Machine Learning (ML) and Linear Programming (LP) to predict and reduce customer churn using the Telco Customer Churn dataset.

---

## Overview

- Use ML to predict churn probabilities
- Use LP to optimize which customers to retain under a fixed budget
- Goal: Maximize retained revenue while staying within cost constraints

---

## ML Model

- Input: Preprocessed customer features (numerical + one-hot encoded categoricals)
- Output: Probability of churn (between 0 and 1)
- Evaluation Metric: ROC AUC Score

---

## LP Optimization

- Decision: Which customers to offer retention efforts to
- Objective: Maximize expected **retained revenue** = `churn_prob × revenue` for targeted customers
- Constraints: Stay within total retention **budget** (defined by business)
- Tool: `PuLP` linear programming solver

---

### Dataset
Source: Telco Customer Churn - Kaggle - https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data
