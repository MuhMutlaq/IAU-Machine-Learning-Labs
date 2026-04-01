# Lab 6: Linear Regression

This repository contains my completed Jupyter Notebook for `Lab 6`, which focuses heavily on training, evaluating, and optimizing continuous prediction models. (The solution is on the same file of the lab).

In this lab, I built a Multiple Linear Regression model to predict `Yearly Amount Spent` using an `Ecommerce Customers` dataset.

## Data Cleaning & Exploratory Data Analysis

Data Profiling: Utilized `info()` and `describe()` to understand the distribution of customer usage metrics and verify the structural integrity of the data.

**Noise Reduction & Feature Selection:** Isolated the core numerical features (Avg. Session Length, Time on App, Time on Website, Length of Membership) as the primary independent variables, deliberately dropping non-predictive categorical strings like Email and Avatar from the training set.

## Tasks

**Task 1:** Baseline Model Training: Built and trained a standard Linear Regression model using an optimal 60/40 train-test split.

**Task 2:** Feature Engineering & Overfitting Analysis: Experimented with advanced feature engineering by extracting `App_Membership_Interaction` a multiplicative interaction term `(Time on App * Length of Membership)`. Testing this engineered model revealed a drop in performance (RMSE increased from `9.68` to `9.69`).

**Task 3:** Train/Test Split: Using `0.4` testing size for the model (Specified from the lab).

**Task 4:** Final Evaluation & Deployment: Evaluated the final, optimized baseline model using strict regression metrics (`Mean Absolute Error`, `Mean Squared Error`, and `Root Mean Squared Error`) and plotted the True vs. Predicted values using matplotlib to visualize the accuracy. Finally, the model was serialized and exported using `joblib` for future deployment testing.
