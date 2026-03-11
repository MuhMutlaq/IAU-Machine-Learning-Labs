# Lab 5: Feature Engineering

This repository contains my completed Jupyter Notebook for **Lab 5**, which focuses heavily on feature engineering and selection for classification tasks. **(The solution is on the same file of the lab)**.

In this lab, I built a baseline Random Forest classifier to predict `Order_Status` (Delivered, Cancelled, or In Transit) using a Talabat-style food delivery dataset **(Requires more work on the dataset)**.

## Data Cleaning & Exploratory Data Analysis

* **Noise Reduction:** Dropped high-cardinality and non-predictive columns such as `Order_ID`, `User_ID`, `Restaurant_ID`, and `Driver_ID` to prevent the model from memorizing the dataset. Exact timestamps were also dropped after extracting useful temporal data to avoid data leakage.
* **Correlation Analysis:** Generated a correlation heatmap for numerical features to check for multicollinearity before modeling.
* **Feature Importances:** Plotted a Gini importance bar chart from a baseline Random Forest to visualize which features actually drove the model's decisions.

## Feature Engineering Tasks

* **Task 1: Temporal Engineering:** Engineered a new `is_weekend` feature derived from the order timestamp. Delivery logistics change drastically on weekends due to traffic and restaurant volume, making this a vital signal.
* **Task 2: Peak Hour Adjustment:** Modified the `is_peak_hour` rule to encompass both the afternoon lunch rush (13:00-15:00) and the evening dinner rush (19:00-21:00), providing the model with clearer bounds for high-stress delivery windows.
* **Task 3: Categorical Dimensionality Reduction:** To isolate the impact of item categories, I temporarily removed other categorical features (like Traffic and City). I reduced the `Item_Name` column to the top $k$ (10, 30, and 50) most frequent items, grouping the rest into "Other", and encoded them using scikit-learn's `OneHotEncoder`. Testing these different thresholds highlighted the direct trade-off between category granularity and model noise (and demonstrated how much accuracy relies on variables like `Traffic_Level`!).
* **Task 4: Automated Feature Selection:** Built a scikit-learn `Pipeline` integrating `SelectFromModel`. By setting a `"median"` threshold based on Random Forest feature importances, the pipeline automatically prunes the weakest 50% of features during training. This resulted in a leaner, more efficient model that successfully discarded noisy variables.
