# Lab8: K-Nearest Neighbors (KNN) Classification

## Overview

This Lab focuses on implementing a **K-Nearest Neighbors (KNN)** `classification model` in Python. Because the dataset contains `hidden/anonymous feature columns`, the primary objective is to rely strictly on data patterns and standard machine learning techniques—specifically feature scaling and hyperparameter tuning—to accurately predict the target class of each observation.

## Dataset

The dataset `KNN_Project_Data.csv` contains observations of anonymous data points. Because the data is highly classified or stripped of context, the features are simply represented by random letter combinations (e.g., XVPM, GWYH, TRVR). The features used for prediction include:

* **Anonymous Features:** A series of continuous numerical variables representing various unknown attributes.
* **Target Class:** `0` or `1`, indicating the binary category each observation belongs to (Target Variable).

## Tasks Completed

* **Exploratory Data Analysis (EDA):** Leveraged Seaborn to create a `pairplot` colored by the target class. Even without knowing what the features represent, this allowed us to visually inspect for distinct clusters and relationships between variables.
* **Feature Standardization:** Utilized Scikit-Learn's `StandardScaler` to scale all independent features. We fit the scaler to the data and transformed the features to ensure all variables were on the same scale before passing them into the model.
* **Data Splitting:** Split the standardized data into a **training set** (70%) and a **testing set** (30%) using Scikit-Learn's `train_test_split`.
* **Baseline Model Training & Evaluation:** Initialized and trained a baseline KNN classifier (`KNeighborsClassifier`) with K=1. We evaluated its predictive power on the test data using a `confusion_matrix` and `classification_report`.
* **Hyperparameter Tuning (The Elbow):** Wrote a `for` loop to train 40 different KNN models with K values ranging from 1 to 40. We tracked the error rate for each model and plotted the results using Matplotlib to visually identify the optimal K value where the error rate drops and stabilizes.
* **Retraining & Optimization:** Retrained the KNN model using the newly discovered optimal K value (e.g., K=30). The resulting classification report showed improved and stabilized precision, recall, and f1-scores compared to the baseline.

## Conclusion

By completing this lab, the following concepts were reinforced:

* **Importance of Feature Scaling:** Understanding *why* scaling matters for distance-based algorithms. Because KNN predicts classes by calculating the distance between data points (like Euclidean distance), standardizing the variables ensures that features with larger magnitudes don't overpower smaller ones.
* **Iterative Model Improvement:** Following a complete optimization cycle: building a naive baseline model (K= 1), evaluating its performance, tuning the parameters via the Elbow Method, and retraining to achieve a superior classification outcome.
* **Handling Anonymous Data:** Realizing that supervised machine learning models can still effectively discover patterns, draw boundaries, and classify data even when the real-world context or meaning of the features is entirely hidden from the data scientist.
* **The Elbow:** How to programmatically test a range of hyperparameter values (K), capture their error rates, and create an "Error Rate vs. K Value" plot. This teaches the importance of empirically justifying the hyperparameters we choose.
