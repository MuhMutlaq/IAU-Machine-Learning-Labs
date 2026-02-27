# Lab 4 Solution: Data Quality Assessment & Preprocessing

## Overview

This repository contains the Jupyter Notebook solution (`Solution.ipynb`) for the **Lab 4 Assignment**. The objective of this lab is to assess data quality and apply standard preprocessing techniques to prepare a dataset for machine learning. 

The dataset used is `Chocolate_Sales.csv`, which contains details on chocolate shipments, sales personnel, geographical regions, and financials.

## Tasks & Methodologies Implemented

This notebook successfully executes 5 primary data preprocessing tasks:

### **Task 1: Identify Data Quality Issues**

* **Identified Issues:** The `Date` column was incorrectly formatted as a string (`object`), and the `Amount` column contained non-numeric currency characters (`$` and `,`).
* **Resolution:** * Converted the `Date` column to standard pandas `datetime64[ns]` objects (`dayfirst=True`).
  * Used Regex to strip the currency symbols and commas from `Amount`, casting it to a `float64` data type.

### **Task 2: Missing Value Strategy**

* **Simulation:** Artificially simulated missing data in the first 6 rows of `Boxes Shipped` to test the imputation logic.
* **Strategy (Linear Interpolation with Boundary Handling):** * Used `interpolate(method="linear", limit=5)` to estimate missing values smoothly based on neighboring data points.
  * Added custom conditional logic to handle boundary edge-cases: if the very first or very last element is `NaN`, it wraps around and borrows the value from the opposite end of the dataset to anchor the interpolation.

### **Task 3: Detect and Handle Outliers (IQR)**

* **Detection:** Visualized the data spread and extreme values using Seaborn `boxplot`s for both `Amount` and `Boxes Shipped`.
* **Handling:** Implemented a custom `handle_outliers_iqr` function. It calculates the 25th (Q1) and 75th (Q3) percentiles to find the Interquartile Range (IQR). Outliers are safely clipped (capped) to the lower bound ($Q1 - 1.5 \times IQR$) and upper bound ($Q3 + 1.5 \times IQR$) using `numpy.where`.

### **Task 4: Normalize Numerical Features**

* Isolated the numerical features: `["Amount", "Boxes Shipped"]`.
* **Min-Max Normalization:** Applied `sklearn.preprocessing.MinMaxScaler` to scale data into a fixed `[0, 1]` range (`df_minmax`).
* **Z-Score Standardization:** Applied `sklearn.preprocessing.StandardScaler` to transform the data to have a mean of 0 and a standard deviation of 1 (`df_zscore`), which is required for PCA.

### **Task 5: Apply PCA and Interpret Explained Variance**

* Applied Principal Component Analysis (`sklearn.decomposition.PCA`) on the Z-score standardized dataset.
* **Interpretation:** The model output an explained variance ratio of exactly `[0.50598606, 0.49401394]`. This indicates that PC1 and PC2 capture an almost equal amount of variance (~50.6% and ~49.4% respectively).
* **Visualizations Generated:**
  1. **Scree Plot:** A bar and line chart illustrating the individual and cumulative explained variance.
  2. **PCA 2D Scatter Plot:** A visual projection of the data points distributed across the newly created Principal Components 1 and 2.
  3. **Correlation Heatmap:** Proves why the variance is split 50/50—the original features (`Amount` and `Boxes Shipped`) have near-zero correlation (`-0.013`), meaning they are independent and both equally important to the dataset's overall variance.

---

## Requirements & Dependencies

To execute the `Solution.ipynb` notebook, ensure you have the following Python libraries installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn