# Lab 3 Solution: Exploratory Data Analysis (EDA) of Bug Reports

**Course:** ARTI308 - Machine Learning  
**Topic:** Exploratory Data Analysis (EDA)  
**File:** `exploring.ipynb`  
**Dataset:** `probset.csv`

## Overview

This notebook contains the solution for the **Lab 3 assignment**. The goal was to perform **Exploratory Data Analysis (EDA)** on a dataset of software bug reports to `understand its structure`, `clean the data`, and `visualize key patterns`.

## Libraries Used

The following Python libraries were utilized for data manipulation and visualization:

* **[Pandas](https://pandas.pydata.org):** For data frame operations and cleaning.
* **[NumPy](https://numpy.org):** For numerical operations.
* **[Matplotlib](https://matplotlib.org) & [Seaborn](https://seaborn.pydata.org):** For plotting and visualizing distributions.

## Key Analysis Steps

### 1. Data Loading & Inspection

* Loaded the dataset `probset.csv` containing **50,000 bug reports**.
* **Initial Checks:**
  * Used `.head()`, `.shape`, and `.dtypes` to understand the data structure.
  * Identified that the dataset consists of **14 columns** (e.g., `bug_id`, `severity`, `error_code`, `created_at`).
* **Missing Value Analysis:**
  * Detected **6,188 missing values** in the `error_code` column using `.isna().sum()`.
  * Confirmed no other columns had missing data.

### 2. Data Cleaning & Preprocessing

To prepare the data for analysis, the following cleaning steps were performed:

* **Imputation:** Filled missing values in the `error_code` column with `0` to handle gaps in the data.
* **Type Conversion:** Converted the `created_at` column from object/string format to **datetime** objects for proper time-series handling.

### 3. Feature Engineering

* **Severity Scoring:** Created a new numerical column `Severity_Score` to quantify the priority of bugs.
  * *Mapping:* **{Low: 1, Medium: 2, High: 3, Critical: 4}**.

### 4. Visualization & Insights

* **Bug Category Distribution:**
  * Generated a horizontal **Count Plot** (using `sns.countplot`) to visualize the frequency of different bug categories.
  * *Insight:* Observed the distribution of issues across categories like "API Bug", "Memory Leak", etc.
* **Environment & Domain Analysis:**
  * Grouped the data by `environment` (e.g., Staging, Production) and `bug_domain` to analyze bug volume across different system areas.

## Conclusion

The raw bug report data was successfully cleaned and enriched with numerical scores. The exploratory visualizations helped identify the most common types of bugs and their distribution across different development environments.

---
*This README documents the steps and code logic covered in the `CCSIT_ARTI308_Lab3.ipynb` notebook.*
