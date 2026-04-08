# Lab7: Logistic Regression

## Overview

This Lab focuses on implementing a Logistic Regression model in Python to predict whether an internet user will click on an advertisement. The predictions are based on consumer features such as age, area income, daily time spent on the site, and daily internet usage.

## Dataset

The dataset `advertising.csv` contains 1,000 observations of fake internet users and their interactions with a company website. The features used for the prediction include:

* **Daily Time Spent on Site:** Consumer time on site in minutes
* **Age:** Customer age in years
* **Area Income:** Avg. Income of geographical area of consumer
* **Daily Internet Usage:** Avg. minutes a day consumer is on the internet
* **Male:** Whether or not the consumer was male (Categorical: 1/0)
* **Clicked on Ad:** 0 or 1 indicating clicking on an Ad (Target Variable)

## Tasks Completed

1. **Exploratory Data Analysis (EDA):** Leveraged data visualization libraries (Seaborn and Matplotlib) to understand feature distributions and relationships. We generated histograms, KDE joint plots, and pair plots separated by our target hue to visually identify separable clusters.
2. **Data Splitting:** Preprocessed the selected features and target variable by splitting the data into a `training set` (**80%**) and a `testing set` (**20%**) using Scikit-Learn's `train_test_split`.
3. **Model Training:** Initialized and trained a Logistic Regression classifier (`LogisticRegression()`) using the training split.
4. **Prediction & Evaluation:** Evaluated the model's predictive power on the unseen testing data. We used a `classification_report` to `output precision`, `recall`, and `f1-scores`. The model achieved a strong overall accuracy of **97%**.

## Conclusion

By completing this lab, the following concepts were reinforced:

* **Visualizing Data Relationships:** How to use Seaborn's joint plots and pair plots to intuitively evaluate how well numerical variables separate target classes (e.g., seeing how users with lower 'Daily Internet Usage' and lower 'Daily Time Spent on Site' are more likely to click on ads).
* **Feature Selection:** Understanding how to parse a pandas DataFrame to separate independent features (X) from the dependent target variable (y).
* **Supervised Machine Learning Workflow:** Following the standard Scikit-Learn modeling sequence: instantiating a model, fitting it to training data, and generating predictions on test data.
* **Classification Metrics:** Interpreting classification reports beyond basic accuracy. Gaining familiarity with precision, recall, and f1-score to evaluate the specific capabilities of our classification model on both "Clicked" (1) and "Did Not Click" (0) outcomes.
