# Lab 9 — Decision Trees and Random Forests

This lab covered supervised classification using tree-based models on two datasets: the small **Kyphosis** dataset (walk-through) and the larger **LendingClub `loan_data.csv`** dataset. The goal of the lab was to predict whether a borrower will pay back their loan in full (`not.fully.paid`).

## What I Learned

### 1. Decision Trees

- A Decision Tree splits the data recursively on feature thresholds to produce pure leaf nodes.
- Sklearn exposes it through `from sklearn.tree import DecisionTreeClassifier`.
- Training is a single call: `dtree.fit(X_train, y_train)`, and prediction is `dtree.predict(X_test)`.
- Single trees overfit easily — they tend to memorize the training data, which hurts performance on new samples.

### 2. Random Forests

- A Random Forest trains many decision trees on bootstrapped samples and averages their votes.
- This reduces variance and usually beats a single tree in overall accuracy.
- Imported from `sklearn.ensemble.RandomForestClassifier`, tuned mainly via `n_estimators` (I used 600 for the project).
- Random Forests are less interpretable than a single tree but much more stable.

### 3. Model Evaluation

- `classification_report` from `sklearn.metrics` shows **precision, recall, f1-score** per class.
- `confusion_matrix` shows the raw TP/FP/FN/TN counts.
- Accuracy alone is misleading on imbalanced data — on the loan dataset, ~84% of borrowers fully repay, so a model that always predicts "paid" still looks ~84% accurate while completely missing the defaulters.

### 4. Handling Categorical Features

- Sklearn estimators need numeric input, so the `purpose` column had to be one-hot encoded.
- Used the pattern `pd.get_dummies(loans, columns=cat_feats, drop_first=True)` to expand `purpose` into dummy columns while avoiding the dummy-variable trap.

### 5. Exploratory Data Analysis (EDA)

- Overlaid histograms of FICO score split by `credit.policy` and `not.fully.paid` to see how FICO relates to each outcome.
- `sns.countplot` with `hue='not.fully.paid'` revealed which loan purposes default the most.
- `sns.jointplot` and `sns.lmplot` showed the negative relationship between FICO and interest rate — higher FICO, lower rate.

### 6. Standard ML Workflow

- Load data → explore → encode categoricals → `train_test_split` → fit model → predict → evaluate.
- Using `random_state=101` in `train_test_split` makes the results reproducible.

## What I Applied in This Lab

- Read and inspected `loan_data.csv` (9578 rows, 14 columns) with `.info()`, `.head()`, `.describe()`.
- Built the required EDA plots (FICO histograms, purpose countplot, FICO vs int.rate jointplot, faceted lmplot).
- One-hot encoded the `purpose` column to produce `final_data` with 19 columns.
- Split the data 70/30 into train/test sets.
- Trained a `DecisionTreeClassifier` and a `RandomForestClassifier(n_estimators=600)`.
- Produced a classification report and confusion matrix for each model and compared them.

## Results Comparison

| Model | Accuracy | Recall (class 1 = not fully paid) |
|-------|----------|-----------------------------------|
| Decision Tree | ~0.73 | ~0.23 |
| Random Forest | ~0.85 | ~0.02 |

The Random Forest wins on overall accuracy but essentially predicts "paid" for almost everyone, so it misses the defaulters — exactly the class we care about. The Decision Tree catches more defaulters but at the cost of more false alarms. **Takeaway:** on imbalanced datasets, accuracy is not enough; recall and precision on the minority class matter more, and techniques like class weighting or resampling would be needed to improve the Random Forest's recall.

## Files

- `01-Decision Trees and Random Forests.ipynb` — walk-through on the Kyphosis dataset.
- `02-Decision Trees and Random Forest Project.ipynb` — completed project on the LendingClub data.
- `kyphosis.csv` — dataset for the walk-through.
- `loan_data.csv` — dataset for the project.
