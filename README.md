# Lab 10: Support Vector Machines (SVM)

**Course:** ARTI308 - Machine Learning

This lab introduces **Support Vector Machines (SVM)**, a supervised learning algorithm used for classification and regression. The lab is split into a guided walkthrough notebook and a hands-on assignment notebook.

---

## Files

- [01-Support Vector Machines.ipynb](01-Support%20Vector%20Machines.ipynb): Guided walkthrough using the **Breast Cancer Wisconsin** dataset.
- [02-SVM Assignment.ipynb](02-SVM%20Assignment.ipynb): Assignment using the famous **Iris flower** dataset.

---

## What I Did

### Notebook 1: Breast Cancer Classification

1. Loaded the built-in `load_breast_cancer` dataset from scikit-learn (569 samples, 30 numeric features, 2 classes: malignant / benign).
2. Built a pandas DataFrame from the feature matrix and target vector.
3. Split the data into training (70%) and testing (30%) sets using `train_test_split`.
4. Trained a default `SVC()` model and evaluated it using a **confusion matrix** and a **classification report** (achieved ~92% accuracy).
5. Used **`GridSearchCV`** to search over `C`, `gamma`, and `kernel` hyperparameters in order to find a better model configuration.

### Notebook 2: Iris Classification (Assignment)

1. Loaded the Iris dataset using `sns.load_dataset("iris")` (150 samples, 4 features, 3 species).
2. Performed **Exploratory Data Analysis**:
   - Created a `pairplot` colored by species to visually identify which class is most separable (setosa was clearly the most separable).
   - Created a `kdeplot` of `sepal_length` vs `sepal_width` for the setosa species.
3. Split the data into training (70%) and testing (30%) sets.
4. Trained an `SVC()` model and evaluated it, achieved **98% accuracy** on the test set.
5. Practiced hyperparameter tuning with `GridSearchCV` over a grid of `C` values `[0.1, 1, 10, 100]` and `gamma` values `[1, 0.1, 0.01, 0.001]` (5-fold CV, 80 total fits).
6. Compared the tuned model's predictions to the baseline, both performed equally well (98%) since the dataset is small and very separable.

---

## What I Learned

- **How SVMs work:** SVM is a supervised classifier that finds the optimal hyperplane that maximally separates classes in the feature space.
- **Using `SVC` from scikit-learn:** how to instantiate, fit, and predict with a Support Vector Classifier in just a few lines of code.
- **The importance of hyperparameters** in SVM:
  - **`C`**: The regularization parameter that controls the trade-off between a smooth decision boundary and classifying training points correctly.
  - **`gamma`**: Defines how far the influence of a single training example reaches (low= far, high= close).
  - **`kernel`**: The kernel function used (e.g., `rbf`, `linear`).
- **Hyperparameter tuning with `GridSearchCV`:** how to define a parameter grid and let scikit-learn try every combination using cross-validation to pick the best one. Best params are accessible via `.best_params_` and `.best_estimator_`.
- **Model evaluation:** how to read a **confusion matrix** and interpret a **classification report** (precision, recall, f1-score, support).
- **Pitfall observed:** in Notebook 1 the GridSearch with limited parameter ranges actually produced a *worse* model that classified everything into one class — showing that a poorly chosen parameter grid can hurt performance, and that the search space matters.
- **EDA first:** visualization tools like `sns.pairplot` and `sns.kdeplot` are very helpful for understanding class separability before training a model.

---

## Libraries Used

- `pandas`, `numpy`: Data handling
- `matplotlib`, `seaborn`: Visualization
- `scikit-learn`, `SVC`, `train_test_split`, `GridSearchCV`, `classification_report`, `confusion_matrix`
- `sklearn.datasets.load_breast_cancer`: Built-in dataset
