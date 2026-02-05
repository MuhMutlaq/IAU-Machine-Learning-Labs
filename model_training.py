import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. Load the dataset
df= pd.read_csv("Housing.csv")

# 2. Data Preprocessing
# Convert binary categorical columns (yes/no) to 0/1 or use One-Hot Encoding
# Use get_dummies to handle all categorical variables automatically
# drop_first=True helps avoid multicollinearity
categorical_cols= ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea", "furnishingstatus"]
df_processed= pd.get_dummies(df, columns= categorical_cols, drop_first= True)

# 3. Define Features (X) and Target (y)
X= df_processed.drop("price", axis= 1)  # All columns except "price"
y= df_processed["price"]               # The target we want to predict

# 4. Train/Test Split (80% Training, 20% Testing)
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.2, random_state= 42)

print("\n\n---- Now training a Random Forest Regressor model ----\n")

# 5. Initialize the Model (Using Random Forest Regressor) because of predicting a continuous value
model= RandomForestRegressor(n_estimators= 100, random_state= 42)

# 6. Train the Model
model.fit(X_train, y_train)
print("The model training complete.")

# 7. Evaluate the Model
predictions= model.predict(X_test)

# Calculate metrics
mse= mean_squared_error(y_test, predictions)
rmse= np.sqrt(mse)
r2= r2_score(y_test, predictions)

print(f"\nModel Evaluation:")
print(f"Accuracy (R^2 Score): {r2:.4f} (The model explains {r2*100:.2f}% of the price variation)")
print(f"Mean Square Error (MSE): {mse:,.2f}")
print(f"Root Mean Square Error (RMSE): {rmse:,.2f}")

list_better= []
list_better.append(r2*100)

# 8. Save the model
trained_model1= "trained_model1.pkl"

joblib.dump(model, trained_model1)
print(f"\nThe model saved successfully as ({trained_model1})")




from sklearn.linear_model import LinearRegression  # Changed import

print("\n\n---- Now training a Linear Regression model ----\n")

# 5. Initialize the Model (Using Linear Regression)
model= LinearRegression()

# 6. Train the Model
model.fit(X_train, y_train)
print("The model training complete.")

# 7. Evaluate the Model
predictions= model.predict(X_test)

# Calculate metrics
mse= mean_squared_error(y_test, predictions)
rmse= np.sqrt(mse)
r2= r2_score(y_test, predictions)

print(f"\nModel Evaluation:")
print(f"Accuracy (R^2 Score): {r2:.4f} (The model explains {r2*100:.2f}% of the price variation)")
print(f"Mean Square Error (MSE): {mse:,.2f}")
print(f"Root Mean Square Error (RMSE): {rmse:,.2f}")

list_better.append(r2*100)

# 8. Save the model
trained_model2= "trained_model2.pkl"

joblib.dump(model, trained_model2)
print(f"\nThe model saved successfully as ({trained_model2})")




from sklearn.ensemble import GradientBoostingRegressor

print("\n\n---- Now training a Gradient Boosting Regressor model ----\n")

# 5. Initialize the Improved Model (Gradient Boosting builds trees one by one to correct previous errors)
model= GradientBoostingRegressor(n_estimators= 100, learning_rate= 0.1, max_depth= 3, random_state= 42)

# 6. Train the model
model.fit(X_train, y_train)
print("The model training complete.")

# 7. Evaluate the model
predictions= model.predict(X_test)

mse= mean_squared_error(y_test, predictions)
rmse= np.sqrt(mse)
r2= r2_score(y_test, predictions)

print(f"\nModel Evaluation:")
print(f"Accuracy (R^2 Score): {r2:.4f} (The model explains {r2*100:.2f}% of the price variation)")
print(f"Mean Square Error (MSE): {mse:,.2f}")
print(f"Root Mean Square Error (RMSE): {rmse:,.2f}")

list_better.append(r2*100)

# 8. Save the model
trained_model3= "trained_model3.pkl"

joblib.dump(model, trained_model3)
print(f"\nModel saved as ({trained_model3})")

if list_better[0] > list_better[1] and list_better[0] > list_better[2]:
    print(f"\nThe better model is (Random Forest Regressor) with R^2: {list_better[0]:.2f}%")
elif list_better[1] > list_better[0] and list_better[1] > list_better[2]:
    print(f"\nThe better model is (Linear Regression) with R^2: {list_better[1]:.2f}%")
else:
    print(f"\nThe better model is (Gradient Boosting Regressor) with R^2: {list_better[2]:.2f}%")