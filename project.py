import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create dataset
data = {
    "customer_id": range(1, 11),
    "age": [25, 30, 22, 40, 35, 28, 50, 45, 33, 38],
    "tenure": [1, 3, 2, 5, 4, 2, 7, 6, 3, 4],
    "salary": [40000, 50000, 35000, 60000, 55000, 45000, 70000, 65000, 50000, 58000]
}

df = pd.DataFrame(data)

# Feature Engineering
df["bonus"] = df["salary"] * 0.10

# Define features and target
X = df[["age", "tenure", "bonus"]]
y = df["salary"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation")
print("MSE:", f"{mse}")
print("R2 Score:", f"{r2}")

# Visualization
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.show()



