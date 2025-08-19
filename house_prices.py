# house_price_prediction.py

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset (make sure train.csv is in the same folder as this file)
data = pd.read_csv("train.csv")

# Select relevant features
X = data[["GrLivArea", "BedroomAbvGr", "FullBath"]]
y = data["SalePrice"]

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("📊 Model Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Show coefficients
coefficients = pd.DataFrame(
    model.coef_, X.columns, columns=["Coefficient"]
)
print("\n📌 Model Coefficients:")
print(coefficients)

# Visualization: Actual vs Predicted
plt.scatter(y_test, y_pred, alpha=0.6, color="blue")
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted Sale Price")
plt.show()

# ---------------- USER INPUT SECTION ----------------
print("\n🏠 House Price Prediction with Your Input")

# Take user input
sqft = float(input("Enter square footage of the house (GrLivArea): "))
bedrooms = int(input("Enter number of bedrooms (BedroomAbvGr): "))
bathrooms = int(input("Enter number of bathrooms (FullBath): "))

# Prepare input data for prediction
user_data = np.array([[sqft, bedrooms, bathrooms]])

# Predict house price
predicted_price = model.predict(user_data)

print(f"\n💰 Predicted House Price: ${predicted_price[0]:,.2f}")
