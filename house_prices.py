import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


features = ["GrLivArea", "BedroomAbvGr", "FullBath"]

X = train[features]
y = train["SalePrice"]


X = X.fillna(X.median())
test_features = test[features].fillna(X.median())


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_val)

mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred)

print("📊 Model Evaluation:")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")


plt.scatter(y_val, y_pred, alpha=0.6, color="blue")
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted Sale Price")
plt.show()


print("\n🏠 House Price Prediction with Your Input")

sqft = float(input("Enter square footage of the house (GrLivArea): "))
bedrooms = int(input("Enter number of bedrooms (BedroomAbvGr): "))
bathrooms = int(input("Enter number of bathrooms (FullBath): "))

user_data = np.array([[sqft, bedrooms, bathrooms]])
predicted_price = model.predict(user_data)

print(f"\n💰 Predicted House Price: ${predicted_price[0]:,.2f}")


predictions = model.predict(test_features)

submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": predictions
})

submission.to_csv("submission.csv", index=False)

print("\n✅ submission.csv file has been created successfully!")

