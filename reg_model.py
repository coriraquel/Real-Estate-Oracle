#Import Libraries 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

#Load Dataset 
df= pd.read_csv("cali_housing_data.csv")

#Select Features and Targets 
X= df.drop(columns=["Price"]) 
y= df["Price"] 

#Split Data into Training and Testing Data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model with Linear Regression 
model = LinearRegression()
model.fit(X_train, y_train)

# Predict model 
y_pred = model.predict(X_test)

# Evaluate model with mse and rmse 
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

#Format results for print 
print(f"Model Performance:\nMAE: {mae}\nRMSE: {rmse}")

# Optional Code to Save model versions 
import joblib
joblib.dump(model, "house_price_model.pkl")
print("Model saved as house_price_model.pkl")