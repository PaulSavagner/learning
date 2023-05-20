from sklearn.linear_model import LinearRegression
import numpy as np

# Data Preparation
X = np.array([5, 10, 20, 30, 40]).reshape(-1, 1)  # Weight of the dogs in kg
y = np.array([14, 13, 10, 8, 7])  # Life expectancy of the dogs in years

# Model Creation
model = LinearRegression()

# Model Training
model.fit(X, y)

# Make Predictions
weight = np.array([25]).reshape(-1, 1)  # Weight of the dog we want to predict the life expectancy of
predicted_life_expectancy = model.predict(weight)

print("The predicted life expectancy of the dog is: ", predicted_life_expectancy[0], "years.")
