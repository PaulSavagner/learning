from sklearn.linear_model import LinearRegression
import numpy as np

# Data Preparation
X = np.array([5, 7, 10, 13, 20, 23, 30, 33, 40, 45, 50]).reshape(-1, 1)  # Weight of the dogs in kg
y = np.array([14, 14, 13, 12, 11, 10, 9, 8, 9, 6, 10])  # Life expectancy of the dogs in years

# Model Creation
model = LinearRegression()

# Model Training
model.fit(X, y)

# Ask the user for a weight
weight_input = input("Please enter a dog's weight in kg: ")
weight = np.array([float(weight_input)]).reshape(-1, 1)

# Make Predictions
predicted_life_expectancy = model.predict(weight)

print("The predicted life expectancy of the dog is: ", predicted_life_expectancy[0], "years.")
