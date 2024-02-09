import random
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the iris dataset
iris = datasets.load_iris()
iris_X = iris.data

# # Split the dataset into training and testing data without using pandas
# iris_X_train = iris_X[:-30]  # Using all but the last 30 samples for training
# iris_X_test = iris_X[-20:]   # Using the last 20 samples for testing

# iris_y_train = iris.target[:-30]  # Corresponding target labels for training data
# iris_y_test = iris.target[-20:]   # Corresponding target labels for testing data


# Convert iris dataset to a pandas DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# Split the dataset into training and testing data using train_test_split
r = random.randint(1,300)
iris_train, iris_test = train_test_split(iris_df, test_size=0.2, random_state=r)

# Extract features and target variables for training set
iris_X_train = iris_train.drop('target', axis=1)
iris_y_train = iris_train['target']

# Extract features and target variables for testing set
iris_X_test = iris_test.drop('target', axis=1)
iris_y_test = iris_test['target']

# Create a linear regression model
model = linear_model.LinearRegression()

# Train the model using the training data
model.fit(iris_X_train, iris_y_train)

# Testing the model on the testing data
iris_y_predict = model.predict(iris_X_test)

# Calculating Mean Squared Error (MSE) as a measure of the model's performance
mse = mean_squared_error(iris_y_test, iris_y_predict)
print(f"Mean Squared Error is: {mse}")

# # Display the weights (coefficients) and intercept of the linear regression model
# print("Weights (Coefficients):", model.coef_)   # 'tan theta' in the linear equation
# print("Intercept:", model.intercept_)            # y-intercept in the linear equation


# Create a DataFrame to display actual and predicted values along with flower types
result_df = pd.DataFrame({
    'Actual Values': iris_y_test,
    'Predicted Values': iris_y_predict,
    'Flower Type': iris.target_names[iris_y_test]
})

# Print the DataFrame
print(result_df)