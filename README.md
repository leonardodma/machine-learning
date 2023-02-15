# Machine Learning

## Machine Learning vs. Artificial Intelligence

Machine learning is a subset of artificial intelligence. Is a method of data analysis that automates analytical model building. Using algorithms that iteratively learn from data, machine learning allows computers to find hidden insights without being explicitly programmed where to look.

## Machine Learning vs. Statistics

Machine learning is foucsed on prediction and inference, while statistics is focused on estimation and hypothesis testing (explaining the data).

## Variables Types

- Ordinal: Categorical variable with an order, e.g. low, medium, high
- Nominal: Categorical variable without an order, e.g. red, blue, green
- Continuous: Numerical variable with an infinite number of possible values, e.g. height, weight
- Discrete: Numerical variable with a finite number of possible values, e.g. number of children

## Linear Regression

Linear regression is a linear approach to modeling the relationship between a scalar response and one or more explanatory variables (also known as dependent and independent variables). The case of one explanatory variable is called simple linear regression. For more than one explanatory variable, the process is called multiple linear regression.

**How to define how good is the model fit/accuracy? The most common method is to use two types of errors:**

- Mean Absolute Error (MAE): The mean absolute error is the average of the absolute value of the errors. It is calculated as:

```math
MAE = 1/n * sum(|y - y_pred|)
```

- Mean Squared Error (MSE): The mean squared error is the average of the squared errors. It is calculated as:

```math
MSE = 1/n * sum((y - y_pred)^2)
```

**Algorithms:**

Function: `y = mx + b` where `m` is the slope and `b` is the y-intercept.

1. Start with a random value for `m` and `b`
2. Calculate the error for each point
3. Calculate the total error
4. Adjust `m` and `b` to reduce the error
5. Repeat steps 2-4 until the error is small enough

Gradient descent:

1. Start with a random value for `m` and `b`
2. Calculate the error for each point
3. Calculate the total error
4. Calculate the gradient of the error function with respect to `m` and `b`
5. Adjust `m` and `b` in the direction of the gradient
6. Repeat steps 2-5 until the error is small enough

**Example of implementation of Gradient Descent:**

```python
import numpy as np

# Generate random data
x = np.random.rand(100, 1)
y = 4 + 3 * x + np.random.rand(100, 1)

# Initialize m and b
m = 0
b = 0

# Learning rate
lr = 0.01

# Number of iterations
n = 1000

# Tolerance
tol = 0.0001

# Number of data points
N = float(len(x))

# Perform gradient descent
while True:
    y_pred = m * x + b
    D_m = (-2 / N) * sum(x * (y - y_pred))
    D_b = (-2 / N) * sum(y - y_pred)
    m = m - lr * D_m
    b = b - lr * D_b

    # Calculate sum of squared errors
    error = sum((y - y_pred) ** 2) / N

    # Check if error is small enough
    if error < tol:
        break

# Print m and b
print(m, b)
```

**Example of implementation Using Sklearn:**

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

# Generate random data
x = np.random.rand(100, 1)
y = 4 + 3 * x + np.random.rand(100, 1)

# Plot data
plt.scatter(x, y)
plt.show()

# Create linear regression model
model = LinearRegression()

# Train the model using the training sets
model.fit(x, y)

# Make predictions using the testing set
y_pred = model.predict(x)

# Plot outputs
plt.scatter(x, y,  color='black')
plt.plot(x, y_pred, color='blue', linewidth=3)
```

For more information, see [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).
