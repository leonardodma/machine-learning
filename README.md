# Machine Learning

## Variables Types

- Ordinal: Categorical variable with an order, e.g. low, medium, high
- Nominal: Categorical variable without an order, e.g. red, blue, green
- Continuous: Numerical variable with an infinite number of possible values, e.g. height, weight
- Discrete: Numerical variable with a finite number of possible values, e.g. number of children

### One Hot Encoding

One hot encoding is a process by which **nominal variables** are converted into a form that could be provided to ML algorithms to do a better job in prediction. One hot encoding creates a new column for each unique category in a **nominal variable.** The new columns are binary (0 or 1) and only one column will have a 1 (for each row of data). The column with a 1 will indicate the category that the row belongs to.

**Example:**

| Color | Size | Label  |
| ----- | ---- | ------ |
| Red   | S    | Apple  |
| Red   | M    | Apple  |
| Red   | L    | Orange |
| Green | S    | Orange |
| Green | M    | Apple  |
| Green | L    | Banana |

**One Hot Encoding:**

| Color_Red | Color_Green | Size_S | Size_M | Size_L | Label  |
| --------- | ----------- | ------ | ------ | ------ | ------ |
| 1         | 0           | 1      | 0      | 0      | Apple  |
| 1         | 0           | 0      | 1      | 0      | Apple  |
| 1         | 0           | 0      | 0      | 1      | Orange |
| 0         | 1           | 1      | 0      | 0      | Orange |
| 0         | 1           | 0      | 1      | 0      | Apple  |
| 0         | 1           | 0      | 0      | 1      | Banana |

**Example of implementation:**

```python
import pandas as pd

# Create a dataframe
df = pd.DataFrame({
    'Color': ['Red', 'Red', 'Red', 'Green', 'Green', 'Green'],
    'Size': ['S', 'M', 'L', 'S', 'M', 'L'],
    'Label': ['Apple', 'Apple', 'Orange', 'Orange', 'Apple', 'Banana']
})

# One hot encoding
df = pd.get_dummies(df, columns=['Color', 'Size'])

# Print the dataframe
print(df)
```

## Feature Engineering

Feature engineering is the process of using domain knowledge of the data to create features that make machine learning algorithms work. Feature engineering is fundamental to the application of machine learning, and is both difficult and expensive. The art of feature engineering is often the most difficult part of a machine learning project.

- **Feature selection:** Selecting a subset of the original features. Example: removing features that are not useful for the problem at hand.
- **Feature extraction:** Deriving information from the features to construct a new feature set. Example: using PCA to reduce the dimensionality of the data or using clustering to summarize the data into groups.
- **Feature creation:** Creating new features from the original feature set. Example: using the size of an image to create a new feature representing the ratio of the width to the height.

## Data Preprocessing

Data preprocessing is a data mining technique that involves transforming raw data into an understandable format. Real-world data is often incomplete, inconsistent, and/or lacking in certain behaviors or trends, and is likely to contain many errors. Data preprocessing is a proven method of resolving such issues.

- **Missing data:** Missing data can be a problem for many machine learning algorithms. There are several ways to deal with missing data, such as:
  - Removing the samples with missing data
  - Removing the features with missing data
  - Imputing the missing values
- **Categorical data:** Categorical data is data that can be placed into categories. There are two types of categorical data:
  - Nominal: Categorical data without an order, e.g. red, blue, green. To solve this problem we can use one hot encoding.
  - Ordinal: Categorical data with an order, e.g. low, medium, high. To solve this problem we can use label encoding, for example, low = 1, medium = 2, high = 3.

## Data Splitting

- **Training set:** The data used to train the model.
- **Validation set:** The data used to evaluate the model during training.
- **Test set:** The data used to evaluate the model after training.

## Overfitting and Underfitting

- **Overfitting:** Overfitting occurs when a model fits the training data too well. The model is too complex and does not generalize well to new data. Overfitting is a problem because the model will have a high accuracy on the training data, but a low accuracy on the test data.

## Models

Models are a way to represent the relationship between the features and the target variable. The goal of a model is to learn the relationship between the features and the target variable so that we can predict the target variable for new data. It is a **Combination of all possible candidate function**s.

**Parametric models:** Find a candidate function that minimizes the error between the predicted values and the actual values trough a set of parameters. The parameters are learned from the training data. The model is trained to find the best parameters.

**No parametric models:** functions that is not defined by a set of parameters, such as decision trees. The model is trained to find the best structure.

## Linear Regression

Linear regression is a linear approach to modeling the relationship between a scalar response and one or more explanatory variables (also known as dependent and independent variables). The case of one explanatory variable is called simple linear regression. For more than one explanatory variable, the process is called multiple linear regression.

- Examples:
  - Predicting the price of a house based on its size
  - Predicting the number of purchases a customer will make based on their income

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

## Multiple Linear Regression

Multiple linear regression is an extension of linear regression that involves more than one explanatory variable. The goal of multiple linear regression is to model the linear relationship between the explanatory variables and the response variable.

## Precision and Recall

**Precision:** The ability of a classifier not to label an instance positive that is actually negative. From all the data, how many did we label correctly?

- Formula: `Precision = TP / (TP + FP)`

**Recall:** The ability of a classifier to find all positive instances. From all the positive instances, how many did we label correctly?

- Formula: `Recall = TP / (TP + FN)`

It is important to note that precision and recall are often in conflict. Increasing precision reduces recall, and vice versa.

### The Confusion Matrix

The confusion matrix is a table that is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known.

**Example:**

| Actual/Predicted | Positive | Negative |
| ---------------- | -------- | -------- |
| Positive         | 10       | 5        |
| Negative         | 2        | 3        |

- **True Positive (TP):** The number of positive cases that were correctly identified as positive.
- **True Negative (TN):** The number of negative cases that were correctly identified as negative.
- **False Positive (FP):** The number of negative cases that were incorrectly identified as positive.
- **False Negative (FN):** The number of positive cases that were incorrectly identified as negative.

- Precision = `TP / (TP + FP)` = `10 / (10 + 5)` = `0.67`
- Recall = `TP / (TP + FN)` = `10 / (10 + 2)` = `0.83`

## Sensitivity and Specificity

**Sensitivity:** The ability of a test to correctly identify positive results. From all the positive instances, how many did we label correctly?

- Formula: `Sensitivity = TP / (TP + FN)`

**Specificity:** The ability of a test to correctly identify negative results. From all the negative instances, how many did we label correctly?

- Formula: `Specificity = TN / (TN + FP)`

## Comparing Models

- cross_val_score
- cross_val_predict

### Why use cross-validation?

- It is a more accurate way of evaluating a model on unseen data than train/test split because testing accuracy can change a lot depending on which observations happen to be in the testing set.
- It allows for more efficient use of data (every observation is used for both training and testing).

### How does cross-validation work?

- Split the dataset into k equal parts, or "folds" (typically k = 10).
- Use fold 1 as the testing set and the union of the other folds as the training set.
- Calculate testing accuracy.
- Repeat steps 2 and 3 k times, using a different fold as the testing set each time.
- Use the average testing accuracy as the estimate of out-of-sample accuracy.
