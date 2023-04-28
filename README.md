# Machine Learning

---

## Types of Machine Learning

- **Supervised learning:** Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs. Example: **classification** and **regression**.

- **Unsupervised learning:** Unsupervised learning is a type of machine learning algorithm used to draw inferences from datasets consisting of input data without labeled responses. The most common unsupervised learning method is **clustering**, which is the task of grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar (in some sense or another) to each other than to those in other groups (clusters).

- **Reinforcement learning:** Reinforcement learning is an area of machine learning concerned with how software agents ought to take actions in an environment in order to maximize the notion of cumulative reward.

---

## Variables

### Types of Variables

- **Ordinal**: Categorical variable with an order, e.g. low, medium, high
- **Nominal**: Categorical variable without an order, e.g. red, blue, green
- **Continuous**: Numerical variable with an infinite number of possible values, e.g. height, weight
- **Discrete**: Numerical variable with a finite number of possible values, e.g. number of children

---

## Data Preprocessing

Data preprocessing is a data mining technique that involves transforming raw data into an understandable format. Real-world data is often incomplete, inconsistent, and/or lacking in certain behaviors or trends, and is likely to contain many errors. Data preprocessing is a proven method of resolving such issues.

- **Missing data:** Missing data can be a problem for many machine learning algorithms. There are several ways to deal with missing data, such as:
  - Removing the samples with missing data
  - Removing the features with missing data
  - Imputing the missing values
- **Categorical data:** Categorical data is data that can be placed into categories. There are two types of categorical data:
  - Nominal: Categorical data without an order, e.g. red, blue, green. To solve this problem we can use one hot encoding.
  - Ordinal: Categorical data with an order, e.g. low, medium, high. To solve this problem we can use label encoding, for example, low = 1, medium = 2, high = 3.

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

### Feature Engineering

Feature engineering is the process of using domain knowledge of the data to create features that make machine learning algorithms work. Feature engineering is fundamental to the application of machine learning, and is both difficult and expensive. The art of feature engineering is often the most difficult part of a machine learning project.

- **Feature selection:** Selecting a subset of the original features. Example: removing features that are not useful for the problem at hand.
- **Feature extraction:** Deriving information from the features to construct a new feature set. Example: using PCA to reduce the dimensionality of the data or using clustering to summarize the data into groups.
- **Feature creation:** Creating new features from the original feature set. Example: using the size of an image to create a new feature representing the ratio of the width to the height.

### Data Splitting

- **Training set:** The data used to train the model.
- **Validation set:** The data used to evaluate the model during training.
- **Test set:** The data used to evaluate the model after training.

**Example:**

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Determine validation set size
val_size = 0.2

# Split training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=1)
```

---

## Models

Models are a way to represent the relationship between the features and the target variable. The goal of a model is to learn the relationship between the features and the target variable so that we can predict the target variable for new data

- **Regression**: Predicting a number. Example: predicting the price of a house.
  - Linear Regression
  - Multiple Linear Regression
  - Polynomial Regression
  - Support Vector Regression (SVR)
  - Decision Tree Regression
  - Random Forest Regression
- **Classification**: Predicting a category. Example: predicting if an email is spam or not.
  - Decision Tree
  - Random Forest
  - Support Vector Machine
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
  - Logistic Regression
- **Clustering**: Grouping similar data points together. Example: grouping customers by purchasing behavior.
  - K-Means
  - Hierarchical Clustering

---

## Algorithms

### Regression

#### Linear Regression

Linear regression is a linear approach to modeling the relationship between a scalar response and one or more explanatory variables (also known as **dependent (y)** and **independent (x)** variables). The case of one explanatory variable is called simple linear regression. For more than one explanatory variable, the process is called multiple linear regression.

- Use cases:
  - After analyzing the data, we found that there is a linear relationship between the features and the target variable.

#### Multiple Linear Regression

Multiple linear regression is an extension of linear regression that involves more than one explanatory variable (independent). The goal of multiple linear regression is to model the linear relationship between the explanatory variables and the response variable.

- Use cases:
  - After analyzing the data, we found that there is a linear relationship between the features and the target variable.

#### Polynomial Regression

Polynomial regression is a form of linear regression in which the relationship between the independent variable x and the dependent variable y is modeled as an nth degree polynomial in x. Polynomial regression fits a nonlinear relationship between the value of x and the corresponding conditional mean of y, denoted E(y | x).

- Use cases:
  - After analyzing the data, we found that there is a non-linear relationship between the features and the target variable.

#### Support Vector Regression (SVR)

Support vector regression (SVR) is a type of supervised learning algorithm that uses support vector machines (SVMs) to perform regression. SVR is similar to SVM for classification, but instead of finding a line that divides the classes, it finds a line that best fits the data.

#### Decision Tree Regression

Decision tree regression is a type of regression algorithm that uses a decision tree to predict the value of a target variable by learning simple decision rules inferred from the data features.

#### Random Forest Regression

Random forest regression is a type of regression algorithm that uses ensemble learning method for regression. Ensemble learning method combines the predictions from multiple machine learning algorithms together to make more accurate predictions than a single model.

### Classification

#### Decision Tree

Decision tree is a supervised learning algorithm that can be used for both classification and regression problems. It is a tree-like model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility.

Parameters:

- **Criterion:** The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
  - **Gini impurity:** A measure of how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset. It reaches its minimum (zero) when all cases in the node fall into a single target category.
  - **Information gain:** The expected reduction in entropy caused by partitioning the examples according to this attribute. It reaches its maximum (one) when all cases in the node fall into a single target category.
- **Max Depth:** The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

#### Random Forest

Random forest is a supervised learning algorithm that can be used for both classification and regression problems. It is a tree-like model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility.

This algorithm is based on the idea of creating multiple decision trees and combining their predictions to get a more accurate prediction than a single decision tree. **This idea is supported by the fact that multiple weak learners can outperform a single strong learner.**

#### Support Vector Machine (SVM)

In an SVM algorithm, we plot each data item as a point in n-dimensional space (where n is number of features you have) with the value of each feature being the value of a particular coordinate. Then, we perform classification by finding the hyper-plane that differentiate the two classes very well. The hyper-plane is the line or plane that separates the two classes.

**It is mandatory to scale the features before training the SVM algorithm.**

#### K-Nearest Neighbors (KNN)

Classifies a data point based on how its neighbors are classified. The data point is assigned to the class that is most common among its k nearest neighbors. The value of k is a positive integer, typically small.

The k value is used to determine the number of nearest neighbors to use for voting. A large k value means that more neighbors will be used in the majority vote, which means that the data point is more likely to be assigned to a class that is not the majority class in the training set.

- Use cases:
  - When the data is labeled.
  - Noise free data.
  - When the data is not too large.

#### Naive Bayes

Bayes' theorem provides a way of calculating posterior probability P(c|x) from P(c), P(x), and P(x|c). Bayes' theorem is stated mathematically as the following equation:

$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

Basically, Bayes' theorem calculates the conditional probability of the occurrence of an event, based on prior knowledge of conditions that might be related to the event.

#### Logistic Regression

This regression is used when the **dependent variable is categorical**. It is used to estimate the probability of a binary outcome (1 / 0, Yes / No, True / False) given a set of independent variables.

---

## Boosting Algorithms

Using a set of learning algorithms to improve the predictive power of the model.

### AdaBoost

AdaBoost is a boosting algorithm that combines multiple weak learners into a single strong learner. AdaBoost works by choosing a base algorithm (e.g. decision trees) and iteratively improving it by accounting for the incorrectly classified examples in the training set.

### Gradient Boosting (MOST USED)

Gradient boosting is a boosting algorithm that combines multiple weak learners into a single strong learner in an iterative fashion. It is a generalization of boosting to arbitrary differentiable loss functions.

- One of the most powerful techniques for building predictive models.

### Others

- XGBoost
- LightGBM
- CatBoost
- Voting Classifier

---

## Enhancing Performance

### Correlation

Correlation is a statistical measure that expresses the extent to which two variables are linearly related. The correlation coefficient is a value between -1 and 1 inclusive, where:

- 1: Total positive linear correlation.
- 0: No linear correlation, the two variables most likely do not affect each other.

Example:

```python
import pandas as pd

df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [5, 6, 7, 8, 9]
})

print(df.corr())
```

### Feature Importance

Feature importance refers to techniques that assign a score to input features based on how useful they are at predicting a target variable.

Every model in scikit-learn has a `feature_importances` attribute that can be used to access a feature's importance relative to the other features. The higher the value, the more important the feature is.

Example:

```python
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# Get numerical feature importances
importances = list(rf.feature_importances_)
```

## Defining Accuracy

### Root Mean Squared Error (RMSE)

The root mean squared error (RMSE) is the square root of the mean of the squared errors. It is calculated as:

```math
RMSE = sqrt(1/n * sum((y - y_pred)^2))
```

### Mean Absolute Error (MAE)

The mean absolute error (MAE) is the mean of the absolute value of the errors. It is calculated as:

```math
MAE = 1/n * sum(|y - y_pred|)
```

### R-Squared

The R-squared is a statistical measure of how close the data are to the fitted regression line. It is also known as the coefficient of determination, or the coefficient of multiple determination for multiple regression. It is calculated as:

```math
R^2 = 1 - (SSres/SStot)
```

- **SSres:** The sum of the squared residuals. The squared residuals are calculated by subtracting the predicted value from the observed value and squaring the difference.
- **SStot:** The total sum of squares. The total sum of squares is calculated by subtracting the mean of the observed values from the observed values and squaring the difference.

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

### Precision and Recall

**Precision:** The ability of a classifier not to label an instance positive that is actually negative. From all the data, how many did we label correctly?

- Formula: `Precision = TP / (TP + FP)`

**Recall:** The ability of a classifier to find all positive instances. From all the positive instances, how many did we label correctly?

- Formula: `Recall = TP / (TP + FN)`

It is important to note that precision and recall are often in conflict. Increasing precision reduces recall, and vice versa.

- Precision = `TP / (TP + FP)` = `10 / (10 + 5)` = `0.67`
- Recall = `TP / (TP + FN)` = `10 / (10 + 2)` = `0.83`

### Sensitivity and Specificity

**Sensitivity:** The ability of a test to correctly identify positive results. From all the positive instances, how many did we label correctly?

- Formula: `Sensitivity = TP / (TP + FN)`

**Specificity:** The ability of a test to correctly identify negative results. From all the negative instances, how many did we label correctly?

- Formula: `Specificity = TN / (TN + FP)`

---

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

## Overfitting and Underfitting

- **Overfitting:** Overfitting occurs when a model fits the training data too well. The model is too complex and does not generalize well to new data. Overfitting is a problem because the model will have a high accuracy on the training data, but a low accuracy on the test data.
- **Underfitting:** Underfitting occurs when a model is not complex enough to capture the underlying structure of the data. The model is too simple and does not generalize well to new data. Underfitting is a problem because the model will have a low accuracy on both the training data and the test data.

---

# Statistics

## Distributions

### Normal Distribution

[Normal distribution](https://en.wikipedia.org/wiki/Normal_distribution) is a continuous probability distribution. It is also called **Gaussian distribution**. It is a bell-shaped curve.

![Normal Distribution](https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Normal_Distribution_PDF.svg/1280px-Normal_Distribution_PDF.svg.png)

### Skewed Distribution

A [skewed distribution](https://en.wikipedia.org/wiki/Skewness) is a distribution that is not symmetrical. It is a result of outliers.

![Skewed Distribution](https://upload.wikimedia.org/wikipedia/commons/thumb/c/cc/Relationship_between_mean_and_median_under_different_skewness.png/1280px-Relationship_between_mean_and_median_under_different_skewness.png)

### Binomial Distribution

[Binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution) is a discrete probability distribution. It is a result of a binary outcome.

![Binomial Distribution](https://upload.wikimedia.org/wikipedia/commons/thumb/7/75/Binomial_distribution_pmf.svg/1280px-Binomial_distribution_pmf.svg.png)

### Poisson Distribution

[Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution) is a discrete probability distribution. It is a result of a count of events.

![Poisson Distribution](https://upload.wikimedia.org/wikipedia/commons/thumb/1/16/Poisson_pmf.svg/1280px-Poisson_pmf.svg.png)

### Uniform Distribution

[Uniform distribution](<https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)>) is a continuous probability distribution. It is a result of a random experiment.

![Uniform Distribution](https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Uniform_Distribution_PDF_SVG.svg/1280px-Uniform_Distribution_PDF_SVG.svg.png)

### Exponential Distribution

[Exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution) is a continuous probability distribution. It is a result of a Poisson process.

![Exponential Distribution](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Exponential_pdf.svg/1280px-Exponential_pdf.svg.png)

### Bernoulli Distribution

[Bernoulli distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution) is a discrete probability distribution. It is a result of a binary outcome.

![Bernoulli Distribution](<https://probabilitycourse.com/images/chapter3/geometric(p=0.3)%20color.png>)

## Hypothesis Testing

### What is hypothesis testing?

Hypothesis testing is a statistical method that is used in making statistical decisions using experimental data. Hypothesis testing is used to evaluate the results of an experiment and draw conclusions.

### What is a hypothesis?

A hypothesis is a statement about a population parameter. The hypothesis is assumed to be true until proven otherwise.

### T Test

H<sub>0</sub>: The null hypothesis is a statement of "no difference," "no effect," or "no association" between two sets of data. It is the hypothesis that the researcher is trying to disprove.

H<sub>a</sub>: The alternative hypothesis is a statement of "difference," "effect," or "association" between two sets of data. It is the hypothesis that the researcher is trying to prove.

The objective of the t-test is to provide evidence **against the null hypothesis**. The t-test is used to determine whether the null hypothesis should be rejected in favor of the alternative hypothesis.
