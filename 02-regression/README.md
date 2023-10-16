# Machine Learning (ML) Regression

Regression is a fundamental technique in machine learning used for predicting continuous outcomes. It's widely used in various domains, such as finance, healthcare, and economics, to forecast trends, analyze relationships, and make predictions based on input variables.

In regression, the goal is to find the best-fitting line or curve that describes the relationship between input features (independent variables) and the target variable (dependent variable). Linear regression is one of the simplest and widely used regression techniques, aiming to fit a linear equation to the data.

## Regression Using Linear Regression: A Step-by-Step Process

### 1. Prepare the Data and Perform Exploratory Data Analysis (EDA):
   - Clean the data, handle missing values, and preprocess features.
   - Explore and understand the data through statistical analysis, visualization, and summary statistics.

### 2. Use Linear Regression to Predict the Target (Price in this example):
   - Select the features (independent variables) and the target variable (e.g., house price).
   - Split the data into training and testing sets.
   - Train a linear regression model on the training data.

### 3. Internal Workings of Linear Regression:
   - Linear regression fits a line (in simple linear regression) or a hyperplane (in multiple linear regression) to minimize the sum of squared differences between the observed and predicted values.
   - It uses techniques like Ordinary Least Squares (OLS) to estimate the model parameters (coefficients) that define the line.

### 4. Evaluate the Model using Root Mean Squared Error (RMSE):
   - RMSE measures the average error between the observed and predicted values.
   - Lower RMSE indicates a better fit of the model to the data.

### 5. Feature Engineering:
   - Enhance the model's predictive power by creating new features or transforming existing ones.
   - Feature engineering may involve scaling, binning, one-hot encoding, or extracting useful information from raw data.

### 6. Regularization (Optional):
   - Implement regularization techniques like Lasso (L1) or Ridge (L2) regression to prevent overfitting and improve model generalization.
   - Regularization adds a penalty to the model parameters to avoid excessively large coefficients.

### 7. Use the Model for Predictions:
   - Apply the trained model to make predictions on new, unseen data.
   - Use the model to forecast prices based on the chosen features.

By following this step-by-step process, you can effectively use linear regression for prediction, understand its internal mechanisms, evaluate its performance, and enhance it through feature engineering and regularization.

## Example: Predicting House Prices

Here's a Python example for each step of the process using a simple linear regression to predict house prices:

### Prepare the Data and Perform Exploratory Data Analysis (EDA):

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load and explore the dataset
df = pd.read_csv('housing.csv')

# Display the first few rows of the dataset
print(df.head())

# Visualize data
sns.pairplot(df, x_vars=['area', 'bedrooms'], y_vars='price', height=5, aspect=1)
plt.show()
```

### Use Linear Regression to Predict the Target (Price in this example):

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Prepare the data
X = df[['area', 'bedrooms']]
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
```

### Evaluate the Model using Root Mean Squared Error (RMSE):

```python
# Predict using the model
y_pred = model.predict(X_test)

# Evaluate the model
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('Root Mean Squared Error:', rmse)
```

### Feature Engineering:

In this simple example, we're using the existing features 'area' and 'bedrooms'.

### Regularization (Optional):

Regularization helps prevent overfitting by adding a penalty term to the model parameters (coefficients) during the training process. This penalty discourages overly complex models with large coefficients. In linear regression, two common types of regularization are Lasso (L1 regularization) and Ridge (L2 regularization).

Here's how you could apply Ridge regularization to the linear regression model:

```python
from sklearn.linear_model import Ridge

# Train the Ridge regression model with regularization (alpha is the regularization parameter)
ridge_model = Ridge(alpha=1.0)  # You can adjust the alpha value
ridge_model.fit(X_train, y_train)

# Evaluate the Ridge model
ridge_y_pred = ridge_model.predict(X_test)
ridge_rmse = mean_squared_error(y_test, ridge_y_pred, squared=False)
print('Ridge Regression RMSE:', ridge_rmse)
```

In a real-world scenario, when dealing with more complex data or when you observe overfitting in your model, you would typically experiment with both Lasso and Ridge regularization techniques to find an optimal value for the regularization parameter (alpha) that balances model complexity and performance. Regularization is a crucial tool in your toolkit for robust and stable model training in machine learning.

### Use the Model for Predictions:

```python
# Predict house prices for new data
new_data = pd.DataFrame({'area': [1500, 2000], 'bedrooms': [3, 4]})
predicted_prices = model.predict(new_data)
print('Predicted Prices:', predicted_prices)
```

Use the trained linear regression model to predict house prices for new, unseen data. Let's break down the code step by step:

**Creating New Data:**
   ```python
   new_data = pd.DataFrame({'area': [1500, 2000], 'bedrooms': [3, 4]})
   ```
   - Here, a new DataFrame named `new_data` is created using pandas. It has two rows, each representing a new house's features (area and bedrooms). For demonstration purposes, we have two hypothetical houses with different areas and bedrooms.

**Predicting House Prices:**
   ```python
   predicted_prices = model.predict(new_data)
   ```
   - The `predict()` method of the trained linear regression model (`model`) is used to predict the house prices for the new data (`new_data`). The `predict()` method takes the features of the new data and returns the predicted prices based on the trained model.

**Printing Predicted Prices:**
   ```python
   print('Predicted Prices:', predicted_prices)
   ```
   - Finally, the predicted house prices are printed to the console. The `predicted_prices` variable holds the predicted prices for the new houses based on the features provided in `new_data`.

This example demonstrates a basic implementation of linear regression for predicting house prices using two features: 'area' and 'bedrooms'.