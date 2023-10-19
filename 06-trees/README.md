# Machine Learning (ML) Decision Trees and Ensemble Learning

### Decision Trees:
Decision trees are a versatile tool in machine learning for classification and regression tasks. They mimic human decision-making by creating a flowchart-like structure to make predictions based on input features.

#### Purpose:
- Decision trees are intuitive and easy to interpret.
- They can handle both categorical and numerical data.
- Efficiently handles feature selection, requiring minimal data preparation.

The random_state parameter in the train_test_split function is used to seed the random number generator that the function uses for shuffling the data and splitting it into training and validation sets. Setting a specific value for random_state ensures that the random split is reproducible.

If you set random_state to a fixed integer (e.g., random_state=1), the data will be shuffled and split in the same way every time you run the code with the same seed. This is useful for reproducibility and allows you to obtain consistent results across different runs.

#### Python Example:
```python
# Import necessary libraries
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction import DictVectorizer

# Load the housing dataset
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
df = pd.read_csv(url)

# Data Preprocessing
# Fill missing values with zeros
df.fillna(0, inplace=True)

# Apply log transform to median_house_value
df['median_house_value'] = np.log1p(df['median_house_value'])

# Split the data into train/val/test sets with 60%/20%/20% distribution
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

# Separate the target variable from the train/val/test sets
y_train = df_train['median_house_value'].values
y_val = df_val['median_house_value'].values

# Create a DictVectorizer to transform data
dv = DictVectorizer(sparse=True)
train_dict = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

# Train a DecisionTreeRegressor model
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)

# Identify the feature used for splitting the data
feature_names = dv.get_feature_names_out(input_features=df_train.columns)
split_feature_idx = dt_model.tree_.feature[0]
split_feature = feature_names[split_feature_idx]

print(f"Feature used for splitting: {split_feature}")

```

### Ensemble Learning (Random Forest):
Random Forest is an ensemble learning method that combines multiple decision trees to improve predictive performance and control overfitting.

#### Purpose:
- Provides higher accuracy and generalization by aggregating predictions from multiple trees.
- Reduces the risk of overfitting compared to a single decision tree.

#### Python Example (using Random Forest for credit risk analysis):
```python
# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load your credit risk data and split into features (X) and labels (y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predict using the model
y_pred_rf = rf_model.predict(X_test)

# Calculate accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy (Random Forest): {accuracy_rf}")

# Additional evaluation (classification report)
print("Classification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))
```

In this example, we've used a Random Forest classifier for credit risk analysis. The model is trained on the training set and evaluated on the test set, providing accuracy and a detailed classification report.

Feel free to replace the dataset with your specific credit risk data and adapt the features and labels accordingly.

#### Estimators

In models like Random Forest, the term "estimators" refers to individual decision trees within the ensemble. Each estimator is a separate decision tree model that is trained on a subset of the data, using a random subset of features (known as feature bagging) to reduce overfitting and improve generalization.

The purpose of having multiple estimators in an ensemble, such as a Random Forest, is to improve the overall model's performance by aggregating the predictions from multiple individual models. This provides several advantages: