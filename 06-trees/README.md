# Machine Learning (ML) Decision Trees and Ensemble Learning

### Decision Trees:
Decision trees are a versatile tool in machine learning for classification and regression tasks. They mimic human decision-making by creating a flowchart-like structure to make predictions based on input features.

#### Purpose:
- Decision trees are intuitive and easy to interpret.
- They can handle both categorical and numerical data.
- Efficiently handles feature selection, requiring minimal data preparation.

#### Python Example:
```python
# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset (e.g., credit risk data)
data = datasets.load_iris()
X, y = data.data, data.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Decision Tree model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict using the model
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
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