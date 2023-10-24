from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from heart_disease_model_base import HeartDiseaseModelBase

class HeartDiseaseLogisticRegression(HeartDiseaseModelBase):
    
    def __init__(self, numeric_features, categorical_features, target_variable):
        # Initialize the preprocessing transformers
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse=False)
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.target_variable = target_variable

        # Define the preprocessing steps
        numeric_transformer = Pipeline(steps=[('scaler', self.scaler)])
        categorical_transformer = Pipeline(steps=[('encoder', self.encoder)])
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])

        # Initialize the logistic regression model
        self.model = LogisticRegression()

    def __str__(self):
        return "HeartDiseaseLogisticRegression"

    def split_data(self, df, features, test_size=0.2, random_state=1):
        return super().split_data(df, features)

    def preprocess_data(self, data, is_training=True):
        # Separate the target variable (e.g., 'heartdisease')
        y = data[self.target_variable]

        # Select only the specified features from the data frame
        X = data[self.numeric_features + self.categorical_features]

        if is_training:
            # Fit and transform for training data
            X_std = self.preprocessor.fit_transform(X)
        else:
            # Only transform for validation data
            X_std = self.preprocessor.transform(X)

        # Return the standardized features and target variable
        return X_std, y

    def train(self, df_train):
        # Preprocess training data
        X_train, y_train = self.preprocess_data(df_train, is_training=True)

        # Train the logistic regression model
        self.model.fit(X_train, y_train)

    def evaluate(self, df_val):
        # Preprocess validation data
        X_val, y_val = self.preprocess_data(df_val, is_training=False)

        y_pred = self.model.predict_proba(X_val)
        # Evaluate the model
        accuracy = self.model.score(X_val, y_val)        
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)

        # Return a dictionary of metrics
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        return metrics
    
    def predict(self, df_val):
        return super().predict(df_val)
