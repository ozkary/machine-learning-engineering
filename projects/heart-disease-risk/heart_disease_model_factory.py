from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class HeartDiseaseTrainData():
    """
    Data class for training data
    """
    def __init__(self, df, target_variable):
        self.df = df        
        self.target_variable = target_variable

        # get the numeric and categorical features
        self.numeric_features = None
        self.categorical_features = None
        
        # list of all features
        self.all_features = None
        
    def process_features(self):
        """
        Process the features
        """
         # get the numeric and categorical features
        self.numeric_features = list(self.df.select_dtypes(include=[np.number]).columns)
        self.categorical_features = list(self.df.select_dtypes(include=['object']).columns)

        # remove the target feature from the list of numeric features
        if self.target_variable in self.numeric_features:
            self.numeric_features.remove(self.target_variable)

        print('Categorical features',self.categorical_features)
        print('Numerical features',self.numeric_features)
        print('Target feature',self.target_variable)
        
        # create a list of all features
        self.all_features = self.categorical_features + self.numeric_features
        
        return  self.categorical_features, self.numeric_features


    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the data into training and validation sets
        """
        # split the data in train/val/test sets, with 60%/20%/20% distribution with seed 1
        X = self.df[self.all_features]
        y = self.df[self.target_variable]
        X_full_train, X_test, y_full_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

        # .25 splits the 80% train into 60% train and 20% val
        X_train, X_val, y_train, y_val  = train_test_split(X_full_train, y_full_train, test_size=0.25, random_state=random_state)

        X_train = X_train.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        # print the shape of all the data splits
        print('X_train shape', X_train.shape)
        print('X_val shape', X_val.shape)
        print('X_test shape', X_test.shape)
        print('y_train shape', y_train.shape)
        print('y_val shape', y_val.shape)
        print('y_test shape', y_test.shape)
        
        return X_train, X_val, y_train, y_val, X_test, y_test
    

class HeartDiseaseModelFactory():
    """
    Factory class for heart disease risk prediction model    
    """    

    def __init__(self, categorical_features, numeric_features):
        # Initialize the preprocessing transformers
        self.scaler = StandardScaler()        
        self.encoder = DictVectorizer(sparse=False)

        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        
        # Define the preprocessing steps
        numeric_transformer = Pipeline(steps=[('scaler', self.scaler)])
        categorical_transformer = Pipeline(steps=[('encoder', self.encoder)])
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        self.models = None
        self.model = None

    def preprocess_data(self, X, is_training=True):      
        """
        Preprocess the data for training or validation
        """  
        X_dict = X.to_dict(orient='records')
        # X_num = X[self.numeric_features]

        # processor = self.preprocessor        
        
        if is_training:
            X_std = self.encoder.fit_transform(X_dict)
            # Fit and transform for training data
            # X_cat_std = processor.fit_transform(X_dict)
            # X_num_std = processor.fit_transform(X_num)
            # X_std = np.concatenate((X_num_std, X_cat_std), axis=1)
        else:
            X_std = self.encoder.transform(X_dict)
            # Only transform for validation data
            #  X_cat_std = processor.transform(X_dict)
            #  X_num_std = processor.transform(X_num)
            #  X_std = np.concatenate((X_num_std, X_cat_std), axis=1)

        # Return the standardized features and target variable
        return X_std

    def train(self, X_train, y_train):
        
        if self.models is None:
            self.models = {
                'logistic_regression': LogisticRegression(C=10, max_iter=1000, random_state=42),
                'random_forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
                'xgboost': XGBClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),                
                'decision_tree': DecisionTreeClassifier(max_depth=5, random_state=42)
            }
        
        for model in self.models.keys():
            print('Training model', model)
            self.models[model].fit(X_train, y_train)            

    def evaluate(self, X_val, y_val, threshold=0.5):
        """
        Evaluate the model on the validation data set and return the predictions
        """

        # create a dataframe to store the metrics
        df_metrics = pd.DataFrame(columns=['model', 'accuracy', 'precision', 'recall', 'f1', 'y_pred'])

        # define the metrics to be calculated
        fn_metrics = { 'accuracy': accuracy_score,'precision': precision_score,'recall': recall_score,'f1': f1_score}

        # loop through the models and get its metrics
        for model_name in self.models.keys():
            
            model = self.models[model_name]

            # The first column (y_pred_proba[:, 0]) is for class 0 ("N")
            # The second column (y_pred_proba[:, 1]) is for class 1 ("Y")            
            y_pred = model.predict_proba(X_val)[:,1]
            # get the binary predictions
            y_pred_binary = np.where(y_pred > threshold, 1, 0)

            # add a new row to the dataframe for each model            
            df_metrics.loc[len(df_metrics)] = [model_name, 0, 0, 0, 0, y_pred_binary]

            # get the row index
            row_index = len(df_metrics)-1

            # Evaluate the model metrics
            for metric in fn_metrics.keys():
                score = fn_metrics[metric](y_val, y_pred_binary)
                df_metrics.at[row_index,metric] = score
           
        return df_metrics

    def save(model_name, path):
        """
        Save the model
        """
        # get the model from the models dictionary
        model = self.models[model_name]

        if model is None:
            print('Model not found')
            return
            
        # save the model
        model.save(path)

            
    def predict(self, X_val):
        """
        Predict the target variable on the validation data set and return the predictions
        """        
        probs = self.model.predict_proba(X_val)
        return probs
