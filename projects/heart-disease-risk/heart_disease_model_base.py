from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split

class HeartDiseaseModelBase(ABC):
    """
    Abstract class for heart disease risk prediction model    
    """    

    def __init__(self):
        # Common initialization code
        pass
    
    def split_data(self, df, features, test_size=0.2, random_state=42, stratify=None):
        """
        Split the data into training and validation sets
        """
        # split the data in train/val/test sets, with 60%/20%/20% distribution with seed 1
        X = df[features]
        y = df[self.target_variable]
        X_full_train, X_test, y_full_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

        # .25 splits the 80% train into 60% train and 20% val
        X_train, X_val, y_train, y_val  = train_test_split(X_full_train, y_full_train, test_size=0.25, random_state=random_state)

        X_train = X_train.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)

        return X_train, X_val, X_test

    @abstractmethod
    def preprocess_data(self, data, is_training=True):
        """
        Preprocess the data and return the standardized features and target variable
        """
        pass

    @abstractmethod
    def train(self, df_train):
        """
        Train the model on the training data set
        """
        pass

    @abstractmethod
    def evaluate(self, df_val, threshold=0.5):
        """
        Evaluate the model on the validation data set and return the accuracy, precision, recall, and F1 score
        """
        pass
    
    def predict(self, df_val):
        """
        Predict the target variable on the validation data set and return the predictions
        """
        X_val, _ = self.preprocess_data(df_val, is_training=False)
        probs = self.model.predict_proba(X_val)
        return probs
