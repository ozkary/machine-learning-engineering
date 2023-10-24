from heart_disease_model_base import HeartDiseaseModelBase

class HeartDiseaseRandomForest(HeartDiseaseModelBase):
    def __init__(self, numeric_features, categorical_features):
        # ... (your class implementation)
        raise NotImplementedError

    def preprocess_data(self, data, is_training=True):
        # ... (your implementation)
        raise NotImplementedError

    def train(self, df_train):
        # ... (your implementation)
        raise NotImplementedError
    

    def evaluate(self, df_val):
        # ... (your implementation)
        raise NotImplementedError
