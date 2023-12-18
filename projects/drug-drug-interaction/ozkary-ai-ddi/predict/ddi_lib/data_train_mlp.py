#!/usr/bin/env python
# coding: utf-8
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout
# from keras.utils import vis_utils
from keras.utils import plot_model
from keras.optimizers import SGD
from keras.callbacks import History
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import the DDI custom library
from ddi_lib import DDIProcessData

# define a python class to build the MLP model using Keras for the DDI dataset

class DDIMLPFactory:
    """
    Build the MLP model using Keras
    """
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes     
        self.model = None   
            
    def build_model1(self):
        """ 
        Build the MLP model using Keras
        """
        model = Sequential([
            Dense(64, activation='relu', input_shape=(self.input_shape,)),
            Dropout(0.2),
            Dense(self.num_classes, activation='softmax')
        ])                
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def build_model2(self):
        model = Sequential([
            Dense(128, activation='relu', input_shape=(self.input_shape,)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(self.num_classes, activation='softmax')
        ])        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def build_model3(self):
        model = Sequential([
            Dense(256, activation='relu', input_shape=(self.input_shape,)),            
            Dense(128, activation='relu'),
            Dense(128,activation='tanh'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(64, activation='tanh'),
            Dropout(0.2),            
            Dense(self.num_classes, activation='softmax')
        ])        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model


    def train(self, model, X_train, y_train, epochs, X_val, y_val):
        """
        Train the model using training data and return the history
        """
        history = History()        
        model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), callbacks=[history])
        return history

    def evaluate(self,model, X_test, y_test):
        """ 
        Evaluate the model using test data
        """
        return model.evaluate(X_test, y_test, verbose=0)

    def predict(self,model, data):
        """ 
        Predict DDI using the trained model
        """
        return model.predict(data)

    def plot_accuracy(self, name, history):
        """ 
        Plot the accuracy of the model
        """        
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'{name} Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(f'./images/{name}_accuracy.png')

    def plot_loss(self,name, history):
        """ 
        Plot the loss of the model
        """        
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'{name} Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(f'./images/{name}_loss.png')        

    def get_models_results(self, hist):
        # display the models accuracy and loss
        results = {}

        # build the three models dictionary
        for i in range(1,4):        
            results['accuracy'] = round(hist.history['accuracy'][-1],4)
            results['val_accuracy'] = round(hist.history['val_accuracy'][-1],4)
            results['loss'] = round(hist.history['loss'][-1],4)
            results['loss_accuracy'] = round(hist.history['val_loss'][-1],4)

        return results 
    
    def save_model(self, model, file_path):
        model.save(file_path)
    
    def clear_session(self):        
        K.clear_session()

# define a python class to process the DDI dataset
class DDIProcessor:
    def __init__(self, file_path, label):
        self.file_path = file_path
        self.label = label
        self.df = self.load_df()        

    def load_df(self):        
        print('Must run data_analysis.ipynb first to generate the csv file')
        df = pd.read_csv(self.file_path, compression='gzip')
        print(df.info())        
        return df

# all the code below is executed when this file is run add a main method
if __name__ == "__main__":
    os.system('clear')    
    print ('Training MLP models for DDI prediction')

    # load the DDI dataset
    target_variable = 'interaction_type'
    processor = DDIProcessor('./data/ssp_interaction_type.csv.gz', target_variable)

    # process the features
    df = processor.df

    # create an instance of the DDIProcessData class to process the data
    train_data = DDIProcessData(df, target_variable=target_variable)

    # get the features and target series
    cat_features, num_features = train_data.process_features()

    # split the data in train/val/test sets
    # use 60%/20%/20% distribution with seed 1
    # use stratified sampling to ensure the distribution of the target feature is the same in all sets
    X_train, X_val, y_train, y_val, X_test, y_test = train_data.split_data(test_size=0.2, random_state=42)
    y_train_encoded = train_data.preprocess_target(y_train)
    y_val_encoded = train_data.preprocess_target(y_val)
    y_test_encoded = train_data.preprocess_target(y_test)

    # get the data shape and number of classes
    input_shape = X_train.shape[1]  
    num_classes = df[target_variable].nunique()
    num_epochs = 5

    print ('input_shape: ', input_shape)
    print ('num_classes: ', num_classes)
    print ('num_epochs: ', num_epochs)

    # create an instance of the DDICNNFactory class to build the models
    model_factory = DDIMLPFactory(input_shape, num_classes)

    # Build and train model 1
    model1 = model_factory.build_model1()
    # plot_model(model1, to_file='./images/ozkary-mlp-neural-network1.png', show_shapes=True, show_layer_names=True)
    hist1 = model1.fit(X_train, y_train_encoded, epochs=num_epochs, validation_data=(X_val, y_val_encoded))

    print ('Model 1 - Training Accuracy: ', round(hist1.history['accuracy'][-1], 4))
    print ('Model 1 - Validation Accuracy: ', round(hist1.history['val_accuracy'][-1],4))
        
    # Build and train model 2
    model2 = model_factory.build_model2()
    # plot_model(model2, to_file='./images/ozkary-mlp-neural-network2.png', show_shapes=True, show_layer_names=True)
    hist2 = model2.fit(X_train, y_train_encoded, epochs=num_epochs, validation_data=(X_val, y_val_encoded))

    # Build and train model 3
    model3 = model_factory.build_model3()
    # plot_model(model3, to_file='./images/ozkary-mlp-neural-network3.png', show_shapes=True, show_layer_names=True)
    hist3 = model3.fit(X_train, y_train_encoded, epochs=num_epochs, validation_data=(X_val, y_val_encoded))
    
    model_results = {}
    model_results['Model 1'] = model_factory.get_models_results(hist1)
    model_results['Model 2'] = model_factory.get_models_results(hist2)
    model_results['Model 3'] = model_factory.get_models_results(hist3)

    # print the summary of the models results in a tabular format
    df_results = pd.DataFrame.from_dict(model_results, orient='index')
    print(df_results)

    # plot the accuracy and loss of all models using df_results
    df_results.plot(kind='bar', y=['accuracy', 'val_accuracy'], title='Accuracy')
    df_results.plot(kind='bar', y=['loss', 'loss_accuracy'], title='Loss')

    # Evaluate the models
    results_model1 = model_factory.evaluate(model1, X_test, y_test_encoded)
    results_model2 = model_factory.evaluate(model2, X_test, y_test_encoded)
    results_model3 = model_factory.evaluate(model3, X_test, y_test_encoded)

    # Extract metrics (accuracy, loss, etc.) for each model
    accuracy_values = [results_model1[1], results_model2[1], results_model3[1]]
    loss_values = [results_model1[0], results_model2[0], results_model3[0]]

    # Plot accuracy
    plt.figure(figsize=(6, 3))
    plt.bar(['Model 1', 'Model 2', 'Model 3'], accuracy_values, color=['blue', 'green', 'orange'])
    plt.title('Models Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)  # Set the y-axis limit to match the accuracy range (0 to 1)
    plt.show()

    # Plot loss
    plt.figure(figsize=(6, 3))
    plt.bar(['Model 1', 'Model 2', 'Model 3'], loss_values, color=['blue', 'green', 'orange'])
    plt.title('Model Loss Comparison')
    plt.ylabel('Loss')
    plt.show()

    plot_model(model1, to_file='./images/ozkary-mlp-neural-network1.png', show_shapes=True, show_layer_names=True)
    plot_model(model2, to_file='./images/ozkary-mlp-neural-network2.png', show_shapes=True, show_layer_names=True)
    plot_model(model3, to_file='./images/ozkary-mlp-neural-network3.png', show_shapes=True, show_layer_names=True)

    # plot model 3 accuracy and loss
    model_factory.plot_accuracy('ozkary-mlp-model3', hist3)
    model_factory.plot_loss('ozkary-mlp-model3', hist3)

    # save the model
    model_factory.save_model(model3, './models/ozkary-ddi.h5')

