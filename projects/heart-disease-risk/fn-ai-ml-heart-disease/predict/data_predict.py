#!/usr/bin/env python
# coding: utf-8

# Heart Disease Risk Analysis Data - Predicting Heart Disease
import os
import pickle

# Determine the path to the model file relative to the script location and load the models
folder = os.path.dirname(__file__)
model_filename = os.path.join(folder, 'hd_xgboost_model.pkl')
dv_filename = os.path.join(folder, 'hd_dictvectorizer.pkl')

# Load the model and dv from the files
with open(model_filename, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open(dv_filename, 'rb') as dv_file:
    loaded_dv = pickle.load(dv_file)

def probability_label(probability):
    
    labels = ['none','low', 'medium', 'high']
    label = 'unknown'

    # return the label based on the probability
    if probability < 0.3:
        label = labels[0]
    elif probability < 0.50:
        label = labels[1]
    elif probability < 0.75:
        label = labels[2]
    elif probability >= 0.75:
        label = labels[3]
    
    return label

def predict(data):
    # Transform the data
    X = loaded_dv.transform(data)
    # Predict the probability
    y_pred = loaded_model.predict_proba(X)[:, 1]
    
    return y_pred

