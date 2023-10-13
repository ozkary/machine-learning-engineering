# app.py - Your Flask application

from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)

# Load the DictVectorizer
with open('dv.bin', 'rb') as f:
    dv = pickle.load(f)

file_path = 'model1.bin' if os.path.exists('model1.bin') else 'model2.bin' 

# Load the Logistic Regression model
with open(file_path, 'rb') as f:
    model = pickle.load(f)

# Define the prediction endpoint and route
@app.route('/predict', methods=['POST'])
def predict():
    # get the json payload
    data = request.get_json()
    print("data",data)
    
    # Transform the new data using the DictVectorizer
    transformed_data = dv.transform([data])

    # Process the data, make predictions using the model, and return the results
    probabilities = model.predict_proba(transformed_data)

    # break down the probabilities into yes or no score
    no_score, yes_score = probabilities[0]

    # get the class labels from the model
    no_label, yes_label = model.classes_    
    return jsonify({'yes': yes_score, 'no': no_score})
    

# load the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
