# import libraries for web API support
from flask import Flask, request, jsonify
import sklearn
import json
# import the data prediction module
from data_predict import predict, probability_label

# create a Flask app instance
app = Flask(__name__)

VERSION = '1.0.0'
LABEL = 'Heart Disease Risk Prediction API'

# define the root endpoint
@app.route('/', methods=['GET'])
# define the root endpoint function
def root():
    return f'{LABEL} {VERSION}'

print(f"Loading {LABEL} {VERSION}")
print("sklearn version", sklearn.__version__)

# define the predict endpoint
@app.route('/predict', methods=['POST'])
# define the predict endpoint function
def predict_endpoint():
    print("Predict endpoint called")
    # get the request body
    data = request.get_json()

    # Parse the JSON string into a dictionary
    data_dict = json.loads(data)

    # get the prediction
    predictions = predict(data_dict)    

    # create a risk_score based on the prediction
    results = []

    # for each prediction, add the risk_score and risk_label
    for score in predictions:
        # create a risk_score based on the prediction
        risk_score = round(score,4)
        risk_label = probability_label(score)
        results.append({'risk_score': risk_score, 'risk_label': risk_label})
    
    # return the prediction
    return jsonify(results)

# load the application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)

