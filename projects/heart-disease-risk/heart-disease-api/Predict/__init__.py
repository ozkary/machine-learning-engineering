# import libraries for web API support
import azure.functions as func
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
@app.route('api/', methods=['GET'])
# define the root endpoint function
def root():
    return f'{LABEL} {VERSION}'

print(f"Loading {LABEL} {VERSION}")
print("sklearn version", sklearn.__version__)

# define the predict endpoint
@app.route('api/predict', methods=['POST'])
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


def main(req: func.HttpRequest) -> func.HttpResponse:
    context = app.test_request_context('/api' + request.url[len(req.url):])
    with context:
        app.preprocess_request()
        response = app.full_dispatch_request()
        response.set_data(response.get_data(as_text=True))
        return func.HttpResponse(
            response.get_data(),
            status_code=response.status_code,
            mimetype=response.mimetype,
            headers=response.headers
        )
