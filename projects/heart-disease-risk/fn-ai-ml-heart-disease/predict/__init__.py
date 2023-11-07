# import libraries for web API support
import azure.functions as func
import sklearn
import json
# import the data prediction module
from .data_predict import predict, probability_label

VERSION = '1.0.0'
LABEL = 'Heart Disease Risk Prediction API'

print(f"Loading {LABEL} {VERSION}")
print("sklearn version", sklearn.__version__)

# define the predict endpoint function
def predict_cases(data):
    print("Predict endpoint called")
    
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
    return results


def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        data = req.get_json()
        if data:
            # Make predictions
            predictions = predict_cases(data)

            # Return the predictions as a JSON response
            return func.HttpResponse(json.dumps(predictions), mimetype="application/json")
        else:
            return func.HttpResponse("Invalid input data.", status_code=400)
    except Exception as e:
        return func.HttpResponse(f"An error occurred: {str(e)}", status_code=500)