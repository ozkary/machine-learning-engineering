import logging
import os
import azure.functions as func
import sklearn
import json
from .ddi_lib import predict

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Predict HTTP trigger function processed a request.')

    try:
        data = req.get_json()
        if data:
            path = os.path.abspath(os.path.dirname(__file__))                      
            # Make predictions
            predictions = predict(json.loads(data), path)

            # Return the predictions as a JSON response
            return func.HttpResponse(json.dumps(predictions), mimetype="application/json")
        else:
            return func.HttpResponse("Invalid input data.", status_code=400)
    except Exception as e:
        return func.HttpResponse(f"An error occurred: {str(e)}", status_code=500)
