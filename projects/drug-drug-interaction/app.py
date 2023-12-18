# import libraries for web API support
from flask import Flask, request, jsonify
import sklearn
import json
# import the data prediction module
from ddi_lib import predict

# create a Flask app instance
app = Flask(__name__)

VERSION = '1.0.0'
LABEL = 'Drug to Drug Interaction Prediction API'

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
    results = predict(data_dict)    
    
    # return the prediction
    return jsonify(results)

# load the application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
