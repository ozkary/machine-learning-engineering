# import libraries for web API support
from flask import Flask, request, jsonify

# import the data prediction module
from data_predict import predict

# create a Flask app instance
app = Flask(__name__)

# define the root endpoint
@app.route('/', methods=['GET'])
# define the root endpoint function
def root():
    return 'Heart Disease Risk Prediction API'

# define the predict endpoint
@app.route('/predict', methods=['POST'])
# define the predict endpoint function
def predict_endpoint():
    # get the request body
    request_body = request.get_json()
    # get the prediction
    prediction = predict(request_body)
    # return the prediction
    return jsonify(prediction)

# load the application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)

