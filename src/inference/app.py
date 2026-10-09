from flask import Flask, request, jsonify
import boto3
from io import BytesIO
import joblib
import os

app = Flask(__name__)

def load_model_from_s3(bucket, key):
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key=key)
    model = joblib.load(BytesIO(response['Body'].read()))
    return model

model = load_model_from_s3("oilprice-models", "model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    current_price = data['current_price']
    # Prepare input for model (assuming it's a single feature: lag_1, which is current_price)
    input_data = [[current_price]]
    prediction = model.predict(input_data)[0]
    return jsonify({'prediction': float(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)