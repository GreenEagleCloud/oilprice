from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# Backend service URL (assuming it's in the same namespace)
BACKEND_URL = "http://inference-service:5000/predict"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        response = requests.post(BACKEND_URL, json=data)
        response.raise_for_status()
        return jsonify(response.json())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)