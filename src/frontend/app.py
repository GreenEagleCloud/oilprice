import os
from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# Backend service URL (overridable via the BACKEND_URL env var, set in the
# k8s frontend deployment). Defaults to the in-cluster inference service.
BACKEND_URL = os.environ.get("BACKEND_URL", "http://inference-service:5000/predict")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Proxy to the inference service.

    Passes the backend response through (JSON + status code) so users see the
    backend's error messages (e.g. "no feature vector for date ...; latest
    available date: ...") instead of a raw HTTPError string.
    """
    data = request.json
    try:
        response = requests.post(BACKEND_URL, json=data, timeout=30)
    except requests.RequestException as e:
        return jsonify({'error': f'prediction service is unreachable: {e}'}), 502

    try:
        payload = response.json()
    except ValueError:
        return jsonify({'error': f'prediction service returned a non-JSON response (HTTP {response.status_code})'}), 502

    if response.status_code != 200:
        if isinstance(payload, dict):
            return jsonify(payload), response.status_code
        return jsonify({'error': str(payload)}), response.status_code

    return jsonify(payload)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)