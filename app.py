import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    features = [
        float(data['step']), int(data['type']), float(data['amount']), float(data['oldbalanceOrg']),
        float(data['newbalanceOrig']), float(data['oldbalanceDest']), float(data['newbalanceDest']),
        float(data['isFraud']), float(data['isFlaggedFraud'])
    ]
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    
    return render_template('result.html', prediction=int(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)
