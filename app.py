from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained XGBoost model
model = joblib.load("fraud_detection_model.pkl")

@app.route('/')
def home():
    return "Bank Loan Fraud Detection API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        amount = float(data['amount'])
        oldbalanceOrg = float(data['oldbalanceOrg'])
        newbalanceOrig = float(data['newbalanceOrig'])
        transaction_type = data['transaction_type']

        # Convert transaction type to numerical
        type_mapping = {"CASH_OUT": 1, "TRANSFER": 2}
        type_encoded = type_mapping.get(transaction_type, 0)

        # Prepare input for model
        input_data = np.array([[amount, oldbalanceOrg, newbalanceOrig, type_encoded]])
        prediction = model.predict(input_data)
        fraud_prediction = int(prediction[0])

        result = {
            "fraudulent": True if fraud_prediction == 1 else False,
            "message": "Fraudulent Transaction Detected!" if fraud_prediction == 1 else "Transaction is Legitimate."
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)





