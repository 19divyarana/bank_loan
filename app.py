from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect all 10 input features (update based on your dataset)
        amount = request.form.get('amount', type=float)
        oldbalanceOrg = request.form.get('oldbalanceOrg', type=float)
        newbalanceOrig = request.form.get('newbalanceOrig', type=float)
        transaction_type = request.form.get('transaction_type')

        # Example: Additional features (Modify based on your dataset)
        step = request.form.get('step', type=int)
        newbalanceDest = request.form.get('newbalanceDest', type=float)
        oldbalanceDest = request.form.get('oldbalanceDest', type=float)
        isFlaggedFraud = request.form.get('isFlaggedFraud', type=int, default=0)

        # Convert transaction type to numerical encoding
        type_mapping = {"CASH_OUT": 1, "TRANSFER": 2}
        type_encoded = type_mapping.get(transaction_type, 0)

        # Ensure all features are included
        feature_array = np.array([[step, amount, oldbalanceOrg, newbalanceOrig, 
                                   oldbalanceDest, newbalanceDest, type_encoded, isFlaggedFraud]])

        # If a scaler was used, apply it
        scaled_features = scaler.transform(feature_array)  

        # Predict fraud
        prediction = model.predict(scaled_features)
        fraud_prediction = int(prediction[0])

        result = "🚨 Fraudulent Transaction Detected!" if fraud_prediction == 1 else "✅ Transaction is Legitimate."

        return render_template("result.html", prediction=result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
