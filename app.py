from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        amount = float(request.form['amount'])
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        transaction_type = request.form['type']

        # Convert transaction type to numerical encoding
        type_mapping = {"CASH_OUT": 1, "TRANSFER": 2}
        type_encoded = type_mapping.get(type, 0)

        # Prepare input for model
        input_data = np.array([[amount, oldbalanceOrg, newbalanceOrig, type_encoded]])
        prediction = model.predict(input_data)
        fraud_prediction = int(prediction[0])

        result = {
            "fraudulent": True if fraud_prediction == 1 else False,
            "message": "🚨 Fraudulent Transaction Detected!" if fraud_prediction == 1 else "✅ Transaction is Legitimate."
        }

        return render_template("result.html", prediction=result["message"])
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
