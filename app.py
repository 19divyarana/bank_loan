from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("models/fraud_detection_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        type_mapping = {"CASH_OUT": 1, "TRANSFER": 2}
        input_data = np.array([[data["amount"], data["oldbalanceOrg"], data["newbalanceOrig"], type_mapping[data["transaction_type"]]]])
        
        prediction = model.predict(input_data)
        fraud_prediction = int(prediction[0])

        return jsonify({
            "fraudulent": fraud_prediction == 1,
            "message": "🚨 Fraudulent Transaction Detected!" if fraud_prediction == 1 else "✅ Transaction is Legitimate."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
