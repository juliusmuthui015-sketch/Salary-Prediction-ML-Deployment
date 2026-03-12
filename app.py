from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))

@app.route("/predict")
def home():
    return "Salary Prediction API"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    exp = data["experience"]
    prediction = model.predict([[exp]])
    return jsonify({"predicted_salary": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)