from flask import Flask, request, jsonify
import pickle
import pandas as pd


app = Flask(__name__)
#load model
with open('model.pkl','rb') as f:
    model=pickle.load

#@app.route("/")
#def home():
   # return "Salary Prediction API"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    experience = data["experience"]
    prediction = model.predict([[experience]])
    return jsonify({"predicted_salary": prediction[0]})

if __name__ == "__main__":
    app.run(port=5000,debug=True)
