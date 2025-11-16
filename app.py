from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # read form values (adjust names to match your HTML)
    features = [
        float(request.form["fixed_acidity"]),
        float(request.form["volatile_acidity"]),
        float(request.form["citric_acid"]),
        float(request.form["residual_sugar"]),
        float(request.form["chlorides"]),
        float(request.form["free_sulfur_dioxide"]),
        float(request.form["total_sulfur_dioxide"]),
        float(request.form["density"]),
        float(request.form["pH"]),
        float(request.form["sulphates"]),
        float(request.form["alcohol"]),]
    pred = model.predict([features])[0]
    pred = round(pred, 2)
    return render_template("index.html", prediction=pred)

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, debug=True)
