# app.py

from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    feature1 = float(request.form['feature1'])
    feature2 = float(request.form['feature2'])

    prediction = model.predict(np.array([[feature1, feature2]]))[0]
    return render_template('index.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
