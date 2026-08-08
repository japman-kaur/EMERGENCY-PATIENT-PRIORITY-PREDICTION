from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('triage_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
feature_columns = joblib.load('feature_columns.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = {col: float(request.form[col]) for col in feature_columns}
    input_df = pd.DataFrame([data])[feature_columns]
    prediction = model.predict(input_df)[0]
    return render_template('home.html', prediction_text=f'Predicted Triage Level: {prediction}')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)