from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load your trained model (replace with your actual model path)
best_log_model = joblib.load('path_to_best_log_model.pkl')

# Pre-fitted encoders from training
loan_intent_encoder = joblib.load('path_to_loan_intent_encoder.pkl')
loan_grade_encoder = joblib.load('path_to_loan_grade_encoder.pkl')
home_ownership_encoder = joblib.load('path_to_home_ownership_encoder.pkl')
default_on_file_encoder = joblib.load('path_to_default_on_file_encoder.pkl')

# Pre-fitted scaler from training
scaler = joblib.load('path_to_scaler.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return "Loan Prediction API is working!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Prepare input features
    input_data = {
        'person_age': data['person_age'],
        'person_income': data['person_income'],
        'person_home_ownership': data['person_home_ownership'],
        'person_emp_length': data['person_emp_length'],
        'loan_intent': data['loan_intent'],
        'loan_grade': data['loan_grade'],
        'loan_amnt': data['loan_amnt'],
        'loan_int_rate': data['loan_int_rate'],
        'loan_percent_income': data['loan_percent_income'],
        'cb_person_default_on_file': data['cb_person_default_on_file'],
        'cb_person_cred_hist_length': data['cb_person_cred_hist_length']
    }

    # Convert to DataFrame
    new_input_df = pd.DataFrame([input_data])

    # Encode categorical variables
    new_input_df['loan_intent'] = loan_intent_encoder.transform(new_input_df['loan_intent'])
    new_input_df['loan_grade'] = loan_grade_encoder.transform(new_input_df['loan_grade'])
    new_input_df['person_home_ownership'] = home_ownership_encoder.transform(new_input_df['person_home_ownership'])
    new_input_df['cb_person_default_on_file'] = default_on_file_encoder.transform(new_input_df['cb_person_default_on_file'])

    # Scale the features
    new_input_scaled = scaler.transform(new_input_df)

    # Make prediction
    prediction = best_log_model.predict(new_input_scaled)

    # Return result
    result = {
        'prediction': 'Approved' if prediction[0] == 1 else 'Not Approved'
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

