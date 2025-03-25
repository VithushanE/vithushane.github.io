from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

app = Flask(__name__)

# Apply CORS to the whole app (allowing all origins)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load your trained model (replace with your actual model path)
best_log_model = joblib.load('/Users/vithushanesan/Desktop/Vithu Portfolio/vithushane.github.io/portfolios/best_log_model.pkl')

# Pre-fitted encoders from training
loan_intent_encoder = joblib.load('/Users/vithushanesan/Desktop/Vithu Portfolio/vithushane.github.io/portfolios/loan_intent_encoder.pkl')
loan_grade_encoder = joblib.load('/Users/vithushanesan/Desktop/Vithu Portfolio/vithushane.github.io/portfolios/loan_grade_encoder.pkl')
home_ownership_encoder = joblib.load('/Users/vithushanesan/Desktop/Vithu Portfolio/vithushane.github.io/portfolios/home_ownership_encoder.pkl')
default_on_file_encoder = joblib.load('/Users/vithushanesan/Desktop/Vithu Portfolio/vithushane.github.io/portfolios/default_on_file_encoder.pkl')

# Pre-fitted scaler from training
scaler = joblib.load('/Users/vithushanesan/Desktop/Vithu Portfolio/vithushane.github.io/portfolios/scaler.pkl')

# Define safe transform function (with fallback for unseen labels)
def safe_transform(encoder, value, column_name):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        print(f"⚠️ Warning: Unseen value '{value}' in column '{column_name}'. Encoding as -1.")
        return -1  # Placeholder for unknown values

@app.route('/')
def home():
    return "Loan Prediction API is working!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)  # Debugging statement

        # Prepare input features (excluding the 'customer_id' or 'id' if present)
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

        # Encode categorical variables using safe_transform
        new_input_df['loan_intent'] = safe_transform(loan_intent_encoder, str(new_input_df['loan_intent'][0]), 'loan_intent')
        new_input_df['loan_grade'] = safe_transform(loan_grade_encoder, str(new_input_df['loan_grade'][0]), 'loan_grade')
        new_input_df['person_home_ownership'] = safe_transform(home_ownership_encoder, str(new_input_df['person_home_ownership'][0]), 'person_home_ownership')
        new_input_df['cb_person_default_on_file'] = safe_transform(default_on_file_encoder, str(new_input_df['cb_person_default_on_file'][0]), 'cb_person_default_on_file')

        # Ensure the feature names match the model's training data
        expected_columns = ['person_age', 'person_income', 'person_home_ownership', 'person_emp_length',
                            'loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                            'cb_person_default_on_file', 'cb_person_cred_hist_length']
        
        # Align columns if necessary
        new_input_df = new_input_df[expected_columns]

        # Scale the features
        new_input_scaled = scaler.transform(new_input_df)

        # Make prediction
        prediction = best_log_model.predict(new_input_scaled)
        print("Prediction:", prediction)  # Debugging statement

        # Return result
        result = {
            'prediction': 'Approved' if prediction[0] == 1 else 'Not Approved'
        }
        print("Result:", result)  # Debugging statement

        return jsonify(result)
    except Exception as e:
        print("Error:", e)  # Debugging statement
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
