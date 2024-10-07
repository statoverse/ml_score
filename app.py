from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
#from your_module import load_data, extract_features_from_custom, predict_score
import joblib
import os

print(os.getcwd)


    
def load_data():
    # Load the CSV file into a DataFrame
    data_path = 'data/customers.csv'
    df = pd.read_csv(data_path)
    
    # Extract the customer IDs (SK_ID_CURR) column
    customer_ids = df['SK_ID_CURR'].tolist()
    
    return df, customer_ids

def extract_features_from_custom(df, customer_id):
    # Get the row of the customer
    customer_data = df[df['SK_ID_CURR'] == customer_id]
    
    # Drop the SK_ID_CURR and TARGET columns
    customer_data = customer_data.drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore')
    
    return customer_data

def predict_score(customer_data):
    # Load the pre-trained model
    model_path = 'score/final_model.joblib'
    model = joblib.load(model_path)
    
    # Predict the score (probability)
    prediction_success = np.round(model.predict_proba(customer_data)[:, 0],2)[0]
    prediction_failure = np.round(model.predict_proba(customer_data)[:, 1],2)[0]
    decision = model.predict(customer_data)[0]
    if prediction_failure > 0.25:
        decision = "Bank loan not granted"
    else:
        decision = "Bank loan granted"
        
    return decision, prediction_success, prediction_failure



# Load data once
df, customer_ids = load_data()

app = Flask(__name__)



@app.route('/')
def welcome():
    # Pass the list of customer IDs to the welcome page
    return render_template('welcome.html', customer_ids=customer_ids)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the selected customer ID from the form
    selected_id = int(request.form.get('customer_id'))
    
    # Extract customer features
    customer_data = extract_features_from_custom(df, selected_id)
    
    # Make a prediction
    decision, prediction_success, prediction_failure = predict_score(customer_data)
    
    # Redirect to the prediction result page with the result
    return redirect(url_for('show_prediction', decision=decision, 
                            prediction_success=prediction_success, 
                            prediction_failure=prediction_failure))

@app.route('/result')
def show_prediction():
    # Get the prediction from the URL arguments
    decision = request.args.get('decision')
    prediction_success = request.args.get('prediction_success')
    prediction_failure = request.args.get('prediction_failure')
    return render_template('prediction.html', 
                           decision=decision, 
                           prediction_success=prediction_success, 
                           prediction_failure=prediction_failure)

if __name__ == '__main__':
    app.run(debug=True)
