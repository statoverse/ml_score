from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import joblib
import os
from functions.functions import *

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
                            prediction_failure=prediction_failure,
                            customer_id=selected_id))

@app.route('/result', methods=['GET'])
def show_prediction():
    decision = request.args.get('decision')
    prediction_success = request.args.get('prediction_success')
    prediction_failure = request.args.get('prediction_failure')
    customer_id = request.args.get('customer_id')
    return render_template('prediction.html', 
                           decision=decision, 
                           prediction_success=prediction_success, 
                           prediction_failure=prediction_failure,
                           customer_id=customer_id)
    
@app.route('/explain/<int:customer_id>')
def explain(customer_id):
    try:
        # Extraire les données du client
        customer_data_raw = extract_features_from_custom(df, customer_id)
        
        # Charger le pipeline complet
        model_path = 'score/final_model.joblib'
        pipeline = joblib.load(model_path)
        
        # Appliquer le prétraitement du pipeline sur les données du client
        preprocessor = pipeline.named_steps['preprocessor']
        customer_data = preprocessor.transform(customer_data_raw)
        
        # Extraire uniquement le modèle final (LogisticRegression)
        model = pipeline.named_steps['model']
        
        # Créer l'explainer avec seulement le modèle final
        explainer = shap.Explainer(model, customer_data)
        
        # Calculer les valeurs SHAP
        shap_values = explainer(customer_data)
        
        # Générer et sauvegarder le graphique SHAP
        plt.figure()
        shap.summary_plot(shap_values, customer_data, show=False)
        plot_path = 'static/shap_global_importance.png'
        plt.savefig(plot_path)
        plt.close()
        
        return render_template('explain.html', plot_path=plot_path)
    
    except Exception as e:
        print("Error in explain route:", str(e))
        return "An error occurred", 500






if __name__ == '__main__':
    app.run(debug=True)
