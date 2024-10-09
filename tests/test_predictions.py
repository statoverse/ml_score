import pytest
import numpy as np
import joblib
from app import predict_score, extract_features_from_custom, df
from functions.functions import *

@pytest.fixture
def load_model():
    # Charger le modèle une fois pour tous les tests
    model_path = 'score/final_model.joblib'
    return joblib.load(model_path)

def test_predict_score_bank_loan_not_granted(load_model):
    # Test pour un client dont le prêt doit être refusé
    customer_id = 272059
    customer_data = extract_features_from_custom(df, customer_id)
    
    # Appliquer le prétraitement
    pipeline = load_model
    preprocessor = pipeline.named_steps['preprocessor']
    processed_data = preprocessor.transform(customer_data)
    
    # Tester la prédiction
    decision, prediction_success, prediction_failure = predict_score(processed_data)
    
    # Vérifier la décision et les prédictions
    assert decision == "Bank loan not granted"
    assert prediction_failure > 0.25  # Seuil pour refuser le prêt

def test_predict_score_bank_loan_granted(load_model):
    # Test pour un client dont le prêt doit être accordé
    customer_id = 218558  # Remplacez avec l'ID du client pour qui le prêt doit être accordé
    customer_data = extract_features_from_custom(df, customer_id)
    
    # Appliquer le prétraitement
    pipeline = load_model
    preprocessor = pipeline.named_steps['preprocessor']
    processed_data = preprocessor.transform(customer_data)
    
    # Tester la prédiction
    decision, prediction_success, prediction_failure = predict_score(processed_data)
    
    # Vérifier la décision et les prédictions
    assert decision == "Bank loan granted"
    assert prediction_failure <= 0.25  # Seuil pour accorder le prêt
