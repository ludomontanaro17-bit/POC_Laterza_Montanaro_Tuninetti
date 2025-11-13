from flask import Flask, request, jsonify, render_template, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from prometheus_flask_exporter import PrometheusMetrics

class API:
    def __init__(self, model_paths=None, scaler_path="model/scaler.pkl"):
        """Inizializza il server Flask e carica tutti i modelli."""
        self.app = Flask(__name__)

        # Questo abilita l'esportazione automatica di metriche HTTP
        self.metrics = PrometheusMetrics(self.app)


        # Paths di default per i modelli
        if model_paths is None:
            model_paths = {
                'keras': "model/keras_model.h5",
                'logreg': "model/logistic_regression_model.pkl", 
                'xgboost': "model/xgboost_model.json"
            }
        
        self.model_paths = model_paths
        self.scaler_path = scaler_path
        self.models = {}
        self.scaler = None

        # 🔹 Le feature effettive usate nel training (ordine corretto)
        self.expected_features = [
            'Dependents',
            'LoanAmount',
            'Loan_Amount_Term',
            'Credit_History',
            'TotalIncome',
            'LoanAmount_to_Income',
            'HasDependents',
            'IsMarried',
            'IsGraduate',
            'Income_bracket', 
            'Credit_x_IncomeHigh', 
            'Gender_Male',
            'Married_Yes', 
            'Education_Not Graduate', 
            'Self_Employed_Yes',
            'Property_Area_Semiurban', 
            'Property_Area_Urban'
        ]

        # Pesi per il soft voting (basati sulle performance)
        self.model_weights = {
            'keras': 0.4,
            'xgboost': 0.35,
            'logreg': 0.25
        }

        # Caricamento automatico all'avvio
        self._load_all_models()
        self._load_scaler()

        # Registrazione endpoints
        self._register_routes()

    def _load_all_models(self):
       
        try:
            # Carica modello Keras
            if os.path.exists(self.model_paths['keras']):
                self.models['keras'] = load_model(self.model_paths['keras'])
                print("✅ Modello Keras caricato con successo")
            else:
                print("⚠️ Modello Keras non trovato")
            
            # Carica modello Logistic Regression
            if os.path.exists(self.model_paths['logreg']):
                self.models['logreg'] = joblib.load(self.model_paths['logreg'])
                print("✅ Modello Logistic Regression caricato con successo")
            else:
                print("⚠️ Modello Logistic Regression non trovato")
            
            # Carica modello XGBoost
            if os.path.exists(self.model_paths['xgboost']):
                # *** MODIFICA APPLICATA ***
                # CARICA XGBOOST MODEL con il metodo corretto per .json
                xgb_instance = xgb.XGBClassifier() # Crea un nuovo oggetto XGBClassifier
                xgb_instance.load_model(self.model_paths['xgboost']) # Carica i pesi/configurazione dal file JSON
                self.models['xgboost'] = xgb_instance # Assegna l'oggetto caricato
                print("✅ Modello XGBoost caricato con successo")
            else:
                print("⚠️ Modello XGBoost non trovato")
                
        except Exception as e:
            print(f"❌ Errore nel caricamento dei modelli: {e}")

    def _load_scaler(self):
        """Carica lo scaler salvato con joblib."""
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
            print("✅ Scaler caricato con successo")
        else:
            print("⚠️ Nessuno scaler trovato — gli input devono essere già normalizzati")

    def _predict_individual_models(self, features):
        """Esegue predizioni con tutti i modelli individualmente."""
        predictions = {}

        try:
            print(f"🔍 Input features shape: {features.shape}")

            # Applica scaler solo alle feature numeriche specifiche
            if self.scaler:
                # Definisci le colonne che sono state scalate durante il training
                scaler_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                                  'TotalIncome', 'LoanAmount_to_Income']

                # Indici di queste colonne nell'array delle feature
                feature_indices = []
                for col in scaler_columns:
                    if col in self.expected_features:
                        feature_indices.append(self.expected_features.index(col))

                print(f"🔍 Scaling columns indices: {feature_indices}")

                # Crea una copia delle feature
                features_processed = features.copy()

                # Applica lo scaler solo alle colonne specifiche
                if feature_indices:
                    features_to_scale = features[:, feature_indices]
                    scaled_features = self.scaler.transform(features_to_scale)
                    # Sostituisci le feature originali con quelle scalate
                    for i, idx in enumerate(feature_indices):
                        features_processed[0, idx] = scaled_features[0, i]

                    print(f"✅ Applied scaler to {len(feature_indices)} numerical features")
            else:
                features_processed = features

            print(f"🔍 Final features shape: {features_processed.shape}")

            # Predizione Keras
            if 'keras' in self.models:
                keras_pred = self.models['keras'].predict(features_processed, verbose=0)
                print(f"🔍 Keras raw prediction: {keras_pred}")
                if keras_pred.shape[1] == 2:
                    predictions['keras'] = float(keras_pred[0][1])
                else:
                    predictions['keras'] = float(keras_pred[0][0])
                print(f"✅ Keras probability: {predictions['keras']}")

            # Predizione Logistic Regression
            if 'logreg' in self.models:
                logreg_pred = self.models['logreg'].predict_proba(features_processed)
                print(f"🔍 Logistic Regression raw prediction: {logreg_pred}")
                predictions['logreg'] = float(logreg_pred[0][1])
                print(f"✅ Logistic Regression probability: {predictions['logreg']}")

            # Predizione XGBoost
            if 'xgboost' in self.models:
                if hasattr(self.models['xgboost'], 'predict_proba'):
                    xgb_pred = self.models['xgboost'].predict_proba(features_processed)
                    print(f"🔍 XGBoost (sklearn) raw prediction: {xgb_pred}")
                    predictions['xgboost'] = float(xgb_pred[0][1])
                else:
                    dmatrix = xgb.DMatrix(features_processed)
                    raw_pred = self.models['xgboost'].predict(dmatrix)
                    print(f"🔍 XGBoost (native) raw prediction: {raw_pred}")
                    from scipy.special import expit
                    predictions['xgboost'] = float(expit(raw_pred[0]))

                print(f"✅ XGBoost probability: {predictions['xgboost']}")

        except Exception as e:
            print(f"❌ Errore nelle predizioni individuali: {e}")
            import traceback
            traceback.print_exc()

        return predictions

    def _soft_voting(self, individual_predictions):
        """Calcola la predizione finale tramite soft voting pesato."""
        total_weight = 0
        weighted_sum = 0
        voting_details = []
        
        for model_name, prob in individual_predictions.items():
            if model_name in self.model_weights:
                weight = self.model_weights[model_name]
                weighted_sum += prob * weight
                total_weight += weight
                
                voting_details.append({
                    'model': model_name,
                    'probability': float(prob),
                    'weight': weight,
                    'weighted_prob': float(prob * weight)
                })
        
        if total_weight > 0:
            final_probability = weighted_sum / total_weight
        else:
            final_probability = sum(individual_predictions.values()) / len(individual_predictions)
        
        final_prediction = 1 if final_probability > 0.5 else 0
        
        return {
            'final_prediction': final_prediction,
            'final_probability': float(final_probability),
            'voting_details': voting_details,
            'individual_predictions': individual_predictions
        }

    def _register_routes(self):
        """Definisce gli endpoint dell'API."""

        @self.app.route("/", methods=["GET"])
        def home():
            return jsonify({
                "message": "API Ensemble per classificazione di Loan",
                "expected_features": self.expected_features,
                "available_models": list(self.models.keys()),
                "model_weights": self.model_weights
            })

        @self.app.route("/predict", methods=["POST"])
        def predict():
            data = request.get_json()

            if not data or "features" not in data:
                return jsonify({
                    "error": "JSON non valido. Usa 'features': [valori...]",
                    "expected_features": self.expected_features
                }), 400

            features = np.array(data["features"]).reshape(1, -1)

            # ✅ Controllo numero di feature
            if features.shape[1] != len(self.expected_features):
                return jsonify({
                    "error": f"Numero di feature errato: atteso {len(self.expected_features)}, ricevuto {features.shape[1]}",
                    "expected_order": self.expected_features
                }), 400

            # Predizioni individuali
            individual_predictions = self._predict_individual_models(features)
            
            if not individual_predictions:
                return jsonify({
                    "error": "Nessun modello disponibile per la predizione"
                }), 500

            # Soft voting
            ensemble_result = self._soft_voting(individual_predictions)

            return jsonify({
                "final_prediction": ensemble_result['final_prediction'],
                "final_probability": ensemble_result['final_probability'],
                "ensemble_details": ensemble_result['voting_details'],
                "individual_predictions": ensemble_result['individual_predictions'],
                "model_weights": self.model_weights,
                "expected_features": self.expected_features
            })

        @self.app.route("/ui", methods=["GET"])
        def serve_ui():
            return render_template("index.html")

        @self.app.route("/static/<path:path>")
        def serve_static(path):
            return send_from_directory('static', path)

        @self.app.route("/models/status", methods=["GET"])
        def models_status():
            """Endpoint per verificare lo stato dei modelli."""
            models_status = {}
            for model_name, model in self.models.items():
                models_status[model_name] = "loaded" if model is not None else "not loaded"
            
            return jsonify({
                "models": models_status,
                "weights": self.model_weights,
                "scaler_loaded": self.scaler is not None
            })

    def run(self, host="0.0.0.0", port=5000, debug=True):
        self.app.run(host=host, port=port, debug=debug)