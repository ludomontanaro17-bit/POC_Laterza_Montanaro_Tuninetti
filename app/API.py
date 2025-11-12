from flask import Flask, request, jsonify, render_template, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
import joblib
import os
import xgboost as xgb
import pandas as pd


class API:
    def __init__(self, model_paths=None, scaler_path="model/scaler.pkl"):
        """Inizializza il server Flask e carica tutti i modelli."""
        self.app = Flask(__name__)

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

        # 🔥 MODIFICA: Ora riceviamo i dati RAW dal frontend
        self.expected_raw_features = [
            'Gender', 'Married', 'Dependents', 'Education',
            'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
            'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
            'Property_Area'
        ]

        # 🔥 MODIFICA: Feature engineering che verrà fatto nel backend
        self.expected_processed_features = [
            'Dependents', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
            'TotalIncome', 'LoanAmount_to_Income', 'HasDependents', 'IsMarried',
            'IsGraduate', 'Income_bracket', 'Credit_x_IncomeHigh', 'Gender_Male',
            'Married_Yes', 'Education_Not Graduate', 'Self_Employed_Yes',
            'Property_Area_Semiurban', 'Property_Area_Urban'
        ]

        # Pesi per il soft voting
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
                xgb_instance = xgb.XGBClassifier()
                xgb_instance.load_model(self.model_paths['xgboost'])
                self.models['xgboost'] = xgb_instance
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
            print("⚠️ Nessuno scaler trovato")

    def _preprocess_features(self, raw_data):
        """
        🔥 NUOVO METODO: Preprocessing centralizzato nel backend
        Converte i dati RAW dal frontend nelle feature che i modelli si aspettano
        """
        try:
            print("🔄 Inizio preprocessing features...")
            print(f"📥 Dati ricevuti dal frontend: {raw_data}")

            # Crea un DataFrame con i dati ricevuti
            df = pd.DataFrame([raw_data])

            # 🔥 FEATURE ENGINEERING (stesso che hai fatto nel training)

            # 1. TotalIncome
            applicant_income = float(raw_data.get('ApplicantIncome', 0))
            coapplicant_income = float(raw_data.get('CoapplicantIncome', 0))
            total_income = applicant_income + coapplicant_income

            # 2. LoanAmount to Income ratio
            loan_amount = float(raw_data.get('LoanAmount', 0))
            loan_amount_to_income = loan_amount / total_income if total_income > 0 else 0

            # 3. Feature binarie derivate
            dependents = int(raw_data.get('Dependents', 0))
            has_dependents = 1 if dependents > 0 else 0

            # 4. Encoding variabili categoriche (stesso del training)
            married = raw_data.get('Married', 'No')
            is_married = 1 if married == 'Yes' else 0

            education = raw_data.get('Education', 'Graduate')
            is_graduate = 1 if education == 'Graduate' else 0
            education_not_graduate = 1 if education == 'Not Graduate' else 0

            gender = raw_data.get('Gender', 'Female')
            gender_male = 1 if gender == 'Male' else 0

            self_employed = raw_data.get('Self_Employed', 'No')
            self_employed_yes = 1 if self_employed == 'Yes' else 0

            # 5. Property Area one-hot encoding
            property_area = raw_data.get('Property_Area', 'Rural')
            property_area_semiurban = 1 if property_area == 'Semiurban' else 0
            property_area_urban = 1 if property_area == 'Urban' else 0

            # 6. Income bracket (stesso threshold del training)
            income_bracket = 1 if total_income > 5000 else 0

            # 7. Interaction feature
            credit_history = int(raw_data.get('Credit_History', 0))
            credit_x_income_high = credit_history * income_bracket

            # 🔥 COSTRUZIONE FEATURE FINALI nell'ordine corretto
            processed_features = [
                dependents,  # Dependents
                loan_amount,  # LoanAmount
                float(raw_data.get('Loan_Amount_Term', 0)),  # Loan_Amount_Term
                credit_history,  # Credit_History
                total_income,  # TotalIncome
                loan_amount_to_income,  # LoanAmount_to_Income
                has_dependents,  # HasDependents
                is_married,  # IsMarried
                is_graduate,  # IsGraduate
                income_bracket,  # Income_bracket
                credit_x_income_high,  # Credit_x_IncomeHigh
                gender_male,  # Gender_Male
                is_married,  # Married_Yes
                education_not_graduate,  # Education_Not Graduate
                self_employed_yes,  # Self_Employed_Yes
                property_area_semiurban,  # Property_Area_Semiurban
                property_area_urban  # Property_Area_Urban
            ]

            print(f"📤 Feature processate: {processed_features}")
            print(f"🔢 Numero feature: {len(processed_features)}")

            return np.array(processed_features).reshape(1, -1)

        except Exception as e:
            print(f"❌ Errore nel preprocessing: {e}")
            raise

    def _apply_scaling(self, features):
        """Applica lo scaling alle feature numeriche (se lo scaler è disponibile)"""
        if self.scaler is None:
            return features

        try:
            print("🔧 Applicazione scaling...")
            # Definisci le colonne che devono essere scalate (stesso del training)
            scaler_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                              'TotalIncome', 'LoanAmount_to_Income']

            # Indici di queste colonne nell'array delle feature
            feature_indices = []
            for col in scaler_columns:
                if col in self.expected_processed_features:
                    feature_indices.append(self.expected_processed_features.index(col))

            print(f"🔍 Colonne da scalare (indici): {feature_indices}")

            # Applica lo scaler solo alle colonne specifiche
            features_processed = features.copy()

            if feature_indices:
                features_to_scale = features[:, feature_indices]
                scaled_features = self.scaler.transform(features_to_scale)

                # Sostituisci le feature originali con quelle scalate
                for i, idx in enumerate(feature_indices):
                    features_processed[0, idx] = scaled_features[0, i]

                print(f"✅ Scaling applicato a {len(feature_indices)} feature numeriche")

            return features_processed

        except Exception as e:
            print(f"❌ Errore nello scaling: {e}")
            return features

    def _predict_individual_models(self, features):
        """Esegue predizioni con tutti i modelli individualmente."""
        predictions = {}

        try:
            print(f"🔍 Input features shape: {features.shape}")

            # Applica scaling
            features_processed = self._apply_scaling(features)
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
            return render_template("index.html")

        @self.app.route("/predict", methods=["POST"])
        def predict():
            try:
                data = request.get_json()
                print(f"📨 Dati ricevuti: {data}")

                if not data:
                    return jsonify({
                        "error": "JSON non valido. Invia i dati del form",
                        "expected_raw_features": self.expected_raw_features
                    }), 400

                # 🔥 MODIFICA: Ora riceviamo i dati RAW e li processiamo nel backend
                raw_features = {
                    'Gender': data.get('Gender', 'Female'),
                    'Married': data.get('Married', 'No'),
                    'Dependents': int(data.get('Dependents', 0)),
                    'Education': data.get('Education', 'Graduate'),
                    'Self_Employed': data.get('Self_Employed', 'No'),
                    'ApplicantIncome': float(data.get('ApplicantIncome', 0)),
                    'CoapplicantIncome': float(data.get('CoapplicantIncome', 0)),
                    'LoanAmount': float(data.get('LoanAmount', 0)),
                    'Loan_Amount_Term': float(data.get('Loan_Amount_Term', 0)),
                    'Credit_History': int(data.get('Credit_History', 0)),
                    'Property_Area': data.get('Property_Area', 'Rural')
                }

                # Preprocessing nel backend
                processed_features = self._preprocess_features(raw_features)

                print(f"✅ Feature processate shape: {processed_features.shape}")
                print(f"✅ Atteso: {len(self.expected_processed_features)} feature")

                # Predizioni individuali
                individual_predictions = self._predict_individual_models(processed_features)

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
                    "raw_features_received": raw_features,
                    "processed_features_count": len(self.expected_processed_features)
                })

            except Exception as e:
                print(f"❌ Errore generale in /predict: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    "error": f"Errore interno del server: {str(e)}"
                }), 500

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
                "scaler_loaded": self.scaler is not None,
                "expected_raw_features": self.expected_raw_features,
                "expected_processed_features": self.expected_processed_features
            })

    def run(self, host="0.0.0.0", port=5000, debug=True):
        self.app.run(host=host, port=port, debug=debug)