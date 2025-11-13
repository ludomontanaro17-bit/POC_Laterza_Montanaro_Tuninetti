from flask import Flask, request, jsonify, render_template, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
import joblib
import os
import pandas as pd
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
                'keras': "../model/keras_model.h5",
                'logreg': "../model/logistic_regression_model.pkl",
                'xgboost': "../model/xgboost_model.json"
            }

        self.model_paths = model_paths
        self.scaler_path = scaler_path
        self.models = {}
        self.scaler = None

        # 🔹 Le feature effettive usate nel training
        self.expected_features = joblib.load("../model/final_columns.pkl")

        # 🔥 MODIFICA: Inizializza pesi e performance
        self.model_weights = {}
        self.performance_metrics = {}

        # Caricamento automatico all'avvio
        self._load_all_models()
        self._load_scaler()
        self._load_performance_weights()  # 🔥 NUOVO: Carica pesi basati su performance

        # Registrazione endpoints
        self._register_routes()

    def _load_performance_weights(self):
        """Carica i pesi dei modelli basati sulle performance ROC AUC."""
        try:
            # Prova a caricare i pesi pre-calcolati dal training
            weights_path = "../model/model_weights.pkl"
            metrics_path = "../model/model_performance_metrics.pkl"

            if os.path.exists(weights_path):
                self.model_weights = joblib.load(weights_path)
                self.performance_metrics = joblib.load(metrics_path)
                print("✅ Pesi dei modelli caricati basati su performance ROC AUC")

                # Mostra i pesi e le performance
                print("🎯 Performance dei modelli (AUC ROC):")
                for model, weight in self.model_weights.items():
                    auc = self.performance_metrics[model]['auc_roc']
                    print(f"   {model}: peso {weight:.3f} (AUC: {auc:.3f})")

            else:
                # Fallback: usa pesi di default
                print("⚠️ File pesi non trovato, uso pesi default")
                self._load_default_weights()

        except Exception as e:
            print(f"❌ Errore nel caricamento pesi: {e}")
            self._load_default_weights()

    def _load_default_weights(self):
        """Carica pesi di default in caso di errore."""
        self.model_weights = {
            'keras': 0.4,
            'xgboost': 0.35,
            'logreg': 0.25
        }
        self.performance_metrics = {}
        print("🔄 Uso pesi di default")

    def _prepare_features_from_input(self, raw_features_dict):
        """
        Prepara le feature dall'input utente nello stesso formato usato durante il training.

        raw_features_dict: dict con i valori originali dall'utente
        """
        try:
            # Crea un DataFrame con tutte le feature attese, inizialmente con valori null
            features_df = pd.DataFrame(columns=self.expected_features)

            # Mappatura dei nomi delle feature dall'input alle feature del modello
            input_mapping = {
                'Gender': 'Gender',
                'Married': 'Married',
                'Dependents': 'Dependents',
                'Education': 'Education',
                'Self_Employed': 'Self_Employed',
                'ApplicantIncome': 'ApplicantIncome',
                'CoapplicantIncome': 'CoapplicantIncome',
                'LoanAmount': 'LoanAmount',
                'Loan_Amount_Term': 'Loan_Amount_Term',
                'Credit_History': 'Credit_History',
                'Property_Area': 'Property_Area'
            }

            # Popola le feature di base
            for input_key, model_key in input_mapping.items():
                if input_key in raw_features_dict:
                    features_df[model_key] = [raw_features_dict[input_key]]

            # 🔥 FEATURE ENGINEERING - DEVE ESSERE IDENTICO AL TRAINING 🔥

            # 1. TotalIncome (come nel training)
            if 'ApplicantIncome' in features_df.columns and 'CoapplicantIncome' in features_df.columns:
                features_df['TotalIncome'] = features_df['ApplicantIncome'] + features_df['CoapplicantIncome']
            elif 'ApplicantIncome' in features_df.columns:
                features_df['TotalIncome'] = features_df['ApplicantIncome']

            # 2. Converti LoanAmount in migliaia (come nel training)
            if 'LoanAmount' in features_df.columns:
                features_df['LoanAmount'] = features_df['LoanAmount'] * 1000.0

            # 3. Encoding delle variabili categoriche (COME NEL TRAINING)
            if 'Married' in features_df.columns:
                features_df['Married'] = features_df['Married'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

            if 'Education' in features_df.columns:
                features_df['Education'] = features_df['Education'].map({'Graduate': 1, 'Not Graduate': 0}).fillna(
                    1).astype(int)

            if 'Gender' in features_df.columns:
                features_df['Gender'] = features_df['Gender'].map({'Male': 1, 'Female': 0}).fillna(1).astype(int)

            if 'Self_Employed' in features_df.columns:
                features_df['Self_Employed'] = features_df['Self_Employed'].map({'No': 1, 'Yes': 0}).fillna(1).astype(
                    int)

            # 4. One-Hot Encoding per Property_Area (COME NEL TRAINING)
            if 'Property_Area' in features_df.columns:
                property_dummies = pd.get_dummies(features_df['Property_Area'], prefix='Property_Area')
                # Assicurati di avere tutte le colonne attese
                for col in ['Property_Area_Semiurban', 'Property_Area_Urban']:
                    if col not in property_dummies.columns:
                        property_dummies[col] = 0

                features_df = pd.concat([features_df.drop('Property_Area', axis=1),
                                         property_dummies[['Property_Area_Semiurban', 'Property_Area_Urban']]], axis=1)

            # 5. Credit_History - imputa con 0 come nel training
            if 'Credit_History' in features_df.columns:
                features_df['Credit_History'] = features_df['Credit_History'].fillna(0).astype(int)

            # 6. Dependents - converti '3+' in 3 come nel training
            if 'Dependents' in features_df.columns:
                features_df['Dependents'] = features_df['Dependents'].replace('3+', '3')
                features_df['Dependents'] = pd.to_numeric(features_df['Dependents'], errors='coerce').fillna(0)

            # 🔍 VERIFICA: Controlla che tutte le feature attese siano presenti
            missing_features = set(self.expected_features) - set(features_df.columns)
            if missing_features:
                print(f"⚠️ Feature mancanti dopo preprocessing: {missing_features}")
                # Aggiungi le feature mancanti con valore 0
                for feature in missing_features:
                    features_df[feature] = 0

            # Riordina le colonne nell'ordine esatto atteso dal modello
            features_df = features_df[self.expected_features]

            print(f"✅ Feature preparate: {len(features_df.columns)} colonne")
            print(f"🔍 Colonne finali: {list(features_df.columns)}")

            return features_df.values.astype(float)

        except Exception as e:
            print(f"❌ Errore nella preparazione delle feature: {e}")
            import traceback
            traceback.print_exc()
            raise

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

            # Carica modello XGBoost - VERSIONE SEMPLIFICATA
            xgb_paths = [
                self.model_paths['xgboost'],  # JSON originale
                'model/xgboost_model.pkl',  # Backup pickle
                'model/xgboost_simple.json',  # Eventuale versione semplice
                'model/xgboost_simple.pkl'  # Eventuale versione semplice pickle
            ]

            for xgb_path in xgb_paths:
                if os.path.exists(xgb_path):
                    try:
                        if xgb_path.endswith('.json'):
                            xgb_instance = xgb.XGBClassifier()
                            xgb_instance.load_model(xgb_path)
                            self.models['xgboost'] = xgb_instance
                            print(f"✅ Modello XGBoost caricato da {xgb_path}")
                            break
                        elif xgb_path.endswith('.pkl'):
                            self.models['xgboost'] = joblib.load(xgb_path)
                            print(f"✅ Modello XGBoost caricato da {xgb_path}")
                            break
                    except Exception as e:
                        print(f"❌ Fallito caricamento da {xgb_path}: {e}")
                else:
                    print(f"⚠️ {xgb_path} non trovato")
            else:
                print("⚠️ Nessun modello XGBoost disponibile")

        except Exception as e:
            print(f"❌ Errore nel caricamento dei modelli: {e}")
            import traceback
            traceback.print_exc()

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
                # Le colonne che sono state scalate durante il training
                scaler_columns = [ 'LoanAmount', 'TotalIncome']

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
        """
        Esegue il soft voting pesato basato sulle performance ROC AUC.
        """
        total_weight = 0
        weighted_sum = 0
        voting_details = []

        # ---------------------------------------------------------
        # 1. Calcolo con pesi basati su performance ROC AUC
        # ---------------------------------------------------------
        for model_name, prob in individual_predictions.items():
            if model_name in self.model_weights:
                weight = self.model_weights[model_name]

                weighted_sum += prob * weight
                total_weight += weight

                # Aggiungi informazioni sulle performance per il frontend
                model_performance = self.performance_metrics.get(model_name, {})
                auc_score = model_performance.get('auc_roc', 'N/A')
                accuracy = model_performance.get('accuracy', 'N/A')

                voting_details.append({
                    'model': model_name,
                    'probability': float(prob),
                    'weight': weight,
                    'weighted_prob': float(prob * weight),
                    'performance': {
                        'auc_roc': auc_score,
                        'accuracy': accuracy
                    }
                })

        # ---------------------------------------------------------
        # 2. Calcolo della probabilità finale combinata
        # ---------------------------------------------------------
        if total_weight > 0:
            final_probability = weighted_sum / total_weight
        else:
            # Caso di fallback
            final_probability = sum(individual_predictions.values()) / len(individual_predictions)

        # ---------------------------------------------------------
        # 3. Conversione probabilità → classe
        # ---------------------------------------------------------
        final_prediction = 1 if final_probability > 0.5 else 0

        # ---------------------------------------------------------
        # 4. Restituzione risultati
        # ---------------------------------------------------------
        return {
            'final_prediction': final_prediction,
            'final_probability': float(final_probability),
            'voting_details': voting_details,
            'individual_predictions': individual_predictions,
            'voting_method': 'roc_auc_based'  # 🔥 NUOVO: indica il metodo usato
        }

    def _register_routes(self):
        """Definisce gli endpoint dell'API."""

        @self.app.route("/", methods=["GET"])
        def serve_ui():
            return render_template("index.html")

        @self.app.route("/predict", methods=["POST"])
        @self.app.route("/predict", methods=["POST"])
        def predict():
            try:
                data = request.get_json()

                if not data:
                    return jsonify({
                        "error": "JSON non valido. Fornisci i dati del mutuo."
                    }), 400

                # 🔥 MODIFICA: Accetta l'input in formato dizionario invece di array
                if "features" in data and isinstance(data["features"], list):
                    # Modalità array (backward compatibility)
                    features = np.array(data["features"]).reshape(1, -1)

                    # ✅ Controllo numero di feature
                    if features.shape[1] != len(self.expected_features):
                        return jsonify({
                            "error": f"Numero di feature errato: atteso {len(self.expected_features)}, ricevuto {features.shape[1]}",
                            "expected_order": self.expected_features
                        }), 400

                elif all(key in data for key in ['Gender', 'Married', 'Dependents', 'Education',
                                                 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
                                                 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']):
                    # 🔥 NUOVA MODALITÀ: Input come oggetto con nomi delle feature
                    features = self._prepare_features_from_input(data)
                else:
                    return jsonify({
                        "error": "Formato input non supportato. Usa 'features': [array] o fornire tutte le feature individualmente",
                        "required_features": [
                            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                            'Loan_Amount_Term', 'Credit_History', 'Property_Area'
                        ]
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
                    "expected_features": self.expected_features,
                    "message": "✅ Predizione completata con successo"
                })

            except Exception as e:
                print(f"❌ Errore durante la predizione: {e}")
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
                "scaler_loaded": self.scaler is not None
            })

        @self.app.route("/debug/features", methods=["GET"])
        def debug_features():
            """Endpoint per debug delle feature attese"""
            return jsonify({
                "expected_features": self.expected_features,
                "expected_count": len(self.expected_features),
                "feature_details": {
                    "numeric": [f for f in self.expected_features if any(x in f for x in ['Income', 'Amount', 'Term'])],
                    "categorical": [f for f in self.expected_features if
                                    f in ['Gender', 'Married', 'Education', 'Credit_History']],
                    "encoded": [f for f in self.expected_features if 'Property_Area' in f]
                }
            })

    def run(self, host="0.0.0.0", port=5000, debug=True):
        self.app.run(host=host, port=port, debug=debug)