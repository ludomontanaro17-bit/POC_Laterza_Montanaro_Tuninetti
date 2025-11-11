# test_load_models.py

import joblib
import os
import xgboost as xgb
from tensorflow import keras

model_dir = "model"

print("--- Test Caricamento Modelli e Oggetti ---")

# Prova a caricare scaler
try:
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
         scaler = joblib.load(scaler_path)
         print("✅ Scaler caricato correttamente.")
    else:
         print("❌ File scaler.pkl non trovato.")
except Exception as e:
    print(f"❌ Errore caricando scaler: {e}")

# Prova a caricare final_columns
try:
    final_cols_path = os.path.join(model_dir, "final_columns.pkl")
    if os.path.exists(final_cols_path):
         final_columns = joblib.load(final_cols_path)
         print(f"✅ Final columns caricato correttamente. Lunghezza: {len(final_columns)}")
         # Stampa solo le prime 10 colonne per brevità, se sono tante
         print(f"Colonne (prime 10): {final_columns[:10]}")
    else:
         print("❌ File final_columns.pkl non trovato.")
except Exception as e:
    print(f"❌ Errore caricando final_columns: {e}")

# Prova a caricare XGBoost (esempio con JSON)
try:
    xgb_path = os.path.join(model_dir, "xgboost_model.json") # Aggiusta il nome se necessario (es. .pkl)
    if os.path.exists(xgb_path):
         xgb_model = xgb.XGBClassifier() # Crea un'istanza vuota
         xgb_model.load_model(xgb_path) # Carica i pesi e la configurazione
         print("✅ XGBoost model caricato correttamente.")
         # Facoltativo: stampa una breve rappresentazione del modello
         # print(xgb_model) # Può essere molto verboso
    else:
         print("❌ File xgboost_model.json non trovato.")
except Exception as e:
    print(f"❌ Errore caricando XGBoost model: {e}")

# Prova a caricare Logistic Regression (salvato con joblib)
try:
    logreg_path = os.path.join(model_dir, "logistic_regression_model.pkl") # Aggiusta il nome se necessario
    if os.path.exists(logreg_path):
         logreg_model = joblib.load(logreg_path)
         print("✅ Logistic Regression model caricato correttamente.")
         # Facoltativo: stampa una breve rappresentazione del modello
         # print(logreg_model)
    else:
         print("❌ File logistic_regression_model.pkl non trovato.")
except Exception as e:
    print(f"❌ Errore caricando Logistic Regression model: {e}")

# Prova a caricare Keras Model (salvato con model.save)
try:
    keras_path = os.path.join(model_dir, "keras_model.h5") # Aggiusta il nome o estensione se necessario (es. directory per SavedModel format)
    if os.path.exists(keras_path):
         keras_model = keras.models.load_model(keras_path)
         print("✅ Keras model caricato correttamente.")
         # Facoltativo: stampa un riepilogo del modello
         # keras_model.summary() # Commenta se troppo verboso
    else:
         print("❌ File keras_model.h5 non trovato.")
except Exception as e:
    print(f"❌ Errore caricando Keras model: {e}")

print("\n--- Fine Test ---")
