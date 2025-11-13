from API import API
import os

if __name__ == "__main__":

    # Percorsi assoluti per i modelli nel container Docker
    model_paths = {
        'keras': "/app/model/keras_model.h5",
        'logreg': "/app/model/logistic_regression_model.pkl",
        'xgboost': "/app/model/xgboost_model.json",  # usa json, è il più comune per XGBoost
    }

    print("🔍 Verifica modelli...")

    # Verifica file modello
    for model_name, path in model_paths.items():
        if os.path.exists(path):
            print(f"✅ {model_name}: trovato → {path}")
        else:
            print(f"❌ {model_name}: NON trovato → {path}")

    # Verifica scaler
    if os.path.exists("/app/model/scaler.pkl"):
        print("✅ Scaler trovato")
    else:
        print("❌ Scaler NON trovato")

    print("🚀 Inizializzazione API...")

    # Inizializza l’API, passando i percorsi assoluti come vuole API.py
    api = API(
        model_paths=model_paths,
        scaler_path="/app/model/scaler.pkl"
    )

    print("🚀 Avvio del server Flask su 0.0.0.0:5000...")

    # Espone il servizio
    api.run(host="0.0.0.0", port=5000, debug=False)
