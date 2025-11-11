from API import API
import os

if __name__ == "__main__":
    # Definisci i percorsi corretti
    model_paths = {
        'keras': "../model/keras_model.h5",
        'logreg': "../model/logistic_regression_model.pkl",
        'xgboost': "../model/xgboost_model.json"  # Cambia l'estensione se necessario
    }

    # Verifica che i file esistano
    for model_name, path in model_paths.items():
        if os.path.exists(path):
            print(f"✅ {model_name}: {path} trovato")
        else:
            print(f"❌ {model_name}: {path} NON trovato")

    if os.path.exists("../model/scaler.pkl"):
        print("✅ Scaler trovato")
    else:
        print("❌ Scaler NON trovato")

    # Inizializza l'API
    api = API(
        model_paths=model_paths,
        scaler_path="../model/scaler.pkl"
    )

    print("🚀 Avvio del server Flask...")
    api.run(debug=True)