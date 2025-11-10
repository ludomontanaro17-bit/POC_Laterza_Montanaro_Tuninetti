import os
import joblib
from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel:
    """
    Classe per addestrare e salvare un modello di Logistic Regression.
    """

    def __init__(self, model_dir: str = "model"):
        """
        Inizializza il modello e imposta la directory di salvataggio.
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.model = LogisticRegression(
            solver='liblinear',  # stabile per dataset piccoli
            random_state=42,
            class_weight='balanced'  # utile per dataset sbilanciati
        )

    def train(self, X_train, y_train):
        """
        Addestra il modello di regressione logistica sui dati forniti.
        """
        if X_train is None or y_train is None:
            raise ValueError("Dati di training non validi o non forniti.")

        print("🚀 Addestramento Logistic Regression in corso...")
        self.model.fit(X_train, y_train)
        print("✅ Addestramento completato.")

    def predict(self, X):
        """
        Effettua previsioni su nuovi dati.
        Restituisce: (classi_predette, probabilità_predette)
        """
        if self.model is None:
            raise RuntimeError("Il modello non è stato ancora addestrato.")
        y_pred_class = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1] if hasattr(self.model, "predict_proba") else None
        return y_pred_class, y_pred_proba

    def save_model(self):
        """
        Salva il modello addestrato in un file .pkl nella cartella specificata.
        """
        model_path = os.path.join(self.model_dir, "logistic_regression_model.pkl")
        joblib.dump(self.model, model_path)
        print(f"💾 Modello salvato in: {model_path}")
        return model_path

    def load_model(self, model_path: str = None):
        """
        Carica un modello salvato da file .pkl.
        """
        if model_path is None:
            model_path = os.path.join(self.model_dir, "logistic_regression_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modello non trovato: {model_path}")
        self.model = joblib.load(model_path)
        print(f"📂 Modello caricato da: {model_path}")
        return self.model
