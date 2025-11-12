import xgboost as xgb
import numpy as np
import pandas as pd
import os
import joblib

class XGBoostModel:
    def __init__(self,
                 objective='binary:logistic',
                 eval_metric='auc',
                 use_label_encoder=False,
                 model_dir="model"):
        """
        Wrapper per il modello XGBoost integrabile nel tuo ensemble Flask.
        """
        self.objective = objective
        self.eval_metric = eval_metric
        self.use_label_encoder = use_label_encoder
        self.model_dir = model_dir

        # Crea il modello di base
        self.model = xgb.XGBClassifier(
            objective=self.objective,
            eval_metric=self.eval_metric,
            use_label_encoder=self.use_label_encoder,
            random_state=42
        )

        # Assicura che la cartella esista
        os.makedirs(self.model_dir, exist_ok=True)

    # ============================================
    # 🔹 TRAINING
    # ============================================
    def train(self, X_train, y_train, X_val=None, y_val=None, **xgb_params):
        """
        Addestra il modello XGBoost, gestendo anche class imbalance.
        """
        neg, pos = np.bincount(y_train)
        scale_pos_weight = neg / pos
        print(f"⚖️ Calcolato scale_pos_weight = {scale_pos_weight:.2f}")

        # Parametri del modello
        xgb_params['scale_pos_weight'] = scale_pos_weight
        self.model.set_params(**xgb_params)

        print("🚀 Inizio addestramento XGBoost...")
        self.model.fit(
            X_train, y_train,
            verbose=True
        )
        print("✅ Addestramento completato.")

    # ============================================
    # 🔹 PREDICTION
    # ============================================
    def predict(self, X):
        """Restituisce la probabilità della classe positiva (1)."""
        return self.model.predict_proba(X)[:, 1]

    def predict_classes(self, X, threshold=0.5):
        """Restituisce la classe (0 o 1) in base alla soglia."""
        return (self.predict(X) > threshold).astype(int)

    # ============================================
    # 🔹 SALVATAGGIO & CARICAMENTO
    # ============================================
    def save_model(self, filename="xgboost_model.pkl"):
        """
        Salva il modello in formato pickle, mantenendo metadati sklearn.
        """
        filepath = os.path.join(self.model_dir, filename)
        joblib.dump(self.model, filepath)
        print(f"💾 Modello XGBoost salvato in formato pickle: {filepath}")

    def load_model(self, filename="xgboost_model.pkl"):
        """
        Carica un modello XGBoost salvato in formato pickle.
        """
        filepath = os.path.join(self.model_dir, filename)
        if os.path.exists(filepath):
            self.model = joblib.load(filepath)
            print(f"✅ Modello XGBoost caricato da: {filepath}")

            # Debug: verifica che siano mantenuti i metadati
            print("🧩 Feature names:", getattr(self.model, "feature_names_in_", None))
            print("🧩 Classi:", getattr(self.model, "classes_", None))
        else:
            print(f"❌ Errore: Il file {filepath} non esiste.")

# =====================================================
# ✅ ESEMPIO DI UTILIZZO (solo per test locali)
# =====================================================
# if __name__ == "__main__":
#     from sklearn.datasets import load_breast_cancer
#     from sklearn.model_selection import train_test_split
#     data = load_breast_cancer()
#     X_train, X_test, y_train, y_test = train_test_split(
#         data.data, data.target, test_size=0.2, random_state=42)
#
#     model = XGBoostModel()
#     model.train(X_train, y_train, n_estimators=100, max_depth=4)
#     model.save_model()
#
#     # Ricarica e testa
#     model2 = XGBoostModel()
#     model2.load_model()
#     preds = model2.predict(X_test)
#     print("🔍 Esempio predizioni:", preds[:5])
