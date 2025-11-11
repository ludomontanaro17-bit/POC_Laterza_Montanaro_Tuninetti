import xgboost as xgb
import numpy as np
import pandas as pd
import os
import joblib

class XGBoostModel:
    def __init__(self, objective='binary:logistic', eval_metric='auc', use_label_encoder=False, model_dir="model"):
        """
        Inizializza il wrapper per il modello XGBoost.

        Args:
            objective (str): Obiettivo del modello (default 'binary:logistic' per classificazione binaria).
            eval_metric (str or list): Metrica(i) da usare per la valutazione durante il training.
            use_label_encoder (bool): Disabilita l'encoder di XGBoost per le classi (raccomandato).
            model_dir (str): Directory dove salvare il modello.
        """
        self.objective = objective
        self.eval_metric = eval_metric
        self.use_label_encoder = use_label_encoder
        self.model_dir = model_dir
        self.model = xgb.XGBClassifier(
            objective=self.objective,
            eval_metric=self.eval_metric,
            use_label_encoder=self.use_label_encoder,
            # Potresti voler impostare altri iperparametri di base qui
            # n_estimators=100,
            # learning_rate=0.1,
            # max_depth=6,
        )
        # Assicura che la cartella per il modello esista
        os.makedirs(self.model_dir, exist_ok=True)

    def train(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=10, **xgb_params):
        """
        Addestra il modello XGBoost.

        Args:
            X_train, y_train: Dati di training.
            X_val, y_val: Dati di validazione per l'early stopping (opzionale).
            early_stopping_rounds (int): Rounds senza miglioramento prima di fermarsi.
            **xgb_params: Altri parametri specifici di XGBoost (es. n_estimators, learning_rate, max_depth, scale_pos_weight).
                         Questi sovrascrivono i valori di default del modello.
        """
        # Gestione dello sbilanciamento con scale_pos_weight
        # Calcola il rapporto tra il numero di osservazioni della classe negativa e quella positiva
        # scale_pos_weight = (numero di osservazioni con classe 0) / (numero di osservazioni con classe 1)
        # Questo è un modo efficace per gestire il bilanciamento in XGBoost.
        neg, pos = np.bincount(y_train)
        scale_pos_weight = neg / pos
        print(f"Calcolato scale_pos_weight: {scale_pos_weight:.2f} per gestire lo sbilanciamento.")
        xgb_params['scale_pos_weight'] = scale_pos_weight # Aggiungi il peso calcolato ai parametri

        # Aggiorna i parametri del modello con quelli forniti
        self.model.set_params(**xgb_params)

        # Prepara le liste per la valutazione durante il training
        eval_set = [(X_train, y_train)]
        eval_names = ['train']
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            eval_names.append('val')

        print("Inizio addestramento del modello XGBoost...")
        # Addestra il modello
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_names=eval_names,
            early_stopping_rounds=early_stopping_rounds,
            verbose=True # Mostra la progressione del training
        )
        print("Addestramento completato.")

    def predict(self, X):
        """
        Fai previsioni delle probabilità sul dataset X.

        Args:
            X (np.array or pd.DataFrame): Dati di input.

        Returns:
            np.array: Array delle probabilità predette per la classe positiva.
        """
        return self.model.predict_proba(X)[:, 1] # Prende la probabilità della classe positiva (1)

    def predict_classes(self, X, threshold=0.5):
        """
        Fai previsioni delle classi sul dataset X.

        Args:
            X (np.array or pd.DataFrame): Dati di input.
            threshold (float): Soglia per la classificazione binaria.

        Returns:
            np.array: Array delle classi predette (0 o 1).
        """
        y_pred_proba = self.predict(X)
        return (y_pred_proba > threshold).astype(int)

    def save_model(self, filename="xgboost_model.json"):
        """
        Salva il modello XGBoost.

        Args:
            filename (str): Nome del file per salvare il modello (formato JSON consigliato).
        """
        filepath = os.path.join(self.model_dir, filename)
        self.model.save_model(filepath)
        print(f"Modello XGBoost salvato in: {filepath}")

    def load_model(self, filename="xgboost_model.json"):
        """
        Carica un modello XGBoost salvato.

        Args:
            filename (str): Nome del file del modello da caricare.
        """
        filepath = os.path.join(self.model_dir, filename)
        if os.path.exists(filepath):
            self.model = xgb.XGBClassifier()
            self.model.load_model(filepath)
            print(f"Modello XGBoost caricato da: {filepath}")
        else:
            print(f"Errore: Il file {filepath} non esiste.")