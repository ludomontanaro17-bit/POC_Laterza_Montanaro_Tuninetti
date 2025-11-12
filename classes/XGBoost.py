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
        # Rimuovi early_stopping_rounds dal costruttore
        self.model = xgb.XGBClassifier(
            objective=self.objective,
            eval_metric=self.eval_metric,
            use_label_encoder=self.use_label_encoder,
            # Potresti voler impostare altri iperparametri di base qui
            # n_estimators=100, # Ad esempio, imposta un numero fisso di alberi
            # learning_rate=0.1,
            # max_depth=6,
        )
        # Assicura che la cartella per il modello esista
        os.makedirs(self.model_dir, exist_ok=True)

    def train(self, X_train, y_train, X_val=None, y_val=None, **xgb_params):
        """
        Addestra il modello XGBoost.

        Args:
            X_train, y_train: Dati di training.
            X_val, y_val: Dati di validazione per l'early stopping (opzionale).
                         ATTENZIONE: Questo parametro è ora ignorato.
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
        # Attenzione: Se X_val e y_val sono forniti, ma non si vuole usare l'early stopping,
        # non devono essere passati a fit() come eval_set.
        # Rimuoviamo eventuali parametri specifici di early stopping dai parametri aggiuntivi se presenti
        xgb_params.pop('early_stopping_rounds', None) # Rimuove il parametro se presente
        xgb_params.pop('eval_set', None) # Rimuove eval_set se accidentalmente fornito
        xgb_params.pop('eval_names', None) # Rimuove eval_names se accidentalmente fornito

        self.model.set_params(**xgb_params)

        # Addestra il modello *senza* specificare eval_set o early_stopping_rounds
        print("Inizio addestramento del modello XGBoost (senza early stopping)...")
        self.model.fit(
            X_train, y_train,
            # eval_set=eval_set, # Non usato
            # early_stopping_rounds=early_stopping_rounds, # Non usato
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

    def predict_proba(self, X):
        """
        Restituisce le probabilità predette per la classe positiva.

        Args:
            X: Dati di input

        Returns:
            np.array: Probabilità per la classe positiva
        """
        return self.model.predict_proba(X)[:, 1]

# Esempio di utilizzo (opzionale)
# if __name__ == "__main__":
#     # Questo richiede dati X_train, y_train, X_val, y_val preprocessati
#     # xgb_model = XGBoostModel()
#     # xgb_model.train(X_train, y_train) # Senza X_val/y_val o con X_val/y_val ora ignorati per early stopping
#     pass