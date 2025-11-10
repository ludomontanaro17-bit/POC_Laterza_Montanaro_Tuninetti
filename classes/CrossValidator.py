import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from scikeras.wrappers import KerasClassifier
from classes.KerasModel import KerasModel # Assumendo che la tua classe KerasModel esista

class CrossValidator:
    def __init__(self, model, cv=5, scoring=None, random_state=42):
        """
        Inizializza il validatore incrociato.

        Args:
            model: Un modello scikit-learn o un KerasClassifier/Regressor.
            cv (int): Numero di fold per la cross-validation.
            scoring (list or str): Metriche da calcolare (es. ['accuracy', 'roc_auc']).
                                   Se None, usa ['accuracy', 'roc_auc'].
            random_state (int): Seme per la riproducibilità.
        """
        self.model = model
        self.cv = cv
        self.scoring = scoring if scoring is not None else ['accuracy', 'roc_auc']
        self.random_state = random_state
        self.cv_results_ = None

    def validate(self, X, y):
        """
        Esegue la cross-validation.

        Args:
            X (np.array or pd.DataFrame): Features.
            y (np.array or pd.Series): Target.

        Returns:
            self: Restituisce l'oggetto validatore con i risultati.
        """
        # Usa StratifiedKFold per mantenere la distribuzione del target
        cv_splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        print(f"Esecuzione di {self.cv}-Fold Cross-Validation...")
        self.cv_results_ = cross_validate(
            estimator=self.model,
            X=X, y=y,
            cv=cv_splitter,
            scoring=self.scoring,
            return_train_score=True, # Opzionale: restituisce anche i punteggi di training
            n_jobs=-1 # Usa tutti i core disponibili
        )

        # Calcola e stampa le medie e le deviazioni standard
        print(f"\n--- Risultati {self.cv}-Fold Cross-Validation ---")
        for metric in self.scoring:
            train_key = f'train_{metric}'
            test_key = f'test_{metric}'
            if train_key in self.cv_results_:
                train_scores = self.cv_results_[train_key]
                print(f"Mean {metric} (Train): {train_scores.mean():.4f} (+/- {train_scores.std() * 2:.4f})")
            test_scores = self.cv_results_[test_key]
            print(f"Mean {metric} (Test): {test_scores.mean():.4f} (+/- {test_scores.std() * 2:.4f})")

        return self

    def get_cv_results(self):
        """Restituisce i risultati completi della cross-validation."""
        return self.cv_results_

    def get_mean_test_scores(self):
        """Restituisce un dizionario con le medie dei punteggi di test per ogni metrica."""
        mean_scores = {}
        for metric in self.scoring:
            test_key = f'test_{metric}'
            if test_key in self.cv_results_:
                mean_scores[metric] = self.cv_results_[test_key].mean()
        return mean_scores

    def get_std_test_scores(self):
        """Restituisce un dizionario con le deviazioni standard dei punteggi di test per ogni metrica."""
        std_scores = {}
        for metric in self.scoring:
            test_key = f'test_{metric}'
            if test_key in self.cv_results_:
                std_scores[metric] = self.cv_results_[test_key].std()
        return std_scores