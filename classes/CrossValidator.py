import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score

class CrossValidator:
    def __init__(self, model, cv=5, scoring=None, random_state=42):
        """
        Valida un modello tramite cross-validation e permette
        di ottenere il miglior modello allenato tra i vari fold.
        """
        self.model = model
        self.cv = cv
        self.scoring = scoring if scoring is not None else ['accuracy', 'roc_auc']
        self.random_state = random_state

        self.cv_results_ = None
        self.fitted_models = []   # ⭐ Modelli dei fold (ESSENZIALE)

    def validate(self, X, y):
        """
        Esegue la cross-validation manualmente, salvando i modelli
        addestrati in ogni fold così da poter selezionare il migliore.
        """
        cv_splitter = StratifiedKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state
        )

        # Dizionario risultati per ogni metrica
        self.cv_results_ = {f"test_{metric}": [] for metric in self.scoring}
        self.fitted_models = []  # reset

        print(f"Esecuzione di {self.cv}-Fold Cross-Validation...")

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y), 1):
            print(f"\n🔹 Fold {fold_idx}/{self.cv}")

            # Split
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            # Clona il modello (fonda essenziale per modelli indipendenti tra fold)
            model_clone = clone(self.model)

            # Addestramento
            model_clone.fit(X_train_fold, y_train_fold)

            # ⭐ Salviamo il modello del fold
            self.fitted_models.append(model_clone)

            # Predizione probabilità (se disponibile)
            if hasattr(model_clone, "predict_proba"):
                y_prob = model_clone.predict_proba(X_val_fold)[:, 1]
            else:
                # fallback per modelli senza predict_proba
                y_prob = model_clone.predict(X_val_fold)

            # Predizione classi
            y_pred = model_clone.predict(X_val_fold)

            # Calcolo metriche
            if "accuracy" in self.scoring:
                acc = accuracy_score(y_val_fold, y_pred)
                self.cv_results_["test_accuracy"].append(acc)

            if "roc_auc" in self.scoring:
                try:
                    auc = roc_auc_score(y_val_fold, y_prob)
                except:
                    auc = 0.0
                self.cv_results_["test_roc_auc"].append(auc)

        # Stampa riepilogo
        print("\n--- Risultati Cross-Validation ---")
        for metric in self.scoring:
            scores = self.cv_results_[f"test_{metric}"]
            print(f"{metric}: mean={pd.Series(scores).mean():.4f}, std={pd.Series(scores).std():.4f}")

        return self

    def get_cv_results(self):
        """Restituisce i risultati completi della cross-validation."""
        return self.cv_results_

    def get_mean_test_scores(self):
        """Restituisce le medie delle metriche sui fold."""
        return {metric: pd.Series(self.cv_results_[f"test_{metric}"]).mean()
                for metric in self.scoring}

    def get_std_test_scores(self):
        """Restituisce le deviazioni standard delle metriche sui fold."""
        return {metric: pd.Series(self.cv_results_[f"test_{metric}"]).std()
                for metric in self.scoring}
