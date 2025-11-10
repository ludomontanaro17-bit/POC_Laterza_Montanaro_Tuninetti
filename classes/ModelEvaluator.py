import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self, y_true, y_pred_proba, y_pred_class=None, model_name="Model"):
        """
        Inizializza il valutatore con i risultati del modello.

        Args:
            y_true (np.array): Etichette vere.
            y_pred_proba (np.array): Probabilità predette dal modello (es. output del sigmoid).
            y_pred_class (np.array, optional): Classi predette (es. output di predict_classes). Se non fornito, viene calcolato con soglia 0.5.
            model_name (str): Nome del modello per i grafici/report.
        """
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        if y_pred_class is None:
            self.y_pred_class = (y_pred_proba > 0.5).astype(int)
        else:
            self.y_pred_class = y_pred_class
        self.model_name = model_name

    def calculate_metrics(self):
        """
        Calcola e stampa le metriche principali.
        """
        val_accuracy = np.mean(self.y_true == self.y_pred_class)
        auc_score = roc_auc_score(self.y_true, self.y_pred_proba)

        print(f"\n--- Risultati per: {self.model_name} ---")
        print(f"Accuracy: {val_accuracy:.4f}")
        print(f"AUC-ROC Score: {auc_score:.4f}")

        print("\n--- Report di Classificazione ---")
        print(classification_report(self.y_true, self.y_pred_class))

        print("\n--- Matrice di Confusione ---")
        print(confusion_matrix(self.y_true, self.y_pred_class))

        return {
            'accuracy': val_accuracy,
            'auc_roc': auc_score,
            'classification_report': classification_report(self.y_true, self.y_pred_class, output_dict=True),
            'confusion_matrix': confusion_matrix(self.y_true, self.y_pred_class)
        }

    def plot_confusion_matrix(self):
        """
        Plotta la matrice di confusione.
        """
        cm = confusion_matrix(self.y_true, self.y_pred_class)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matrice di Confusione - {self.model_name}')
        plt.xlabel('Predetto')
        plt.ylabel('Reale')
        plt.show()

    def plot_roc_curve(self):
        """
        Plotta la curva ROC.
        """
        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_proba)
        auc_score = roc_auc_score(self.y_true, self.y_pred_proba)

        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{self.model_name} (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.show()

    def evaluate(self):
        """
        Esegue l'intero processo di valutazione: calcola metriche e crea i grafici.
        """
        metrics = self.calculate_metrics()
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        return metrics