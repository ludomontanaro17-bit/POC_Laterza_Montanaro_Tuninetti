import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, ClassifierMixin
import os


class FeatureImportanceAnalyzer:
    def __init__(self, model_dir="model", fig_dir="presentation_figures"):
        self.model_dir = model_dir
        self.fig_dir = fig_dir
        os.makedirs(fig_dir, exist_ok=True)

    def analyze_logistic_regression_importance(self, model, feature_names, X_val, y_val):
        """Analizza l'importanza delle feature per Logistic Regression."""
        if hasattr(model, 'coef_'):
            coefficients = model.coef_[0]
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': coefficients,
                'abs_coefficient': np.abs(coefficients)
            }).sort_values('abs_coefficient', ascending=False)

            # Calcola permutation importance
            perm_importance = permutation_importance(
                model, X_val, y_val, n_repeats=10, random_state=42, scoring='accuracy'
            )

            importance_df['permutation_importance'] = perm_importance.importances_mean
            importance_df['permutation_std'] = perm_importance.importances_std

            return importance_df
        return None

    def analyze_xgboost_importance(self, model, feature_names, importance_type='weight'):
        """Analizza l'importanza delle feature per XGBoost."""
        try:
            importance_scores = model.get_booster().get_score(importance_type=importance_type)
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': [importance_scores.get(f, 0) for f in feature_names]
            }).sort_values('importance', ascending=False)

            # Normalizza i punteggi
            importance_df['importance_normalized'] = (
                    importance_df['importance'] / importance_df['importance'].sum() * 100
            )

            return importance_df
        except:
            # Fallback per i metodi standard di XGBoost
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            importance_df['importance_normalized'] = (
                    importance_df['importance'] / importance_df['importance'].sum() * 100
            )

            return importance_df

    def analyze_keras_importance(self, model, feature_names, X_val, y_val):
        """Analizza l'importanza delle feature per il modello Keras usando permutation importance."""

        # Crea un wrapper per il modello Keras che implementa l'interfaccia scikit-learn
        class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
            def __init__(self, keras_model):
                self.keras_model = keras_model

            def fit(self, X, y):
                # Per permutation importance, il fit può essere un no-op
                # poiché il modello è già addestrato
                return self

            def predict(self, X):
                # Restituisce predizioni binarie (0 o 1) invece di probabilità
                y_pred_proba = self.keras_model.predict(X, verbose=0)
                return (y_pred_proba > 0.5).astype(int).flatten()

            def score(self, X, y):
                # Metodo score richiesto da alcuni estimator
                from sklearn.metrics import accuracy_score
                y_pred = self.predict(X)
                return accuracy_score(y, y_pred)

        # Usa il wrapper per permutation importance
        wrapped_model = KerasClassifierWrapper(model)

        # Permutation importance per modelli neurali
        perm_importance = permutation_importance(
            wrapped_model, X_val, y_val, n_repeats=10, random_state=42, scoring='accuracy'
        )

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'permutation_importance': perm_importance.importances_mean,
            'permutation_std': perm_importance.importances_std
        }).sort_values('permutation_importance', ascending=False)

        return importance_df

    def plot_feature_importance_comparison(self, importance_dict, top_n=15):
        """Crea un grafico comparativo dell'importanza delle feature tra i modelli."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()

        models = list(importance_dict.keys())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for idx, (model_name, importance_df) in enumerate(importance_dict.items()):
            if importance_df is not None:
                top_features = importance_df.head(top_n)

                if 'importance_normalized' in top_features.columns:
                    # XGBoost
                    ax = axes[idx]
                    sns.barplot(data=top_features, y='feature', x='importance_normalized',
                                ax=ax, palette='viridis')
                    ax.set_title(f'{model_name} - Feature Importance', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Importance (%)')

                elif 'coefficient' in top_features.columns:
                    # Logistic Regression
                    ax = axes[idx]
                    top_features = top_features.sort_values('coefficient', ascending=True)
                    sns.barplot(data=top_features, y='feature', x='coefficient',
                                ax=ax, palette='coolwarm')
                    ax.set_title(f'{model_name} - Coefficient Magnitude', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Coefficient Value')
                    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)

                elif 'permutation_importance' in top_features.columns:
                    # Keras o permutation importance
                    ax = axes[idx]
                    sns.barplot(data=top_features, y='feature', x='permutation_importance',
                                ax=ax, palette='plasma')
                    ax.set_title(f'{model_name} - Permutation Importance', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Mean Decrease in Accuracy')

                ax.set_ylabel('')
                ax.tick_params(axis='y', labelsize=10)

        # Rimuovi gli assi vuoti
        for idx in range(len(models), 4):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir, 'feature_importance_comparison.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        return fig

    def create_combined_importance_heatmap(self, importance_dict, top_n=10):
        """Crea una heatmap combinata dell'importanza delle feature."""
        # Prendi le top features da ogni modello
        all_top_features = set()
        for model_name, importance_df in importance_dict.items():
            if importance_df is not None:
                top_features = importance_df.head(top_n)['feature'].tolist()
                all_top_features.update(top_features)

        # Crea una matrice comparativa
        comparison_data = []
        for feature in all_top_features:
            row = {'feature': feature}
            for model_name, importance_df in importance_dict.items():
                if importance_df is not None:
                    feature_data = importance_df[importance_df['feature'] == feature]
                    if not feature_data.empty:
                        if 'importance_normalized' in importance_df.columns:
                            row[model_name] = feature_data['importance_normalized'].values[0]
                        elif 'abs_coefficient' in importance_df.columns:
                            row[model_name] = feature_data['abs_coefficient'].values[0]
                        elif 'permutation_importance' in importance_df.columns:
                            row[model_name] = feature_data['permutation_importance'].values[0]
                    else:
                        row[model_name] = 0
            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.set_index('feature')

        # Normalizza per riga per una migliore visualizzazione
        comparison_df_normalized = comparison_df.div(comparison_df.max(axis=1), axis=0)

        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(comparison_df_normalized, annot=True, cmap='YlOrRd',
                    cbar_kws={'label': 'Normalized Importance'})
        plt.title('Comparative Feature Importance Across Models', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir, 'feature_importance_heatmap.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        return comparison_df

    def plot_horizontal_importance_bars(self, importance_dict, top_n=15):
        """Crea istogrammi orizzontali per l'importanza delle feature per ogni modello."""

        # Determina il numero di modelli
        n_models = len(importance_dict)
        fig, axes = plt.subplots(n_models, 1, figsize=(14, 5 * n_models))

        # Se c'è solo un modello, axes non è una lista
        if n_models == 1:
            axes = [axes]

        for idx, (model_name, importance_df) in enumerate(importance_dict.items()):
            if importance_df is not None:
                # Prendi le top N feature
                top_features = importance_df.head(top_n).copy()

                # Ordina per importanza (crescente per barplot orizzontale)
                top_features = top_features.sort_values(self._get_importance_column(importance_df),
                                                        ascending=True)

                ax = axes[idx]

                # Seleziona il colore in base al tipo di modello
                color = self._get_model_color(model_name)

                # Crea il barplot orizzontale
                bars = ax.barh(top_features['feature'],
                               top_features[self._get_importance_column(importance_df)],
                               color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

                # Aggiungi i valori sulle barre
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + (width * 0.01), bar.get_y() + bar.get_height() / 2,
                            f'{width:.3f}', ha='left', va='center', fontsize=9)

                # Personalizza il grafico
                ax.set_title(f'{model_name} - Top {top_n} Feature Importance',
                             fontsize=14, fontweight='bold', pad=20)
                ax.set_xlabel(self._get_importance_label(importance_df), fontsize=12)
                ax.set_ylabel('Features', fontsize=12)
                ax.grid(axis='x', alpha=0.3, linestyle='--')

                # Migliora la leggibilità
                ax.tick_params(axis='y', labelsize=10)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir, 'horizontal_feature_importance.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        return fig

    def _get_importance_column(self, importance_df):
        """Identifica la colonna di importanza corretta per il DataFrame."""
        if 'importance_normalized' in importance_df.columns:
            return 'importance_normalized'
        elif 'abs_coefficient' in importance_df.columns:
            return 'abs_coefficient'
        elif 'permutation_importance' in importance_df.columns:
            return 'permutation_importance'
        elif 'coefficient' in importance_df.columns:
            return 'coefficient'
        else:
            return importance_df.columns[1]  # Fallback sulla seconda colonna

    def _get_importance_label(self, importance_df):
        """Restituisce l'etichetta corretta per l'asse x in base al tipo di importanza."""
        if 'importance_normalized' in importance_df.columns:
            return 'Importance (%)'
        elif 'abs_coefficient' in importance_df.columns:
            return 'Absolute Coefficient Value'
        elif 'permutation_importance' in importance_df.columns:
            return 'Permutation Importance (Mean Decrease in Accuracy)'
        elif 'coefficient' in importance_df.columns:
            return 'Coefficient Value'
        else:
            return 'Importance Score'

    def _get_model_color(self, model_name):
        """Restituisce un colore specifico per ogni tipo di modello."""
        color_map = {
            'Logistic Regression': '#1f77b4',  # Blu
            'XGBoost': '#ff7f0e',  # Arancione
            'Keras': '#2ca02c',  # Verde
            'Random Forest': '#d62728',  # Rosso
            'SVM': '#9467bd',  # Viola
            'Neural Network': '#8c564b'  # Marrone
        }
        return color_map.get(model_name, '#17becf')  # Default: ciano

    def plot_stacked_importance_comparison(self, importance_dict, top_n=10):
        """Crea un grafico a barre impilate per confrontare le top feature tra i modelli."""

        # Trova le feature più importanti complessivamente
        all_scores = {}
        for model_name, importance_df in importance_dict.items():
            if importance_df is not None:
                for _, row in importance_df.head(top_n).iterrows():
                    feature = row['feature']
                    score = row[self._get_importance_column(importance_df)]
                    if feature not in all_scores:
                        all_scores[feature] = []
                    all_scores[feature].append(score)

        # Calcola lo score medio per ogni feature
        avg_scores = {feature: np.mean(scores) for feature, scores in all_scores.items()}

        # Prendi le top N feature globali
        top_global_features = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_features = [feature for feature, score in top_global_features]

        # Prepara i dati per il grafico impilato
        plot_data = []
        for feature in top_features:
            for model_name, importance_df in importance_dict.items():
                if importance_df is not None:
                    feature_data = importance_df[importance_df['feature'] == feature]
                    if not feature_data.empty:
                        score = feature_data[self._get_importance_column(importance_df)].values[0]
                        plot_data.append({
                            'feature': feature,
                            'model': model_name,
                            'importance': score
                        })

        plot_df = pd.DataFrame(plot_data)

        # Crea il grafico
        plt.figure(figsize=(15, 10))

        # Pivot per avere modelli come colonne
        pivot_df = plot_df.pivot(index='feature', columns='model', values='importance').fillna(0)

        # Ordina le feature per importanza totale
        pivot_df['total'] = pivot_df.sum(axis=1)
        pivot_df = pivot_df.sort_values('total', ascending=True)
        pivot_df = pivot_df.drop('total', axis=1)

        # Crea il barplot orizzontale impilato
        colors = [self._get_model_color(model) for model in pivot_df.columns]
        bars = pivot_df.plot.barh(stacked=True, color=colors, ax=plt.gca(),
                                  edgecolor='black', linewidth=0.5)

        plt.title(f'Top {top_n} Global Features - Importance Across Models',
                  fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Combined Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='x', alpha=0.3, linestyle='--')

        # Rimuovi bordi
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir, 'stacked_feature_importance.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def save_importance_results(self, importance_dict, out_dir="presentation_tables"):
        """Salva i risultati dell'importanza delle feature in file CSV."""
        os.makedirs(out_dir, exist_ok=True)

        for model_name, importance_df in importance_dict.items():
            if importance_df is not None:
                filename = f"feature_importance_{model_name.lower().replace(' ', '_')}.csv"
                filepath = os.path.join(out_dir, filename)
                importance_df.to_csv(filepath, index=False)
                print(f"✅ Importanza features {model_name} salvata in: {filepath}")