# classes/FeatureImportance.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.inspection import permutation_importance
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
                model, X_val, y_val, n_repeats=10, random_state=42
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
        # Permutation importance per modelli neurali
        perm_importance = permutation_importance(
            model, X_val, y_val, n_repeats=10, random_state=42, scoring='accuracy'
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
    
    def save_importance_results(self, importance_dict, out_dir="presentation_tables"):
        """Salva i risultati dell'importanza delle feature in file CSV."""
        os.makedirs(out_dir, exist_ok=True)
        
        for model_name, importance_df in importance_dict.items():
            if importance_df is not None:
                filename = f"feature_importance_{model_name.lower().replace(' ', '_')}.csv"
                filepath = os.path.join(out_dir, filename)
                importance_df.to_csv(filepath, index=False)
                print(f"✅ Importanza features {model_name} salvata in: {filepath}")