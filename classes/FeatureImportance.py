# classes/FeatureImportance.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.inspection import permutation_importance
import os
import warnings
warnings.filterwarnings('ignore')

class FeatureImportanceAnalyzer:
    def __init__(self, model_dir="model", fig_dir="presentation_figures"):
        self.model_dir = model_dir
        self.fig_dir = fig_dir
        os.makedirs(fig_dir, exist_ok=True)
        
        # Colori per i diversi modelli
        self.model_colors = {
            'Logistic Regression': '#1f77b4',
            'XGBoost': '#ff7f0e', 
            'Keras NN': '#2ca02c'
        }
        
    def analyze_logistic_regression_importance(self, model, feature_names, X_val, y_val):
        """Analizza l'importanza delle feature per Logistic Regression."""
        try:
            if hasattr(model, 'coef_'):
                coefficients = model.coef_[0]
                
                # Crea DataFrame con i coefficienti
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': coefficients,
                    'Absolute_Coefficient': np.abs(coefficients)
                }).sort_values('Absolute_Coefficient', ascending=False)
                
                # Calcola permutation importance
                print("    Calcolo permutation importance per Logistic Regression...")
                perm_importance = permutation_importance(
                    model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1
                )
                
                importance_df['Permutation_Importance'] = perm_importance.importances_mean
                importance_df['Permutation_Std'] = perm_importance.importances_std
                
                return importance_df
            else:
                print("    ⚠️ Modello Logistic Regression non ha coefficienti")
                return None
                
        except Exception as e:
            print(f"    ❌ Errore nell'analisi Logistic Regression: {e}")
            return None
    
    def analyze_xgboost_importance(self, model, feature_names, importance_type='weight'):
        """Analizza l'importanza delle feature per XGBoost."""
        try:
            # Prova diversi metodi per ottenere l'importanza
            if hasattr(model, 'get_booster'):
                importance_scores = model.get_booster().get_score(importance_type=importance_type)
                importance_values = [importance_scores.get(f, 0) for f in feature_names]
            else:
                importance_values = model.feature_importances_
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance_values
            }).sort_values('Importance', ascending=False)
            
            # Normalizza i punteggi in percentuale
            total_importance = importance_df['Importance'].sum()
            if total_importance > 0:
                importance_df['Importance_Percentage'] = (
                    importance_df['Importance'] / total_importance * 100
                )
            else:
                importance_df['Importance_Percentage'] = 0
                
            return importance_df
            
        except Exception as e:
            print(f"    ❌ Errore nell'analisi XGBoost: {e}")
            return None
    
    def analyze_keras_importance(self, model, feature_names, X_val, y_val, n_repeats=5):
        """Analizza l'importanza delle feature per il modello Keras usando permutation importance."""
        try:
            print("    Calcolo permutation importance per Keras (può richiedere tempo)...")
            
            # Usa un subset per velocizzare il calcolo
            if len(X_val) > 1000:
                indices = np.random.choice(len(X_val), 1000, replace=False)
                X_sample = X_val[indices] if hasattr(X_val, 'iloc') else X_val[indices]
                y_sample = y_val.iloc[indices] if hasattr(y_val, 'iloc') else y_val[indices]
            else:
                X_sample = X_val
                y_sample = y_val
            
            # Permutation importance
            perm_importance = permutation_importance(
                model, X_sample, y_sample, 
                n_repeats=n_repeats, 
                random_state=42,
                scoring='accuracy',
                n_jobs=-1
            )
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Permutation_Importance': perm_importance.importances_mean,
                'Permutation_Std': perm_importance.importances_std
            }).sort_values('Permutation_Importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            print(f"    ❌ Errore nell'analisi Keras: {e}")
            return None
    
    def plot_individual_importance(self, importance_df, model_name, top_n=15):
        """Crea un grafico individuale per l'importanza delle feature di un modello."""
        if importance_df is None or len(importance_df) == 0:
            return None
            
        # Prendi le top feature
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        
        # Determina quale metrica usare per il plotting
        if 'Absolute_Coefficient' in top_features.columns:
            # Logistic Regression - usa coefficienti assoluti
            data_to_plot = top_features.sort_values('Absolute_Coefficient', ascending=True)
            sns.barplot(data=data_to_plot, x='Absolute_Coefficient', y='Feature', 
                       palette='viridis', alpha=0.8)
            plt.xlabel('Absolute Coefficient Value')
            plt.title(f'Feature Importance - {model_name}\n(Absolute Coefficients)')
            
        elif 'Importance_Percentage' in top_features.columns:
            # XGBoost - usa importanza normalizzata
            data_to_plot = top_features.sort_values('Importance_Percentage', ascending=True)
            sns.barplot(data=data_to_plot, x='Importance_Percentage', y='Feature', 
                       palette='plasma', alpha=0.8)
            plt.xlabel('Importance (%)')
            plt.title(f'Feature Importance - {model_name}')
            
        elif 'Permutation_Importance' in top_features.columns:
            # Keras - usa permutation importance
            data_to_plot = top_features.sort_values('Permutation_Importance', ascending=True)
            sns.barplot(data=data_to_plot, x='Permutation_Importance', y='Feature', 
                       palette='coolwarm', alpha=0.8)
            plt.xlabel('Mean Decrease in Accuracy')
            plt.title(f'Feature Importance - {model_name}\n(Permutation Importance)')
        
        plt.ylabel('')
        plt.tight_layout()
        
        # Salva il grafico
        filename = f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
        filepath = os.path.join(self.fig_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ Grafico salvato: {filename}")
        return filepath
    
    def plot_comparison(self, importance_dict, top_n=12):
        """Crea un grafico comparativo dell'importanza delle feature tra i modelli."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        models_plotted = 0
        for idx, (model_name, importance_df) in enumerate(importance_dict.items()):
            if importance_df is not None and len(importance_df) > 0:
                top_features = importance_df.head(top_n)
                ax = axes[models_plotted]
                
                if 'Absolute_Coefficient' in top_features.columns:
                    # Logistic Regression
                    data_to_plot = top_features.sort_values('Absolute_Coefficient', ascending=True)
                    bars = ax.barh(data_to_plot['Feature'], data_to_plot['Absolute_Coefficient'], 
                                 color=self.model_colors.get(model_name, '#1f77b4'), alpha=0.8)
                    ax.set_xlabel('Absolute Coefficient')
                    
                elif 'Importance_Percentage' in top_features.columns:
                    # XGBoost
                    data_to_plot = top_features.sort_values('Importance_Percentage', ascending=True)
                    bars = ax.barh(data_to_plot['Feature'], data_to_plot['Importance_Percentage'],
                                 color=self.model_colors.get(model_name, '#ff7f0e'), alpha=0.8)
                    ax.set_xlabel('Importance (%)')
                    
                elif 'Permutation_Importance' in top_features.columns:
                    # Keras
                    data_to_plot = top_features.sort_values('Permutation_Importance', ascending=True)
                    bars = ax.barh(data_to_plot['Feature'], data_to_plot['Permutation_Importance'],
                                 color=self.model_colors.get(model_name, '#2ca02c'), alpha=0.8)
                    ax.set_xlabel('Permutation Importance')
                
                ax.set_title(f'{model_name}', fontsize=14, fontweight='bold', 
                           color=self.model_colors.get(model_name, 'black'))
                ax.tick_params(axis='y', labelsize=10)
                
                # Aggiungi valori sulle barre
                for bar in bars:
                    width = bar.get_width()
                    if width > 0:  # Solo se il valore è positivo
                        ax.text(width + width * 0.01, bar.get_y() + bar.get_height()/2, 
                               f'{width:.2f}', ha='left', va='center', fontsize=9)
                
                models_plotted += 1
        
        # Nascondi gli assi vuoti
        for idx in range(models_plotted, 4):
            axes[idx].set_visible(False)
        
        plt.suptitle('Comparative Feature Importance Across Models', 
                    fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        
        # Salva il grafico comparativo
        filepath = os.path.join(self.fig_dir, 'feature_importance_comparison.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print("    ✅ Grafico comparativo salvato: feature_importance_comparison.png")
        return filepath
    
    def create_combined_heatmap(self, importance_dict, top_n=10):
        """Crea una heatmap combinata dell'importanza delle feature."""
        try:
            # Raccoglie le top feature da tutti i modelli
            all_top_features = set()
            for model_name, importance_df in importance_dict.items():
                if importance_df is not None and len(importance_df) > 0:
                    top_features = importance_df.head(top_n)['Feature'].tolist()
                    all_top_features.update(top_features)
            
            # Crea matrice comparativa
            comparison_data = []
            for feature in all_top_features:
                row = {'Feature': feature}
                for model_name, importance_df in importance_dict.items():
                    if importance_df is not None:
                        feature_data = importance_df[importance_df['Feature'] == feature]
                        if not feature_data.empty:
                            # Usa la metrica appropriata per ogni modello
                            if 'Absolute_Coefficient' in importance_df.columns:
                                row[model_name] = feature_data['Absolute_Coefficient'].values[0]
                            elif 'Importance_Percentage' in importance_df.columns:
                                row[model_name] = feature_data['Importance_Percentage'].values[0]
                            elif 'Permutation_Importance' in importance_df.columns:
                                row[model_name] = feature_data['Permutation_Importance'].values[0]
                        else:
                            row[model_name] = 0
                comparison_data.append(row)
            
            comparison_df = pd.DataFrame(comparison_data)
            if len(comparison_df) > 0:
                comparison_df = comparison_df.set_index('Feature')
                
                # Normalizza per riga per una migliore visualizzazione
                comparison_df_normalized = comparison_df.div(comparison_df.max(axis=1), axis=0)
                comparison_df_normalized = comparison_df_normalized.fillna(0)
                
                # Plot heatmap
                plt.figure(figsize=(12, 10))
                sns.heatmap(comparison_df_normalized, annot=True, cmap='YlOrRd', 
                           cbar_kws={'label': 'Normalized Importance'}, 
                           fmt='.2f', linewidths=0.5)
                plt.title('Comparative Feature Importance Heatmap\n(Normalized by Row)', 
                         fontsize=16, fontweight='bold')
                plt.tight_layout()
                
                filepath = os.path.join(self.fig_dir, 'feature_importance_heatmap.png')
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                plt.close()
                
                print("    ✅ Heatmap salvata: feature_importance_heatmap.png")
                return comparison_df_normalized
                
        except Exception as e:
            print(f"    ❌ Errore nella creazione della heatmap: {e}")
            return None
    
    def save_importance_results(self, importance_dict, out_dir="presentation_tables"):
        """Salva i risultati dell'importanza delle feature in file CSV."""
        os.makedirs(out_dir, exist_ok=True)
        
        for model_name, importance_df in importance_dict.items():
            if importance_df is not None and len(importance_df) > 0:
                filename = f"feature_importance_{model_name.lower().replace(' ', '_')}.csv"
                filepath = os.path.join(out_dir, filename)
                importance_df.to_csv(filepath, index=False)
                print(f"    ✅ Importanza features {model_name} salvata in: {filename}")
    
    def generate_summary_report(self, importance_dict, consensus_df):
        """Genera un report di sintesi dell'analisi delle feature."""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("FEATURE IMPORTANCE ANALYSIS SUMMARY")
        report_lines.append("=" * 60)
        
        # Top features per ogni modello
        for model_name, importance_df in importance_dict.items():
            if importance_df is not None and len(importance_df) > 0:
                report_lines.append(f"\n📊 {model_name.upper()}:")
                top_5 = importance_df.head(5)
                for idx, row in top_5.iterrows():
                    if 'Absolute_Coefficient' in row:
                        report_lines.append(f"   {row['Feature']}: {row['Absolute_Coefficient']:.4f}")
                    elif 'Importance_Percentage' in row:
                        report_lines.append(f"   {row['Feature']}: {row['Importance_Percentage']:.2f}%")
                    elif 'Permutation_Importance' in row:
                        report_lines.append(f"   {row['Feature']}: {row['Permutation_Importance']:.4f}")
        
        # Top features per consensus
        if consensus_df is not None and len(consensus_df) > 0:
            report_lines.append(f"\n🎯 CONSENSUS TOP FEATURES:")
            top_5_consensus = consensus_df.head(5)
            for idx, row in top_5_consensus.iterrows():
                report_lines.append(f"   {row['Feature']}: Score {row['consensus_score']} "
                                  f"({row['models_agreeing']} modelli)")
        
        report_lines.append("\n" + "=" * 60)
        
        report_text = "\n".join(report_lines)
        print(report_text)
        
        # Salva il report
        report_path = os.path.join(self.fig_dir, "feature_importance_summary.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"    ✅ Report di sintesi salvato: feature_importance_summary.txt")
        return report_text