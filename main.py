# main.py (aggiornato con CrossValidator, XGBoost, e valutazioni estese)

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from classes.Datapreprocessing import DataPreprocessing
from classes.KaggleLoader import KaggleLoader
from classes.LogisticRegression import LogisticRegressionModel
from classes.ModelEvaluator import ModelEvaluator
from classes.KerasModel import KerasModel
from classes.FeatureImportance import FeatureImportanceAnalyzer
# --- Nuovi Import ---
from classes.XGBoost import XGBoostModel
from classes.CrossValidator import CrossValidator
# --------------------

def main():
    # === Directory di output ================================================
    out_dir = "presentation_tables"
    fig_dir = "presentation_figures"
    model_dir = "model"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # === Step 1: Caricamento del dataset ====================================
    loader = KaggleLoader(dataset="rishikeshkonapure/home-loan-approval", download_dir="./data")
    loader.load()
    loaded = loader.get_full_dataset()

    # Gestione formati diversi del loader
    if isinstance(loaded, tuple):
        df_train = next((x for x in loaded if isinstance(x, pd.DataFrame)), None)
    elif isinstance(loaded, dict):
        for k in ('train', 'train_df', 'train_data', 'data'):
            if k in loaded and isinstance(loaded[k], pd.DataFrame):
                df_train = loaded[k]
                break
        else:
            df_train = next((v for v in loaded.values() if isinstance(v, pd.DataFrame)), None)
    else:
        df_train = loaded if isinstance(loaded, pd.DataFrame) else None

    if df_train is None:
        raise SystemExit("❌ Dataset non trovato o formato inatteso.")

    print(f"✅ Dataset caricato: {df_train.shape[0]} righe, {df_train.shape[1]} colonne")

    # === Step 2: Analisi esplorativa ========================================
    prep = DataPreprocessing(df_train, target_column='Loan_Status')
    prep.display_info()
    prep.display_statistics()
    prep.display_missing_values()

    # --- Distribuzioni e Boxplot ---
    plt.figure(figsize=(6, 4))
    sns.countplot(x=prep.y, palette="viridis")
    plt.title("Distribuzione Loan Status")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "target_distribution.png"), dpi=150)
    plt.close()

    ## Plot distribuzione colonne categoriche
    categorical_cols = ['Gender', 'Married', 'Education', 'Property_Area']
    for col in categorical_cols:
        if col in df_train.columns:
            plt.figure(figsize=(6, 4))
            sns.countplot(
                x=col, data=df_train, palette="mako",
                order=df_train[col].value_counts().index
            )
            plt.title(f"Distribuzione di {col}")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"{col.lower()}_distribution.png"), dpi=150)
            plt.close()

## Plot distribuzione colonne numeriche
    numeric_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    if 'TotalIncome' in df_train.columns:
        numeric_cols.append('TotalIncome')

    for col in numeric_cols:
        if col in df_train.columns:
            # Istogramma
            plt.figure(figsize=(6, 4))
            sns.histplot(df_train[col], bins=30, kde=False, color="skyblue")
            plt.title(f"Istogramma di {col}")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"{col.lower()}_histogram.png"), dpi=150)
            plt.close()



    # === Step 3: Preprocessing ==============================================
    df_preprocessed = prep.advanced_preprocessing(
        impute_numeric='median',
        create_features=True
    )
    print(f"✅ Preprocessing completato. Nuova shape: {df_preprocessed.shape}")

    ## === 🔍 AGGIUNGI: VERIFICA DATI PREPROCESSATI ===========================
    #print("\n🔍 VERIFICA DATI PREPROCESSATI:")
    #print("Statistiche features numeriche dopo preprocessing:")
    #numeric_features = ['LoanAmount', 'TotalIncome', 'LoanAmount_to_Income']
    #for col in numeric_features:
    #    if col in df_preprocessed.columns:
    #        print(f"\n--- {col} ---")
    #        print(f"  Min: {df_preprocessed[col].min():.3f}")
    #        print(f"  Max: {df_preprocessed[col].max():.3f}")
    #        print(f"  Mean: {df_preprocessed[col].mean():.3f}")
    #        print(f"  Std: {df_preprocessed[col].std():.3f}")
#
    ## Verifica che i valori siano ragionevoli
    #print("\n✅ CONTROLLO VALORI RAGIONEVOLI:")
    #if 'LoanAmount_to_Income' in df_preprocessed.columns:
    #    reasonable_ratio = (df_preprocessed['LoanAmount_to_Income'] <= 10).all()
    #    print(f"  LoanAmount_to_Income <= 10: {reasonable_ratio}")
    #if 'TotalIncome' in df_preprocessed.columns:
    #    positive_income = (df_preprocessed['TotalIncome'] >= 0).all()
    #    print(f"  TotalIncome >= 0: {positive_income}")



    # === Step 4: Matrice di correlazione (DOPO preprocessing) ===============
    num_df = df_preprocessed.select_dtypes(include=['int64', 'float64']).copy()
    if 'Loan_Status' in df_preprocessed.columns and not pd.api.types.is_numeric_dtype(df_preprocessed['Loan_Status']):
        num_df['Loan_Status_bin'] = df_preprocessed['Loan_Status'].map({'Y': 1, 'N': 0})

    if num_df.shape[1] > 1:
        corr = num_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
        plt.title("Matrice di correlazione (post-preprocessing)")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "correlation_matrix_post_preprocessing.png"), dpi=150)
        plt.close()
        print("✅ Matrice di correlazione salvata.")


    ## DEBUG
    #print("\n🔍 DEBUG - Distribuzione classi dopo preprocessing:")
    #print("Valori unici in y:", df_preprocessed['Loan_Status'].unique())
    #print("Conteggi:")
    #print(df_preprocessed['Loan_Status'].value_counts())
    #print("Proporzioni:")
    #print(df_preprocessed['Loan_Status'].value_counts(normalize=True))
#
    ## Verifica che '1' corrisponda effettivamente a 'Y' (approvato)
    #target_mapping = joblib.load("model/target_mapping_Loan_Status.pkl")
    #print("Mappatura target:", target_mapping)



    # === Step 5: Split dei dati =============================================
    X_train, X_val, y_train, y_val = prep.split_data(df_preprocessed, test_size=0.1, random_state=42)
    print("✅ Train-test split completato.")
    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")
    print(X_train.columns)

    # === Step 5.5: Prepara i nomi delle feature =============================
    final_feature_list = df_preprocessed.drop(columns=['Loan_Status']).columns.tolist()

    # Salva UNA sola volta i nomi delle feature usate dai modelli
    joblib.dump(final_feature_list, os.path.join(model_dir, "feature_names.pkl"))
    joblib.dump(final_feature_list, os.path.join(model_dir, "final_columns.pkl"))

    # === Step 6: Salvataggio oggetti utili ==================================
    if hasattr(prep, 'scaler'):
        joblib.dump(prep.scaler, os.path.join(model_dir, "scaler.pkl"))

    print(f"\n📁 Figure salvate in: {fig_dir}")
    print(f"📁 Oggetti salvati in: {model_dir}")

    # Rimuovi i print di debug se non servono più
    # print(X_train.dtypes[X_train.dtypes == 'category'])
    # for col in X_train.columns:
    #     if str(X_train[col].dtype) == 'category':
    #         print(f"\n🔍 {col} → {X_train[col].unique()[:10]}")

    # --- STEP 6: Addestramento Logistic Regression ==========================
    print("\n--- Logistic Regression: Cross-Validation (Train Set) ---")

    logreg_base = LogisticRegressionModel(model_dir="model")

    cv_lr = CrossValidator(
        model=logreg_base.model,
        cv=5,
        scoring=['accuracy', 'roc_auc']
    )

    cv_lr.validate(X_train, y_train)

    # Seleziona il modello migliore (in base all'AUC)
    best_idx_lr = np.argmax(cv_lr.cv_results_['test_roc_auc'])
    best_lr_model = cv_lr.fitted_models[best_idx_lr]

    print(f"🎯 Miglior modello LR è il fold #{best_idx_lr} con AUC={cv_lr.cv_results_['test_roc_auc'][best_idx_lr]:.4f}")

    # --- SALVA IL MODELLO SCELTO DALLA CROSS ---
    joblib.dump(best_lr_model, os.path.join(model_dir, "logistic_regression_model.pkl"))
    print("💾 Logistic Regression salvato (best fold dalla cross validation).")

    # --- Predizioni sul validation set usando il modello migliore ---
    y_prob_lr = best_lr_model.predict_proba(X_val)[:, 1]
    y_pred_lr = (y_prob_lr > 0.5).astype(int)

    # --- STEP 7: Valutazione Logistic Regression (Test Set) =================
    print("\n🔍 Valutazione Logistic Regression (Test Set):")
    evaluator_lr = ModelEvaluator(
        y_true=y_val,
        y_pred_proba=y_prob_lr,
        y_pred_class=y_pred_lr,
        model_name="Logistic Regression"
    )
    metrics_lr = evaluator_lr.evaluate() # Ottieni i risultati per il confronto




    # --- STEP 9: Addestramento Rete Neurale Keras ===========================
    print("\n--- Addestramento Rete Neurale Keras ---")
    keras_model = KerasModel(input_dim=X_train.shape[1], model_dir="model")
    keras_model.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=50,
        batch_size=32,
        verbose=1
    )
    keras_model.save_model()

    # Predizioni Keras
    y_prob_keras = keras_model.predict(X_val).flatten()
    y_pred_keras = (y_prob_keras > 0.5).astype(int)

    # --- STEP 10: Valutazione Keras (Test Set) ==============================
    print("\n🔍 Valutazione Rete Neurale Keras (Test Set):")
    evaluator_nn = ModelEvaluator(
        y_true=y_val,
        y_pred_proba=y_prob_keras,
        y_pred_class=y_pred_keras,
        model_name="Keras Neural Network"
    )
    metrics_keras = evaluator_nn.evaluate() # Ottieni i risultati per il confronto

    # --- STEP 11: Cross-Validation Keras (su Train Set) =====================
    print("\n🔍 Cross-Validation Keras (Train Set):")
    # Per Keras, dobbiamo usare il wrapper KerasClassifier da scikeras
    # Assicurati di averlo installato: pip install scikeras
    # from scikeras.wrappers import KerasClassifier
    # def build_fn():
    #     # Copia la logica di _build_model da KerasModel qui
    #     # o passa l'istanza del modello compilato
    #     pass # Implementazione richiesta
    # cv_keras_model = KerasClassifier(model=build_fn, epochs=10, batch_size=32) # Usa poche epoche per CV
    # cv_keras = CrossValidator(model=cv_keras_model, cv=3, scoring=['accuracy', 'roc_auc']) # CV rapida
    # cv_keras.validate(X_train, y_train)
    # cv_results_keras = cv_keras.get_mean_test_scores()
    # print(f"Mean CV Accuracy: {cv_results_keras['accuracy']:.4f}, Mean CV AUC: {cv_results_keras['roc_auc']:.4f}")
    # Per ora, saltiamo la CV per Keras per semplicità, a meno che tu non implementi il wrapper scikeras.

    # --- STEP 12: Addestramento XGBoost ====================================

    print("\n--- XGBoost: Cross-Validation (Train Set) ---")

    xgb_base = XGBoostModel(model_dir="model")

    cv_xgb = CrossValidator(
        model=xgb_base.model,
        cv=5,
        scoring=['accuracy', 'roc_auc']
    )

    cv_xgb.validate(X_train, y_train)

    # scegliamo il modello migliore sui fold
    best_idx_xgb = np.argmax(cv_xgb.cv_results_['test_roc_auc'])
    best_xgb_model = cv_xgb.fitted_models[best_idx_xgb]

    print(f"🎯 Miglior XGBoost fold = {best_idx_xgb}, AUC = {cv_xgb.cv_results_['test_roc_auc'][best_idx_xgb]:.4f}")

    # --- SALVA IL MIGLIOR MODELLO CON FORMATO CORRETTO ---
    try:
        # Metodo 1: Prova a salvare con encoding esplicito
        best_xgb_model.save_model('model/xgboost_model.json')
        print("💾 XGBoost salvato in formato JSON")

        # Verifica che il file sia leggibile
        with open('model/xgboost_model.json', 'r', encoding='utf-8') as f:
            content = f.read()
            print("✅ File JSON verificato e leggibile")

    except Exception as e:
        print(f"❌ Errore nel salvataggio JSON: {e}")
        # Metodo alternativo: salva come pickle
        try:
            joblib.dump(best_xgb_model, 'model/xgboost_model.pkl')
            print("💾 XGBoost salvato come file .pkl (alternativa)")
        except Exception as e2:
            print(f"❌ Errore anche nel salvataggio pickle: {e2}")
    # --- Predizioni ---
    y_prob_xgb = best_xgb_model.predict(X_val)
    y_pred_xgb = (y_prob_xgb > 0.5).astype(int)


    # --- STEP 13: Valutazione XGBoost (Test Set) ============================
    print("\n🔍 Valutazione XGBoost (Test Set):")
    evaluator_xgb = ModelEvaluator(
        y_true=y_val,
        y_pred_proba=y_prob_xgb,
        y_pred_class=y_pred_xgb,
        model_name="XGBoost"
    )
    metrics_xgb = evaluator_xgb.evaluate() # Ottieni i risultati per il confronto


    # --- STEP 15: Confronto Finale Modelli (Test Set) =======================
    print("\n--- Confronto Finale Modelli (Test Set) ---")
    print(f"Logistic Regression - Accuracy: {metrics_lr['accuracy']:.4f}, AUC: {metrics_lr['auc_roc']:.4f}")
    print(f"Keras Neural Network - Accuracy: {metrics_keras['accuracy']:.4f}, AUC: {metrics_keras['auc_roc']:.4f}")
    print(f"XGBoost - Accuracy: {metrics_xgb['accuracy']:.4f}, AUC: {metrics_xgb['auc_roc']:.4f}")


    #print("\n🔍 DEBUG PREDIZIONI:")
    #print("=== Logistic Regression ===")
    #print(f"Probabilità range: [{y_prob_lr.min():.3f}, {y_prob_lr.max():.3f}]")
    #print(f"Classi predette: {np.unique(y_pred_lr, return_counts=True)}")
#
    #print("\n=== XGBoost ===")
    #print(f"Probabilità range: [{y_prob_xgb.min():.3f}, {y_prob_xgb.max():.3f}]")
    #print(f"Classi predette: {np.unique(y_pred_xgb, return_counts=True)}")
#
    #print("\n=== Keras ===")
    #print(f"Probabilità range: [{y_prob_keras.min():.3f}, {y_prob_keras.max():.3f}]")
    #print(f"Classi predette: {np.unique(y_pred_keras, return_counts=True)}")
#
    ## Verifica threshold
    #print(f"\n🔍 Soglia attuale: 0.5")
    #print(f"Logistic - Numero sopra soglia: {np.sum(y_prob_lr > 0.5)}")
    #print(f"XGBoost - Numero sopra soglia: {np.sum(y_prob_xgb > 0.5)}")
    #print(f"Keras - Numero sopra soglia: {np.sum(y_prob_keras > 0.5)}")
#
    #print("\n🏁 Tutte le valutazioni e confronti completati!")

    # --- STEP 16: Analisi Importanza Features ===============================
    print("\n--- Analisi Importanza Features ---")

    # Inizializza l'analizzatore
    feature_analyzer = FeatureImportanceAnalyzer(model_dir=model_dir, fig_dir=fig_dir)

    # Analizza l'importanza per ogni modello
    importance_results = {}

    # Logistic Regression
    print("🔍 Analizzando importanza features Logistic Regression...")
    lr_importance = feature_analyzer.analyze_logistic_regression_importance(
        best_lr_model, final_feature_list, X_val, y_val
    )
    importance_results['Logistic Regression'] = lr_importance
    if lr_importance is not None:
        print("Top 5 features Logistic Regression:")
        print(lr_importance.head())

    # XGBoost
    print("🔍 Analizzando importanza features XGBoost...")
    xgb_importance = feature_analyzer.analyze_xgboost_importance(
        best_xgb_model, final_feature_list
    )
    importance_results['XGBoost'] = xgb_importance
    if xgb_importance is not None:
        print("Top 5 features XGBoost:")
        print(xgb_importance.head())

    # Keras Neural Network
    print("🔍 Analizzando importanza features Keras...")
    keras_importance = feature_analyzer.analyze_keras_importance(
        keras_model.model, final_feature_list, X_val, y_val
    )
    importance_results['Keras NN'] = keras_importance
    if keras_importance is not None:
        print("Top 5 features Keras:")
        print(keras_importance.head())

    # Crea visualizzazioni
    print("📊 Creando visualizzazioni importanza features...")
    feature_analyzer.plot_feature_importance_comparison(importance_results, top_n=12)
    combined_importance = feature_analyzer.create_combined_importance_heatmap(importance_results, top_n=8)

    # Salva risultati
    feature_analyzer.save_importance_results(importance_results, out_dir=out_dir)

    # --- STEP 17: Analisi Feature Consensus ================================
    print("\n--- Analisi Consensus Features ---")

    def analyze_feature_consensus(importance_dict, top_n=10):
        """Analizza il consenso tra i modelli sulle feature più importanti."""
        consensus_scores = {}

        for feature in final_feature_list:
            score = 0
            appearances = 0

            for model_name, importance_df in importance_dict.items():
                if importance_df is not None:
                    feature_rank = importance_df[importance_df['feature'] == feature].index
                    if not feature_rank.empty:
                        rank = feature_rank[0] + 1  # +1 perché l'indice parte da 0
                        if rank <= top_n:
                            score += (top_n - rank + 1)  # Punteggio più alto per rank migliori
                            appearances += 1

            if appearances > 0:
                consensus_scores[feature] = {
                    'total_score': score,
                    'appearances': appearances,
                    'average_score': score / appearances
                }

        # Crea DataFrame con i risultati
        consensus_df = pd.DataFrame([
            {
                'feature': feature,
                'consensus_score': data['total_score'],
                'models_agreeing': data['appearances'],
                'average_rank_score': data['average_score']
            }
            for feature, data in consensus_scores.items()
        ]).sort_values('consensus_score', ascending=False)

        return consensus_df

    # Calcola il consenso
    consensus_df = analyze_feature_consensus(importance_results, top_n=10)
    print("\n🎯 Top 10 Features per Consensus tra Modelli:")
    print(consensus_df.head(10))

    # Salva il consensus
    consensus_path = os.path.join(out_dir, "feature_consensus_ranking.csv")
    consensus_df.to_csv(consensus_path, index=False)
    print(f"✅ Consensus features salvato in: {consensus_path}")

    # Plot consensus
    plt.figure(figsize=(12, 8))
    top_consensus = consensus_df.head(15)

    fig, ax1 = plt.subplots(figsize=(14, 10))

    # Plot consensus score
    ax1.barh(top_consensus['feature'], top_consensus['consensus_score'],
             color='skyblue', label='Consensus Score')
    ax1.set_xlabel('Consensus Score')
    ax1.set_ylabel('Features')
    ax1.set_title('Top 15 Features - Consensus tra Modelli', fontsize=16, fontweight='bold')

    # Aggiungi numero di modelli che concordano
    ax2 = ax1.twiny()
    ax2.scatter(top_consensus['models_agreeing'], top_consensus['feature'],
                color='red', s=100, alpha=0.7, label='Models Agreeing')
    ax2.set_xlabel('Number of Models Agreeing')

    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'feature_consensus_ranking.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # --- STEP 18: Salva le metriche di performance per il soft voting =======
    print("\n--- Salvataggio metriche per soft voting ---")

    # Crea un dizionario con le metriche disponibili di ogni modello
    performance_metrics = {
        'logreg': {
            'auc_roc': metrics_lr['auc_roc'],
            'accuracy': metrics_lr['accuracy'],
            # Usa f1 se disponibile, altrimenti calcola una metrica alternativa
            'f1_score': metrics_lr.get('f1_score', metrics_lr.get('f1', 0.0))
        },
        'xgboost': {
            'auc_roc': metrics_xgb['auc_roc'],
            'accuracy': metrics_xgb['accuracy'],
            'f1_score': metrics_xgb.get('f1_score', metrics_xgb.get('f1', 0.0))
        },
        'keras': {
            'auc_roc': metrics_keras['auc_roc'],
            'accuracy': metrics_keras['accuracy'],
            'f1_score': metrics_keras.get('f1_score', metrics_keras.get('f1', 0.0))
        }
    }

    # 🔥 DEBUG: Verifica quali metriche sono disponibili
    print("🔍 Metriche disponibili per Logistic Regression:")
    for key, value in metrics_lr.items():
        print(f"  {key}: {value}")

    print("🔍 Metriche disponibili per XGBoost:")
    for key, value in metrics_xgb.items():
        print(f"  {key}: {value}")

    print("🔍 Metriche disponibili per Keras:")
    for key, value in metrics_keras.items():
        print(f"  {key}: {value}")

    # Salva le metriche (solo quelle che esistono)
    joblib.dump(performance_metrics, os.path.join(model_dir, "model_performance_metrics.pkl"))
    print("✅ Metriche di performance salvate per il soft voting")

    # Calcola e mostra i pesi basati sull'AUC
    auc_scores = {
        'logreg': metrics_lr['auc_roc'],
        'xgboost': metrics_xgb['auc_roc'],
        'keras': metrics_keras['auc_roc']
    }

    total_auc = sum(auc_scores.values())
    model_weights = {model: auc / total_auc for model, auc in auc_scores.items()}

    print("🎯 Pesi calcolati basati su AUC ROC:")
    for model, weight in model_weights.items():
        print(f"  {model}: {weight:.3f} (AUC: {auc_scores[model]:.3f})")

    # Salva anche i pesi calcolati
    joblib.dump(model_weights, os.path.join(model_dir, "model_weights.pkl"))
    print("✅ Pesi dei modelli salvati")


if __name__ == "__main__":
    main()

