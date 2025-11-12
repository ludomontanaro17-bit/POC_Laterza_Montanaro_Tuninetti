# main.py (aggiornato con CrossValidator, XGBoost, e valutazioni estese)

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import joblib
from classes.Datapreprocessing import DataPreprocessing
from classes.KaggleLoader import KaggleLoader
from classes.LogisticRegression import LogisticRegressionModel
from classes.ModelEvaluator import ModelEvaluator
from classes.KerasModel import KerasModel
from classes.FeatureImportance import FeatureImportanceAnalyzer
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

    ## Categorical
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

## Numeric + TotalIncome
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

            # Swarm plot (ogni punto)
            plt.figure(figsize=(6, 4))
            sns.swarmplot(y=df_train[col], color="mediumseagreen", size=3)
            plt.title(f"Swarm plot di {col}")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"{col.lower()}_swarm.png"), dpi=150)
            plt.close()

    # --- Mappa dei valori mancanti ---
    plt.figure(figsize=(10, 5))
    sns.heatmap(df_train.isnull(), cbar=False, cmap='viridis')
    plt.title("Mappa dei valori mancanti")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "missing_values_heatmap.png"), dpi=150)
    plt.close()

    # === Step 3: Preprocessing ==============================================
    df_preprocessed = prep.advanced_preprocessing(
        impute_numeric='median',
        encode_categoricals='onehot',
        create_features=True
    )
    print(f"✅ Preprocessing completato. Nuova shape: {df_preprocessed.shape}")

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

    # === Step 5: Split dei dati =============================================
    X_train, X_val, y_train, y_val = prep.split_data(df_preprocessed, test_size=0.2, random_state=42)
    print("✅ Train-test split completato.")
    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")
    print(X_train.columns) 
    
    # === Step 5.5: Prepara i nomi delle feature =============================
    feature_names = X_train.columns.tolist()
    print(f"Numero di feature: {len(feature_names)}")
    print("Prime 10 feature:", feature_names[:10])

    # === Step 6: Salvataggio oggetti utili ==================================
    joblib.dump(feature_names, os.path.join(model_dir, "feature_names.pkl"))
    joblib.dump(df_preprocessed.drop(columns=['Loan_Status']).columns.tolist(),
                os.path.join(model_dir, "final_columns.pkl"))

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
    print("\n--- Addestramento Logistic Regression ---")
    logreg_model = LogisticRegressionModel(model_dir="model")
    logreg_model.train(X_train, y_train)
    y_pred_lr, y_prob_lr = logreg_model.predict(X_val)
    logreg_model.save_model()

    # --- STEP 7: Valutazione Logistic Regression (Test Set) =================
    print("\n🔍 Valutazione Logistic Regression (Test Set):")
    evaluator_lr = ModelEvaluator(
        y_true=y_val,
        y_pred_proba=y_prob_lr,
        y_pred_class=y_pred_lr,
        model_name="Logistic Regression"
    )
    metrics_lr = evaluator_lr.evaluate() # Ottieni i risultati per il confronto

    # --- STEP 8: Cross-Validation Logistic Regression (su Train Set) ========
    print("\n🔍 Cross-Validation Logistic Regression (Train Set):")
    cv_lr = CrossValidator(model=logreg_model.model, cv=5, scoring=['accuracy', 'roc_auc'])
    cv_lr.validate(X_train, y_train)
    cv_results_lr = cv_lr.get_mean_test_scores()
    print(f"Mean CV Accuracy: {cv_results_lr['accuracy']:.4f}, Mean CV AUC: {cv_results_lr['roc_auc']:.4f}")


    # --- STEP 9: Addestramento Rete Neurale Keras ===========================
    print("\n--- Addestramento Rete Neurale Keras (Versione Migliorata) ---")

    keras_model = KerasModel(input_dim=X_train.shape[1], model_dir="model")
    history, y_prob_keras, y_pred_keras, keras_report = keras_model.train(
        X_train, y_train, X_val, y_val, epochs=100, batch_size=16, verbose=1
    )

    print(f"✅ Training completato. "
          f"AUC={keras_report['val_auc']:.3f}, "
          f"BalancedAcc={keras_report['val_balanced_acc']:.3f}, "
          f"F1={keras_report['val_f1']:.3f}, "
          f"Soglia={keras_report['best_threshold']:.2f}, "
          f"Distribuzione={keras_report['dist_pred']})")

    keras_model.save_model()

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
    print("\n--- Addestramento XGBoost ---")
    xgb_model = XGBoostModel(model_dir="model")
    xgb_model.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_estimators=100, # Esempio di iperparametro
        learning_rate=0.1,
        max_depth=6
        # scale_pos_weight viene calcolato automaticamente in train()
    )
    xgb_model.save_model()

    # Predizioni XGBoost
    y_prob_xgb = xgb_model.predict(X_val)
    y_pred_xgb = xgb_model.predict_classes(X_val)




# INIZIO DEBUG
    # === DEBUG: Analisi Probabilità ============================================

    print("\n" + "=" * 60)
    print("🔍 DEBUG - ANALISI PROBABILITÀ MODELLI")
    print("=" * 60)

    # Ottieni le probabilità di tutti i modelli
    y_prob_lr = logreg_model.predict_proba(X_val)  # Assicurati che questo metodo esista
    y_prob_keras = keras_model.predict_proba(X_val).flatten()
    y_prob_xgb = xgb_model.predict_proba(X_val)  # Assicurati che questo metodo esista

    print("\n📊 DISTRIBUZIONE PROBABILITÀ:")
    print(f"Logistic Regression - Min: {y_prob_lr.min():.4f}, Max: {y_prob_lr.max():.4f}, Mean: {y_prob_lr.mean():.4f}")
    print(
        f"Keras NN           - Min: {y_prob_keras.min():.4f}, Max: {y_prob_keras.max():.4f}, Mean: {y_prob_keras.mean():.4f}")
    print(
        f"XGBoost            - Min: {y_prob_xgb.min():.4f}, Max: {y_prob_xgb.max():.4f}, Mean: {y_prob_xgb.mean():.4f}")

    # Calcola manualmente il soft voting per debug
    soft_voting_manual = (y_prob_lr + y_prob_keras + y_prob_xgb) / 3
    print(
        f"\n🧮 SOFT VOTING MANUALE - Min: {soft_voting_manual.min():.4f}, Max: {soft_voting_manual.max():.4f}, Mean: {soft_voting_manual.mean():.4f}")

    # Conta quanti esempi hanno probabilità estreme per Keras
    keras_extreme_low = (y_prob_keras < 0.01).sum()
    keras_extreme_high = (y_prob_keras > 0.99).sum()
    print(
        f"\n⚠️  KERAS - Probabilità < 0.01: {keras_extreme_low}/{len(y_prob_keras)} ({keras_extreme_low / len(y_prob_keras) * 100:.1f}%)")
    print(
        f"⚠️  KERAS - Probabilità > 0.99: {keras_extreme_high}/{len(y_prob_keras)} ({keras_extreme_high / len(y_prob_keras) * 100:.1f}%)")

    # Visualizza distribuzioni
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(y_prob_lr, bins=50, alpha=0.7, color='blue')
    plt.title('Logistic Regression Probabilities')
    plt.xlabel('Probability')
    plt.ylabel('Count')

    plt.subplot(1, 3, 2)
    plt.hist(y_prob_keras, bins=50, alpha=0.7, color='red')
    plt.title('Keras NN Probabilities')
    plt.xlabel('Probability')
    plt.ylabel('Count')

    plt.subplot(1, 3, 3)
    plt.hist(y_prob_xgb, bins=50, alpha=0.7, color='green')
    plt.title('XGBoost Probabilities')
    plt.xlabel('Probability')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "debug_probability_distributions.png"), dpi=150)
    plt.close()

    print(f"\n📈 Grafico distribuzioni salvato in: {os.path.join(fig_dir, 'debug_probability_distributions.png')}")













    #INIZIO DEBUG

    # --- STEP 13: Valutazione XGBoost (Test Set) ============================
    print("\n🔍 Valutazione XGBoost (Test Set):")
    evaluator_xgb = ModelEvaluator(
        y_true=y_val,
        y_pred_proba=y_prob_xgb,
        y_pred_class=y_pred_xgb,
        model_name="XGBoost"
    )
    metrics_xgb = evaluator_xgb.evaluate() # Ottieni i risultati per il confronto

    # === DEBUG: Confronto Predizioni Soft Voting ================================

    print("\n" + "=" * 60)
    print("🔍 DEBUG - CONFRONTO PREDIZIONI SOFT VOTING")
    print("=" * 60)

    # Calcola predizioni individuali
    y_pred_lr = (y_prob_lr > 0.5).astype(int)
    y_pred_keras = (y_prob_keras > 0.5).astype(int)
    y_pred_xgb = (y_prob_xgb > 0.5).astype(int)
    y_pred_soft_voting = (soft_voting_manual > 0.5).astype(int)

    print("\n🎯 CONFRONTO PREDIZIONI (0=Rifiutato, 1=Approvato):")
    print(f"Logistic Regression: {np.bincount(y_pred_lr)}")
    print(f"Keras NN:           {np.bincount(y_pred_keras)}")
    print(f"XGBoost:            {np.bincount(y_pred_xgb)}")
    print(f"Soft Voting Manual: {np.bincount(y_pred_soft_voting)}")

    # Verifica discrepanze
    discrepancies_keras_vs_lr = (y_pred_keras != y_pred_lr).sum()
    discrepancies_keras_vs_xgb = (y_pred_keras != y_pred_xgb).sum()
    discrepancies_all = len(set([tuple(y_pred_lr), tuple(y_pred_keras), tuple(y_pred_xgb)])) > 1

    print(f"\n❌ DISCREPANZE:")
    print(
        f"Keras vs Logistic Regression: {discrepancies_keras_vs_lr}/{len(y_val)} ({discrepancies_keras_vs_lr / len(y_val) * 100:.1f}%)")
    print(
        f"Keras vs XGBoost:            {discrepancies_keras_vs_xgb}/{len(y_val)} ({discrepancies_keras_vs_xgb / len(y_val) * 100:.1f}%)")
    print(f"Tutti i modelli discordano:  {discrepancies_all}")
    # === DEBUG CRITICO: Verifica Encoding Target ===============================

    print("\n" + "=" * 60)
    print("🔍 DEBUG - VERIFICA ENCODING TARGET")
    print("=" * 60)

    print("Valori unici target:")
    print(f"y_train: {np.unique(y_train)}")
    print(f"y_val: {np.unique(y_val)}")

    # Verifica la correlazione tra probabilità e target reale
    corr_lr = np.corrcoef(y_prob_lr, y_val)[0, 1]
    corr_keras = np.corrcoef(y_prob_keras, y_val)[0, 1]
    corr_xgb = np.corrcoef(y_prob_xgb, y_val)[0, 1]

    print(f"\nCorrelazione probabilità vs target:")
    print(f"LR: {corr_lr:.3f} | Keras: {corr_keras:.3f} | XGB: {corr_xgb:.3f}")

    # Se Keras ha correlazione negativa, sta predendo l'opposto!














    # --- STEP 14: Cross-Validation XGBoost (su Train Set) ===================
    print("\n🔍 Cross-Validation XGBoost (Train Set):")
    cv_xgb = CrossValidator(model=xgb_model.model, cv=5, scoring=['accuracy', 'roc_auc'])
    cv_xgb.validate(X_train, y_train)
    cv_results_xgb = cv_xgb.get_mean_test_scores()
    print(f"Mean CV Accuracy: {cv_results_xgb['accuracy']:.4f}, Mean CV AUC: {cv_results_xgb['roc_auc']:.4f}")


    # --- STEP 15: Confronto Finale Modelli (Test Set) =======================
    print("\n--- Confronto Finale Modelli (Test Set) ---")
    print(f"Logistic Regression - Accuracy: {metrics_lr['accuracy']:.4f}, AUC: {metrics_lr['auc_roc']:.4f}")
    print(f"Keras Neural Network - Accuracy: {metrics_keras['accuracy']:.4f}, AUC: {metrics_keras['auc_roc']:.4f}")
    print(f"XGBoost - Accuracy: {metrics_xgb['accuracy']:.4f}, AUC: {metrics_xgb['auc_roc']:.4f}")


    print("\n🏁 Tutte le valutazioni e confronti completati!")

    # --- STEP 16: Analisi Importanza Features ===============================
    print("\n--- Analisi Importanza Features ---")

    # Inizializza l'analizzatore
    feature_analyzer = FeatureImportanceAnalyzer(model_dir=model_dir, fig_dir=fig_dir)

    # Analizza l'importanza per ogni modello
    importance_results = {}

    # Logistic Regression
    print("🔍 Analizzando importanza features Logistic Regression...")
    lr_importance = feature_analyzer.analyze_logistic_regression_importance(
        logreg_model.model, feature_names, X_val, y_val
    )
    importance_results['Logistic Regression'] = lr_importance
    if lr_importance is not None:
        print("Top 5 features Logistic Regression:")
        print(lr_importance.head())

    # XGBoost
    print("🔍 Analizzando importanza features XGBoost...")
    xgb_importance = feature_analyzer.analyze_xgboost_importance(
        xgb_model.model, feature_names
    )
    importance_results['XGBoost'] = xgb_importance
    if xgb_importance is not None:
        print("Top 5 features XGBoost:")
        print(xgb_importance.head())

    # Keras Neural Network
    print("🔍 Analizzando importanza features Keras...")
    keras_importance = feature_analyzer.analyze_keras_importance(
        keras_model.model, feature_names, X_val, y_val
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

        for feature in feature_names:
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


if __name__ == "__main__":
    main()

