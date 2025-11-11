import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from classes.Datapreprocessing import DataPreprocessing
from classes.KaggleLoader import KaggleLoader
from classes.LogisticRegression import LogisticRegressionModel
from classes.ModelEvaluator import ModelEvaluator
#from classes.KerasModel import KerasModel


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

    # === Step 6: Salvataggio oggetti utili ==================================
    joblib.dump(df_preprocessed.drop(columns=['Loan_Status']).columns.tolist(),
                os.path.join(model_dir, "final_columns.pkl"))

    if hasattr(prep, 'scaler'):
        joblib.dump(prep.scaler, os.path.join(model_dir, "scaler.pkl"))

    print(f"\n📁 Figure salvate in: {fig_dir}")
    print(f"📁 Oggetti salvati in: {model_dir}")

    print(X_train.dtypes[X_train.dtypes == 'category'])
    for col in X_train.columns:
        if str(X_train[col].dtype) == 'category':
            print(f"\n🔍 {col} → {X_train[col].unique()[:10]}")

    # === STEP 6: Addestramento Logistic Regression ==========================

    logreg_model = LogisticRegressionModel(model_dir="model")
    logreg_model.train(X_train, y_train)
    y_pred_lr, y_prob_lr = logreg_model.predict(X_val)
    logreg_model.save_model()

    # === STEP 7: Valutazione Logistic Regression ============================
    print("\n🔍 Valutazione Logistic Regression:")
    evaluator_lr = ModelEvaluator(
        y_true=y_val,
        y_pred_proba=y_prob_lr,
        y_pred_class=y_pred_lr,
        model_name="Logistic Regression"
    )
    evaluator_lr.evaluate()

    """
    # === STEP 8: Addestramento Rete Neurale Keras ===========================
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

    # === STEP 9: Valutazione Keras ==========================================
    print("\n🔍 Valutazione Rete Neurale Keras:")
    evaluator_nn = ModelEvaluator(
        y_true=y_val,
        y_pred_proba=y_prob_keras,
        y_pred_class=y_pred_keras,
        model_name="Keras Neural Network"
    )
    evaluator_nn.evaluate()

    print("\n🏁 Tutte le valutazioni completate!")
    """

if __name__ == "__main__":
    main()

