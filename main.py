'''
step 1: caricamento del dataset
step 2: analisi esplorativa del dataset (inizializzo la classe di preprocessing, mostro informazioni generali e caratteristiche del dataset, visualizzo distribuzioni grafiche, matrice di correlazione, noto eventuali valori mancanti)
step 3: data preprocessing (rimuovo le variabili collineari, tratto i valori mancanti, standardizzazione o normalizzazione e aggiorno il dataframe 'modificato', suddivisione testing e training set)
step 4: creazione e addestramento dei modelli, previsioni sul validation set, report di classificazione, salvataggio del modello nel percorso corretto
'''
import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from classes.Datapreprocessing import DataPreprocessing
from classes.KaggleLoader import KaggleLoader  # file contenente la tua classe KaggleLoader

def safe_numeric_stats(df, out_dir):
    num_cols = [c for c in ['ApplicantIncome','CoapplicantIncome','LoanAmount'] if c in df.columns]
    if num_cols:
        num_stats = df[num_cols].agg(['mean','median','std','min','max']).T.round(2).reset_index().rename(columns={'index':'feature'})
        num_stats.to_csv(os.path.join(out_dir, "numeric_statistics.csv"), index=False)

def main(args):
    out_dir = "presentation_tables"
    fig_dir = "presentation_figures"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # 1) Scarica / carica dataset usando KaggleLoader se richiesto
    if args.use_kaggle:
        loader = KaggleLoader(dataset=args.kaggle_dataset, download_dir=args.data_dir)
        loader.load()
        df_train, df_test = loader.get_full_dataset()
        if df_train is None:
            raise SystemExit("Errore: dataset non trovato dopo il download.")
        df = df_train.copy()
    else:
        if args.data_path is None:
            raise SystemExit("Errore: specifica --data_path o usa --use_kaggle")
        df = pd.read_csv(args.data_path)

    # Inizializza preprocessing
    prep = DataPreprocessing(df, target_column='Loan_Status')

    # 1) Distribuzione target (tabella + figura)
    target_counts = prep.y.value_counts(dropna=False).rename_axis('Loan_Status').reset_index(name='count')
    target_counts['percent'] = (target_counts['count'] / target_counts['count'].sum() * 100).round(2)
    target_counts.to_csv(os.path.join(out_dir, "target_distribution.csv"), index=False)
    plt.figure(figsize=(6,4))
    sns.countplot(x=prep.y, palette='viridis')
    plt.title('Distribuzione Loan Status')
    plt.xlabel('Loan Status')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "target_distribution.png"), dpi=150)
    plt.close()

    # 2) Conteggi e percentuali per categorie selezionate
    cats = ['Gender','Married','Education','Property_Area']
    for col in cats:
        if col in df.columns:
            ct = df[col].value_counts(dropna=False).rename_axis(col).reset_index(name='count')
            ct['percent'] = (ct['count'] / ct['count'].sum() * 100).round(2)
            ct.to_csv(os.path.join(out_dir, f"{col.lower()}_counts.csv"), index=False)

    # 3) Statistiche numeriche per Income e LoanAmount (con controllo colonne)
    safe_numeric_stats(df, out_dir)

    # 4) Percentuale Credit_History positivo (gestione caso vuoto)
    if 'Credit_History' in df.columns:
        ch_valid = df['Credit_History'].dropna()
        if len(ch_valid) > 0:
            pct_pos = (ch_valid == 1).sum() / len(ch_valid) * 100
            pct_str = f"{pct_pos:.2f}%"
        else:
            pct_str = "N/A (no non-mancanti)"
        with open(os.path.join(out_dir, "credit_history_summary.txt"), "w") as f:
            f.write(f"Percentuale Credit_History positivo (su non-mancanti): {pct_str}")

    # 5) Correlazioni numeriche e heatmap (crea colonna binaria target se necessario)
    num_df = df.select_dtypes(include=['int64','float64']).copy()
    if 'Loan_Status' not in num_df.columns:
        if 'Loan_Status' in df.columns:
            num_df['Loan_Status_bin'] = df['Loan_Status'].map({'Y':1,'N':0})
    else:
        if not pd.api.types.is_numeric_dtype(num_df['Loan_Status']):
            num_df['Loan_Status_bin'] = df['Loan_Status'].map({'Y':1,'N':0})

    corr = num_df.corr()
    corr.to_csv(os.path.join(out_dir, "numeric_correlations.csv"))
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='vlag', center=0)
    plt.title("Matrice di correlazione (numeriche)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "numeric_correlation_heatmap.png"), dpi=150)
    plt.close()

    # 6) Cross-tab Credit_History vs Loan_Status
    if 'Credit_History' in df.columns and 'Loan_Status' in df.columns:
        ct = pd.crosstab(df['Credit_History'], df['Loan_Status'], normalize='index') * 100
        ct = ct.round(2).reset_index()
        ct.to_csv(os.path.join(out_dir, "credit_history_vs_loan_status.csv"), index=False)

    print(f"Tabelle CSV salvate in: {out_dir}")
    print(f"Figure PNG salvate in: {fig_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_kaggle", action="store_true", help="Scarica dataset da Kaggle usando KaggleLoader")
    parser.add_argument("--kaggle_dataset", type=str, default="rishikeshkonapure/home-loan-approval")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--data_path", type=str, help="Percorso CSV locale (se non si usa --use_kaggle)")
    args = parser.parse_args()
    main(args)

