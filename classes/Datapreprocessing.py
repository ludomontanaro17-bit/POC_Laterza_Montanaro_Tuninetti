import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from KaggleLoader import KaggleLoader

class DataPreprocessing:
    def __init__(self, dataframe, target_column):
        self.dataframe = dataframe.copy()
        self.target_column = target_column
        self.X = self.dataframe.drop(columns=[target_column])
        self.y = self.dataframe[target_column]
        self.scaler = StandardScaler()
        if self.target_column not in dataframe.columns:
          raise ValueError(f"Target column '{self.target_column}' non presente nel dataframe")
 
    # --- Info & basic checks -------------------------------------------------
    def display_info(self):
        print("\nInformazioni sul dataset:")
        dtypes = self.X.dtypes
        for t in sorted(dtypes.unique(), key=str):
            cols = dtypes[dtypes == t].index.tolist()
            print(f"\n🔹 Tipo {t}: {len(cols)} colonne")
            print(cols)

    def display_statistics(self):
        print("\nStatistiche descrittive (tutte le colonne):")
        display(self.X.describe(include='all'))

    def display_missing_values(self):
        print("\nConteggio valori mancanti per colonna:")
        print(self.dataframe.isnull().sum().sort_values(ascending=False))

    # --- Target & categorical distributions ---------------------------------
    def target_distribution(self):
        counts = self.y.value_counts(dropna=False)
        pct = self.y.value_counts(normalize=True, dropna=False) * 100
        df = pd.concat([counts, pct.round(2)], axis=1)
        df.columns = ['count', 'percent']
        print("\nDistribuzione della classe target:")
        print(df)
        # Plot
        plt.figure(figsize=(6,4))
        sns.countplot(x=self.y, palette='viridis')
        plt.title('Distribuzione Loan Status')
        plt.xlabel('Loan Status')
        plt.ylabel('Count')
        plt.show()

    def categorical_counts(self, columns=None):
        if columns is None:
            columns = ['Gender', 'Married', 'Education', 'Property_Area']
        print("\nConteggi e percentuali per categorie:")
        for col in columns:
            if col in self.dataframe.columns:
                ct = self.dataframe[col].value_counts(dropna=False)
                pct = self.dataframe[col].value_counts(normalize=True, dropna=False) * 100
                summary = pd.concat([ct, pct.round(2)], axis=1)
                summary.columns = ['count', 'percent']
                print(f"\n--- {col} ---")
                print(summary)
            else:
                print(f"\n--- {col} non presente nel dataset ---")

    # --- Numeric stats ------------------------------------------------------
    def numeric_stats(self, columns=None):
        if columns is None:
            columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
        present = [c for c in columns if c in self.dataframe.columns]
        print("\nStatistica su variabili numeric:")
        print(self.dataframe[present].agg(['mean','median','std','min','max']).T)

    def percent_credit_history_positive(self):
        col = 'Credit_History'
        if col in self.dataframe.columns:
            valid = self.dataframe[col].dropna()
            pct = (valid == 1).sum() / len(valid) * 100 if len(valid)>0 else np.nan
            print(f"\nPercentuale con Credit History positivo (su non-mancanti): {pct:.2f}%")
        else:
            print("\nCredit_History non presente nel dataset.")

    # --- Correlation & cross-tabs ------------------------------------------
    def numeric_correlations(self):
        num = self.dataframe.select_dtypes(include=['int64','float64'])
        if self.target_column in num.columns:
            corr = num.corr()[self.target_column].drop(self.target_column).sort_values(key=abs, ascending=False)
            print("\nCorrelazioni (valori numerici) rispetto al target:")
            print(corr)
            plt.figure(figsize=(8,6))
            sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap='vlag', center=0)
            plt.title("Matrice di correlazione (numeriche)")
            plt.show()
        else:
            print("\nTarget non numerico: convertilo temporaneamente a 0/1 per vedere correlazioni numeriche.")
       


    def cross_tab_credit_target(self):
        if 'Credit_History' in self.dataframe.columns:
            ct = pd.crosstab(self.dataframe['Credit_History'], self.dataframe[self.target_column], normalize='index') * 100
            print("\nCross-tab Credit_History vs Loan_Status (percentuale per riga):")
            print(ct.round(2))
        else:
            print("\nCredit_History non presente nel dataset.")

    # --- Cleaning & preprocessing recommendations ----------------------------
    def summary_missing_treatment(self):
        print("\nRaccomandazioni per gestione valori mancanti:")
        print("- Income (Applicant/Coapplicant): imputare con mediana o usare TotalIncome + log-transform")
        print("- LoanAmount: imputare con mediana; verificare unità (k?)")
        print("- Categorical: imputare con 'Unknown' o modalità; mantenere NA come categoria separata se informativa")
        print("- Credit_History: considerare categoria separata per NA; è spesso predittivo")
        print("- Dependents: convertire '3+' in 3 o lasciare categoria '3+'")

    def advanced_preprocessing(self, impute_numeric='median', encode_categoricals='onehot', create_features=True):
        df = self.dataframe.copy()

        # Standard fixes
        if 'Dependents' in df.columns:
            df['Dependents'] = df['Dependents'].replace('3+', '3')
            df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce')

        # Impute numeric
        numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != self.target_column]
        if impute_numeric == 'median':
            for c in numeric_cols:
                df[c] = df[c].fillna(df[c].median())
        elif impute_numeric == 'mean':
            for c in numeric_cols:
                df[c] = df[c].fillna(df[c].mean())

        # Impute categoricals
        cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
        cat_cols = [c for c in cat_cols if c != self.target_column]
        for c in cat_cols:
            df[c] = df[c].fillna('Unknown')

        # Feature engineering
        if create_features:
            if 'ApplicantIncome' in df.columns and 'CoapplicantIncome' in df.columns:
                df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
            elif 'ApplicantIncome' in df.columns:
                df['TotalIncome'] = df['ApplicantIncome']

            if 'LoanAmount' in df.columns and 'TotalIncome' in df.columns:
                # LoanAmount sometimes in thousands — be careful; assume same scale
                df['LoanAmount_to_Income'] = df['LoanAmount'] / (df['TotalIncome'].replace({0:np.nan}))
                df['LoanAmount_to_Income'] = df['LoanAmount_to_Income'].fillna(0)

            # Binary indicators
            if 'Dependents' in df.columns:
                df['HasDependents'] = (df['Dependents'] > 0).astype(int)
            if 'Married' in df.columns:
                df['IsMarried'] = df['Married'].map({'Yes':1,'No':0}).fillna(0).astype(int)
            if 'Education' in df.columns:
                df['IsGraduate'] = df['Education'].map({'Graduate':1,'Not Graduate':0}).fillna(0).astype(int)

            # Income binning
            if 'TotalIncome' in df.columns:
                df['Income_bracket'] = pd.qcut(df['TotalIncome'].rank(method='first'), q=3, labels=['low','medium','high'])

            # Interaction example
            if 'Credit_History' in df.columns and 'Income_bracket' in df.columns:
                df['Credit_x_IncomeHigh'] = ((df['Credit_History'] == 1) & (df['Income_bracket'] == 'high')).astype(int)

        # Encoding categoricals
        if encode_categoricals == 'onehot':
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        elif encode_categoricals == 'label':
            for c in cat_cols:
                df[c] = df[c].astype('category').cat.codes

        # Scaling numeric (only ApplicantIncome, CoapplicantIncome, LoanAmount, TotalIncome, LoanAmount_to_Income)
        scaler_cols = [c for c in ['ApplicantIncome','CoapplicantIncome','LoanAmount','TotalIncome','LoanAmount_to_Income']
                       if c in df.columns]
        if scaler_cols:
            self.scaler.fit(df[scaler_cols])
            df[scaler_cols] = self.scaler.transform(df[scaler_cols])
            model_dir = os.path.join("model")
            os.makedirs(model_dir, exist_ok=True)
            joblib.dump(self.scaler, os.path.join(model_dir, "scaler.pkl"))

        # Ensure target preserved
        if self.target_column in df.columns:
            df[self.target_column] = self.dataframe[self.target_column].values

        print("\n✅ Preprocessing avanzato completato. Colonne finali:", df.columns.tolist())
        return df

    # --- Utilities ----------------------------------------------------------
    def detect_outliers(self, column):
        if column not in self.dataframe.columns:
            print(f"{column} non presente.")
            return
        s = self.dataframe[column].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
        out = s[(s < lower) | (s > upper)]
        print(f"\nOutliers rilevati in {column}: {len(out)} (limiti: {lower:.2f}, {upper:.2f})")
        return out.index.tolist()

    def split_data(self, df=None, test_size=0.2, random_state=42):
        if df is None:
            df = self.dataframe.copy()
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size,
                                                          random_state=random_state, stratify=y)
        print(f"Split: {len(X_train)} train / {len(X_val)} val")
        return X_train, X_val, y_train, y_val
