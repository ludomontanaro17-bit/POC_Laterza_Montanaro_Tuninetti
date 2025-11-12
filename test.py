import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
import xgboost as xgb

# =============================
# 🔹 Caricamento modelli e scaler
# =============================
model_dir = "model"

keras_model = load_model(os.path.join(model_dir, "keras_model.h5"))
logreg_model = joblib.load(os.path.join(model_dir, "logistic_regression_model.pkl"))
xgb_model = joblib.load(os.path.join(model_dir, "xgboost_model.pkl"))
scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
feature_names = joblib.load(os.path.join(model_dir, "feature_names.pkl"))

print("✅ Modelli e scaler caricati con successo")

# =============================
# 🔹 Funzione per generare feature coerenti
# =============================
def make_features(loan_income_ratio):
    """
    Crea un array di feature coerenti per testare l'impatto di LoanAmount_to_Income.
    Si basano su un profilo medio 'ragionevole'.
    """
    Dependents = 1
    LoanAmount = 100 * loan_income_ratio  # in migliaia di euro (coerente con training)
    Loan_Amount_Term = 360
    Credit_History = 1
    TotalIncome = 100.0  # 100.000 €
    HasDependents = 1
    IsMarried = 1
    IsGraduate = 1
    Income_bracket = 1
    Credit_x_IncomeHigh = 1
    Gender_Male = 0
    Married_Yes = 1
    Education_Not_Graduate = 0
    Self_Employed_Yes = 0
    Property_Area_Semiurban = 1
    Property_Area_Urban = 0

    return np.array([
        Dependents,
        LoanAmount,
        Loan_Amount_Term,
        Credit_History,
        TotalIncome,
        loan_income_ratio,  # LoanAmount_to_Income
        HasDependents,
        IsMarried,
        IsGraduate,
        Income_bracket,
        Credit_x_IncomeHigh,
        Gender_Male,
        Married_Yes,
        Education_Not_Graduate,
        Self_Employed_Yes,
        Property_Area_Semiurban,
        Property_Area_Urban
    ]).reshape(1, -1)

# =============================
# 🔹 Test su range di valori
# =============================
ratios = np.linspace(2, 20, 20)
results = []

for ratio in ratios:
    features = make_features(ratio)
    X_df = pd.DataFrame(features, columns=feature_names)

    # Applica scaling (solo alle colonne numeriche)
    scaler_columns = [c for c in ['LoanAmount', 'TotalIncome', 'LoanAmount_to_Income'] if c in X_df.columns]
    X_df[scaler_columns] = scaler.transform(X_df[scaler_columns])

    # Predizioni
    keras_pred = float(keras_model.predict(X_df, verbose=0)[0][0])
    logreg_pred = float(logreg_model.predict_proba(X_df)[0][1])
    xgb_pred = float(xgb_model.predict_proba(X_df)[0][1])

    results.append({
        'LoanAmount_to_Income': ratio,
        'Keras': keras_pred,
        'LogReg': logreg_pred,
        'XGBoost': xgb_pred
    })

# =============================
# 🔹 Plot dei risultati
# =============================
df_results = pd.DataFrame(results)
plt.figure(figsize=(9,6))
plt.plot(df_results['LoanAmount_to_Income'], df_results['Keras'], label='Keras (NN)', marker='o')
plt.plot(df_results['LoanAmount_to_Income'], df_results['LogReg'], label='Logistic Regression', marker='s')
plt.plot(df_results['LoanAmount_to_Income'], df_results['XGBoost'], label='XGBoost', marker='^')

plt.title('Sensibilità della Probabilità di Approvazione vs LoanAmount_to_Income')
plt.xlabel('LoanAmount_to_Income (rapporto prestito / reddito)')
plt.ylabel('Probabilità di approvazione (classe 1)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
output_path = os.path.join("model", "approval_vs_ratio.png")
plt.savefig(output_path, dpi=200)
print(f"📊 Grafico salvato in: {output_path}")
