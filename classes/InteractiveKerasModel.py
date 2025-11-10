import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from tensorflow import keras

class InteractiveKerasModel:
    def __init__(self, model_path, scaler_path, feature_names_path):
        """
        Inizializza l'interfaccia interattiva caricando modello, scaler e feature names.

        Args:
            model_path (str): Percorso del file del modello Keras (.h5 o SavedModel).
            scaler_path (str): Percorso del file del scaler salvato con joblib.
            feature_names_path (str): Percorso del file delle feature names salvato con joblib.
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.feature_names_path = feature_names_path

        self.model = None
        self.scaler = None
        self.feature_names = None

        self._load_assets()

    def _load_assets(self):
        """Carica modello, scaler e feature names."""
        try:
            print(f"Caricamento modello da {self.model_path}...")
            self.model = keras.models.load_model(self.model_path)
            print("Modello caricato con successo.")

            print(f"Caricamento scaler da {self.scaler_path}...")
            self.scaler = joblib.load(self.scaler_path)
            print("Scaler caricato con successo.")

            print(f"Caricamento feature names da {self.feature_names_path}...")
            self.feature_names = joblib.load(self.feature_names_path)
            print(f"Feature names caricate con successo. Numero: {len(self.feature_names)}")

        except FileNotFoundError as e:
            st.error(f"Errore: File non trovato - {e}")
            st.stop()
        except Exception as e:
            st.error(f"Errore durante il caricamento degli asset: {e}")
            st.stop()

    def run_interface(self):
        """Avvia l'interfaccia Streamlit."""
        st.title('Previsione Concessione Mutuo - Modello Keras')

        if not all([self.model, self.scaler, self.feature_names]):
            st.error("Errore: Uno o più asset (modello, scaler, feature names) non sono stati caricati correttamente.")
            return

        # --- Input dell'utente tramite controlli Streamlit ---
        # Devi creare un controllo per ogni feature richiesta dal modello finale.
        # Esempio per alcune features comuni basate sul preprocessing visto prima:

        # Feature Numeriche (es. quelle scalate)
        applicant_income = st.number_input('Reddito Applicant', min_value=0.0, value=0.0)
        coapplicant_income = st.number_input('Reddito Co-Applicant', min_value=0.0, value=0.0)
        loan_amount = st.number_input('Importo del Mutuo', min_value=0.0, value=0.0)

        # Feature derivate (calcolate come in DataPreprocessing)
        total_income = applicant_income + coapplicant_income
        loan_to_income = loan_amount / total_income if total_income > 0 else 0.0

        # Feature categoriche binarie (es. codificate come 0/1)
        is_graduate = st.selectbox('Istruzione (Graduato)', ['Graduate', 'Not Graduate'])
        is_graduate_encoded = 1 if is_graduate == 'Graduate' else 0

        is_married = st.selectbox('Stato Civile (Sposato)', ['Yes', 'No'])
        is_married_encoded = 1 if is_married == 'Yes' else 0

        has_dependents = st.selectbox('Ha Dipendenti?', ['No', 'Yes'])
        has_dependents_encoded = 1 if has_dependents == 'Yes' else 0

        credit_history = st.selectbox('Storia Creditizia', [0, 1], format_func=lambda x: 'Buona (1)' if x == 1 else 'Cattiva (0)')
        # Nota: Credit_History è trattato come categorico binario, quindi non è scalato,
        # ma deve essere presente nel formato corretto (es. 0 o 1) se era numerico nel dataset originale
        # e non è stato convertito in stringa prima dell'encoding finale.

        # Feature categoriche OneHot (esempio con Property Area)
        property_area = st.selectbox('Zona Proprietà', ['Rural', 'Semiurban', 'Urban'])
        # Crea le colonne OneHot come richiesto dal modello finale
        # Supponiamo che il modello si aspetti colonne tipo: 'Property_Area_Rural', 'Property_Area_Semiurban', 'Property_Area_Urban'
        # (anche se con drop_first=True, potrebbe mancarne una)
        # In questo esempio, assumiamo che le colonne siano: 'Property_Area_Semiurban', 'Property_Area_Urban'
        property_area_semiurban = 1 if property_area == 'Semiurban' else 0
        property_area_urban = 1 if property_area == 'Urban' else 0
        # Property_Area_Rural sarebbe implicitamente 1 se entrambe le altre sono 0

        # EDUCAZIONE (esempio OneHot)
        education_encoded = is_graduate_encoded # Se era solo una feature binaria dopo encoding, usa questa
        # Se era una categoria con più valori (es. Graduate, Not Graduate), e hai fatto OneHot,
        # allora devi creare colonne separate come fatto per Property Area.
        # Ad esempio, se il modello finale ha 'Education_Not Graduate', 'Education_Graduate':
        # education_graduate = 1 if is_graduate == 'Graduate' else 0
        # education_not_graduate = 1 if is_graduate == 'Not Graduate' else 0
        # Usa solo quella che NON è stata rimossa con drop_first (es. 'Education_Graduate')
        # education_encoded = education_graduate # E rimuovi la riga precedente is_graduate_encoded


        # ... (aggiungi tutti i controlli necessari per tutte le features del modello finale) ...
        # Ad esempio, se hai Income_bracket (Low, Medium, High) codificato come categoria,
        # e hai fatto OneHot, devi gestirlo qui allo stesso modo di Property Area.
        # income_bracket = st.selectbox('Fascia di Reddito', ['low', 'medium', 'high'])
        # income_bracket_medium = 1 if income_bracket == 'medium' else 0
        # income_bracket_high = 1 if income_bracket == 'high' else 0
        # income_bracket_low sarebbe implicito se entrambe le altre sono 0


        # Bottone per fare la previsione
        if st.button('Prevedi Concessione Mutuo'):
            # --- APPLICA LO STESSO PREPROCESSING ---
            # Crea il dizionario dei dati come atteso dal modello
            # ATTENZIONE: L'ordine e la presenza di *tutte* le features richieste dal modello è cruciale
            # Supponendo che le colonne siano nell'ordine di self.feature_names
            # Devi conoscere ESATTAMENTE il nome di ogni colonna richiesta dal modello finale
            # e assegnargli il valore corretto (scalato o codificato).

            # Esempio di costruzione del dizionario (aggiusta i nomi delle chiavi!)
            # I nomi devono corrispondere esattamente a quelli in self.feature_names
            user_data = {}
            # Supponiamo che il modello finale richieda queste colonne (esempio NON VERIFICATO):
            # Ordine fittizio: 'TotalIncome', 'LoanAmount_to_Income', 'IsGraduate', 'IsMarried', 'HasDependents',
            # 'Credit_History', 'Property_Area_Semiurban', 'Property_Area_Urban'
            # Devi sostituire questi nomi con quelli effettivi in self.feature_names

            # Esempio (da adattare!):
            # user_data = {
            #     'TotalIncome': total_income,
            #     'LoanAmount_to_Income': loan_to_income,
            #     'IsGraduate': is_graduate_encoded, # Se non OneHot
            #     'IsMarried': is_married_encoded,  # Se non OneHot
            #     'HasDependents': has_dependents_encoded,
            #     'Credit_History': credit_history, # Assumendo sia rimasto numerico (0/1) o codificato come categoria
            #     'Property_Area_Semiurban': property_area_semiurban,
            #     'Property_Area_Urban': property_area_urban,
            #     # ... altre features ...
            # }

            # OPPURE, crea un dizionario con tutte le features richieste, inizializzando a 0
            # e poi assegnando i valori calcolati solo per le features che hanno un valore specifico.
            # Questo è spesso più robusto se hai molte features OneHot.
            user_data = {name: 0 for name in self.feature_names} # Inizializza tutto a 0

            # Ora assegna i valori calcolati alle chiavi corrette
            # Devi sapere esattamente come si chiamano queste features dopo il preprocessing finale!
            # Esempi (aggiusta i nomi!):
            # user_data['TotalIncome'] = total_income # Assicurati che 'TotalIncome' sia il nome corretto
            # user_data['LoanAmount_to_Income'] = loan_to_income # Assicurati del nome
            # user_data['IsGraduate'] = is_graduate_encoded # Assicurati del nome
            # user_data['IsMarried'] = is_married_encoded # Assicurati del nome
            # user_data['HasDependents'] = has_dependents_encoded # Assicurati del nome
            # user_data['Credit_History'] = credit_history # Assicurati del nome
            # user_data['Property_Area_Semiurban'] = property_area_semiurban # Assicurati del nome
            # user_data['Property_Area_Urban'] = property_area_urban # Assicurati del nome
            # user_data['Education_Graduate'] = education_encoded # Se era una OneHot e 'Graduate' è rimasto

            # *** CRITICO: ASSICURATI CHE I NOMI DELLE FEATURES SIANO CORRETTI ***
            # Il dizionario user_data DEVE avere le stesse chiavi di self.feature_names
            # e nello STESSO ordine (anche se usando un DataFrame con reindex, l'ordine finale è quello del modello).

            # Crea DataFrame e ordina le colonne come richiesto dal modello
            input_df = pd.DataFrame([user_data])
            # Riordina e assicura che tutte le colonne richieste esistano
            input_df = input_df.reindex(columns=self.feature_names, fill_value=0)

            # Identifica le colonne numeriche che richiedono scaling
            # Devi sapere quali sono queste colonne in base al tuo preprocessing
            # Esempio fittizio: supponiamo che siano queste (aggiusta!)
            # scaler_cols = ['TotalIncome', 'LoanAmount_to_Income']
            # scaler_cols = [col for col in scaler_cols if col in input_df.columns] # Assicura che esistano
            scaler_cols = [] # DA DEFINIRE IN BASE AL TUO DATASET FINALE
            # Ad esempio, se hai salvato la lista delle colonne scalate:
            # scaler_cols = joblib.load('model/scaler_columns.pkl') # Se lo hai fatto
            # Altrimenti, devi definirle manualmente qui sapendo cosa hai scalato.

            # Applica scaling alle features numeriche (se scalate e presenti)
            if scaler_cols:
                try:
                    input_df[scaler_cols] = self.scaler.transform(input_df[scaler_cols])
                    print(f"Scaling applicato alle colonne: {scaler_cols}")
                except ValueError as e:
                    st.error(f"Errore durante l'applicazione dello scaling: {e}")
                    st.error(f"Controlla che le colonne scalate {scaler_cols} abbiano valori numerici validi.")
                    return

            # Prepara i dati per il modello (assicura formato numpy array)
            X_new = input_df.values

            try:
                # Fai la previsione
                prediction_proba = self.model.predict(X_new, verbose=0) # verbose=0 per nascondere output durante la previsione
                prediction_class = (prediction_proba > 0.5).astype(int)

                # Mostra il risultato
                st.subheader("Risultato della Previsione")
                prob_concessione = prediction_proba[0][0]
                classe_predetta = "Concesso" if prediction_class[0][0] == 1 else "Rifiutato"

                st.metric(label="Classe Predetta", value=classe_predetta)
                st.metric(label="Probabilità di Concessione", value=f"{prob_concessione:.4f}")

                # Puoi anche mostrare un messaggio più dettagliato
                if prediction_class[0][0] == 1:
                    st.success(f"Previsto: Concesso (Probabilità: {prob_concessione:.2%})")
                else:
                    st.error(f"Previsto: Rifiutato (Probabilità di concessione: {prob_concessione:.2%})")

            except Exception as e:
                st.error(f"Errore durante la previsione: {e}")

# --- Esempio di utilizzo ---
# Assicurati di avere i file:
# - model/best_keras_model.h5 (o il nome che hai dato al tuo modello)
# - model/scaler.pkl (salvato da DataPreprocessing)
# - model/feature_names.pkl (es. salvato con: joblib.dump(X_train.columns.tolist(), 'model/feature_names.pkl'))

# if __name__ == "__main__":
#     # Percorsi ai file salvati
#     MODEL_PATH = "model/best_keras_model.h5" # Aggiusta il percorso
#     SCALER_PATH = "model/scaler.pkl"         # Aggiusta il percorso
#     FEATURE_NAMES_PATH = "model/feature_names.pkl" # Aggiusta il percorso

#     # Crea l'istanza della classe
#     interactive_model = InteractiveKerasModel(
#         model_path=MODEL_PATH,
#         scaler_path=SCALER_PATH,
#         feature_names_path=FEATURE_NAMES_PATH
#     )

#     # Avvia l'interfaccia
#     interactive_model.run_interface()

# Per eseguire lo script con Streamlit:
# salva il codice in un file, ad esempio 'interactive_model.py'
# apri il terminale
# vai nella directory dove hai salvato il file
# esegui: streamlit run interactive_model.py