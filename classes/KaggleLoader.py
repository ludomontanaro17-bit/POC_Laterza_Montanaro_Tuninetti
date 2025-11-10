import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

class KaggleLoader:
    def __init__(self, dataset: str = "rishikeshkonapure/home-loan-approval", download_dir: str = "./data"):
        """
        Inizializza il loader con il nome del dataset e la directory di destinazione.
        Usa il dataset: rishikeshkonapure/home-loan-approval
        Aggiorna i nomi dei file con quelli reali nel dataset.
        """
        self.dataset = dataset
        self.download_dir = download_dir
        self.api = KaggleApi()
        # Assicurati di aver configurato il token di autenticazione Kaggle
        self.api.authenticate()

        # Aggiorna i nomi dei file con i nomi reali nel dataset
        self.train_filename = "home_sanction_train.csv"
        self.test_filename = "home_sanction_test.csv"
        self.train_path = os.path.join(self.download_dir, self.train_filename)
        self.test_path = os.path.join(self.download_dir, self.test_filename)

    def load(self) -> str:
        """
        Scarica e decomprime il dataset generico Kaggle.
        """
        os.makedirs(self.download_dir, exist_ok=True)

        print(f"⬇️  Downloading dataset: {self.dataset}")
        # Scarica il dataset
        self.api.dataset_download_files(self.dataset, path=self.download_dir, unzip=True)

        print(f"✅ Dataset estratto in: {os.path.abspath(self.download_dir)}")
        return os.path.abspath(self.download_dir)

    def print_information(self):
        """
        Carica i file home_sanction_train.csv e home_sanction_test.csv e mostra informazioni.
        """
        # Controlla l'esistenza dei file con i nomi corretti
        if not os.path.exists(self.train_path):
            print(f"❌ File non trovato: {self.train_path}")
            return
        if not os.path.exists(self.test_path):
            print(f"❌ File non trovato: {self.test_path}")
            return

        # Carica i dataset
        df_train = pd.read_csv(self.train_path)
        df_test = pd.read_csv(self.test_path)

        # --- TRAIN ---
        print("\n=== 🔹 HOME_SANCTION_TRAIN.CSV ===")
        print(f"Dimensioni: {df_train.shape[0]} righe × {df_train.shape[1]} colonne")
        print("\n📄 Prime righe:")
        print(df_train.head())
        print("\n📊 Statistiche descrittive:")
        print(df_train.describe(include='all'))
        print(f"\n Tipi di dati per colonna:\n{df_train.dtypes}")
        print(f"\n Valori nulli per colonna:\n{df_train.isnull().sum()}")

        # --- TEST ---
        print("\n=== 🔸 HOME_SANCTION_TEST.CSV ===")
        print(f"Dimensioni: {df_test.shape[0]} righe × {df_test.shape[1]} colonne")
        print("\n📄 Prime righe:")
        print(df_test.head())
        print("\n📊 Statistiche descrittive:")
        print(df_test.describe(include='all'))
        print(f"\n Tipi di dati per colonna:\n{df_test.dtypes}")
        print(f"\n Valori nulli per colonna:\n{df_test.isnull().sum()}")

    def get_full_dataset(self):
        """
        Restituisce i due DataFrame: train e test.
        """
        if not os.path.exists(self.train_path) or not os.path.exists(self.test_path):
            print("❌ Uno o entrambi i file non esistono. Chiama prima il metodo `load()`.")
            return None, None

        df_train = pd.read_csv(self.train_path)
        df_test = pd.read_csv(self.test_path)
        print(f"✅ Dataset caricati. Train shape: {df_train.shape}, Test shape: {df_test.shape}")
        return df_train, df_test


if __name__ == "__main__":
    # Usa il nome del dataset corretto
    loader = KaggleLoader(dataset="rishikeshkonapure/home-loan-approval", download_dir="./data")
    loader.load()
    loader.print_information()
    # df_train, df_test = loader.get_full_dataset()