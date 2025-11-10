import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import os
import joblib

class KerasModel:
    def __init__(self, input_dim, dropout_rate=0.3, l2_reg=0.01, model_dir="model"):
        """
        Inizializza il wrapper per il modello Keras.

        Args:
            input_dim (int): Numero di features in input.
            dropout_rate (float): Tasso di dropout.
            l2_reg (float): Fattore di regolarizzazione L2.
            model_dir (str): Directory dove salvare il modello.
        """
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.model_dir = model_dir
        self.model = self._build_model()
        # Assicura che la cartella per il modello esista
        os.makedirs(self.model_dir, exist_ok=True)

    def _build_model(self):
        """Costruisce il modello Keras."""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.input_dim,),
                         kernel_regularizer=keras.regularizers.l2(self.l2_reg)),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),

            layers.Dense(64, activation='relu',
                         kernel_regularizer=keras.regularizers.l2(self.l2_reg)),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),

            layers.Dense(32, activation='relu',
                         kernel_regularizer=keras.regularizers.l2(self.l2_reg)),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),

            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, verbose=1):
        """
        Addestra il modello.

        Args:
            X_train, y_train: Dati di training.
            X_val, y_val: Dati di validazione.
            epochs (int): Numero massimo di epoche.
            batch_size (int): Dimensione del batch.
            verbose (int): Verbosità del training.

        Returns:
            history: Oggetto History del training.
        """
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001
        )

        print("Inizio addestramento del modello Keras...")
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose
        )
        print("Addestramento completato.")
        return history

    def predict(self, X):
        """
        Fai previsioni sul dataset X.

        Args:
            X (np.array or pd.DataFrame): Dati di input.

        Returns:
            np.array: Array delle probabilità predette.
        """
        return self.model.predict(X)

    def predict_classes(self, X, threshold=0.5):
        """
        Fai previsioni delle classi sul dataset X.

        Args:
            X (np.array or pd.DataFrame): Dati di input.
            threshold (float): Soglia per la classificazione binaria.

        Returns:
            np.array: Array delle classi predette (0 o 1).
        """
        y_pred_proba = self.predict(X)
        return (y_pred_proba > threshold).astype(int)

    def save_model(self, filename="keras_model.h5"):
        """
        Salva il modello Keras.

        Args:
            filename (str): Nome del file per salvare il modello.
        """
        filepath = os.path.join(self.model_dir, filename)
        self.model.save(filepath)
        print(f"Modello Keras salvato in: {filepath}")

    def load_model(self, filename="keras_model.h5"):
        """
        Carica un modello Keras salvato.

        Args:
            filename (str): Nome del file del modello da caricare.
        """
        filepath = os.path.join(self.model_dir, filename)
        if os.path.exists(filepath):
            self.model = keras.models.load_model(filepath)
            print(f"Modello Keras caricato da: {filepath}")
        else:
            print(f"Errore: Il file {filepath} non esiste.")