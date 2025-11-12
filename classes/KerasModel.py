import os
import json
import joblib
import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc as sk_auc

@dataclass
class TrainResult:
    history: object
    best_threshold: float
    val_auc: float

class KerasModel:
    """
    Rete binaria con:
      - Dense -> BN -> ReLU -> Dropout (xN)
      - L2 lieve, Dropout lieve
      - metriche AUC/Precision/Recall
      - early stopping su val_auc
      - calibratore (Platt) + soglia ottimale (Youden J)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_layers: Sequence[int] = (128, 64, 32),
        dropout: float = 0.15,
        l2_reg: float = 1e-3,
        lr: float = 1e-3,
        label_smoothing: float = 0.0,
        model_dir: str = "model",
        name: str = "keras_model.h5",
    ):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.lr = lr
        self.label_smoothing = label_smoothing
        self.model_dir = model_dir
        self.name = name

        os.makedirs(self.model_dir, exist_ok=True)

        self.model = self._build_model()
        self.calibrator: Optional[LogisticRegression] = None
        self.best_threshold: float = 0.5

    # -------------------------- build/compile -------------------------------
    def _block(self, x, units):
        x = layers.Dense(
            units,
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l2(self.l2_reg),
            use_bias=False,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(self.dropout)(x)
        return x

    def _build_model(self):
        inputs = keras.Input(shape=(self.input_dim,))
        x = inputs
        for units in self.hidden_layers:
            x = self._block(x, units)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inputs, outputs)

        loss = keras.losses.BinaryCrossentropy(label_smoothing=self.label_smoothing)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.lr),
            loss=loss,
            metrics=[
                "accuracy",
                keras.metrics.AUC(name="auc"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
            ],
        )
        return model

    # ------------------------------ training --------------------------------
    def train(
            self,
            X_train, y_train,
            X_val, y_val,
            epochs: int = 100,
            batch_size: int = 32,
            class_weight: Optional[dict] = None,
            verbose: int = 1,
    ):
        """Addestra la rete, calcola soglia ottimale e restituisce i valori richiesti dal main."""

        # Pesi di classe bilanciati automaticamente
        if class_weight is None:
            cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
            class_weight = {0: float(cw[0]), 1: float(cw[1])}
            print(f"🎯 Class weights: {class_weight}")

        ckpt_path = os.path.join(self.model_dir, self.name)
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                ckpt_path, monitor="val_auc", mode="max", save_best_only=True, verbose=0
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_auc", mode="max", factor=0.5, patience=6, min_lr=1e-5, verbose=0
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=15, restore_best_weights=True, verbose=0
            ),
        ]

        # Training
        hist = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=verbose,
        )

        # Ricarica pesi migliori
        if os.path.exists(ckpt_path):
            self.model = keras.models.load_model(ckpt_path)

        # Probabilità sul validation set
        y_prob = self.model.predict(X_val, verbose=0).ravel()

        # Ricerca soglia ottimale (Youden J)
        from sklearn.metrics import roc_curve, auc, balanced_accuracy_score, f1_score
        fpr, tpr, thr = roc_curve(y_val, y_prob)
        youden = tpr - fpr
        self.best_threshold = float(thr[np.argmax(youden)])
        val_auc = float(auc(fpr, tpr))

        # Predizioni binarie
        y_pred = (y_prob >= self.best_threshold).astype(int)

        # Report sintetico (per il main)
        keras_report = {
            "val_auc": val_auc,
            "val_balanced_acc": float(balanced_accuracy_score(y_val, y_pred)),
            "val_f1": float(f1_score(y_val, y_pred)),
            "best_threshold": self.best_threshold,
            "dist_pred": np.bincount(y_pred).tolist(),
        }

        # Salva soglia
        with open(os.path.join(self.model_dir, "keras_meta.json"), "w") as f:
            json.dump({"best_threshold": self.best_threshold}, f)

        print(f"✅ Best threshold: {self.best_threshold:.3f} | AUC: {val_auc:.3f}")
        return hist, y_prob, y_pred, keras_report

    # ------------------------------ inference --------------------------------
    def predict_proba(self, X, calibrated: bool = True) -> np.ndarray:
        raw = self.model.predict(X, verbose=0).reshape(-1)
        if calibrated and self.calibrator is not None:
            return raw  # calibrazione disattivata
        return raw

    def predict_classes(self, X, threshold: Optional[float] = None, calibrated: bool = True) -> np.ndarray:
        probs = self.predict_proba(X, calibrated=calibrated)
        thr = self.best_threshold if threshold is None else threshold
        return (probs >= thr).astype(int)

    # ------------------------------- IO --------------------------------------
    def save_model(self):
        self.model.save(os.path.join(self.model_dir, self.name))
        if self.calibrator is not None:
            joblib.dump(self.calibrator, os.path.join(self.model_dir, "keras_calibrator.pkl"))
        with open(os.path.join(self.model_dir, "keras_meta.json"), "w") as f:
            json.dump({"best_threshold": self.best_threshold}, f)

    def load(self):
        self.model = keras.models.load_model(os.path.join(self.model_dir, self.name))
        cal_path = os.path.join(self.model_dir, "keras_calibrator.pkl")
        meta_path = os.path.join(self.model_dir, "keras_meta.json")
        self.calibrator = joblib.load(cal_path) if os.path.exists(cal_path) else None
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.best_threshold = json.load(f).get("best_threshold", 0.5)
