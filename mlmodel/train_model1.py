"""
Anaemia Detection - ML Model Training Script
============================================
Uses EfficientNetB0 (transfer learning) to classify blood smear images
into 3 classes: L1, L2, L3.

Dataset structure expected:
  dataset/
    train/
      L1/
      L2/
      L3/
    val/
      L1/
      L2/
      L3/
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ─── CONFIG ────────────────────────────────────────────────────────────────────

IMG_SIZE         = (224, 224)
BATCH_SIZE       = 32
EPOCHS           = 30
FINE_TUNE_EPOCHS = 20
LEARNING_RATE    = 1e-4
DATASET_DIR      = "dataset"
MODEL_DIR        = "saved_model"
LABELS           = ["L1", "L2", "L3"]   # ✅ your 3 classes

os.makedirs(MODEL_DIR, exist_ok=True)


# ─── DATA AUGMENTATION ─────────────────────────────────────────────────────────

def build_data_generators():
    """Build train/val data generators with augmentation."""

    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )

    val_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_ds = train_gen.flow_from_directory(
        os.path.join(DATASET_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",        # ✅ 3 classes → categorical
        classes=LABELS,
        shuffle=True,
    )

    val_ds = val_gen.flow_from_directory(
        os.path.join(DATASET_DIR, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",        # ✅ 3 classes → categorical
        classes=LABELS,
        shuffle=False,
    )

    return train_ds, val_ds              # ✅ removed test_ds (no test folder)


# ─── MODEL ARCHITECTURE ────────────────────────────────────────────────────────

def build_model(trainable_base=False):
    """
    EfficientNetB0 with custom classification head.
    Phase 1: train only the head (base frozen).
    Phase 2: fine-tune top layers of base.
    """
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
    )
    base.trainable = trainable_base

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=trainable_base)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(3, activation="softmax")(x)  # ✅ 3 neurons + softmax

    model = Model(inputs, outputs)
    return model, base


# ─── TRAINING ──────────────────────────────────────────────────────────────────

def get_callbacks(phase: str):
    return [
        callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, f"best_{phase}.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.CSVLogger(os.path.join(MODEL_DIR, f"history_{phase}.csv")),
    ]


def compute_class_weights(train_ds):
    """Handle class imbalance with weighted loss."""
    counts = np.bincount(train_ds.classes)
    total  = counts.sum()
    return {i: total / (len(counts) * c) for i, c in enumerate(counts)}


def train():
    print("=" * 60)
    print("  Anaemia Detection — Model Training")
    print("=" * 60)

    train_ds, val_ds = build_data_generators()          # ✅ only 2 return values
    class_weights = compute_class_weights(train_ds)
    print(f"\nClass weights: {class_weights}")
    print(f"Train samples : {train_ds.samples}")
    print(f"Val samples   : {val_ds.samples}\n")

    # ── Phase 1: head only ────────────────────────────────────────────────────
    print("Phase 1: Training classification head (base frozen)...")
    model, base = build_model(trainable_base=False)
    model.compile(
        optimizer=Adam(LEARNING_RATE),
        loss="categorical_crossentropy",                # ✅ not binary
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    model.summary()

    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=get_callbacks("phase1"),
    )

    # ── Phase 2: fine-tune top 30 layers ──────────────────────────────────────
    print("\nPhase 2: Fine-tuning top layers of EfficientNet...")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(LEARNING_RATE / 10),
        loss="categorical_crossentropy",                # ✅ not binary
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )

    hist2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=get_callbacks("phase2"),
    )

    # ── Save final model ───────────────────────────────────────────────────────
    final_path = os.path.join(MODEL_DIR, "anaemia_model.keras")
    model.save(final_path)
    print(f"\nModel saved → {final_path}")

    # Save metadata
    meta = {
        "img_size": list(IMG_SIZE),
        "labels": LABELS,
        "model_path": final_path,
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── Evaluation on val set ─────────────────────────────────────────────────
    evaluate(model, val_ds)
    plot_history(hist1, hist2)


# ─── EVALUATION ────────────────────────────────────────────────────────────────

def evaluate(model, val_ds):
    print("\nEvaluating on validation set...")
    val_ds.reset()
    preds = model.predict(val_ds, verbose=1)
    y_pred = np.argmax(preds, axis=1)               # ✅ argmax for multi-class
    y_true = val_ds.classes

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=LABELS))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=LABELS, yticklabels=LABELS)
    plt.title("Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
    print("Confusion matrix saved.")


def plot_history(*hists):
    all_acc  = []
    all_val  = []
    all_loss = []
    all_vloss= []
    for h in hists:
        all_acc  += h.history["accuracy"]
        all_val  += h.history["val_accuracy"]
        all_loss += h.history["loss"]
        all_vloss+= h.history["val_loss"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(all_acc) + 1)

    ax1.plot(epochs, all_acc,  label="Train accuracy")
    ax1.plot(epochs, all_val,  label="Val accuracy")
    ax1.set_title("Accuracy"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, all_loss,  label="Train loss")
    ax2.plot(epochs, all_vloss, label="Val loss")
    ax2.set_title("Loss"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "training_curves.png"), dpi=150)
    print("Training curves saved.")


# ─── RBC ANALYSIS UTILITIES ────────────────────────────────────────────────────

def analyze_rbc_morphology(image_rgb: np.ndarray) -> dict:
    """
    Morphological analysis of RBCs using classical CV.
    Called by the backend for per-cell feature extraction.
    Returns counts and flags.
    """
    import cv2
    from scipy import ndimage

    img = (image_rgb * 255).astype(np.uint8) if image_rgb.max() <= 1 else image_rgb
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rbc_areas, pallor_ratios, circularities = [], [], []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300 or area > 8000:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < 0.4:
            continue

        rbc_areas.append(area)
        circularities.append(circularity)

        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        x, y, w, h = cv2.boundingRect(cnt)
        inner_r = int(min(w, h) * 0.25)
        cx = x + w // 2; cy = y + h // 2
        inner_mask = np.zeros_like(mask)
        cv2.circle(inner_mask, (cx, cy), max(inner_r, 1), 255, -1)
        combined = cv2.bitwise_and(mask, inner_mask)
        inner_mean = cv2.mean(gray, mask=combined)[0]
        outer_mask = cv2.bitwise_and(mask, cv2.bitwise_not(inner_mask))
        outer_mean = cv2.mean(gray, mask=outer_mask)[0]
        if outer_mean > 0:
            pallor_ratios.append(inner_mean / outer_mean)

    n = len(rbc_areas)
    mean_area   = float(np.mean(rbc_areas))     if rbc_areas     else 0
    mean_pallor = float(np.mean(pallor_ratios)) if pallor_ratios else 0
    mean_circ   = float(np.mean(circularities)) if circularities else 0

    flags = []
    if mean_pallor > 0.75: flags.append("Hypochromia")
    if mean_area   < 500:  flags.append("Microcytosis")
    if mean_area   > 2000: flags.append("Macrocytosis")
    if mean_circ   < 0.70: flags.append("Poikilocytosis")
    if n < 50:             flags.append("Low RBC density")

    return {
        "rbc_count":      n,
        "mean_cell_area": round(mean_area, 1),
        "pallor_ratio":   round(mean_pallor, 3),
        "circularity":    round(mean_circ, 3),
        "flags":          flags,
    }


if __name__ == "__main__":
    train()
