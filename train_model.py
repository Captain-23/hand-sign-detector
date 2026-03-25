"""
Train a hand sign detection model on the ASL MNIST dataset.
Produces a cvzone-compatible Keras .h5 model + labels.txt.

Usage:
    /opt/anaconda3/envs/handsign/bin/python train_model.py
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split

# ============================================================
# Config
# ============================================================
CSV_PATH = "DATA/sign_mnist_train.csv"
MODEL_OUT = "Model/keras_model.h5"
LABELS_OUT = "Model/labels.txt"
IMG_SIZE = 224          # cvzone Classifier expects 224x224x3
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42

# ASL MNIST labels → letter mapping
# Labels 0-24 map to A-Z, but 9 (J) is skipped because it requires motion
# So label 0=A, 1=B, ..., 8=I, 10=K, ..., 24=Y
# Z (25) is also absent from the dataset
LABEL_TO_LETTER = {}
letter_idx = 0
for i in range(26):
    if i == 9:  # skip J
        continue
    if i == 25:  # skip Z
        continue
    LABEL_TO_LETTER[i] = chr(ord('A') + i)

print(f"Label mapping ({len(LABEL_TO_LETTER)} classes):")
for k, v in sorted(LABEL_TO_LETTER.items()):
    print(f"  {k} → {v}")

# ============================================================
# Load & prepare data
# ============================================================
print("\n📂 Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")

y_raw = df['label'].values
X_raw = df.drop('label', axis=1).values

# Map raw labels to consecutive indices 0..23
unique_labels = sorted(np.unique(y_raw))
raw_to_idx = {raw: idx for idx, raw in enumerate(unique_labels)}
idx_to_letter = {idx: LABEL_TO_LETTER[raw] for raw, idx in raw_to_idx.items()}
NUM_CLASSES = len(unique_labels)

print(f"  Found {NUM_CLASSES} classes: {[idx_to_letter[i] for i in range(NUM_CLASSES)]}")

y = np.array([raw_to_idx[label] for label in y_raw])

# Reshape pixels to 28x28 grayscale images
X = X_raw.reshape(-1, 28, 28).astype(np.float32)

print(f"  X shape: {X.shape}, y shape: {y.shape}")

# ============================================================
# Preprocessing: 28x28 gray → 224x224 RGB, normalize to [-1,1]
# ============================================================
print("\n🔄 Preprocessing images (resize to 224×224, convert to RGB)...")
print("   This may take a minute...")

import cv2

def preprocess_batch(images, batch_size=500):
    """Resize and convert grayscale to RGB in batches to save memory."""
    results = []
    total = len(images)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = images[start:end]
        resized = []
        for img in batch:
            # Resize 28x28 → 224x224
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
            # Convert grayscale to 3-channel
            img_rgb = np.stack([img_resized] * 3, axis=-1)
            resized.append(img_rgb)
        results.append(np.array(resized))
        pct = min(100, int((end / total) * 100))
        print(f"   Processed {end}/{total} ({pct}%)", end="\r")
    print()
    return np.concatenate(results, axis=0)

X_processed = preprocess_batch(X)

# Normalize to [-1, 1] (same as cvzone's preprocessing: pixel/127 - 1)
X_processed = (X_processed / 127.0) - 1.0

print(f"  Preprocessed X shape: {X_processed.shape}")
print(f"  Value range: [{X_processed.min():.2f}, {X_processed.max():.2f}]")

# ============================================================
# Train/validation split
# ============================================================
X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"\n  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

# ============================================================
# Build model with MobileNetV2 transfer learning
# ============================================================
print("\n🏗️ Building MobileNetV2 model...")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model — we only train the top layers
base_model.trainable = False

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================
# Train
# ============================================================
print("\n🚀 Training...")

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        verbose=1
    )
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# Evaluate
# ============================================================
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\n✅ Validation Accuracy: {val_acc*100:.1f}%")
print(f"   Validation Loss: {val_loss:.4f}")

# ============================================================
# Save model + labels
# ============================================================
os.makedirs("Model", exist_ok=True)

model.save(MODEL_OUT)
print(f"\n💾 Model saved to: {MODEL_OUT}")

with open(LABELS_OUT, "w") as f:
    for idx in range(NUM_CLASSES):
        f.write(f"{idx} {idx_to_letter[idx]}\n")
print(f"📝 Labels saved to: {LABELS_OUT}")
print(f"   Labels: {[idx_to_letter[i] for i in range(NUM_CLASSES)]}")

print("\n🎉 Training complete! Restart the app to use the new model.")
