import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import os
import json
import numpy as np
from sklearn.metrics import cohen_kappa_score

# --- Configuration ---
DATA_DIR = 'EuroSAT_RGB'
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

# --- 1. Load Data ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training",
    seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation",
    seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)

train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# --- 2. Build Model (One-Shot Strategy) ---
print("\n🏗️ Building One-Shot Model...")
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3), include_top=False, weights='imagenet'
)
base_model.trainable = True # Train everything from the jump

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# --- 3. Compile & Train ---
# 1e-4 is the "sweet spot" for training a full MobileNetV2 without crashing
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping to catch it when it hits the peak
early_stop = callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)

print("\n🚀 Training started. It will start low and climb steadily. No more phases!")
model.fit(train_ds, validation_data=val_ds, epochs=25, callbacks=[early_stop])

# --- 4. Final Assessment ---
print("\n🧪 Final Assessment...")
model.save(os.path.join(MODELS_DIR, 'your_model.h5'))

y_true, y_pred = [], []
for x, y in val_ds:
    y_true.extend(y.numpy())
    preds = model.predict(x, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))

kappa = cohen_kappa_score(np.array(y_true), np.array(y_pred))
final_acc = model.evaluate(val_ds, verbose=0)[1]

metrics = {
    "accuracy": round(float(final_acc) * 100, 2),
    "kappa": round(float(kappa), 4),
    "classes": class_names
}

with open(os.path.join(MODELS_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics, f)

print(f"✅ DONE! Final Accuracy: {metrics['accuracy']}% | Kappa: {metrics['kappa']}")