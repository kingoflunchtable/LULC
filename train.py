import tensorflow as tf
from tensorflow.keras import layers, models
import os
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import json

# 1. DIRECTORY SETUP
data_dir = r'C:\Users\Praney\OneDrive\Desktop\satellite\EuroSAT_RGB'

if not os.path.exists(data_dir):
    print(f"❌ ERROR: Folder not found at {data_dir}")
else:
    print(f"✅ Data Found. Initializing 70/30 Research Split...")

    # 2. LOAD DATASET (70-30 Split)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.3, # 70% Training
        subset="training",
        seed=123,
        image_size=(256, 256),
        batch_size=32
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.3, # 30% Testing/Validation
        subset="validation",
        seed=123,
        image_size=(256, 256),
        batch_size=32
    )

    # Pre-fetching for better performance
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # 3. BUILD CNN MODEL (Transfer Learning)
    # Using MobileNetV2 as the CNN backbone for Sentinel-2 feature extraction
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(256, 256, 3), 
        include_top=False, 
        weights='imagenet'
    )
    base_model.trainable = False 

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2), # Prevents overfitting
        layers.Dense(10, activation='softmax') # 10 LULC Classes
    ])

    model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )

    # 4. TRAINING (15 Epochs)
    print("🚀 Starting 15-Epoch Training...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=15)

    # 5. ACCURACY ASSESSMENT (Kappa & Confusion Matrix)
    print("\n🧪 Performing Accuracy Assessment...")
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    # Kappa Coefficient calculation
    kappa = cohen_kappa_score(y_true, y_pred)
    # Confusion Matrix generation
    cm = confusion_matrix(y_true, y_pred)

    print(f"✅ Training Complete!")
    print(f"✅ Final Kappa Coefficient: {kappa:.4f}")
    print("✅ Confusion Matrix:\n", cm)

    # 6. SAVE THE SMART MODEL
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save('models/your_model.h5')
    print("📂 Model saved as 'models/your_model.h5'")


# Create the metrics dictionary
stats = {
    "accuracy": f"{history.history['accuracy'][-1]*100:.2f}%",
    "kappa": f"{kappa:.4f}"
}

# Save it to the models folder
with open('models/metrics.json', 'w') as f:
    json.dump(stats, f)

print("📊 Metrics saved! Your app will now show real performance data.")