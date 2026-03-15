import tensorflow as tf
import os

# Create the directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Define a simple architecture based on your paper
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(256, 256, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name="target_conv_layer"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(5, activation='softmax') 
])

# Save it
model.save('models/your_model.h5')
print("✅ Dummy model created successfully in /models/your_model.h5")