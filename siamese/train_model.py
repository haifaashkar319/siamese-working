import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_custom_objects
import tensorflow.keras.backend as K
from data_loader import X_train, Y_train
from siamese import create_base_network, create_head_model, SiameseNetwork

# 🔹 Step 1: Define and Register `l1_distance`
@tf.keras.utils.register_keras_serializable()
def l1_distance(vects):
    """Computes L1 distance (Manhattan distance) between two input vectors."""
    x, y = vects
    return K.abs(x - y)

# 🔹 Register globally for saving
get_custom_objects()["l1_distance"] = l1_distance

# 🔹 Step 2: Define Model Parameters
num_features = X_train.shape[-1]
input_shape = (num_features,)

# 🔹 Step 3: Create Base Model & Siamese Network
base_model = create_base_network(input_shape)
head_model = create_head_model(base_model.output_shape)
siamese_network = SiameseNetwork(base_model, head_model)

# 🔹 Step 4: Compile the Model
siamese_network.compile(loss='binary_crossentropy',
                        optimizer=Adam(learning_rate=0.001),
                        metrics=['accuracy'])

# Convert to NumPy Arrays (Ensure Correct Shape)
X_train = np.array(X_train, dtype=np.float32)
Y_train = np.array(Y_train, dtype=np.float32)

# Split into training and validation sets (80/20 split)
split_index = int(0.8 * len(X_train))
X_train_data, Y_train_data = X_train[:split_index], Y_train[:split_index]
X_val_data, Y_val_data = X_train[split_index:], Y_train[split_index:]

# 🔹 Step 5: Train the Model
siamese_network.fit(X_train_data, Y_train_data,
                    batch_size=32,
                    epochs=20,
                    validation_data=(X_val_data, Y_val_data))

# 🔹 Step 6: Save the Model with `l1_distance`
siamese_network.siamese_model.save("models/siamese_model.h5")

# 🔹 Step 7: Training Complete Message
print("✅ Siamese network training complete. Model saved.")
