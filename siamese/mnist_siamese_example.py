import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from data_loader import X_train, Y_train
from siamese import create_base_network, create_head_model, SiameseNetwork

# 🔹 Define the input shape based on the feature vector size
num_features = X_train.shape[-1]
input_shape = (num_features,)

# 🔹 Create the base model for keystroke feature extraction
base_model = create_base_network(input_shape)

# 🔹 Create the head model to compare feature embeddings
head_model = create_head_model(base_model.output_shape)

# 🔹 Initialize the Siamese network
siamese_network = SiameseNetwork(base_model, head_model)

# 🔹 Compile the model
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

# 🔹 Train the model
siamese_network.fit(X_train_data, Y_train_data,
                    batch_size=32,
                    epochs=20,
                    validation_data=(X_val_data, Y_val_data))


# 🔹 Save the trained model
siamese_network.siamese_model.save("models/siamese_model.h5")

# 🔹 Print training completion message
print("✅ Siamese network training complete. Model saved.")
