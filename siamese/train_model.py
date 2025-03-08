import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_custom_objects
import tensorflow.keras.backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Local imports
from data_loader import (
    load_data, 
    validate_data, 
    preprocess_data, 
    extract_features_for_session, 
    create_training_pairs
)
from siamese import (
    create_base_network, 
    create_head_model, 
    SiameseNetwork
)

# 🔹 Step 1: Load and preprocess the dataset
print("📥 Loading data...")
df = load_data()

print("✅ Validating data...")
df = validate_data(df)

print("🔍 Preprocessing data...")
df = preprocess_data(df)

print("🧑‍💻 Extracting features per session...")
features_by_session = extract_features_for_session(df)

print("📊 Creating training pairs...")
X_train, Y_train = create_training_pairs(features_by_session)

# Validate and prepare input data
if len(X_train) == 0:
    raise ValueError("No training pairs generated!")

# 🔹 Step 2: Convert training pairs into separate arrays
X1_train_array = np.array([pair[0] for pair in X_train], dtype=np.float32)
X2_train_array = np.array([pair[1] for pair in X_train], dtype=np.float32)
Y_train_array = np.array(Y_train, dtype=np.float32)

scaler = StandardScaler()
X1_train_array = scaler.fit_transform(X1_train_array)
X2_train_array = scaler.transform(X2_train_array)



# Verify shapes match
if X1_train_array.shape != X2_train_array.shape:
    raise ValueError(f"❌ Input shape mismatch: {X1_train_array.shape} vs {X2_train_array.shape}")

# Define model input shape based on feature vector size
input_shape = (X1_train_array.shape[1],)

# 🔹 Step 3: Split data for training and validation
X1_train_data, X1_val_data, X2_train_data, X2_val_data, Y_train_data, Y_val_data = train_test_split(
    X1_train_array, X2_train_array, Y_train_array, test_size=0.2, random_state=42
)

print(f"✅ Training set size: {len(X1_train_data)} pairs")
print(f"✅ Validation set size: {len(X1_val_data)} pairs")

# 🔹 Step 4: Define and Register `l1_distance`
@tf.keras.utils.register_keras_serializable()
def l1_distance(vects):
    """Computes L1 distance (Manhattan distance) between two input vectors."""
    x, y = vects
    return K.abs(x - y)

# Register globally for saving
get_custom_objects()["l1_distance"] = l1_distance

# 🔹 Step 5: Create and compile the Siamese model
print("🛠️ Building the Siamese network...")
base_model = create_base_network(input_shape)
head_model = create_head_model(base_model.output_shape)
siamese_network = SiameseNetwork(base_model, head_model)

print("🛠️ Compiling the model...")
siamese_network.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.0005),
    metrics=['accuracy']
)

# Debug: Print model summary
print(f"✅ Base Model Summary:")
base_model.summary()

print(f"✅ Siamese Model Summary:")
siamese_network.siamese_model.summary()


# 🔹 Step 6: Train the Model
print("🚀 Training the Siamese network...")

def debug_training_data(epoch, logs):
    print(f"\n🔹 Epoch {epoch+1}: Training on {len(X1_train_data)} pairs")
    
history = siamese_network.fit(
    [X1_train_data, X2_train_data],  # Training inputs as a list of two arrays
    Y_train_data,
    batch_size=32,
    validation_data=([X1_val_data, X2_val_data], Y_val_data),
    epochs=30,
    verbose=1,
    callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=debug_training_data)]
)

# 🔹 Step 7: Evaluate Model Performance
print("📊 Evaluating model performance...")
val_loss, val_accuracy = siamese_network.evaluate(
    [X1_val_data, X2_val_data], 
    Y_val_data,
    verbose=1
)
print(f"✅ Final validation accuracy: {val_accuracy:.4f}")

# 🔹 Step 8: Save the Model
print("💾 Saving the trained model...")
siamese_network.siamese_model.save("models/siamese_model.h5")

print("✅ Siamese network training complete. Model saved successfully!")
