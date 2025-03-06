import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import pandas as pd
import tensorflow.keras.backend as K
from data_loader import extract_features_from_csv  # ✅ Reuse feature extraction

# 🔹 Step 1: Define and Register `l1_distance`
@tf.keras.utils.register_keras_serializable()
def l1_distance(vects):
    """Computes L1 distance (Manhattan distance) between two input vectors."""
    x, y = vects
    return K.abs(x - y)

# 🔹 Step 2: Register Custom Function Before Loading Model
custom_objects = {"l1_distance": l1_distance}

# 🔹 Step 3: Load trained model
print("📥 Loading trained Siamese model...")
try:
    siamese_model = load_model("models/siamese_model.h5", custom_objects=custom_objects, compile=False)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ ERROR: Could not load model! Reason: {e}")
    exit()

# 🔹 Step 4: Find the correct embedding layer
print("\n🔍 Checking Model Layers:")
for i, layer in enumerate(siamese_model.layers):
    if hasattr(layer, "output_shape"):  # ✅ Skip Input Layers
        print(f"🔹 Layer {i}: {layer.name}, Output Shape: {layer.output_shape}")
    else:
        print(f"⚠️ Skipping Layer {i}: {layer.name} (No output shape)")

# Extract the correct embedding layer
embedding_layer_index = 2  # Adjust if necessary
base_model = siamese_model.get_layer(index=embedding_layer_index)
expected_input_shape = base_model.input_shape[-1]  # Get expected feature size
print(f"✅ Using `{base_model.name}` as the embedding model (Expected Input Shape: {expected_input_shape})")

# 🔹 Step 5: Load dataset again to extract features
file_path = os.path.join(os.path.dirname(__file__), "FreeDB.csv")
df = pd.read_csv(file_path, low_memory=False)

# Ensure dataset isn't empty
if df.empty:
    print("❌ ERROR: Dataset is empty!")
    exit()

# Extract features from the dataset
print("🔍 Extracting features from dataset...")
keystroke_features = extract_features_from_csv(df)

# Ensure feature extraction worked
if not keystroke_features:
    print("❌ ERROR: No features extracted! Exiting.")
    exit()

# 🔹 Step 6: Define a fixed feature order based on training (adjust to match training)
fixed_feature_keys = [
    "avg_dwell_time", "std_dwell_time",
    "avg_flight_time", "std_flight_time",
    "avg_latency", "std_latency",
    "avg_UU_time", "std_UU_time"
]  # ✅ Ensure this matches the model's expected input!

# Ensure that we have the correct number of features
if len(fixed_feature_keys) != expected_input_shape:
    print(f"❌ ERROR: Model expects {expected_input_shape} features, but we have {len(fixed_feature_keys)}")
    exit()

# 🔹 Step 7: Convert extracted features to embeddings
user_embeddings = {}

for user, features in keystroke_features.items():
    # ✅ Ensure all features are numerical & maintain a fixed order
    numerical_features = []
    for key in fixed_feature_keys:
        value = features.get(key, 0.0)  # Default to 0 if missing
        try:
            numerical_features.append(float(value))  # Convert to float
        except (ValueError, TypeError):
            print(f"⚠ Skipping non-numerical value for {user}, feature '{key}': {value}")
            numerical_features.append(0.0)  # Replace invalid values with 0

    # ✅ Convert to NumPy array
    feature_vector = np.array(numerical_features, dtype=np.float32).reshape(1, -1)

    # ✅ Debugging feature shape
    print(f"🔍 {user} -> Feature Shape BEFORE embedding: {feature_vector.shape}")

    # Ensure feature vector has correct shape
    if feature_vector.shape[-1] != expected_input_shape:
        print(f"❌ ERROR: Feature vector for {user} has incorrect shape {feature_vector.shape}, expected ({expected_input_shape},)")
        continue  # Skip incorrect features

    # ✅ Generate embedding
    embedding = base_model.predict(feature_vector)[0]
    print(f"✅ {user} -> Embedding Shape AFTER base_model: {embedding.shape}")

    if embedding.shape[0] != 32:
        print(f"🚨 WARNING: {user} has incorrect embedding shape: {embedding.shape}")

    user_embeddings[user] = embedding  # Store computed embeddings

# 🔹 Step 8: Save embeddings
embedding_path = "user_embeddings.npy"
np.save(embedding_path, user_embeddings)
print("✅ User embeddings saved to `user_embeddings.npy` successfully!")
