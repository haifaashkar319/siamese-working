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

# Fix: Get input shape correctly from model's config
base_model = siamese_model.get_layer(index=2)  # Adjust index if necessary
expected_input_shape = base_model.input_shape[-1]  # Get expected feature size
print(f"✅ Using `{base_model.name}` as the embedding model (Expected Input Shape: {expected_input_shape})")

# 🔹 Step 4: Load dataset again to extract features
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

# 🔹 Step 5: Define a fixed feature order based on training (adjust to match training)
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

# Before processing features
print(f"🔍 Number of fixed features: {len(fixed_feature_keys)}")

# 🔹 Step 6: Convert extracted features to embeddings
user_embeddings = {}

# Modify the embedding generation section
for user, features in keystroke_features.items():
    numerical_features = []
    for key in fixed_feature_keys:
        value = features.get(key, 0.0)
        try:
            numerical_features.append(float(value))
        except (ValueError, TypeError):
            print(f"⚠️ Invalid value for {user}, feature '{key}': {value}")
            numerical_features.append(0.0)

    feature_vector = np.array(numerical_features, dtype=np.float32).reshape(1, -1)

    # Ensure correct shape for model input
    if feature_vector.shape != (1, expected_input_shape):  # ✅ Fixed shape validation
        print(f"❌ Shape mismatch for {user}: Got {feature_vector.shape}, expected {(1, expected_input_shape)}")
        continue

    # Generate embedding
    embedding = base_model.predict(feature_vector, verbose=0)[0]
    print(f"✅ {user} -> Feature shape: {feature_vector.shape}, Embedding shape: {embedding.shape}")
    user_embeddings[user] = embedding

# Ensure embeddings were created
if not user_embeddings:
    print("❌ ERROR: No embeddings were created! Check input data.")
    exit()

print(f"\n🔍 Final embedding dimensions: {next(iter(user_embeddings.values())).shape}")

# 🔹 Step 7: Save embeddings
embedding_path = "user_embeddings.npy"
np.save(embedding_path, user_embeddings)
print("✅ User embeddings saved to `user_embeddings.npy` successfully!")
