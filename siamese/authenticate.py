import pandas as pd
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pynput import keyboard
from data_loader import extract_features_from_csv  # ✅ Extracts raw features

@tf.keras.utils.register_keras_serializable()
def l1_distance(vects):
    """Computes L1 distance (Manhattan distance) between two input vectors."""
    x, y = vects
    return K.abs(x - y)

custom_objects = {"l1_distance": l1_distance}
print("checkpoint 1")
# 🔹 Load the trained Siamese model
print("📥 Loading Siamese model...")
siamese_model = load_model("models/siamese_model.h5", custom_objects=custom_objects, compile=False)
print("checkpoint 2")
print("✅ Siamese model loaded.")

# 🔍 **Print Model Summary**
print("\n🔍 Model Summary:")
siamese_model.summary()
print("checkpoint 3")
# 🔍 **Check Layers and Shapes**
print("\n🔍 Siamese Model Layers & Shapes:")
for layer in siamese_model.layers:
    print(f"🔹 Layer: {layer.name}, Input Shape: {layer.input_shape}, Output Shape: {layer.output_shape}")

# 🔹 Identify the correct embedding layer
# If your base_model is at index 2 and it expects input shape (None, 32), it means we are using embeddings, not raw features.
base_model = siamese_model.get_layer(index=2)
print(f"\n✅ Selected Base Model: {base_model.name}")
print(f"🔹 Base Model Input Shape: {base_model.input_shape}, Output Shape: {base_model.output_shape}")

# 🔹 Load stored embeddings
embedding_path = "user_embeddings.npy"
if os.path.exists(embedding_path):
    user_embeddings = np.load(embedding_path, allow_pickle=True).item()
    print("✅ User embeddings loaded.")
    print(f"🔍 Available user embeddings: {list(user_embeddings.keys())}")
else:
    print("❌ No stored embeddings found! Run training first.")
    user_embeddings = {}

# 🔹 Define dataset path
file_path = "FreeDB.csv"

def authenticate_user():
    """Runs authentication using stored embeddings."""
    user_id = input("Enter your user ID (e.g., p101, p102): ").strip()

    if user_id not in user_embeddings:
        print(f"❌ Authentication failed. No stored embedding for user {user_id}.")
        return False

    print("\nStart typing a short sentence (10-15 words) to verify your identity. Press 'Enter' when done.\n")

    # 🔹 Keystroke Recording
    keystrokes = []
    start_time = time.time()
    prev_key = None
    prev_down_time = None
    prev_up_time = None

    def on_press(key):
        """Captures key press events and records timing data."""
        nonlocal prev_key, prev_down_time

        if key == keyboard.Key.enter:
            print("\n✅ Typing sample recorded. Verifying your identity...\n")
            return False  # Stop listening

        key_name = key.name.capitalize() if isinstance(key, keyboard.Key) else key.char
        timestamp = time.time()

        # Compute inter-key timing values
        du_time = round(timestamp - prev_down_time, 3) if prev_down_time else 0
        dd_time = round(timestamp - prev_up_time, 3) if prev_up_time else 0
        ud_time = round(timestamp - prev_down_time, 3) if prev_down_time else 0
        uu_time = round(timestamp - prev_up_time, 3) if prev_up_time else 0

        key2 = prev_key if prev_key else key_name  # Use previous key as key1
        key1 = key_name  # Current key as key2

        # Append keystroke data
        keystrokes.append([user_id, "1", key1, key2, du_time, dd_time, ud_time, uu_time])

        # Update previous key states
        prev_key = key_name
        prev_down_time = timestamp

    def on_release(key):
        """Records key release timestamps."""
        nonlocal prev_up_time
        prev_up_time = time.time()

    # 🔹 Start keystroke listener
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    # 🔹 Convert keystrokes to DataFrame
    new_keystroke_data = pd.DataFrame(keystrokes, columns=["participant", "session", "key1", "key2",
                                                            "DU.key1.key1", "DD.key1.key2", "UD.key1.key2", "UU.key1.key2"])
    
    # 🔹 Extract raw features from new input
    new_features_dict = extract_features_from_csv(new_keystroke_data)

    if user_id not in new_features_dict:
        print("❌ Failed to extract features from input.")
        return False

    # 🔥 **DEBUG: Check Raw Features Shape Before Embedding**
    new_features = np.array(list(new_features_dict[user_id].values()), dtype=np.float32).reshape(1, -1)
    print(f"✅ New Features Shape (before embedding): {new_features.shape}")

    # 🔥 **DEBUG: Check Expected Input Shape for `base_model`**
    if new_features.shape[-1] != base_model.input_shape[-1]:
        print(f"⚠️ Mismatch: New features have {new_features.shape[-1]} dimensions but `base_model` expects {base_model.input_shape[-1]}!")

    # 🔹 Convert raw features to embeddings using `base_model`
    new_embedding = base_model.predict(new_features)[0]  # ✅ Convert to embedding

    # 🔹 Retrieve stored embedding for this user
    stored_embedding = np.array(user_embeddings[user_id], dtype=np.float32)
    print(f"✅ Stored Embedding for {user_id} -> Shape: {stored_embedding.shape}")
    print(f"✅ New Embedding for {user_id} -> Shape: {new_embedding.shape}")

    # 🔥 **DEBUG: Check if stored and new embeddings have the same shape**
    if stored_embedding.shape != new_embedding.shape:
        print(f"⚠️ Shape Mismatch! Stored Embedding: {stored_embedding.shape}, New Embedding: {new_embedding.shape}")
        return False

    # 🔹 Compare stored vs. new embedding using the Siamese model
    similarity_score = siamese_model.predict([stored_embedding.reshape(1, -1), new_embedding.reshape(1, -1)])[0][0]

    print(f"\n📊 Similarity Score: {similarity_score}")

    # 🔹 Authentication threshold
    threshold = 0.7

    if similarity_score >= threshold:
        print(f"✅ Authentication successful! Welcome back, {user_id}.")
        return True
    else:
        print("❌ Authentication failed. Typing pattern does not match.")
        return False

# 🔹 Run authentication
authenticate_user()
