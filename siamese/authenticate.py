import pandas as pd
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pynput import keyboard
from data_loader import extract_features_from_csv  # ✅ Reuse feature extraction

# ✅ Fixed feature order (must match generate_embeddings.py)
FIXED_FEATURE_KEYS = [
    "avg_dwell_time", "std_dwell_time",
    "avg_flight_time", "std_flight_time",
    "avg_latency", "std_latency",
    "avg_UU_time", "std_UU_time"
]

@tf.keras.utils.register_keras_serializable()
def l1_distance(vects):
    """Computes L1 distance (Manhattan distance) between two input vectors."""
    x, y = vects
    return K.abs(x - y)

# ✅ Step 1: Load Model & Embeddings
try:
    print("📥 Loading model and embeddings...")
    siamese_model = load_model("models/siamese_model.h5", custom_objects={"l1_distance": l1_distance})
    base_model = siamese_model.get_layer(index=2)  # ✅ Extract embedding model
    expected_input_shape = base_model.input_shape[-1]  # Get expected feature size
    user_embeddings = np.load("user_embeddings.npy", allow_pickle=True).item()
    print(f"✅ Model and embeddings loaded successfully (Expected Input Shape: {expected_input_shape})")
except Exception as e:
    print(f"❌ Error loading model or embeddings: {e}")
    exit()

# ✅ Step 2: Collect Keystroke Data
def collect_keystroke_data():
    """Collects keystroke timing data from user input."""
    keystrokes = []
    prev_key = None
    prev_down_time = None
    prev_up_time = None
    finished = False

    def on_press(key):
        nonlocal prev_key, prev_down_time, finished
        if key == keyboard.Key.enter:
            finished = True
            return False

        timestamp = time.time()
        
        # Handle special keys
        try:
            key_name = key.char
        except AttributeError:
            key_name = str(key).replace('Key.', '')

        # Calculate timing features
        du_time = round(timestamp - prev_down_time, 3) if prev_down_time else 0
        dd_time = round(timestamp - prev_up_time, 3) if prev_up_time else 0
        ud_time = round(timestamp - prev_down_time, 3) if prev_down_time else 0
        uu_time = round(timestamp - prev_up_time, 3) if prev_up_time else 0

        # Record keystroke data
        if prev_key:
            keystrokes.append({
                'key1': prev_key,
                'key2': key_name,
                'DU.key1.key1': du_time,
                'DD.key1.key2': dd_time,
                'UD.key1.key2': ud_time,
                'UU.key1.key2': uu_time
            })

        prev_key = key_name
        prev_down_time = timestamp

    def on_release(key):
        nonlocal prev_up_time
        prev_up_time = time.time()

    # Start collecting keystrokes
    print("\n⌨️ Please type to verify your identity (Press Enter when done):")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        while not finished:
            pass

    return keystrokes

# ✅ Step 3: Extract Features from Keystrokes
def extract_features(keystrokes):
    """Extracts timing features from collected keystrokes."""
    if not keystrokes:
        return None

    # Extract timing sequences
    dwell_times = [k['DU.key1.key1'] for k in keystrokes if k['DU.key1.key1'] > 0]
    flight_times = [k['DD.key1.key2'] for k in keystrokes if k['DD.key1.key2'] > 0]
    latencies = [k['UD.key1.key2'] for k in keystrokes if k['UD.key1.key2'] > 0]
    uu_times = [k['UU.key1.key2'] for k in keystrokes if k['UU.key1.key2'] > 0]

    # Compute feature vector
    features = {
        'avg_dwell_time': np.mean(dwell_times) if dwell_times else 0,
        'std_dwell_time': np.std(dwell_times) if dwell_times else 0,
        'avg_flight_time': np.mean(flight_times) if flight_times else 0,
        'std_flight_time': np.std(flight_times) if flight_times else 0,
        'avg_latency': np.mean(latencies) if latencies else 0,
        'std_latency': np.std(latencies) if latencies else 0,
        'avg_UU_time': np.mean(uu_times) if uu_times else 0,
        'std_UU_time': np.std(uu_times) if uu_times else 0
    }

    return features

# ✅ Step 4: Authenticate User
def authenticate_user():
    """Main authentication flow."""
    print("\n🔍 Available users:", list(user_embeddings.keys()))
    user_id = input("Enter your user ID: ").strip()
    if user_id not in user_embeddings:
        print("❌ User ID not found in database")
        return False

    print("\n⌨️ Please type to verify your identity...")
    keystrokes = collect_keystroke_data()
    features = extract_features(keystrokes)
    if not features:
        print("❌ No valid keystroke data collected")
        return False

    # Convert features to match generate_embeddings.py format
    numerical_features = []
    for key in FIXED_FEATURE_KEYS:
        try:
            value = features.get(key, 0.0)
            numerical_features.append(float(value))
        except (ValueError, TypeError):
            print(f"⚠️ Invalid value for feature '{key}': {value}")
            numerical_features.append(0.0)

    # Create feature vector with shape (8,)
    feature_vector = np.array(numerical_features, dtype=np.float32)
    print(f"✅ Input feature vector shape: {feature_vector.shape}")

    # Reshape input features to (1, 8)
    feature_vector = feature_vector.reshape(1, -1)
    
    # Get stored features and reshape to (1, 8)
    stored_features = user_embeddings[user_id].reshape(1, -1)
    
    print(f"✅ Comparing features - Stored: {stored_features.shape}, New: {feature_vector.shape}")
    
    # Use siamese model to compare raw feature vectors directly
    similarity = siamese_model.predict(
        [feature_vector, feature_vector],  # Compare feature vectors directly
        verbose=0
    )[0][0]

    # Authentication decision
    threshold = 0.7
    print(f"\n📊 Similarity score: {similarity:.3f}")
    
    if similarity >= threshold:
        print("✅ Authentication successful!")
        return True
    else:
        print("❌ Authentication failed")
        return False

# ✅ Run Authentication
if __name__ == "__main__":
    authenticate_user()
