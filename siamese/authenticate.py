import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pynput import keyboard
from data_loader import extract_keystroke_features  # ✅ Use same extraction function

# ✅ Fixed feature order (must match training)
FEATURE_KEYS = [
    "avg_dwell_time", "std_dwell_time",
    "avg_flight_time", "std_flight_time",
    "avg_latency", "std_latency",
    "avg_UU_time", "std_UU_time"
]

@tf.keras.utils.register_keras_serializable()
def l1_distance(vects):
    """Computes L1 (Manhattan distance) between two input vectors."""
    x, y = vects
    return K.abs(x - y)

# ✅ Step 1: Load Model & Stored Features
def load_model_and_features():
    """Loads the trained Siamese model and stored user feature vectors."""
    try:
        print("📥 Loading model and feature vectors...")
        siamese_model = load_model("models/siamese_model.h5", custom_objects={"l1_distance": l1_distance})
        user_feature_vectors = np.load("user_features.npy", allow_pickle=True).item()
        print("✅ Model and user feature vectors loaded successfully!")
        return siamese_model, user_feature_vectors
    except Exception as e:
        print(f"❌ Error loading model or features: {e}")
        exit()

# ✅ Step 2: Collect Keystroke Data (SAME AS collect_data.py)
def collect_keystroke_data():
    """Records user keystrokes for authentication using the same logic as `collect_data.py`."""
    
    keystrokes = []
    down_times = {}  # ✅ Stores key press timestamps
    up_times = {}  # ✅ Stores key release timestamps
    prev_key = None
    finished = False

    def on_press(key):
        nonlocal prev_key, finished

        if key == keyboard.Key.enter:
            finished = True
            return False  # ✅ Stop listener when Enter is pressed

        try:
            key_name = key.char
        except AttributeError:
            key_name = str(key).replace('Key.', '')

        # ✅ Store key press timestamp
        down_times[key_name] = pd.Timestamp.now()

        # ✅ Compute Flight Time (DD) & Latency (UD)
        dd_time = (down_times[key_name] - down_times.get(prev_key, down_times[key_name])).total_seconds() if prev_key else 0.0
        ud_time = (down_times[key_name] - up_times.get(prev_key, down_times[key_name])).total_seconds() if prev_key else 0.0

        keystrokes.append({
            'key1': prev_key if prev_key else key_name,
            'key2': key_name,
            'DU.key1.key1': 0.0,  # Placeholder (updated on release)
            'DD.key1.key2': dd_time,
            'UD.key1.key2': ud_time,
            'UU.key1.key2': 0.0  # Placeholder
        })

        prev_key = key_name  # ✅ Update previous key

    def on_release(key):
        try:
            key_name = key.char
        except AttributeError:
            key_name = str(key).replace('Key.', '')

        up_times[key_name] = pd.Timestamp.now()  # ✅ Store release time

        # ✅ Compute Dwell Time (DU)
        if key_name in down_times:
            dwell_time = (up_times[key_name] - down_times[key_name]).total_seconds()
            for k in keystrokes:
                if k['key1'] == key_name:
                    k['DU.key1.key1'] = dwell_time  # ✅ Update dwell time

        # ✅ Compute Up-Up Time (UU)
        last_up_time = max(up_times.values(), default=up_times[key_name])
        uu_time = (up_times[key_name] - last_up_time).total_seconds() if last_up_time else 0.0

        if keystrokes:
            keystrokes[-1]['UU.key1.key2'] = uu_time

    print("\n⌨️ Type for authentication (Press Enter when done):")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        while not finished:
            pass  # ✅ Wait for Enter key

    return keystrokes

# ✅ Step 3: Extract Features from Keystroke Data
def extract_features_from_keystrokes(keystrokes):
    """Extracts keystroke features using the same feature extraction as training."""
    if not keystrokes:
        print("❌ No valid keystroke data collected")
        return None

    df_keystrokes = pd.DataFrame(keystrokes)
    features = extract_keystroke_features(df_keystrokes)

    if not features:
        print("❌ No valid features extracted")
        return None

    # ✅ Convert to numerical feature vector (same order as training)
    numerical_features = np.array([float(features.get(key, 0.0)) for key in FEATURE_KEYS], dtype=np.float32).reshape(1, -1)

    print(f"✅ Feature vector generated: {numerical_features.shape}")
    return numerical_features

# ✅ Step 4: Compare Features with Stored User Data
def compare_features(new_features, user_id, user_feature_vectors, siamese_model):
    """Compares new features with stored user feature vectors using the trained model."""
    stored_features = user_feature_vectors.get(user_id)
    if stored_features is None:
        print(f"❌ User {user_id} not found in database.")
        return False

    # ✅ Convert stored features to NumPy array (same order as training)
    stored_features = np.array([float(stored_features[key]) for key in FEATURE_KEYS], dtype=np.float32).reshape(1, -1)
    new_features = new_features.reshape(1, -1)

    print(f"✅ Comparing vectors - Stored: {stored_features.shape}, New: {new_features.shape}")

    # ✅ Compute similarity using Siamese model
    similarity = siamese_model.predict([stored_features, new_features], verbose=0)[0][0]

    print(f"\n📊 Similarity Score: {similarity:.3f}")

    threshold = 0.7  # ✅ Adjust threshold based on validation results
    if similarity >= threshold:
        print("✅ Authentication successful!")
        return True
    else:
        print("❌ Authentication failed")
        return False

# ✅ Step 5: Run Authentication Flow
def authenticate_user():
    """Handles full authentication flow."""
    siamese_model, user_feature_vectors = load_model_and_features()

    print("\n🔍 Available users:", list(user_feature_vectors.keys()))
    user_id = input("Enter your user ID: ").strip()

    if user_id not in user_feature_vectors:
        print("❌ User ID not found in database")
        return False

    print("\n⌨️ Type to verify your identity...")
    keystrokes = collect_keystroke_data()

    new_features = extract_features_from_keystrokes(keystrokes)
    if new_features is None:
        return False

    return compare_features(new_features, user_id, user_feature_vectors, siamese_model)

# ✅ Run Authentication
if __name__ == "__main__":
    authenticate_user()
