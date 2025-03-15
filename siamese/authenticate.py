import pandas as pd
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pynput import keyboard
from data_loader import extract_keystroke_features  # ✅ Use existing function

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

# ✅ Step 1: Load Model & User Features
try:
    print("📥 Loading model and user features...")
    siamese_model = load_model("models/siamese_model.h5", custom_objects={"l1_distance": l1_distance})
    expected_input_shape = siamese_model.input_shape[0][-1]  # ✅ Get expected feature size
    user_features = np.load("user_features.npy", allow_pickle=True).item()  # ✅ Load raw user features
    print(f"✅ Model and user features loaded successfully (Expected Input Shape: {expected_input_shape})")
except Exception as e:
    print(f"❌ Error loading model or user features: {e}")
    exit()

# ✅ Step 2: Collect Keystroke Data
def collect_keystroke_data():
    """Collects keystroke timing data from user input."""
    keystrokes = []
    press_times = {}  # ✅ Store key press timestamps
    release_times = {}  # ✅ Store key release timestamps
    finished = False

    def on_press(key):
        """Store key press timestamp."""
        nonlocal finished
        if key == keyboard.Key.enter:
            finished = True
            return False

        try:
            key_name = key.char
        except AttributeError:
            key_name = str(key).replace('Key.', '')

        press_times[key_name] = time.time()  # ✅ Store press timestamp

    def on_release(key):
        """Store key release timestamp."""
        try:
            key_name = key.char
        except AttributeError:
            key_name = str(key).replace('Key.', '')

        release_times[key_name] = time.time()  # ✅ Store release timestamp

    # ✅ Start collecting keystrokes
    print("\n⌨️ Please type to verify your identity (Press Enter when done):")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        while not finished:
            pass  # ✅ Wait until Enter is pressed

    # ✅ Process keystroke data
    keys = list(press_times.keys())
    for i in range(len(keys) - 1):
        key1, key2 = keys[i], keys[i + 1]
        
        if key1 in press_times and key1 in release_times and key2 in press_times:
            du_time = round(release_times[key1] - press_times[key1], 3)
            dd_time = round(press_times[key2] - press_times[key1], 3)
            ud_time = round(press_times[key2] - release_times[key1], 3)
            uu_time = round(release_times[key2] - release_times[key1], 3) if key2 in release_times else 0

            keystrokes.append({
                'key1': key1,
                'key2': key2,
                'DU.key1.key1': du_time,
                'DD.key1.key2': dd_time,
                'UD.key1.key2': ud_time,
                'UU.key1.key2': uu_time
            })

    return keystrokes

# ✅ Step 3: Extract Features from Keystrokes
def extract_features(keystrokes):
    """Extracts keystroke features using existing function."""
    if not keystrokes:
        print("❌ No valid keystroke data collected")
        return None

    df_keystrokes = pd.DataFrame(keystrokes)
    print(f"✅ Converted keystrokes to DataFrame with shape: {df_keystrokes.shape}")

    # ✅ Extract features using **existing function**
    features = extract_keystroke_features(df_keystrokes)
    print(f"🔍 Debug: Extracted features -> Type: {type(features)}, Value: {features}")

    # ✅ Convert to numerical vector (shape: (1, 8))
    numerical_features = np.array([float(features.get(key, 0.0)) for key in FIXED_FEATURE_KEYS], dtype=np.float32).reshape(1, -1)

    print(f"✅ Generated feature vector shape: {numerical_features.shape}")
    return numerical_features

# ✅ Step 4: Authenticate User
def authenticate_user():
    """Main authentication flow."""
    print("\n🔍 Available users:", list(user_features.keys()))
    user_id = input("Enter your user ID: ").strip()
    if user_id not in user_features:
        print("❌ User ID not found in database")
        return False

    print("\n⌨️ Please type to verify your identity...")
    keystrokes = collect_keystroke_data()
    features = extract_features(keystrokes)

    print(f"🔍 Debug: Extracted features -> Type: {type(features)}, Value: {features}")

    # ✅ Convert features to match format in `user_features.npy`
    numerical_features = []
    for key in FIXED_FEATURE_KEYS:
        try:
            value = features.get(key, 0.0)
            numerical_features.append(float(value))
        except (ValueError, TypeError):
            print(f"⚠️ Invalid value for feature '{key}': {value}")
            numerical_features.append(0.0)

    # ✅ Create feature vector with shape (8,)
    feature_vector = np.array(numerical_features, dtype=np.float32).reshape(1, -1)
    
    # ✅ Get stored features and reshape to (1, 8)
    stored_features = np.array([float(user_features[user_id].get(key, 0.0)) for key in FIXED_FEATURE_KEYS], dtype=np.float32).reshape(1, -1)
    
    print(f"✅ Comparing features - Stored: {stored_features.shape}, New: {feature_vector.shape}")
    
    # ✅ Use siamese model to compare user’s feature vector
    similarity = siamese_model.predict([feature_vector, stored_features], verbose=0)[0][0]

    # ✅ Authentication decision
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
