import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pynput import keyboard
from data_loader import extract_keystroke_features  # ✅ Use verified function

# ✅ Fixed feature order (must match `generate_features.py`)
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

# ✅ Step 1: Load Model & Feature Vectors
def load_model_and_features():
    """Loads the Siamese model and stored user feature vectors."""
    try:
        print("📥 Loading model and feature vectors...")
        siamese_model = load_model("models/siamese_model.h5", custom_objects={"l1_distance": l1_distance})
        expected_input_shape = siamese_model.input_shape[0][-1]  # ✅ Get correct feature input size
        user_feature_vectors = np.load("user_features.npy", allow_pickle=True).item()  # ✅ Use stored feature vectors
        print(f"✅ Model and feature vectors loaded successfully (Expected Input Shape: {expected_input_shape})")
        return siamese_model, expected_input_shape, user_feature_vectors
    except Exception as e:
        print(f"❌ Error loading model or feature vectors: {e}")
        exit()

# ✅ Step 2: Collect Keystroke Data (With Timing Features)
def collect_keystroke_data():
    """Collects raw keystroke data with timing information."""
    keystrokes = []
    down_times = {}  # ✅ Track key down timestamps
    prev_key = None
    finished = False

    def on_press(key):
        nonlocal prev_key, finished
        if key == keyboard.Key.enter:
            finished = True
            return False  # ✅ Stop listening when Enter is pressed
        
        try:
            key_name = key.char  # ✅ Get character keys
        except AttributeError:
            key_name = str(key).replace('Key.', '')  # ✅ Handle special keys
        
        timestamp = pd.Timestamp.now()  # ✅ Record timestamp

        # Store key press timing
        down_times[key_name] = timestamp

        if prev_key and prev_key in down_times:
            # Compute **Flight Time (DD)** and **Latency (UD)**
            dd_time = (timestamp - down_times[prev_key]).total_seconds()
        else:
            dd_time = 0.0

        prev_key = key_name

        keystrokes.append({
            'key1': prev_key,
            'key2': key_name,
            'DU.key1.key1': 0.0,  # ✅ Placeholder (will be set on release)
            'DD.key1.key2': dd_time,
            'UD.key1.key2': dd_time,  # ✅ UD same as DD initially
            'UU.key1.key2': 0.0  # ✅ Placeholder
        })

    def on_release(key):
        try:
            key_name = key.char
        except AttributeError:
            key_name = str(key).replace('Key.', '')
        
        timestamp = pd.Timestamp.now()

        if key_name in down_times:
            dwell_time = (timestamp - down_times[key_name]).total_seconds()
            for k in keystrokes:
                if k['key1'] == key_name:
                    k['DU.key1.key1'] = dwell_time  # ✅ Set dwell time

        # ✅ Update **UU Time**
        if keystrokes:
            keystrokes[-1]['UU.key1.key2'] = dwell_time if keystrokes else 0.0

    print("\n⌨️ Please type to verify your identity (Press Enter when done):")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        while not finished:
            pass  # ✅ Wait until Enter is pressed

    return keystrokes

# ✅ Step 3: Extract Features (Using `data_loader` Function)
def extract_features_from_keystrokes(keystrokes):
    """Extracts keystroke features using existing feature extraction function."""
    if not keystrokes:
        print("❌ No valid keystroke data collected")
        return None

    df_keystrokes = pd.DataFrame(keystrokes)
    print(f"✅ Converted keystrokes to DataFrame with shape: {df_keystrokes.shape}")

    # ✅ Extract features using **existing** function
    features = extract_keystroke_features(df_keystrokes)

    if not features:
        print("❌ No valid features extracted from user input")
        return None

    # ✅ Convert to numerical vector (shape: (1, 8))
    numerical_features = np.array([float(features.get(key, 0.0)) for key in FIXED_FEATURE_KEYS], dtype=np.float32).reshape(1, -1)

    print(f"✅ Generated feature vector shape: {numerical_features.shape}")
    return numerical_features

# ✅ Step 4: Compare Feature Vectors
def compare_features(new_features, user_id, user_feature_vectors, siamese_model):
    """Compares new feature vector against stored user feature vectors."""
    stored_features = user_feature_vectors.get(user_id)
    if stored_features is None:
        print(f"❌ User {user_id} not found in database.")
        return False

    # ✅ Convert stored features from dictionary to NumPy array
    stored_features = np.array([float(stored_features[key]) for key in FIXED_FEATURE_KEYS], dtype=np.float32).reshape(1, -1)
    new_features = new_features.reshape(1, -1)  # ✅ Reshape to (1, 8)

    print(f"✅ Comparing feature vectors - Stored: {stored_features.shape}, New: {new_features.shape}")

    # ✅ Use Siamese model to compute similarity score
    similarity = siamese_model.predict([stored_features, new_features], verbose=0)[0][0]

    print(f"\n📊 Similarity score: {similarity:.3f}")

    threshold = 0.7  # Adjust based on validation results
    if similarity >= threshold:
        print("✅ Authentication successful!")
        return True
    else:
        print("❌ Authentication failed")
        return False

# ✅ Step 5: Authenticate User
def authenticate_user():
    """Main authentication flow."""
    siamese_model, expected_input_shape, user_feature_vectors = load_model_and_features()

    print("\n🔍 Available users:", list(user_feature_vectors.keys()))
    user_id = input("Enter your user ID: ").strip()

    if user_id not in user_feature_vectors:
        print("❌ User ID not found in database")
        return False

    print("\n⌨️ Please type to verify your identity...")
    keystrokes = collect_keystroke_data()

    # ✅ Extract raw feature vector instead of embedding
    new_features = extract_features_from_keystrokes(keystrokes)
    if new_features is None:
        return False

    return compare_features(new_features, user_id, user_feature_vectors, siamese_model)

# ✅ Run Authentication
if __name__ == "__main__":
    authenticate_user()
