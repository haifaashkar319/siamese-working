import pandas as pd
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pynput import keyboard
from collections import deque
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
    siamese_model = load_model("models/siamese_model.keras", custom_objects={"l1_distance": l1_distance})
    expected_input_shape = siamese_model.input_shape[0][-1]  # ✅ Get expected feature size
    user_features = np.load("user_features.npy", allow_pickle=True).item()  # ✅ Load raw user features
    print(f"✅ Model and user features loaded successfully (Expected Input Shape: {expected_input_shape})")
except Exception as e:
    print(f"❌ Error loading model or user features: {e}")
    exit()

# ✅ Step 2: Collect Keystroke Data (Fixed)
def collect_keystroke_data():
    """Collects keystroke timing data from user input."""
    keystroke_events = deque()  # ✅ Stores (key, press_time, release_time)
    press_times = {}  # ✅ Stores latest key press timestamps
    finished = False

    def on_press(key):
        """Store key press timestamp."""
        nonlocal finished
        if key == keyboard.Key.enter:
            finished = True
            return False  # ✅ Stop listener when Enter is pressed

        try:
            key_name = key.char
        except AttributeError:
            key_name = str(key).replace('Key.', '')

        press_times[key_name] = time.time()  # ✅ Store press timestamp

    def on_release(key):
        """Store key release timestamp and save keystroke event."""
        try:
            key_name = key.char
        except AttributeError:
            key_name = str(key).replace('Key.', '')

        release_time = time.time()
        if key_name in press_times:
            keystroke_events.append((key_name, press_times[key_name], release_time))  # ✅ Save event
            del press_times[key_name]  # ✅ Remove used press time

    # ✅ Start collecting keystrokes
    print("\n⌨️ Please type to verify your identity (Press Enter when done):")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        while not finished:
            pass  # ✅ Wait until Enter is pressed

    # ✅ Ensure keystrokes are sorted by press time
    sorted_keystrokes = sorted(keystroke_events, key=lambda x: x[1])

    # 🔍 Debugging: Print all collected key timestamps
    # print(f"🔍 Sorted Keystroke Events: {keystroke_events}")

    keystrokes = []
    for i in range(len(keystroke_events) - 1):
        key1, press1, release1 = keystroke_events[i]
        key2, press2, release2 = keystroke_events[i + 1]

        # ✅ Compute keystroke timing features
        du_self = round(release1 - press1, 3)  # ✅ Down-Up (DU) of key1
        dd_time = round(press2 - press1, 3)  # ✅ Down-Down (DD) between key1 and key2
        du_time = round(release2 - press1, 3)  # ✅ Down-Up between key1 & key2
        ud_time = round(press2 - release1, 3)  # ✅ Up-Down (UD) between key1 and key2
        uu_time = round(release2 - release1, 3)  # ✅ Up-Up (UU) between key1 and key2

        # 🚨 Ignore extreme or negative delays
        if any(t > 5 or t < -5 for t in [du_self, dd_time, du_time, ud_time, uu_time]):
            print(f"⚠️ Ignoring extreme/negative delay: {key1} → {key2}")
            continue

        # ✅ Save computed keystroke values
        keystrokes.append({
            "key1": key1, "key2": key2,
            "DU.key1.key1": du_self,
            "DD.key1.key2": dd_time,
            "DU.key1.key2": du_time,
            "UD.key1.key2": ud_time,
            "UU.key1.key2": uu_time
        })

    # 🔍 Debugging: Print final collected keystroke data
    print(f"✅ Collected Keystroke Data: {keystrokes}")
    return keystrokes

# ✅ Step 3: Extract Features from Keystrokes
def extract_features(keystrokes):
    """Extracts keystroke features using existing function."""
    if not keystrokes:
        print("❌ No valid keystroke data collected")
        return None

    df_keystrokes = pd.DataFrame(keystrokes)
    features = extract_keystroke_features(df_keystrokes)

    # ✅ Convert to numerical vector (shape: (1, 8))
    numerical_features = np.array([float(features.get(key, 0.0)) for key in FIXED_FEATURE_KEYS], dtype=np.float32).reshape(1, -1)
    return numerical_features

# ✅ Step 4: Authenticate User
def authenticate_user():
    """Main authentication flow."""
    print("\n🔍 Available users:", list(user_features.keys()))
    user_id = input("Enter your user ID: ").strip()

    if user_id not in user_features:
        print("❌ User ID not found in database")
        return False

    keystrokes = collect_keystroke_data()
    features = extract_features(keystrokes)

    if not isinstance(features, np.ndarray):
        print(f"❌ Unexpected features format: {type(features)}")
        return False  # Exit if features are invalid

    # ✅ Reshape features to match expected input shape (1, 8)
    feature_vector = features.reshape(1, -1)
    
    # ✅ Get stored features and reshape to (1, 8)
    stored_features = np.array(
        [float(user_features[user_id].get(key, 0.0)) for key in FIXED_FEATURE_KEYS], 
        dtype=np.float32
    ).reshape(1, -1)

    # ✅ Use Siamese model to compare user’s feature vector
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
