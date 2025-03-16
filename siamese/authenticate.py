import pandas as pd
import os
import time
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from pynput import keyboard
from collections import deque
from data_loader import extract_keystroke_features  # ✅ Use existing function

# ✅ Temp files for storing keystrokes
TEMP_RAW_FILE = "temp_keystrokes.csv"
TEMP_FEATURE_FILE = "authenticate_temp.csv"

# ✅ Load model with safe deserialization
try:
    print("📥 Loading model and user features...")
    siamese_model = load_model(
        "models/siamese_model.keras",
        custom_objects={"l1_distance": lambda x: x},  # ✅ Bypass Lambda deserialization issue
        safe_mode=False  # ✅ Allow unsafe deserialization
    )
    user_features = np.load("user_features.npy", allow_pickle=True).item()  # ✅ Load raw user features
    print(f"✅ Model and user features loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model or user features: {e}")
    exit()

# ✅ Step 1: Collect Keystroke Data (Same as in Data Collection)
def collect_keystroke_data():
    """Collects keystroke timing data from user input, ensuring correct DD time."""
    keystroke_events = deque()
    press_times = {}
    finished = False

    def on_press(key):
        """Record key press timestamp."""
        nonlocal finished
        if key == keyboard.Key.enter:
            finished = True
            return False  # ✅ Stop listener when Enter is pressed

        key_name = key.char if hasattr(key, 'char') else key.name
        press_times[key_name] = time.time()

    def on_release(key):
        """Record key release timestamp."""
        key_name = key.char if hasattr(key, 'char') else key.name
        release_time = time.time()

        if key_name in press_times:
            keystroke_events.append((key_name, press_times[key_name], release_time))
            del press_times[key_name]

    # ✅ Start collecting keystrokes
    print("\n⌨️ Please type to verify your identity (Press Enter when done):")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        while not finished:
            pass

    sorted_keystrokes = sorted(keystroke_events, key=lambda x: x[1])
    
    # ✅ Save raw keystrokes to a temp file
    with open(TEMP_RAW_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["key", "press_time", "release_time"])  # ✅ Write header
        writer.writerows(sorted_keystrokes)

    print(f"✅ Raw keystroke data saved to {TEMP_RAW_FILE}")
    return sorted_keystrokes

# ✅ Step 2: Process Keystroke Data (Fix Negative DD Time)
def process_keystroke_data(keystroke_events):
    """Compute UD, DU, DD, UU features using the correct formula and store in a temp file."""
    keystroke_data = []

    for i in range(len(keystroke_events) - 1):
        key1, press1, release1 = keystroke_events[i]
        key2, press2, release2 = keystroke_events[i + 1]

        du_self = round(release1 - press1, 3)  # ✅ Down-Up (DU) of key1
        dd_time = max(round(press2 - press1, 3), 0.001)  # ✅ Down-Down (DD) (always positive)
        du_time = round(release2 - press1, 3)  # ✅ Down-Up between key1 & key2
        ud_time = round(press2 - release1, 3)  # ✅ Up-Down (UD) between key1 and key2
        uu_time = round(release2 - release1, 3)  # ✅ Up-Up (UU) between key1 and key2

        if any(t > 5 for t in [du_self, dd_time, du_time, ud_time, uu_time]):
            print(f"⚠️ Ignoring extreme delay: {key1} → {key2}")
            continue

        keystroke_data.append([key1, key2, du_self, dd_time, du_time, ud_time, uu_time])

    # ✅ Save processed keystroke features to a temp file
    with open(TEMP_FEATURE_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["key1", "key2", "DU.key1.key1", "DD.key1.key2", "DU.key1.key2", "UD.key1.key2", "UU.key1.key2"])
        writer.writerows(keystroke_data)

    print(f"✅ Processed keystroke features saved to {TEMP_FEATURE_FILE}")
    return keystroke_data

# ✅ Step 3: Authenticate User
def authenticate_user():
    """Main authentication flow."""
    print("\n🔍 Available users:", list(user_features.keys()))
    user_id = input("Enter your user ID: ").strip()

    if user_id not in user_features:
        print("❌ User ID not found in database")
        return False

    keystrokes = collect_keystroke_data()
    processed_data = process_keystroke_data(keystrokes)

    if not processed_data:
        print("❌ No valid keystroke data collected")
        return False

    # ✅ Load processed keystroke features from the temp file
    df_keystrokes = pd.read_csv(TEMP_FEATURE_FILE)
    features = extract_keystroke_features(df_keystrokes)

    # ✅ Convert to numerical vector (shape: (1, 8))
    FIXED_FEATURE_KEYS = ["avg_dwell_time", "std_dwell_time", "avg_flight_time", "std_flight_time", 
                          "avg_latency", "std_latency", "avg_UU_time", "std_UU_time"]

    numerical_features = np.array(
        [float(features.get(key, 0.0)) for key in FIXED_FEATURE_KEYS], dtype=np.float32
    ).reshape(1, -1)

    stored_features = np.array(
        [float(user_features[user_id].get(key, 0.0)) for key in FIXED_FEATURE_KEYS], dtype=np.float32
    ).reshape(1, -1)

    # ✅ Use the Siamese model to compare user's typing with stored features
    similarity = siamese_model.predict([numerical_features, stored_features], verbose=0)[0][0]

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
