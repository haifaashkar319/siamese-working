import pandas as pd
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from pynput import keyboard
from data_loader import extract_features_from_csv  # ✅ Keep only valid imports

# 🔹 Load the trained Siamese model
print("📥 Loading Siamese model...")
siamese_model = load_model("models/siamese_model.h5", compile=False)
print("✅ Siamese model loaded successfully.")

# 🔹 Define dataset path
file_path = "FreeDB.csv"

def record_keystrokes():
    """Records user keystrokes for authentication."""
    print("\nStart typing... Press 'Enter' when done.\n")

    keystrokes = []
    start_time = time.time()
    prev_key = None
    prev_down_time = None
    prev_up_time = None
    key_count = 0  

    def on_press(key):
        """Captures key press events."""
        nonlocal prev_key, prev_down_time, key_count

        if key == keyboard.Key.enter:
            print("\n✅ Typing sample recorded. Processing authentication...\n")
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
        keystrokes.append(["user_input", "1", key1, key2, du_time, dd_time, ud_time, uu_time])

        # Display character count
        key_count += 1
        print(f"\r🔢 Characters typed: {key_count}", end="", flush=True)

        # Update previous key states
        prev_key = key_name
        prev_down_time = timestamp

    def on_release(key):
        """Records key release timestamps."""
        nonlocal prev_up_time
        prev_up_time = time.time()

    # Start keystroke listener
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    return pd.DataFrame(keystrokes, columns=["participant", "session", "key1", "key2", 
                                             "DU.key1.key1", "DD.key1.key2", "UD.key1.key2", "UU.key1.key2"])

def authenticate_user():
    """Runs authentication using the existing feature extraction from data_loader.py."""
    user_id = input("Enter your user ID (e.g., p101, p102): ").strip()

    if not os.path.exists(file_path):
        print("❌ Error: Data file not found.")
        return False

    # 🔹 Load dataset
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    # 🔹 Verify user exists
    if user_id not in df["participant"].values:
        print(f"❌ Authentication failed. User {user_id} not found.")
        return False

    # 🔹 Record user input keystrokes
    new_keystroke_data = record_keystrokes()

    # 🔹 Extract stored features using `data_loader.py`
    extracted_features = extract_features_from_csv(df)

    if user_id + "_session1" not in extracted_features:
        print("❌ Failed to extract stored features. Cannot authenticate.")
        return False

    stored_features = np.array(list(extracted_features[user_id + "_session1"].values()), dtype=np.float32).reshape(1, -1)

    # 🔹 Extract new input features using `data_loader.py`
    new_features_dict = extract_features_from_csv(new_keystroke_data)

    if "user_input_session1" not in new_features_dict:
        print("❌ Failed to extract features from input.")
        return False

    new_features = np.array(list(new_features_dict["user_input_session1"].values()), dtype=np.float32).reshape(1, -1)

    # 🔹 Handle NaN Issues: Replace all NaNs with 0
    stored_features = np.nan_to_num(stored_features, nan=0)
    new_features = np.nan_to_num(new_features, nan=0)

    # 🔹 Debug: Print Feature Shapes
    print(f"📊 Stored Features Shape: {stored_features.shape}")
    print(f"📊 New Features Shape: {new_features.shape}")

    # 🔹 Use Siamese model to compare stored vs new features
    similarity_score = siamese_model.predict([stored_features, new_features])[0][0]

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
