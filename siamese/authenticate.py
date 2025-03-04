import pandas as pd
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Lambda
from pynput import keyboard
from siamese import create_base_network

# 🔹 Define the Lambda function used in the model
def l1_distance(vectors):
    x, y = vectors
    return tf.abs(x - y)

# 🔹 Load the trained Siamese model with custom objects
print("📥 Loading Siamese model...")
siamese_model = load_model("models/siamese_model.h5", compile=False, custom_objects={"l1_distance": Lambda(l1_distance)})

# 🔹 Print model expected input shape
print("🔍 Siamese model expected input shape:", siamese_model.input_shape)

num_features = 7  # Adjust based on dataset
base_model = create_base_network((num_features,))
print("✅ Siamese model loaded successfully.")

# 🔹 Define dataset path
file_path = "free-text (1).csv"

def extract_features(user_data):
    """Extracts only computable feature vectors from keystroke timestamps."""
    if user_data.empty:
        print("❌ No historical data for this user.")
        return None

    timestamps = user_data["Timestamp"].values

    if len(timestamps) < 2:
        return None  # Not enough keystrokes to compute features

    # Compute flight times (time between consecutive keystrokes)
    flight_times = np.diff(timestamps)

    # Compute meaningful features
    feature_vector = [
        np.mean(flight_times) if len(flight_times) > 0 else 0,
        np.std(flight_times) if len(flight_times) > 0 else 0,
        np.median(flight_times) if len(flight_times) > 0 else 0,
        np.min(flight_times) if len(flight_times) > 0 else 0,
        np.max(flight_times) if len(flight_times) > 0 else 0,
        len(timestamps),  # Total keystrokes recorded
        len(timestamps) / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0  # Typing speed
    ]

    # Convert to NumPy array with correct shape
    features = np.array(feature_vector).reshape(1, -1)

    # Debugging output
    print(f"📊 Extracted Features: {feature_vector}")
    print(f"📊 Extracted Features Shape: {features.shape}")

    return features

def match_feature_sizes(stored_features, new_features):
    """Ensures stored and new features have the same number of samples."""
    stored_len, new_len = len(stored_features), len(new_features)

    print(f"🔄 Matching feature sizes: Stored = {stored_len}, New = {new_len}")

    if stored_len > new_len:
        stored_features = stored_features[:new_len]  # Truncate stored data
    elif new_len > stored_len:
        new_features = new_features[:stored_len]  # Truncate new data

    return stored_features, new_features

def authenticate_user():
    """Runs a single authentication test using the Siamese Network."""
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

    # 🔹 Ensure required columns exist
    if "DU.key1.key2" not in df.columns or "Timestamp" not in df.columns:
        print("❌ Error: Required columns 'DU.key1.key2' or 'Timestamp' are missing.")
        return False

    print("\nStart typing a short sentence (10-15 words) to verify your identity. Press 'Enter' when done.\n")

    keystrokes = []
    start_time = time.time()
    last_timestamp = None

    def on_press(key):
        nonlocal last_timestamp

        if key == keyboard.Key.enter:
            print("\n✅ Typing sample recorded. Verifying your identity...\n")
            return False  # Stop listening

        key_name = key.name.capitalize() if isinstance(key, keyboard.Key) else key.char
        timestamp = time.time()
        abs_time = round(timestamp - start_time, 3)
        latency = round(timestamp - last_timestamp, 3) if last_timestamp else 0.000

        keystrokes.append([key_name, latency, abs_time])
        last_timestamp = timestamp

    # 🔹 Start keystroke listener
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    # 🔹 Convert keystrokes to DataFrame
    new_keystroke_data = pd.DataFrame(keystrokes, columns=["key1", "DU.key1.key2", "Timestamp"])

    # 🔹 Extract stored raw features
    user_data = df[df["participant"] == user_id].sort_values(by="Timestamp")
    stored_features = extract_features(user_data)

    if stored_features is None:
        return False

    # 🔹 Extract raw features from new input
    new_features = extract_features(new_keystroke_data)

    if new_features is None:
        print("❌ Failed to extract features from input.")
        return False

    # 🔹 Debug print
    print(f"📊 Stored Features Shape (raw): {stored_features.shape}")
    print(f"📊 New Features Shape (raw): {new_features.shape}")

    # 🔹 Ensure both feature sets have the same length
    stored_features, new_features = match_feature_sizes(stored_features, new_features)

    # Debugging print before passing to model
    print(f"📊 Final Stored Features Shape: {stored_features.shape}")
    print(f"📊 Final New Features Shape: {new_features.shape}")

    # 🔹 Print actual input values before passing to the model
    print("\n🔍 First 5 stored features:\n", stored_features[:5])
    print("\n🔍 First 5 new features:\n", new_features[:5])

    # 🔹 Use Siamese model to compare stored vs new raw features
    similarity_score = siamese_model.predict([stored_features, new_features])[0][0]

    print(f"\n📊 Similarity Score: {similarity_score}")

# 🔹 Run a single authentication test
authenticate_user()
