import time  # Used for measuring keystroke timings
import os  # Used for checking if the file exists
import numpy as np  # Used for numerical operations
import pandas as pd  # Used for handling data in DataFrames
import tensorflow as tf  # Core TensorFlow library
from tensorflow.keras.models import load_model  # Used for loading the trained model
from data_loader import extract_features_from_csv # Extract features from
import tensorflow.keras.backend as K  # Keras backend for custom functions
from pynput import keyboard  # Used for capturing keystrokes


# 🔹 Step 1: Define `l1_distance` Again (Required for Loading)
@tf.keras.utils.register_keras_serializable()
def l1_distance(vects):
    """Computes L1 (Manhattan) distance between two input vectors."""
    x, y = vects
    return K.abs(x - y)

# 🔹 Step 2: Load the Trained Siamese Model with Custom Object
siamese_model = load_model("models/siamese_model.h5",
                           custom_objects={"l1_distance": l1_distance})

print("✅ Siamese model loaded successfully.")

file_path = os.path.join(os.path.dirname(__file__), "FreeDB.csv")

def record_keystrokes():
    """
    Records user keystrokes for authentication, capturing:
    - key1 (Current Key)
    - key2 (Previous Key)
    - DU (Dwell Time)
    - DD (Flight Time)
    - UD (Latency)
    - UU (Release Time)
    
    Stops after **400 characters** or when 'Esc' is pressed.
    """
    print("\n🖊️ Start typing a short phrase. **Press 'Esc' or reach 400 characters to stop.**\n")

    keystrokes = []
    start_time = time.time()
    prev_key, prev_down_time, prev_up_time = None, None, None
    char_count = 0  # Track number of characters typed

    def on_press(key):
        """Captures key press events and calculates timings."""
        nonlocal prev_key, prev_down_time, char_count

        if key == keyboard.Key.esc:  # Stop when 'Esc' is pressed
            print("\n✅ Typing session complete.\n")
            return False  # Stop listener

        # Convert key into a readable format
        try:
            key_name = key.char if hasattr(key, 'char') else key.name.capitalize()
        except AttributeError:
            return  # Ignore unknown keys

        timestamp = time.time()

        # Compute timing features
        du_time = round(timestamp - prev_down_time, 3) if prev_down_time else 0  # Dwell Time
        dd_time = round(timestamp - prev_up_time, 3) if prev_up_time else 0  # Flight Time
        ud_time = round(timestamp - prev_down_time, 3) if prev_down_time else 0  # Latency
        uu_time = round(timestamp - prev_up_time, 3) if prev_up_time else 0  # Release Timing

        key2 = prev_key if prev_key else key_name  # Previous key as key1
        key1 = key_name  # Current key as key2

        # Append keystroke data
        keystrokes.append(["user_input", "1", key1, key2, du_time, dd_time, ud_time, uu_time])

        # Display character count
        char_count += 1
        print(f"\r🔢 Characters typed: {char_count}/400", end="", flush=True)

        # Update previous key states
        prev_key = key_name
        prev_down_time = timestamp

        # Stop recording if 400 characters are reached
        if char_count >= 200:
            print("\n✅ Character limit reached (400). Processing authentication...\n")
            return False  # Stop listener

    def on_release(key):
        """Records key release timestamps."""
        nonlocal prev_up_time
        prev_up_time = time.time()

    # Start keystroke listener
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    # Convert recorded keystrokes to a DataFrame
    return pd.DataFrame(keystrokes, columns=["participant", "session", "key1", "key2", 
                                             "DU.key1.key1", "DD.key1.key2", "UD.key1.key2", "UU.key1.key2"])

def authenticate_user():
    """
    Prompts the user to enter their ID, then records their typing pattern.
    Stores the keystroke data in a list for later authentication.
    """
    # 🔹 Ask for User ID
    user_id = input("\n👤 Enter your user ID (e.g., p101, p102): ").strip()

    # 🔹 Verify User ID
    if not user_id:
        print("❌ Error: User ID cannot be empty.")
        return None, None

    print(f"\n✅ User ID '{user_id}' recognized. Please start typing for authentication...\n")

    # 🔹 Record Keystroke Data
    user_keystroke_data = record_keystrokes()  # Calls the function you created earlier

    # 🔹 Store data temporarily (not saved in the database)
    recorded_data = user_keystroke_data.values.tolist()

    print("\n✅ Keystroke data captured successfully!\n")

    return user_id, recorded_data  # Returns both user ID and recorded keystrokes as a list
def authenticate_with_model(user_id, recorded_data):
    """
    Compares recorded keystroke data against stored user data using the trained Siamese model.
    Returns an accuracy score for authentication.
    """

    # 🔹 Load stored dataset
    if not os.path.exists(file_path):
        print("❌ Error: Data file not found.")
        return False

    df = pd.read_csv(file_path, low_memory=False)
    df.columns = df.columns.str.strip()  # Remove extra spaces in column names

    # 🔹 Check if user exists in stored dataset
    if user_id not in df["participant"].values:
        print(f"❌ Authentication failed. User {user_id} not found in dataset.")
        return False

    # 🔹 Extract features from stored data
    extracted_features = extract_features_from_csv(df)

    session_key = f"{user_id}_session1"
    if session_key not in extracted_features:
        print("❌ Failed to extract stored features. Cannot authenticate.")
        return False

    stored_features = np.array(list(extracted_features[session_key].values()), dtype=np.float32).reshape(1, -1)

    # 🔹 Convert recorded keystroke list to DataFrame
    user_keystroke_df = pd.DataFrame(recorded_data, columns=["participant", "session", "key1", "key2",
                                                             "DU.key1.key1", "DD.key1.key2", "UD.key1.key2", "UU.key1.key2"])

    # 🔹 Extract features from new input
    new_features_dict = extract_features_from_csv(user_keystroke_df)

    if "user_input_session1" not in new_features_dict:
        print("❌ Failed to extract features from input keystrokes.")
        return False

    new_features = np.array(list(new_features_dict["user_input_session1"].values()), dtype=np.float32).reshape(1, -1)

    # 🔹 Handle NaN values (Replace NaNs with 0)
    stored_features = np.nan_to_num(stored_features, nan=0)
    new_features = np.nan_to_num(new_features, nan=0)

    # 🔹 Ensure feature shapes match
    if stored_features.shape[1] != new_features.shape[1]:
        print(f"❌ Feature shape mismatch! Stored: {stored_features.shape}, Input: {new_features.shape}")
        return False

    # 🔹 Compare stored vs. new keystroke data using the trained model
    similarity_score = siamese_model.predict([stored_features, new_features])[0][0]

    # 🔹 Print authentication result
    print(f"\n📊 Similarity Score: {similarity_score:.4f}")

    # 🔹 Define authentication threshold
    threshold = 0.7  # Adjust this based on testing

    if similarity_score >= threshold:
        print(f"\n✅ Authentication successful! Welcome back, {user_id}.")
        return True
    else:
        print("\n❌ Authentication failed. Typing pattern does not match.")
        return False

# 🔹 Run authentication
user_id, recorded_data = authenticate_user()
if user_id and recorded_data:
    authenticate_with_model(user_id, recorded_data)
