import csv
import os
import time
from pynput import keyboard  # To listen for keystrokes

# Define the dataset file path
file_path = "FreeDB.csv"

def record_keystrokes():
    """
    Records individual keystrokes with timestamps, capturing:
    - `key1`: Previous key
    - `key2`: Current key
    - `UU`: Time between releasing previous key and releasing current key
    - `UD`: Time between releasing previous key and pressing current key
    - `DU`: Time between pressing and releasing the same key (Dwell Time)
    - `DD`: Time between pressing previous key and pressing current key
    - `Timestamp`: Absolute time of key press
    
    Stops after 2500 key presses.
    """

    # 🔹 Ask user for ID and session
    user_id = input("Enter your user ID (e.g., p101, p102): ").strip()
    session_number = input("Enter session number (1 or 2): ").strip()

    print("\n🖊️ Start typing... Press 'Esc' to stop.\n")

    keystrokes = []  # Stores all recorded keystrokes
    start_time = time.time()  # Reference start time
    prev_key = None
    prev_down_time = None  # Last key press timestamp
    prev_up_time = None  # Last key release timestamp
    last_release_time = None  # Last release time for UU calculation
    key_count = 0  # Counter to track 50 keystrokes

    def on_press(key):
        """Handles key press events and computes DD, UD timings."""
        nonlocal prev_key, prev_down_time, prev_up_time, last_release_time, key_count

        if key == keyboard.Key.esc:
            print("\n✅ Typing session complete. Saving data...\n")
            if keystrokes:
                keystrokes[-1][3] = ""  # Ensure last key2 is empty
            save_to_csv(keystrokes)
            return False  # Stop listener

        # Convert key into a readable format
        key_name = key.char if hasattr(key, 'char') else key.name.capitalize()

        timestamp = time.time()  # Current key press time
        abs_time = round(timestamp - start_time, 3)  # Absolute timing

        # Compute UU (Up-Up) - Time between releasing previous key and releasing current key
        uu_time = round(timestamp - last_release_time, 3) if last_release_time else 0

        # Compute UD (Up-Down) - Time between releasing previous key and pressing current key
        ud_time = round(timestamp - prev_up_time, 3) if prev_up_time else 0

        # Compute DD (Down-Down) - Time between pressing previous key and pressing current key
        dd_time = round(timestamp - prev_down_time, 3) if prev_down_time else 0

        # Assign previous key and current key
        key2 = key_name
        key1 = prev_key if prev_key else key_name

        # Append the keystroke data (DU will be updated in `on_release()`)
        keystrokes.append([user_id, session_number, key1, key2, uu_time, ud_time, 0, dd_time, abs_time])

        # Update previous key states
        prev_key = key_name
        prev_down_time = timestamp
        key_count += 1

        if key_count >= 2500:
            print("\n✅ Keystroke limit reached (2500 keys). Saving data...\n")
            if keystrokes:
                keystrokes[-1][3] = ""  # Ensure last key2 is empty
            save_to_csv(keystrokes)
            return False  # Stop listener

    def on_release(key):
        """Handles key release events and computes DU (Press to Release) timing."""
        nonlocal prev_up_time, last_release_time

        timestamp = time.time()  # Current key release time

        # Compute DU (Down-Up) - Time between pressing and releasing the same key (Dwell Time)
        if keystrokes:
            keystrokes[-1][6] = round(timestamp - prev_down_time, 3) if prev_down_time else 0

        # Update previous release times
        prev_up_time = timestamp
        last_release_time = timestamp

    # Start listening for keystrokes
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

def save_to_csv(data):
    """
    Saves the recorded keystroke session into the CSV file.
    Ensures proper formatting and new line insertion.
    """

    file_exists = os.path.exists(file_path)

    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")  # Ensure proper newline formatting

        # Write header if file is new
        if not file_exists:
            writer.writerow(["participant", "session", "key1", "key2", "UU.key1.key2", 
                             "UD.key1.key1", "DU.key1.key1", "DD.key1.key2", "Timestamp"])

        # Write keystroke data row by row
        for row in data:
            writer.writerow(row)  # Explicitly writing a new line for each row

    print(f"✅ Session data saved to {file_path}")

# 🔹 Run keystroke recording
record_keystrokes()
