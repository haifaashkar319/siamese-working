import csv
import os
import time
from pynput import keyboard  # To listen for keystrokes

# Define the dataset file
file_path = "FreeDB.csv"

def record_keystrokes():
    """
    Records individual keystrokes with timestamps, capturing `key1`, `key2`, `UU`, `UD`, `DU`, `DD`.
    Stops after 3000 key presses.
    """
    user_id = input("Enter your user ID (e.g., p101, p102): ").strip()
    session_number = input("Enter session number (1 or 2): ").strip()
    
    print("\nStart typing... Press 'Esc' to stop.\n")

    keystrokes = []
    start_time = time.time()  # Start absolute timestamp
    prev_key = None
    prev_up_time = None
    prev_down_time = None
    key_count = 0  # Stop at 1000 key presses

    def on_press(key):
        nonlocal prev_key, prev_down_time, key_count

        if key == keyboard.Key.esc:  # Stop when 'Esc' is pressed
            print("\n✅ Typing session complete. Saving data...\n")
            if keystrokes:
                keystrokes[-1][3] = ""  # Ensure last key2 is empty
            save_to_csv(keystrokes)
            return False

        # Convert key into a readable format
        if isinstance(key, keyboard.Key):  # Special keys
            key_name = key.name.capitalize()  # Example: "Space", "Enter"
        else:
            key_name = key.char  # Regular character keys

        timestamp = time.time()
        abs_time = round(timestamp - start_time, 3)

        # Compute inter-key timing values
        du_time = round(timestamp - prev_down_time, 3) if prev_down_time else 0  # DU.key1.key1
        dd_time = round(timestamp - prev_up_time, 3) if prev_up_time else 0  # DD.key1.key2

        # Save the previous key as `key1`, ensure it's never empty
        key2 = key_name
        key1 = prev_key if prev_key else key_name

        # Append the keystroke data
        keystrokes.append([user_id, session_number, key1, key2, du_time, dd_time, abs_time])

        # Update previous key states
        prev_key = key_name
        prev_down_time = timestamp
        key_count += 1

        if key_count >= 2000:
            print("\n✅ Keystroke limit reached (3000 keys). Saving data...\n")
            if keystrokes:
                keystrokes[-1][3] = ""  # Ensure last key2 is empty
            save_to_csv(keystrokes)
            return False  # Stop listener

    def on_release(key):
        nonlocal prev_up_time
        prev_up_time = time.time()

    # Listen for keystrokes
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

def save_to_csv(data):
    """
    Appends the new keystroke session to the existing dataset.
    Fixes the issue where new participant rows were merging into the previous row.
    """
    file_exists = os.path.exists(file_path)

    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")  # Ensure proper new lines
        
        # Write header if file is new
        if not file_exists:
            writer.writerow(["participant", "session", "key1", "key2", "DU.key1.key1", "DD.key1.key2", "Timestamp"])

        # Write keystroke data row by row
        for row in data:
            writer.writerow(row)  # Explicitly writing a new line for each row

    print(f"✅ Session data saved to {file_path}")

# Run the function to record new user data
record_keystrokes()
