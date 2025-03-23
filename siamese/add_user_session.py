import time
import csv
from collections import deque
from pynput import keyboard  # Keystroke listener

# Prompt for participant ID and session
participant_id = input("Enter Participant ID: ")
session_number = input("Enter Session Number: ")

# File paths
raw_file = "FreeDB_Raw.csv"
processed_file = "FreeDB2.csv"

# Store raw keystroke timestamps
keystroke_events = deque()  # Stores (key, press_time, release_time)
press_times = {}  # Tracks latest press time
press_count = 0  # Global counter for key presses

def on_press(key):
    """Record key press time and debug the press count."""
    global press_count
    key_name = key.char if hasattr(key, 'char') else key.name
    timestamp = time.time()
    press_times[key_name] = timestamp  # Store latest press time

    # Increment and display the key press count
    press_count += 1
    print(f"DEBUG: Key pressed: {key_name}, Total keys pressed: {press_count}")

def on_release(key):
    """Record key release time and store full event."""
    key_name = key.char if hasattr(key, 'char') else key.name
    timestamp = time.time()

    if key_name in press_times:
        keystroke_events.append((key_name, press_times[key_name], timestamp))  # Save press & release
        del press_times[key_name]  # Remove used press time

    if key == keyboard.Key.esc:
        print("\n✅ Typing session complete. Saving raw data...")
        save_raw_keystrokes()
        print("\n✅ Processing keystroke features...")
        process_keystroke_data()
        return False  # Stop listener

def save_raw_keystrokes():
    """Save raw keystrokes to CSV."""
    with open(raw_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        
        # Write header if new file
        if file.tell() == 0:
            writer.writerow(["participant", "session", "key", "press_time", "release_time"])
        
        for key, press_time, release_time in keystroke_events:
            writer.writerow([participant_id, session_number, key, press_time, release_time])

    print(f"✅ Raw keystroke data saved to {raw_file}")

def process_keystroke_data():
    """Compute UD, DU, DD, UU features using the correct formula."""
    keystroke_data = []

    for i in range(len(keystroke_events) - 1):
        key1, press1, release1 = keystroke_events[i]
        key2, press2, release2 = keystroke_events[i + 1]

        # Compute keystroke timing features
        du_self = round(release1 - press1, 3)  # Down-Up (DU) of key1
        dd_time = round(press2 - press1, 3)  # Down-Down (DD) between key1 and key2
        du_time = round(release2 - press1, 3)  # Down-Up between key1 & key2
        ud_time = round(press2 - release1, 3)  # Up-Down (UD) between key1 and key2
        uu_time = round(release2 - release1, 3)  # Up-Up (UU) between key1 and key2

        # 🚨 Ignore extreme values (>5s)
        if any(t > 5 for t in [du_self, dd_time, du_time, ud_time, uu_time]):
            print(f"⚠️ Ignoring extreme delay: {key1} → {key2}")
            continue

        # Save the computed values
        keystroke_data.append([participant_id, session_number, key1, key2, du_self, dd_time, du_time, ud_time, uu_time])

    save_to_csv(keystroke_data)

def save_to_csv(data):
    """Save processed keystroke features to CSV."""
    with open(processed_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        
        # Write header if new file
        if file.tell() == 0:
            writer.writerow(["participant", "session", "key1", "key2", "DU.key1.key1", "DD.key1.key2", "DU.key1.key2", "UD.key1.key2", "UU.key1.key2"])
        
        writer.writerows(data)

    print(f"✅ Processed keystroke features saved to {processed_file}")

# 🔹 Run keystroke recording
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
