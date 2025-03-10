import time
import csv
from collections import deque
from pynput import keyboard  # Keystroke listener

# File paths
raw_file = "KeystrokeData.csv"
processed_file = "ProcessedKeystrokeData.csv"

# Store raw keystroke timestamps
keystroke_events = deque()  # Stores (key, press_time, release_time)
press_times = {}  # Tracks latest press time

def on_press(key):
    """Record key press time."""
    key_name = key.char if hasattr(key, 'char') else key.name
    timestamp = time.time()
    press_times[key_name] = timestamp  # Store latest press time

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
    with open(raw_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["key", "press_time", "release_time"])  # Header
        writer.writerows(keystroke_events)
    print(f"✅ Raw keystroke data saved to {raw_file}")

def process_keystroke_data():
    """Compute UD, DU, DD, UU features using the **correct** formula."""
    keystroke_data = []

    print("\n🔍 **Processing Keystroke Features** 🔍")
    print("────────────────────────────────────────────────────────")
    print("KEY1 → KEY2  | DU.key1.key2 | UD.key1.key2 | DD.key1.key2 | UU.key1.key2")

    for i in range(len(keystroke_events) - 1):
        key1, press1, release1 = keystroke_events[i]
        key2, press2, release2 = keystroke_events[i + 1]

        # Apply **corrected** keystroke feature computation
        du_time = round(release1 - press1, 3)  # Down-Up (DU) key1 → key2
        ud_time = round(press2 - release1, 3)  # Up-Down (UD) key1 → key2
        dd_time = round(press2 - press1, 3)  # Down-Down (DD) key1 → key2
        uu_time = round(release2 - release1, 3)  # Up-Up (UU) key1 → key2

        # 🚨 Ignore extreme values (>10s)
        if any(t > 10 for t in [du_time, ud_time, dd_time, uu_time]):
            print(f"⚠️ Ignoring extreme delay: {key1} → {key2} (DU = {du_time}s)")
            continue

        # Save the computed values
        keystroke_data.append([key1, key2, du_time, ud_time, dd_time, uu_time])

        # Print debug info
        print(f"{key1} → {key2}  | {du_time}  | {ud_time}  | {dd_time}  | {uu_time}")

    save_to_csv(keystroke_data)

def save_to_csv(data):
    """Save processed keystroke features to CSV."""
    with open(processed_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["key1", "key2", "DU.key1.key2", "UD.key1.key2", "DD.key1.key2", "UU.key1.key2"])
        writer.writerows(data)

    print(f"✅ Processed keystroke features saved to {processed_file}")

# 🔹 Run keystroke recording
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
