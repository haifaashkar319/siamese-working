import pandas as pd
import time
import csv
import os

# Define the dataset file
file_path = "free-text (1).csv"

def record_keystrokes():
    """
    Records keystroke data for a new user session.
    The user will type about 200 words, and timestamps will be recorded.
    """
    user_id = input("Enter your user ID (e.g., p101, p102, etc.): ").strip()
    session_number = input("Enter session number (1 or 2): ").strip()
    
    print("\nStart typing... (type about 200 words). Press 'Enter' when done.\n")
    
    # Initialize keystroke data storage
    keystrokes = []
    start_time = time.time()  # Start timestamp
    
    try:
        while True:
            key = input("")  # Capture input (simulate real typing session)
            if key.lower() == "exit":  # Stop when user types 'exit'
                break
            timestamp = time.time() - start_time  # Calculate relative time
            keystrokes.append([user_id, session_number, key, timestamp])
            
            # Stop recording after approx 200 words
            if len(keystrokes) >= 200:
                print("\n✅ Typing session complete. Saving data...\n")
                break
                
    except KeyboardInterrupt:
        print("\nSession interrupted. Saving recorded data...\n")

    # Save to CSV file
    save_to_csv(keystrokes)

def save_to_csv(data):
    """
    Appends the new keystroke session to the existing dataset.
    """
    file_exists = os.path.exists(file_path)
    
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        
        # Write header if file is new
        if not file_exists:
            writer.writerow(["participant", "session", "key1", "Timestamp"])
        
        # Write new keystroke data
        writer.writerows(data)

    print(f"✅ Session data saved to {file_path}")

# Run the function to record new user data
record_keystrokes()
