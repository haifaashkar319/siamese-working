import pandas as pd
import time
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from pynput import keyboard
from collections import deque

# Import functions from your modules
from add_user_session import collect_keystroke_data, process_keystroke_data
from clean_database import load_raw_data, validate_raw_data, clean_and_overwrite_data
from data_loader import extract_keystroke_features

# Temporary file names
TEMP_RAW_FILE = "temp_keystrokes.csv"          # To store raw keystroke events
TEMP_FEATURE_FILE = "authenticate_temp.csv"    # To store processed (and later cleaned) features

# Define similarity-to-percentage conversion (using a base of 0.01; adjust if needed)
def similarity_to_percentage(similarity, base=0.01):
    percentage = 100 * (1 - np.power(base, similarity))
    return percentage

# --- Block 1: Data Collection ---
def data_collection():
    print("\n--- Authentication: Data Collection ---")
    # Collect keystroke events using the imported function
    events = collect_keystroke_data()
    
    # Save the collected raw events to TEMP_RAW_FILE for debugging/tracking
    with open(TEMP_RAW_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["key", "press_time", "release_time"])
        writer.writerows(events)
    print(f" Raw keystroke data saved to {TEMP_RAW_FILE}")
    
    print("\nCollected keystroke events:")
    for event in events:
        print(event)
    return events

# --- Block 2: Process Keystroke Data ---
def data_processing(events):
    print("\n--- Authentication: Processing Keystroke Data ---")
    # Process raw events into computed timing features (DU, DD, UD, UU)
    processed_data = process_keystroke_data(events)
    
    # Save processed data to TEMP_FEATURE_FILE
    with open(TEMP_FEATURE_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write header (assumed same as in process function)
        writer.writerow(["participant", "session", "key1", "key2", "DU.key1.key1", "DD.key1.key2", "DU.key1.key2", "UD.key1.key2", "UU.key1.key2"])
        writer.writerows(processed_data)
    print(f" Processed keystroke features saved to {TEMP_FEATURE_FILE}")
    return processed_data

# --- Block 3: Data Cleaning ---
def clean_authentication_data(file_path=TEMP_FEATURE_FILE):
    """
    Load the processed authentication data, validate it, and clean it using the cleaning functions.
    The cleaned data is saved back to the same file.
    """
    print("\n--- Cleaning Authentication Data ---")
    df = load_raw_data(file_path)
    df = validate_raw_data(df)
    df_cleaned = clean_and_overwrite_data(df, file_path)
    return df_cleaned

# --- Block 4: Feature Extraction ---
def feature_extraction():
    """
    Load the cleaned authentication data from TEMP_FEATURE_FILE,
    then extract a feature vector using the data_loader function.
    """
    print("\n--- Feature Extraction ---")
    df_cleaned = pd.read_csv(TEMP_FEATURE_FILE)
    print("\n🔍 DEBUG: Cleaned DataFrame (first 5 rows):")
    print(df_cleaned.head())
    
    # Extract a feature dictionary using your data_loader function
    features = extract_keystroke_features(df_cleaned)
    print("\n🔍 DEBUG: Extracted Feature Dictionary:")
    print(features)
    return features

# --- Main Execution Flow ---
if __name__ == "__main__":
    # Step 1: Data Collection
    events = data_collection()
    
    # Step 2: Process the collected raw keystroke data
    data_processing(events)
    
    # Step 3: Clean the processed data (this overwrites TEMP_FEATURE_FILE with cleaned data)
    cleaned_df = clean_authentication_data()
    print("\n🔍 DEBUG: Cleaned Authentication Data (first 5 rows):")
    print(cleaned_df.head())
    
    # Step 4: Feature Extraction (extract feature dictionary for this authentication session)
    features = feature_extraction()
    
    # (Next steps would involve converting these features into a numerical vector,
    # comparing with stored features, and authenticating using the trained Siamese model.)
