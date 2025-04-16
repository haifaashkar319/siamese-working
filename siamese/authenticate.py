import pandas as pd
import time
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from pynput import keyboard
from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Import functions from your modules
from add_user_session import collect_auth_data, process_auth_data
from data_loader import standardize_features, extract_keystroke_features, get_user_percentile_thresholds, extract_pause_features

# Temporary file names
TEMP_RAW_FILE = "temp_keystrokes.csv"          # To store raw keystroke events
TEMP_FEATURE_FILE = "authenticate_temp.csv"    # To store processed (and later cleaned) features

# Define similarity-to-percentage conversion (using a base of 0.01; adjust if needed)
def similarity_to_percentage(similarity, base=0.01):
    percentage = 100 * (1 - np.power(base, similarity))
    return percentage

# --- Block 1: Data Collection ---
def data_collection():
    """Collect keystroke data for authentication."""
    print("\n--- Authentication: Data Collection ---")
    events = collect_auth_data()
    
    print("\nCollected keystroke events:")
    for event in events:
        print(event)
    return events

# --- Block 2: Process Keystroke Data ---
def data_processing(events):
    print("\n--- Authentication: Processing Keystroke Data ---")
    processed_data = process_auth_data(events, TEMP_FEATURE_FILE, save_to_db=False)
    print(f" Processed keystroke features saved to {TEMP_FEATURE_FILE}")
    return processed_data


# --- Block 4: Feature Extraction ---
def feature_extraction(user_id):
    """
    Load the cleaned authentication data and extract features
    """
    print("\n--- Feature Extraction ---")
    df_cleaned = pd.read_csv(TEMP_FEATURE_FILE)
    
    # Get reference features and print them for debugging
    ref_features = pd.read_csv("features.csv", nrows=1)
    expected_features = [col for col in ref_features.columns if col not in ['session', 'participant']]
    print(f"\nActual features used in training (from features.csv):")
    print(ref_features.columns.tolist())
    
    # Calculate pause thresholds from the authentication data itself
    pause_cols = ["DD.key1.key2", "DU.key1.key2", "UD.key1.key2", "UU.key1.key2"]
    
    # Convert columns to numeric
    for col in pause_cols:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce")
    
    # Get thresholds from the authentication data
    thresholds = get_user_percentile_thresholds(df_cleaned, pause_cols)
    
    # Extract pause features using actual timing data
    pause_features, active_data = extract_pause_features(df_cleaned, thresholds)
    
    # Extract regular keystroke features from non-pause data
    keystroke_features = extract_keystroke_features(active_data)
    
    # Print debug info about the session
    print("\n🔍 DEBUG: Session Text Analysis:")
    print("Typing sequence:", ' '.join([f"{row['key1']}->{row['key2']}" for _, row in df_cleaned.iterrows()]))
    print(f"Total keystrokes: {len(df_cleaned)}")
    
    # Combine features
    features = {**keystroke_features, **pause_features}
    
    # Filter features to match exactly what the model expects
    features = {k: features[k] for k in expected_features if k in features}
    
    # Convert to float64
    features = {k: np.float64(v) for k, v in features.items()}
    
    print("\n🔍 DEBUG: Raw features extracted (filtered to match model):")
    for key, value in features.items():
        print(f"{key}: {float(value):.4f}")
    
    # Create session key and features dictionary
    session_key = f"{user_id}_auth"
    features_dict = {session_key: features}
    
    # Use data_loader's standardization function
    df_transformed = standardize_features(features_dict)
    
    return df_transformed.loc[session_key].to_dict()

def load_user_features(user_id, features_path="features.csv"):
    """Load existing feature vectors for the given user ID."""
    print(f"\n--- Loading Features for User {user_id} ---")
    try:
        df = pd.read_csv(features_path, index_col=0)
    except FileNotFoundError:
        raise FileNotFoundError(f"Features file '{features_path}' not found. Please ensure the file exists.")

    # Clean and normalize input
    user_id = user_id.strip().lower()

    # Normalize session index in case it has trailing spaces or inconsistencies
    df.index = df.index.astype(str).str.strip()

    # Use robust regex to extract the participant (e.g., "p001" from "p001_s1")
    df['participant'] = df.index.str.extract(r'^(p\d+)', expand=False)

    # Drop any rows where extraction failed
    df = df.dropna(subset=['participant'])

    # Match user
    user_features = df[df['participant'].str.lower() == user_id]

    if user_features.empty:
        available_users = sorted(set(df['participant'].dropna().unique()))
        error_msg = (
            f"No features found for user {user_id}.\n"
            f"Available users: {', '.join(available_users)}\n"
            "Please try again with one of the available user IDs."
        )
        raise ValueError(error_msg)

    print(f"Found {len(user_features)} sessions for user {user_id}")
    return user_features

def get_feature_columns(features_path="features.csv"):
    """
    Extract feature column names from features.csv to ensure consistency
    """
    try:
        # Read just the header row from features.csv
        df = pd.read_csv(features_path, nrows=0)
        # Get column names excluding 'session' and 'participant' if present
        feature_cols = [col for col in df.columns if col not in ['session', 'participant']]
        return feature_cols
    except FileNotFoundError:
        raise FileNotFoundError(f"Features file '{features_path}' not found.")

def authenticate_user(features, user_id, threshold=0.7):
    """
    Authenticate user by comparing features against stored features.
    """
    print("\n--- Authenticating User ---")
    
    # Load the trained model
    model = load_model("models/siamese_model.keras")
    
    # Get feature columns from features.csv
    feature_cols = get_feature_columns()
    
    # Load stored features for this user (all sessions)
    user_features = load_user_features(user_id)
    
    # Convert current features to array format using correct feature names
    current_features = np.array([[features[col] for col in feature_cols]])
    
    # Get feature columns from stored features
    stored_features = user_features[feature_cols].values
    
    # Compare against each stored session
    similarities = []
    for i, stored_feature in enumerate(stored_features):
        pred = model.predict([current_features, stored_feature.reshape(1, -1)], verbose=0)
        similarity = pred[0][0]
        session_id = user_features.index[i]
        similarities.append((session_id, similarity))
        print(f" Session {session_id}: Similarity {similarity:.3f} ({similarity_to_percentage(similarity):.1f}%)")
    
    # Get best matching session
    best_session, max_similarity = max(similarities, key=lambda x: x[1])
    
    # Authenticate if any session matches above threshold
    is_authenticated = max_similarity > threshold
    print(f"\nAuthentication {'successful' if is_authenticated else 'failed'}!")
    print(f"Best match: Session {best_session} with confidence {max_similarity:.3f} ({similarity_to_percentage(max_similarity):.1f}%)")
    
    return is_authenticated

# --- Main Execution Flow ---
if __name__ == "__main__":
    try:
        # Get user ID to authenticate without int conversion
        user_id = input("Enter user ID to authenticate (format: p<number>, e.g. p001): ")
        
        # Step 1: Data Collection
        events = data_collection()
        
        # Step 2: Process the collected raw keystroke data
        data_processing(events)
        
        # Step 4: Feature Extraction
        features = feature_extraction(user_id)
        
        # Step 5: Authenticate user
        authenticate_user(features, user_id)
        
    except ValueError as e:
        print(f"\n❌ Error: {str(e)}")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {str(e)}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
    finally:
        print("\nAuthentication process completed.")
