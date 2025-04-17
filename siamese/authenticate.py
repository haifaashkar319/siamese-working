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

def feature_extraction(user_id):
    """Extract and prepare features for authentication."""
    print("\n--- Authentication: Feature Extraction ---")
    
    try:
        print("Loading reference data...")
        ref_features_df = pd.read_csv("features.csv")
        auth_features_df = pd.read_csv("authenticate_temp.csv")
        print(f"Reference data shape: {ref_features_df.shape}")
        print(f"Auth data shape: {auth_features_df.shape}")
        
        print("\nGetting user thresholds...")
        user_data = ref_features_df[ref_features_df['session'].str.startswith(user_id)]
        print(f"Found {len(user_data)} samples for user {user_id}")
        
        if len(user_data) == 0:
            raise ValueError(f"No reference data found for user {user_id}")
            
        # First extract basic timing features
        print("\nExtracting keystroke features...")
        timing_cols = ['DU.key1.key1', 'DD.key1.key2', 'DU.key1.key2', 'UD.key1.key2', 'UU.key1.key2']
        
        # Calculate statistics for each timing measure
        features = {}
        for col in timing_cols:
            values = auth_features_df[col].values
            features[f'avg_{col}'] = np.mean(values)
            features[f'std_{col}'] = np.std(values)
            features[f'med_{col}'] = np.median(values)
            features[f'iqr_{col}'] = np.percentile(values, 75) - np.percentile(values, 25)
            
        # Calculate rhythm and pause features
        dd_values = auth_features_df['DD.key1.key2'].values
        features['rhythm_mean'] = np.mean(dd_values)
        features['rhythm_std'] = np.std(dd_values)
        
        # Calculate pause features using thresholds
        threshold = user_data['avg_DD.key1.key2'].values[0]  # Use average as threshold
        long_pauses = dd_values[dd_values > threshold]
        
        features['pause_ratio'] = len(long_pauses) / len(dd_values)
        features['avg_pause'] = np.mean(long_pauses) if len(long_pauses) > 0 else np.mean(dd_values)
        features['std_pause'] = np.std(long_pauses) if len(long_pauses) > 0 else 0
        
        print("Features computed:", list(features.keys()))
        
        # Create DataFrame with computed features
        feature_df = pd.DataFrame([features])
        
        # Ensure columns match reference data
        ref_numeric_cols = ref_features_df.select_dtypes(include=[np.number]).columns
        feature_df = feature_df.reindex(columns=ref_numeric_cols, fill_value=0)
        
        # Apply standardization
        scaler = StandardScaler()
        scaler.fit(ref_features_df[ref_numeric_cols])
        std_features = scaler.transform(feature_df)
        
        print("✓ Features extracted and standardized")
        return std_features
        
    except Exception as e:
        print("\nDetailed error in feature extraction:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"Error location: {e.__traceback__.tb_frame.f_code.co_name}")
        print("Full traceback:")
        print(traceback.format_exc())
        raise

def authenticate_user(feature_vector, user_id, threshold=0.7):
    """
    Authenticate a user based on their typing pattern.
    
    Args:
        feature_vector: Extracted keystroke features as list/array
        user_id: ID of user claiming identity (string starting with 'p')
        threshold: Authentication threshold (0-1)
    """
    
    print("\n--- Authentication: User Verification ---")
    
    try:
        # Convert feature vector to numpy array and reshape
        features = np.array(feature_vector).reshape(1, -1)
        
        # Load the trained model
        model = load_model("models/siamese_model.keras", compile=False)
        
        # Load reference data
        ref_features_df = pd.read_csv("features.csv")
        
        # Filter reference data for claimed user ID
        user_pattern = f"^{user_id}"
        ref_features = ref_features_df[ref_features_df['session'].str.match(user_pattern, na=False)]
        
        if len(ref_features) == 0:
            raise ValueError(f"No reference data found for user {user_id}")
            
        # Drop session column and convert to numpy array
        ref_features = ref_features.drop('session', axis=1).values
        
        # Calculate similarity against all reference samples
        similarities = []
        for ref_sample in ref_features:
            ref_array = ref_sample.reshape(1, -1)
            similarity = model.predict([features, ref_array], verbose=0)[0][0]
            similarities.append(similarity)
            
        # Calculate confidence score
        avg_similarity = np.mean(similarities)
        confidence = avg_similarity * 100
        
        print("\nAuthentication Results:")
        print(f"Average similarity: {avg_similarity:.4f}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Threshold: {threshold*100:.2f}%")
        
        # Make authentication decision
        if confidence >= (threshold * 100):
            print(f"\n✅ Authentication Successful for user {user_id}")
            print(f"   Confidence {confidence:.2f}% >= Threshold {threshold*100:.2f}%")
            return True
        else:
            print(f"\n❌ Authentication Failed for user {user_id}")
            print(f"   Confidence {confidence:.2f}% < Threshold {threshold*100:.2f}%")
            return False
            
    except Exception as e:
        print(f"\n❌ Authentication error: {str(e)}")
        raise

# --- Main Execution Flow ---
if __name__ == "__main__":
    try:
        user_id = input("Enter user ID to authenticate (format: p<number>, e.g. p001): ")
        
        events = data_collection()
        data_processing(events)
        
        # Get features and thresholds
        features = feature_extraction(user_id)
        
        # Authenticate user with features
        authenticate_user(features, user_id)
        
    except ValueError as e:
        print(f"\n❌ Error: {str(e)}")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {str(e)}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
    finally:
        print("\nAuthentication process completed.")
