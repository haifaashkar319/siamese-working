import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from data_loader import extract_features_for_session, load_data, validate_data, preprocess_data

### ----------------------- Step 1: Define & Register Custom Functions -----------------------

@tf.keras.utils.register_keras_serializable()
def l1_distance(vects):
    """Computes L1 distance (Manhattan distance) between two input vectors."""
    x, y = vects
    return K.abs(x - y)

# Register custom function before loading the model
custom_objects = {"l1_distance": l1_distance}

### ----------------------- Step 2: Load Pretrained Siamese Model -----------------------

def load_siamese_model(model_path="models/siamese_model.h5"):
    """Loads the trained Siamese model."""
    print("📥 Loading trained Siamese model...")
    try:
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        print("✅ Model loaded successfully.")
        return model
    except Exception as e:
        print(f"❌ ERROR: Could not load model! Reason: {e}")
        exit()

### ----------------------- Step 3: Extract Features from Dataset -----------------------

def extract_features(file_path="FreeDB.csv"):
    """Loads the dataset, validates it, preprocesses it, and extracts keystroke features."""
    print("📥 Loading dataset...")
    df = load_data(file_path)

    print("✅ Validating dataset...")
    df = validate_data(df)

    print("🔍 Preprocessing dataset (Applying Min-Max Normalization)...")
    df = preprocess_data(df)

    print("🧑‍💻 Extracting features per session...")
    features_by_session = extract_features_for_session(df)

    if not features_by_session:
        print("❌ ERROR: No features extracted! Exiting.")
        exit()

    print(f"✅ Extracted features for {len(features_by_session)} sessions.")
    return features_by_session

### ----------------------- Step 4: Aggregate Features (Hybrid Approach) -----------------------

def aggregate_features(features_by_session):
    """
    Aggregates per-session features into per-user features.

    :param features_by_session: Dictionary containing keystroke feature vectors per session.
    :return: user_features (aggregated per-user), session_features (individual sessions).
    """

    print("\n🛠 Aggregating features...")

    user_features = {}   # 🔹 Stores **user-level** aggregated features
    session_features = {}  # 🔹 Stores **session-level** features
    user_sessions = {}  # 🔹 Temporary storage for per-user session features

    for session_key, features in features_by_session.items():
        user_id = session_key.split("_s")[0]  # Extract user from session key
        session_id = session_key.split("_s")[1]  # Extract session ID

        # Store session-level features
        session_features[session_key] = features
        print(f"✅ {user_id} (Session {session_id}) -> Features extracted.")

        # Collect session features for user-level aggregation
        if user_id not in user_sessions:
            user_sessions[user_id] = []
        user_sessions[user_id].append(features)

    # Compute **per-user aggregated features** by averaging session features
    for user_id, session_feature_list in user_sessions.items():
        if session_feature_list:
            # Convert list of session feature dicts to DataFrame
            df_user_sessions = pd.DataFrame(session_feature_list)
            user_features[user_id] = df_user_sessions.mean().to_dict()  # 🔹 Hybrid: Compute mean per-user
            print(f"🟢 Computed User-Level Features for {user_id}")

    if not user_features:
        print("❌ ERROR: No user features created! Check input data.")
        exit()

    print(f"📊 Final User Feature Count: {len(user_features)}")
    return user_features, session_features

### ----------------------- Step 5: Save Features -----------------------

def save_features(user_features, session_features, user_path="user_features.npy", session_path="session_features.npy"):
    """Saves user and session features as numpy files."""
    np.save(user_path, user_features)
    np.save(session_path, session_features)
    print(f"✅ User features saved to `{user_path}`")
    print(f"✅ Session features saved to `{session_path}`")

### ----------------------- Execution Flow -----------------------

if __name__ == "__main__":
    # Load model
    siamese_model = load_siamese_model()
    
    # Extract features from dataset
    features_by_session = extract_features()

    # Aggregate features (Hybrid Approach)
    user_features, session_features = aggregate_features(features_by_session)

    # Save features
    save_features(user_features, session_features)
