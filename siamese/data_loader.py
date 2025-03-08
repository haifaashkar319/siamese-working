import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

### ----------------------- Data Loading & Preprocessing -----------------------

def load_data(file_path="FreeDB.csv"):
    """Load the keystroke dataset."""
    file_path = os.path.join(os.path.dirname(__file__), file_path)
    df = pd.read_csv(file_path, low_memory=False)
    df.columns = df.columns.str.strip()  # Clean column names
    return df

def validate_data(df):
    """Ensure required columns exist and perform basic checks."""
    required_columns = {"participant", "session", "DU.key1.key1", "DD.key1.key2"}
    
    if not required_columns.issubset(df.columns):
        missing_cols = required_columns - set(df.columns)
        raise KeyError(f"❌ Missing required columns: {missing_cols}")

    return df

def preprocess_data(df):
    """Convert columns to numeric and sort the dataset."""
    timing_columns = ["DU.key1.key1", "DD.key1.key2", "UD.key1.key2", "UU.key1.key2"]
    
    for col in timing_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["DU.key1.key1"])  # Ensure required columns have values
    df = df.sort_values(by=["participant", "session"])  # Maintain session order
    
    # Apply Min-Max Normalization per user
    scaler = MinMaxScaler()
    df[timing_columns] = df.groupby("participant")[timing_columns].transform(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten())
    
    return df

### ----------------------- Feature Extraction -----------------------

def extract_features_for_session(df):
    """Extract features while preserving session information."""
    features_by_session = {}

    for (participant, session), group in df.groupby(['participant', 'session']):
        if len(group) < 3:  # Skip sessions with insufficient keystrokes
            continue
            
        # Extract statistical features
        features = extract_keystroke_features(group)
        
        # Store features with participant and session key
        session_key = f"{participant}_s{session}"
        features_by_session[session_key] = features

    return features_by_session

def extract_keystroke_features(group):
    """Compute statistical keystroke features for a given participant or session."""
    timing_columns = ["DU.key1.key1", "DD.key1.key2", "UD.key1.key2", "UU.key1.key2"]
    
    # Ensure numeric conversion
    for col in timing_columns:
        if col in group.columns:
            group[col] = pd.to_numeric(group[col], errors="coerce")

    group = group.dropna(subset=["DU.key1.key1", "DD.key1.key2"])
    
    if len(group) < 3:
        return {}

    # Compute statistical features
    dwell_times = group["DU.key1.key1"].dropna().tolist()
    flight_times = group["DD.key1.key2"].dropna().tolist()
    latencies = group["UD.key1.key2"].dropna().tolist() if "UD.key1.key2" in group.columns else []
    uu_timings = group["UU.key1.key2"].dropna().tolist() if "UU.key1.key2" in group.columns else []

    return {
        "avg_dwell_time": np.nan_to_num(np.mean(dwell_times), nan=0),
        "std_dwell_time": np.nan_to_num(np.std(dwell_times), nan=0),
        "avg_flight_time": np.nan_to_num(np.mean(flight_times), nan=0),
        "std_flight_time": np.nan_to_num(np.std(flight_times), nan=0),
        "avg_latency": np.nan_to_num(np.mean(latencies), nan=0),
        "std_latency": np.nan_to_num(np.std(latencies), nan=0),
        "avg_UU_time": np.nan_to_num(np.mean(uu_timings), nan=0),
        "std_UU_time": np.nan_to_num(np.std(uu_timings), nan=0)
    }

### ----------------------- Pair Generation for Model Training -----------------------

def create_training_pairs(features_by_session):
    """Create positive and negative training pairs for a Siamese network."""
    pairs = []
    labels = []
    users = list(set([k.split('_s')[0] for k in features_by_session.keys()]))  # Extract unique users
    
    for user in users:
        # Get all session keys for this user
        user_sessions = [k for k in features_by_session.keys() if k.startswith(user)]
        
        # Create positive pairs (same user, different sessions)
        for i in range(len(user_sessions)):
            for j in range(i + 1, len(user_sessions)):
                vec1 = np.array(list(features_by_session[user_sessions[i]].values()), dtype=np.float32)
                vec2 = np.array(list(features_by_session[user_sessions[j]].values()), dtype=np.float32)
                pairs.append((vec1, vec2))
                labels.append(1)  # Same user -> Positive pair

                # Create a negative pair with a different user
                other_user = np.random.choice([u for u in users if u != user])
                other_session = np.random.choice([k for k in features_by_session.keys() if k.startswith(other_user)])
                vec_impostor = np.array(list(features_by_session[other_session].values()), dtype=np.float32)
                pairs.append((vec1, vec_impostor))
                labels.append(0)  # Different users -> Negative pair
       

    return np.array(pairs), np.array(labels)

### ----------------------- Execution Flow -----------------------

if __name__ == "__main__":
    print("📥 Loading data...")
    df = load_data()
    
    print("✅ Validating data...")
    df = validate_data(df)
    
    print("🔍 Preprocessing data (Min-Max Normalization)...")
    df = preprocess_data(df)
    
    print("🧑‍💻 Extracting features per session...")
    features_by_session = extract_features_for_session(df)
    
    print("📊 Creating training pairs...")
    X_train, Y_train = create_training_pairs(features_by_session)

    print(f"✅ Training data ready! Generated {len(X_train)} pairs.")
