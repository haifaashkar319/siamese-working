import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

### ----------------------- Step 1: Data Loading & Validation -----------------------

def load_data(file_path="FreeDB2.csv"):
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

### ----------------------- Step 2: Preprocessing & Scaling -----------------------

def preprocess_data(df):
    """Convert columns to numeric, filter based on adjusted ranges, and apply Z-score normalization."""
    timing_columns = ["DU.key1.key1", "DD.key1.key2", "UD.key1.key2", "UU.key1.key2"]
    
    for col in timing_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df = df.dropna(subset=timing_columns)
    df = df.sort_values(by=["participant", "session"])
    
    # Apply filtering based on acceptable ranges
    df = df[
        (df["DU.key1.key1"].between(0.03, 3.0)) &
        (df["DD.key1.key2"].between(-0.3, 5.0)) &
        (df["UD.key1.key2"].between(-0.5, 4.0)) &
        (df["UU.key1.key2"].between(0.0, 5.0))
    ]
    
    # Apply Z-score normalization to timing columns
    scaler = StandardScaler()
    df[timing_columns] = scaler.fit_transform(df[timing_columns])
    return df

### ----------------------- Step 2a: Percentile Computation -----------------------

def compute_percentiles(file_path="FreeDB2.csv", 
                        columns=["DU.key1.key1", "DD.key1.key2", "UD.key1.key2", "UU.key1.key2"],
                        percentiles=[0.001, 0.01, 0.5, 0.99, 0.999]):
    df = pd.read_csv(file_path, low_memory=False)
    df.columns = df.columns.str.strip()
    stats = df[columns].describe(percentiles=percentiles)
    print("\n🔍 Debugging Percentiles for Features:")
    for col in columns:
        print(f"\nFeature: {col}")
        print(f"  Min:  {stats.loc['min', col]}")
        print(f"  Mean: {stats.loc['mean', col]}")
        print(f"  Std:  {stats.loc['std', col]}")
        print(f"  Median (50th %ile): {stats.loc['50%', col]}")
        for p in percentiles:
            p_str = f"{p:.3f}"
            if p_str in stats.index:
                print(f"  {p*100:.1f}%ile: {stats.loc[p_str, col]}")
            else:
                print(f"  {p*100:.1f}%ile: Not found in index")
        print(f"  Max:  {stats.loc['max', col]}")
    return stats

### ----------------------- Step 3: Feature Extraction with Pause Computation -----------------------

def extract_features_for_session(df):
    """Extract features for each session including pause-related features."""
    features_by_session = {}
    for (participant, session), group in df.groupby(['participant', 'session']):
        if len(group) < 3:
            continue
        
        # Compute pause features and remove pauses from the group for computing other features.
        pause_threshold = 0.5  # Define pause threshold in raw seconds (adjust as needed)
        pause_stats, group_active = extract_pause_features(group, pause_threshold)
        active_features = extract_keystroke_features(group_active)
        
        # Merge pause stats with active typing features
        features = {**active_features, **pause_stats}
        session_key = f"{participant}_s{session}"
        features_by_session[session_key] = features
    return features_by_session

def extract_pause_features(group, pause_threshold):
    """
    Compute pause-related features:
      - pause_count: number of pauses (where DD.key1.key2 exceeds pause_threshold)
      - avg_pause: mean pause duration (using DD.key1.key2 values deemed pauses)
      - std_pause: std deviation of pause durations
    Also, return the group with pause rows removed.
    """
    # Make sure the key column is numeric
    group["DD.key1.key2"] = pd.to_numeric(group["DD.key1.key2"], errors="coerce")
    
    # Identify pause rows: where DD exceeds the threshold
    pause_mask = group["DD.key1.key2"] > pause_threshold
    pause_count = pause_mask.sum()
    if pause_count > 0:
        avg_pause = group.loc[pause_mask, "DD.key1.key2"].mean()
        std_pause = group.loc[pause_mask, "DD.key1.key2"].std()
    else:
        avg_pause = 0
        std_pause = 0

    # Debug logs for pauses
    print(f"\n🔍 Pause Features:")
    print(f"  Pause Threshold: {pause_threshold} sec")
    print(f"  Pause Count: {pause_count}")
    print(f"  Avg Pause Duration: {avg_pause}")
    print(f"  Std Pause Duration: {std_pause}")

    # Remove pause rows for computing the remaining features
    group_active = group[~pause_mask].copy()
    return {"pause_count": pause_count, "avg_pause": avg_pause, "std_pause": std_pause}, group_active

def extract_keystroke_features(group):
    """Compute standard keystroke features for a given session (ignoring pauses)."""
    timing_columns = ["DU.key1.key1", "DD.key1.key2", "UD.key1.key2", "UU.key1.key2"]
    for col in timing_columns:
        group[col] = pd.to_numeric(group[col], errors="coerce")
    group = group.dropna(subset=["DU.key1.key1", "DD.key1.key2"])
    if len(group) < 3:
        return {}
    return {
        "avg_dwell_time": np.mean(group["DU.key1.key1"]),
        "std_dwell_time": np.std(group["DU.key1.key1"]),
        "avg_flight_time": np.mean(group["DD.key1.key2"]),
        "std_flight_time": np.std(group["DD.key1.key2"]),
        "avg_latency": np.mean(group["UD.key1.key2"]) if "UD.key1.key2" in group.columns else 0,
        "std_latency": np.std(group["UD.key1.key2"]) if "UD.key1.key2" in group.columns else 0,
        "avg_UU_time": np.mean(group["UU.key1.key2"]) if "UU.key1.key2" in group.columns else 0,
        "std_UU_time": np.std(group["UU.key1.key2"]) if "UU.key1.key2" in group.columns else 0
    }

### ----------------------- Step 4: Creating Training Pairs -----------------------

def create_training_pairs(features_by_session):
    """Create training pairs and debug distances for feature separation."""
    pairs = []
    labels = []
    users = list(set([k.split('_s')[0] for k in features_by_session.keys()]))
    for user in users:
        user_sessions = [k for k in features_by_session.keys() if k.startswith(user)]
        for i in range(len(user_sessions)):
            for j in range(i + 1, len(user_sessions)):
                vec1 = np.array(list(features_by_session[user_sessions[i]].values()), dtype=np.float32)
                vec2 = np.array(list(features_by_session[user_sessions[j]].values()), dtype=np.float32)
                pairs.append((vec1, vec2))
                labels.append(1)
    pairs = np.array(pairs)
    labels = np.array(labels)
    debug_feature_separation(pairs, users)
    return pairs, labels

### ----------------------- Step 5: Debugging Feature Separation -----------------------

def debug_feature_separation(pairs, users):
    """Check L1 distances and scaling impact."""
    print("\n🔍 Debugging Feature Separation:")
    print("\n✅ Checking Min-Max Scaling Impact:")
    print(f"Min Feature Value: {np.min(pairs)}")
    print(f"Max Feature Value: {np.max(pairs)}")
    sample1 = pairs[0][0]
    sample2 = pairs[0][0]
    identical_distance = np.abs(sample1 - sample2)
    print(f"\n✅ Identical Sample L1 Distance (Expected ~0): {np.mean(identical_distance):.6f}")
    different_user_pair = np.random.choice(len(users) - 1)
    sample3 = pairs[0][0]
    sample4 = pairs[different_user_pair][1]
    different_distance = np.abs(sample3 - sample4)
    print(f"⚠️ Different User Sample L1 Distance (Expected higher than identical): {np.mean(different_distance):.6f}")
    if users[0] == users[different_user_pair]:
        print("🚨 WARNING: Negative pair may be coming from the same user!")
    else:
        print("✅ Negative pair is correctly from different users.")

### ----------------------- Step 6: Percentile Computation -----------------------

def print_feature_percentiles(df, column, percentiles=[0.99, 0.999]):
    for p in percentiles:
        value = df[column].quantile(p)
        print(f"{p*100:.1f}%ile for {column}: {value}")

### ----------------------- Execution Flow -----------------------

if __name__ == "__main__":
    print("📥 Loading data...")
    df = load_data()
    
    print("✅ Validating data...")
    df = validate_data(df)
    
    print("🔍 Preprocessing data (Min-Max Normalization)...")
    df = preprocess_data(df)
    
    # Compute and print percentiles
    compute_percentiles()
    print_feature_percentiles(df, "DU.key1.key1")
    print_feature_percentiles(df, "DD.key1.key2")    
    print_feature_percentiles(df, "DU.key1.key2")
    print_feature_percentiles(df, "UU.key1.key2")    
    print_feature_percentiles(df, "UD.key1.key2")    
    
    print("🧑‍💻 Extracting features per session...")
    features_by_session = extract_features_for_session(df)
    
    print("📊 Creating training pairs...")
    X_train, Y_train = create_training_pairs(features_by_session)
    
    print(f"✅ Training Pairs: {len(X_train)}")
