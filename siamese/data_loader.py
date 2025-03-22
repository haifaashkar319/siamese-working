import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

### ----------------------- Data Loading & Validation -----------------------

def load_data(file_path="FreeDB2.csv"):
    """
    Load the keystroke dataset from a CSV file.
    Make sure the CSV is in the same directory or provide an absolute path.
    """
    file_path = os.path.join(os.path.dirname(__file__), file_path)
    df = pd.read_csv(file_path, low_memory=False)
    df.columns = df.columns.str.strip()  # Clean column names
    return df

def validate_data(df):
    """
    Ensure required columns exist and perform basic checks.
    """
    required_columns = {"participant", "session", "DU.key1.key1", "DD.key1.key2"}
    if not required_columns.issubset(df.columns):
        missing_cols = required_columns - set(df.columns)
        raise KeyError(f"❌ Missing required columns: {missing_cols}")
    return df

### ----------------------- Percentile-Based Thresholds (Per User) -----------------------

def get_user_percentile_thresholds(user_df, columns, lower_pct=0.01, upper_pct=0.99):
    """
    For each column in 'columns', compute percentile-based thresholds on data from a single user.
    'lower_pct' and 'upper_pct' define the quantiles for lower and upper cutoffs.
    Returns a dictionary mapping column names to (lower_threshold, upper_threshold).
    """
    thresholds = {}
    for col in columns:
        user_df[col] = pd.to_numeric(user_df[col], errors="coerce")
        lower = user_df[col].quantile(lower_pct)
        upper = user_df[col].quantile(upper_pct)
        thresholds[col] = (lower, upper)
    return thresholds

### ----------------------- Feature Extraction with Pause Computation -----------------------

def extract_features_for_session(df):
    """
    Extract features for each (participant, session) group.
    Uses per-user percentile-based thresholds for pause detection.
    Rows flagged as pauses (all 4 out-of-range among selected columns) are excluded
    from keystroke feature calculations.
    Returns a dictionary where each key is a session identifier and each value is a dict of features.
    """
    features_by_session = {}
    pause_cols = ["DD.key1.key2", "DU.key1.key2", "UD.key1.key2", "UU.key1.key2"]
    
    # For each participant, compute percentile thresholds, then process each session
    for participant, user_df in df.groupby("participant"):
        user_thresholds = get_user_percentile_thresholds(user_df.copy(), pause_cols, lower_pct=0.01, upper_pct=0.99)
        for session, group in user_df.groupby("session"):
            if len(group) < 3:
                continue
            
            # Identify pause rows and remove them
            pause_stats, group_active = extract_pause_features(group.copy(), user_thresholds)
            # Compute keystroke features on the active (non-pause) rows
            active_features = extract_keystroke_features(group_active)
            # Merge features into one dictionary
            features = {**active_features, **pause_stats}
            session_key = f"{participant}_s{session}"
            features_by_session[session_key] = features
    return features_by_session

def extract_pause_features(group, thresholds):
    """
    Identify rows that are considered pauses.
    A row is flagged as a pause if all 4 specified columns are out-of-range
    based on the percentile thresholds.
    Returns a dictionary of pause stats and a subset of the group with pause rows removed.
    """
    pause_cols = ["DD.key1.key2", "DU.key1.key2", "UD.key1.key2", "UU.key1.key2"]
    for col in pause_cols:
        group[col] = pd.to_numeric(group[col], errors="coerce")
    
    def is_pause(row):
        count = 0
        for col in pause_cols:
            lower, upper = thresholds[col]
            val = row[col]
            if pd.notna(val) and (val < lower or val > upper):
                count += 1
        # Flag as pause only if all 4 columns are out-of-range
        return count >= 4
    
    pause_mask = group.apply(is_pause, axis=1)
    pause_count = pause_mask.sum()
    total_count = len(group)
    # Use pause_ratio for further interpretation (but not printed in feature vector)
    pause_ratio = pause_count / total_count if total_count > 0 else 0
    
    if pause_count > 0:
        avg_pause = group.loc[pause_mask, "DD.key1.key2"].mean()
        std_pause = group.loc[pause_mask, "DD.key1.key2"].std()
    else:
        avg_pause = 0
        std_pause = 0

    group_active = group[~pause_mask].copy()
    
    return {
        "pause_ratio": pause_ratio,
        "avg_pause": avg_pause,
        "std_pause": std_pause
    }, group_active

def extract_keystroke_features(group):
    """
    Compute standard keystroke features for a given session (ignoring pause rows).
    """
    timing_columns = ["DU.key1.key1", "DD.key1.key2", "UD.key1.key2", "UU.key1.key2"]
    for col in timing_columns:
        group[col] = pd.to_numeric(group[col], errors="coerce")
    
    group = group.dropna(subset=["DU.key1.key1", "DD.key1.key2"])
    if len(group) < 3:
        return {}
    
    return {
        "avg_dwell_time":   np.mean(group["DU.key1.key1"]),
        "std_dwell_time":   np.std(group["DU.key1.key1"]),
        "avg_flight_time":  np.mean(group["DD.key1.key2"]),
        "std_flight_time":  np.std(group["DD.key1.key2"]),
        "avg_latency":      np.mean(group["UD.key1.key2"]) if "UD.key1.key2" in group.columns else 0,
        "std_latency":      np.std(group["UD.key1.key2"])  if "UD.key1.key2" in group.columns else 0,
        "avg_UU_time":      np.mean(group["UU.key1.key2"]) if "UU.key1.key2" in group.columns else 0,
        "std_UU_time":      np.std(group["UU.key1.key2"])  if "UU.key1.key2" in group.columns else 0
    }

def create_training_pairs(features_by_session):
    """
    Create training pairs from the dictionary of features_by_session.
    """
    pairs = []
    labels = []
    users = list(set(k.split('_s')[0] for k in features_by_session.keys()))
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
    return pairs, labels

### ----------------------- Standardization Function -----------------------

def standardize_features(features_by_session):
    """
    Convert the features dictionary into a DataFrame, standardize numeric features using StandardScaler,
    and return the standardized DataFrame.
    """
    # Convert the dictionary to a DataFrame (session as index)
    session_list = []
    for session_key, feats in features_by_session.items():
        row = {"session": session_key}
        row.update(feats)
        session_list.append(row)
    df_features = pd.DataFrame(session_list)
    df_features.set_index("session", inplace=True)
    
    # Select numeric columns to scale
    numeric_cols = df_features.columns
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_features[numeric_cols]),
                             columns=numeric_cols,
                             index=df_features.index)
    return df_scaled

### ----------------------- Distance Calculation -----------------------

def calculate_pair_distances(df_scaled, true_pairs, false_pairs):
    """
    Given two lists of session keys for true pairs and false pairs, compute and print
    the Euclidean distances between the corresponding standardized feature vectors.
    """
    # For convenience, create a dictionary mapping session key to vector
    session_vectors = df_scaled.to_dict(orient="index")
    
    print("\nTrue Pair Distances (same participant):")
    for s1, s2 in true_pairs:
        v1 = np.array(list(session_vectors[s1].values()))
        v2 = np.array(list(session_vectors[s2].values()))
        dist = np.linalg.norm(v1 - v2)
        print(f"{s1} vs {s2}: {dist:.4f}")
    
    print("\nFalse Pair Distances (different participants):")
    for s1, s2 in false_pairs:
        v1 = np.array(list(session_vectors[s1].values()))
        v2 = np.array(list(session_vectors[s2].values()))
        dist = np.linalg.norm(v1 - v2)
        print(f"{s1} vs {s2}: {dist:.4f}")

### ----------------------- Example Usage -----------------------

if __name__ == "__main__":
    # Load and validate data
    df = load_data("FreeDB2.csv")
    df = validate_data(df)
    
    # Extract features per session using per-user percentile-based thresholds
    features = extract_features_for_session(df)
    
    # Save features to CSV (optional) and print them
    # (Uncomment the following line if you want to save to CSV)
    # save_features_to_csv(features, output_file="features.csv")
    
    
    # Standardize the features
    df_scaled = standardize_features(features)
    print("\nFirst 5 Standardized Feature Vectors:")
    print(df_scaled.head(5))
    df_scaled.to_csv("features.csv")
    print("\nStandardized features saved to features.csv")
    
    # Create training pairs
    pairs, labels = create_training_pairs(features)
    
    # For evaluation, create lists of true pairs and false pairs manually.
    # True pairs: sessions from the same participant.
    # False pairs: sessions from different participants.
    true_pairs = []
    false_pairs = []
    
    # Group sessions by participant based on session key format "participant_s<session>"
    session_groups = {}
    for session_key in df_scaled.index:
        participant = session_key.split('_s')[0]
        session_groups.setdefault(participant, []).append(session_key)
    
    # Select up to 5 true pairs (from participants with at least 2 sessions)
    for participant, sessions in session_groups.items():
        if len(sessions) >= 2:
            # Take the first two sessions as a true pair
            true_pairs.append((sessions[0], sessions[1]))
            if len(true_pairs) >= 5:
                break
    
    # Select up to 5 false pairs (from two different participants)
    participants = list(session_groups.keys())
    if len(participants) >= 2:
        for i in range(min(5, len(participants)-1)):
            s1 = session_groups[participants[i]][0]
            s2 = session_groups[participants[i+1]][0]
            false_pairs.append((s1, s2))
            if len(false_pairs) >= 5:
                break
    
    # Calculate and print pair distances
    calculate_pair_distances(df_scaled, true_pairs, false_pairs)
