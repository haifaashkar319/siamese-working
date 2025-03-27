import os
import pandas as pd
import numpy as np

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
        raise KeyError(f" Missing required columns: {missing_cols}")
    return df

### ----------------------- Dynamic Threshold Calculation (IQR Method, Per User) -----------------------

def get_dynamic_thresholds(user_df, columns, multiplier=2.0):
    """
    For each column in 'columns', compute dynamic thresholds using the IQR method,
    on data from a single user.
    Lower threshold = Q1 - multiplier * IQR; Upper threshold = Q3 + multiplier * IQR.
    Returns a dictionary mapping column names to (lower_threshold, upper_threshold).
    """
    thresholds = {}
    for col in columns:
        user_df[col] = pd.to_numeric(user_df[col], errors="coerce")
        Q1 = user_df[col].quantile(0.25)
        Q3 = user_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        thresholds[col] = (lower, upper)
    return thresholds

def save_user_thresholds_to_csv(df, columns, output_file="user_thresholds.csv", multiplier=2.0):
    """
    Compute per-user dynamic thresholds for the given columns and save them to a CSV.
    Each row in the CSV corresponds to one user, with lower and upper thresholds for each column.
    """
    user_thresholds_list = []
    for participant, user_df in df.groupby("participant"):
        thresholds = get_dynamic_thresholds(user_df.copy(), columns, multiplier=multiplier)
        row = {"participant": participant}
        for col in columns:
            lower, upper = thresholds[col]
            row[f"{col}_lower"] = lower
            row[f"{col}_upper"] = upper
        user_thresholds_list.append(row)
    df_thresh = pd.DataFrame(user_thresholds_list)
    df_thresh.to_csv(output_file, index=False)
    print(f"User thresholds saved to {output_file}")

### ----------------------- Feature Extraction with Pause Computation -----------------------

def extract_features_for_session(df):
    """
    Extract features for each (participant, session) group.
    Includes pause-related features using per-user dynamic IQR thresholds.
    Rows flagged as pauses (all 4 out-of-range among selected columns) are excluded
    from keystroke feature calculations.
    """
    features_by_session = {}
    # Define the columns for pause detection (excluding DU.key1.key1)
    pause_cols = ["DD.key1.key2", "DU.key1.key2", "UD.key1.key2", "UU.key1.key2"]
    
    # Loop over each participant, compute thresholds per user,
    # then process each session for that user.
    for participant, user_df in df.groupby("participant"):
        # Compute user-specific thresholds using all data for this participant
        user_thresholds = get_dynamic_thresholds(user_df.copy(), pause_cols, multiplier=2.0)
        
        for session, group in user_df.groupby("session"):
            if len(group) < 3:
                continue
            
            # Compute pause features and remove pause rows from the group
            pause_stats, group_active = extract_pause_features(group.copy(), user_thresholds)
            # Compute other keystroke features on the active (non-pause) rows
            active_features = extract_keystroke_features(group_active)
            
            # Merge pause stats with active typing stats
            features = {**active_features, **pause_stats}
            session_key = f"{participant}_s{session}"
            features_by_session[session_key] = features
    return features_by_session

def extract_pause_features(group, thresholds):
    """
    Identify rows that are considered pauses.
    A row is flagged as a pause if all 4 specified columns are out-of-range
    based on the provided thresholds.
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
        return count >= 4  # Flag as pause only if all 4 columns are out-of-range
    
    pause_mask = group.apply(is_pause, axis=1)
    pause_count = pause_mask.sum()
    
    # Debug: report pause count and total rows for this session
    if 'participant' in group.columns and 'session' in group.columns:
        participant = group["participant"].iloc[0]
        session = group["session"].iloc[0]
        print(f"Debug: Participant {participant}, Session {session} -> Total rows: {len(group)}, Pause count: {pause_count}")
    
    if pause_count > 0:
        avg_pause = group.loc[pause_mask, "DD.key1.key2"].mean()
        std_pause = group.loc[pause_mask, "DD.key1.key2"].std()
    else:
        avg_pause = 0
        std_pause = 0

    group_active = group[~pause_mask].copy()
    
    return {
        "pause_count": pause_count,
        "avg_pause": avg_pause,
        "std_pause": std_pause
    }, group_active

def extract_keystroke_features(group):
    """
    Compute standard keystroke features for a given session (ignoring rows flagged as pauses).
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
    Create training pairs from the dictionary of features_by_session
    and run debug_feature_separation to evaluate distances.
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
                labels.append(1)  # same-user pair
    pairs = np.array(pairs)
    labels = np.array(labels)
    
    debug_feature_separation(pairs, users)
    return pairs, labels

def debug_feature_separation(pairs, users):
    """
    Check L1 distances and any anomalies in feature space.
    """
    print("\n🔍 Debugging Feature Separation:")
    if len(pairs) == 0:
        print("No pairs available to debug.")
        return
    sample1 = pairs[0][0]
    sample2 = pairs[0][0]
    identical_distance = np.abs(sample1 - sample2)
    print(f"\n Identical Sample L1 Distance (Expected ~0): {np.mean(identical_distance):.6f}")
    
    if len(users) > 1 and len(pairs) > 1:
        different_user_pair_idx = np.random.choice(len(pairs), 1, replace=False)[0]
        sample3 = pairs[0][0]
        sample4 = pairs[different_user_pair_idx][1]
        different_distance = np.abs(sample3 - sample4)
        print(f"⚠️ Different User Sample L1 Distance (Expected > identical): {np.mean(different_distance):.6f}")
    else:
        print("Only one user or one pair found; cannot compare negative pair distances.")

# Example usage (this part would typically be in your main script)
if __name__ == "__main__":
    # Load the CSV file (adjust file path if needed)
    df = pd.read_csv("FreeDB2.csv")
    df.columns = df.columns.str.strip()
    
    # Save per-user thresholds to a CSV for inspection
    pause_cols = ["DD.key1.key2", "DU.key1.key2", "UD.key1.key2", "UU.key1.key2"]
    save_user_thresholds_to_csv(df, pause_cols, output_file="user_thresholds.csv", multiplier=2.0)
    
    # Extract features per session using per-user dynamic thresholds
    features = extract_features_for_session(df)
    
    print("\nExtracted features by session:")
    for session, feats in features.items():
        print(f"{session}: {feats}")
