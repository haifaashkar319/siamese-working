import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import skew, kurtosis

### ----------------------- Data Loading & Validation -----------------------

def load_and_clean_data(file_path="FreeDB2.csv"):
    """
    Load and clean the keystroke dataset.
    Combines loading and cleaning steps from all files.
    """
    file_path = os.path.join(os.path.dirname(__file__), file_path)
    df = pd.read_csv(file_path, low_memory=False)
    df.columns = df.columns.str.strip()

    timing_columns = ["DU.key1.key1", "DD.key1.key2", "UD.key1.key2", "UU.key1.key2"]
    
    # Convert to numeric
    for col in timing_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop NaN values and apply filters
    df = df.dropna(subset=timing_columns)
    df = df[
        (df["DU.key1.key1"].between(0.03, 3.0)) &
        (df["DD.key1.key2"].between(-0.3, 5.0)) &
        (df["UD.key1.key2"].between(-0.5, 4.0)) &
        (df["UU.key1.key2"].between(0.0, 5.0))
    ]
    
    return df

def check_session_entries(df):
    """
    Print the number of entries per session for each participant.
    """
    print("\n=== Session Entry Count Validation ===")
    for participant, user_df in df.groupby("participant"):
        print(f"\nParticipant: {participant}")
        session_counts = user_df.groupby("session").size()
        print("Sessions and their entry counts:")
        for session, count in session_counts.items():
            print(f"Session {session}: {count} entries")
        print(f"Total sessions: {len(session_counts)}")
        print(f"Average entries per session: {session_counts.mean():.2f}")

def validate_data(df):
    """
    Ensure required columns exist and perform basic checks.
    """
    required_columns = {"participant", "session", "DU.key1.key1", "DD.key1.key2"}
    if not required_columns.issubset(df.columns):
        missing_cols = required_columns - set(df.columns)
        raise KeyError(f"❌ Missing required columns: {missing_cols}")
    
    # Add session entry validation
    check_session_entries(df)
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
    Enhanced feature extraction combining best features from all files.
    """
    features_by_session = {}
    pause_cols = ["DD.key1.key2", "DU.key1.key2", "UD.key1.key2", "UU.key1.key2"]
    
    for participant, user_df in df.groupby("participant"):
        user_thresholds = get_user_percentile_thresholds(user_df.copy(), pause_cols)
        
        for session, group in user_df.groupby("session"):
            if len(group) < 3:
                continue
            
            pause_stats, group_active = extract_pause_features(group.copy(), user_thresholds)
            active_features = extract_keystroke_features(group_active)
            features = {**active_features, **pause_stats}
            session_key = f"{participant}_s{session}"
            features_by_session[session_key] = features
    
    return features_by_session

def extract_pause_features(group, thresholds):
    """
    Identify rows that are considered pauses with improved NaN handling.
    """
    pause_cols = ["DD.key1.key2", "DU.key1.key2", "UD.key1.key2", "UU.key1.key2"]
    
    # Clean and convert data first
    for col in pause_cols:
        group[col] = pd.to_numeric(group[col], errors="coerce")
    
    def is_pause(row):
        count = 0
        valid_cols = 0
        for col in pause_cols:
            if pd.notna(row[col]):  # Only count valid columns
                valid_cols += 1
                lower, upper = thresholds[col]
                if row[col] < lower or row[col] > upper:
                    count += 1
        # Flag as pause if all valid columns are out-of-range
        return valid_cols > 0 and count == valid_cols
    
    pause_mask = group.apply(is_pause, axis=1)
    pause_count = pause_mask.sum()
    total_count = len(group)
    pause_ratio = pause_count / total_count if total_count > 0 else 0
    
    if pause_count > 0:
        pause_data = group.loc[pause_mask, "DD.key1.key2"]
        valid_pauses = pause_data.dropna()
        
        if len(valid_pauses) > 0:
            avg_pause = valid_pauses.mean()
            std_pause = valid_pauses.std() if len(valid_pauses) > 1 else 0
        else:
            avg_pause = 0
            std_pause = 0
    else:
        avg_pause = 0
        std_pause = 0
    
    # Replace NaN with 0 for consistency
    avg_pause = 0 if pd.isna(avg_pause) else avg_pause
    std_pause = 0 if pd.isna(std_pause) else std_pause
    
    group_active = group[~pause_mask].copy()
    
    pause_features = {
        "pause_ratio": pause_ratio,
        "avg_pause": avg_pause,
        "std_pause": std_pause
    }
    
    # Validate features before returning
    for k, v in pause_features.items():
        if pd.isna(v):
            print(f"Warning: NaN detected in {k}")
            pause_features[k] = 0
    
    return pause_features, group_active

def extract_keystroke_features(group):
    """Enhanced feature extraction with most important metrics."""
    timing_columns = ["DU.key1.key1", "DD.key1.key2", "UD.key1.key2", "UU.key1.key2"]
    for col in timing_columns:
        group[col] = pd.to_numeric(group[col], errors="coerce")
    
    group = group.dropna(subset=["DU.key1.key1", "DD.key1.key2"])
    if len(group) < 3:
        return {}
    
    features = {}
    
    # Most important statistics for each timing column
    for col in timing_columns:
        if col in group.columns:
            values = group[col].values
            features.update({
                f"avg_{col}": np.mean(values),
                f"std_{col}": np.std(values),
                f"med_{col}": np.median(values),
                f"iqr_{col}": np.percentile(values, 75) - np.percentile(values, 25)
            })
    
    # Add rhythm features
    if "DD.key1.key2" in group.columns:
        diffs = np.diff(group["DD.key1.key2"].values)
        features.update({
            "rhythm_mean": np.mean(diffs),
            "rhythm_std": np.std(diffs)
        })
    
    return features

### ----------------------- PCA Function -----------------------

def apply_pca(features_matrix, n_components=0.95):
    """
    Apply PCA to reduce feature dimensionality while retaining specified variance.
    
    Args:
        features_matrix: Standardized feature matrix
        n_components: Either float (0-1) for variance retention or int for specific number of components
    
    Returns:
        transformed_features: PCA-transformed features
        pca: Fitted PCA object
    """
    pca = PCA(n_components=n_components)
    transformed_features = pca.fit_transform(features_matrix)
    
    # Print variance explanation analysis
    print("\n=== PCA Analysis ===")
    print(f"Number of components: {pca.n_components_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.4f}")
    print("\nComponent-wise explained variance ratio:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"Component {i+1}: {ratio:.4f}")
    
    return transformed_features, pca

### ----------------------- Standardization Function -----------------------

def standardize_features(features_by_session):
    """
    Convert the features dictionary into a DataFrame, standardize numeric features,
    and return the transformed DataFrame with original feature names.
    """
    # Convert the dictionary to a DataFrame (session as index)
    session_list = []
    for session_key, feats in features_by_session.items():
        row = {"session": session_key}
        row.update(feats)
        session_list.append(row)
    df_features = pd.DataFrame(session_list)
    
    # Save raw features before standardization
    print("\nSaving raw features before standardization...")
    df_features.to_csv('raw_features_before_std.csv', index=False)
    print("Raw features saved to 'raw_features_before_std.csv'")

    # Save detailed feature information
    with open('feature_analysis.txt', 'w') as f:
        f.write("=== Pre-Standardization Feature Analysis ===\n\n")
        f.write("Basic Statistics:\n")
        f.write(df_features.describe().to_string())
        f.write("\n\nFeature Value Ranges:\n")
        for col in df_features.columns:
            if col != 'session':
                f.write(f"\n{col}:\n")
                f.write(f"  Min: {df_features[col].min()}\n")
                f.write(f"  Max: {df_features[col].max()}\n")
                f.write(f"  Mean: {df_features[col].mean()}\n")
                f.write(f"  Std: {df_features[col].std()}\n")
                f.write(f"  NaN count: {df_features[col].isna().sum()}\n")

    df_features.set_index("session", inplace=True)
    
    # Select numeric columns to scale
    numeric_cols = df_features.columns
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_features[numeric_cols])
    
    # Create DataFrame with scaled features (before PCA)
    df_scaled = pd.DataFrame(
        scaled_features,
        index=df_features.index,
        columns=numeric_cols  # Keep original feature names for scaled data
    )
    
    # Save scaled features with original names
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
    df = load_and_clean_data("FreeDB2.csv")
    df = validate_data(df)
    
    # Extract features per session using per-user percentile-based thresholds
    features = extract_features_for_session(df)
    
    # Standardize the features
    df_scaled = standardize_features(features)
    print("\nFirst 5 Standardized Feature Vectors:")
    print(df_scaled.head(5))
    df_scaled.to_csv("features.csv")
    print("\nStandardized features saved to features.csv")
    
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
