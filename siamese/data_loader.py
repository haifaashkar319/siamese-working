import pandas as pd
import numpy as np
from collections import Counter, defaultdict

import os
file_path = os.path.join(os.path.dirname(__file__), "FreeDB.csv")
df = pd.read_csv(file_path, low_memory=False)  # Do NOT use dtype={"DU.key1.key1": "float"}
df["DU.key1.key1"] = pd.to_numeric(df["DU.key1.key1"], errors="coerce")
df = df.dropna(subset=["DU.key1.key1"])
df.columns = df.columns.str.strip()

print(df.columns)
print("HEYYYY")

# Ensure data is sorted by participant and session
df = df.sort_values(by=["participant", "session"])

def extract_features_from_csv(df, split_factor=2):
    """
    Extracts advanced keystroke dynamics features for behavioral authentication.
    Splits each session into smaller mini-sessions.
    
    Parameters:
    df (pd.DataFrame): Keystroke data with columns like 'DU.key1.key1', 'DD.key1.key2', etc.
    split_factor (int): Number of parts to split each session into.

    Returns:
    dict: A dictionary where keys are session identifiers and values are feature vectors.
    """
    if df.empty:
        print("Dataset is empty. No features extracted.")
        return {}

    formatted_data = {}

    # Ensure data is sorted by participant and session
    df = df.sort_values(by=["participant", "session"])

    for (participant, session), group in df.groupby(["participant", "session"]):
        split_size = len(group) // split_factor

        # Split session into mini-sessions
        for split_index in range(split_factor):
            start_idx = split_index * split_size
            end_idx = start_idx + split_size
            mini_df = group.iloc[start_idx:end_idx]

            if len(mini_df) < 2:
                continue  # Skip empty sessions

            # Extract keystroke timing features
            dwell_times = mini_df["DU.key1.key1"].dropna().tolist()
            flight_times = mini_df["DD.key1.key2"].dropna().tolist()
            latencies = mini_df["UD.key1.key2"].dropna().tolist()
            uu_timings = mini_df["UU.key1.key2"].dropna().tolist()

            # Compute statistical features
            feature_vector = {
                "avg_dwell_time": np.mean(dwell_times) if dwell_times else 0,
                "std_dwell_time": np.std(dwell_times) if dwell_times else 0,
                "avg_flight_time": np.mean(flight_times) if flight_times else 0,
                "std_flight_time": np.std(flight_times) if flight_times else 0,
                "avg_latency": np.mean(latencies) if latencies else 0,
                "std_latency": np.std(latencies) if latencies else 0,
                "avg_UU_time": np.mean(uu_timings) if uu_timings else 0,
                "std_UU_time": np.std(uu_timings) if uu_timings else 0,
            }

            # Compute Typing Speed
            total_chars = len(mini_df)
            session_duration = mini_df.iloc[-1]["DU.key1.key1"] if len(mini_df) > 1 else 1
            words_typed = total_chars / 5  # Approximate word length

            feature_vector["characters_per_second"] = total_chars / session_duration if session_duration > 0 else 0
            feature_vector["words_per_minute"] = (words_typed / session_duration) * 60 if session_duration > 0 else 0

            # Compute Key Frequency Variability
            key_counts = Counter(mini_df["key1"])  # Count occurrences of each key
            key_freq_std = np.std(list(key_counts.values())) if key_counts else 0
            feature_vector["key_frequency_std"] = key_freq_std

            # Compute Digraph and Trigraph Latencies
            digraph_latencies = []
            trigraph_latencies = []

            for i in range(len(mini_df) - 1):
                digraph_latencies.append(mini_df.iloc[i + 1]["DD.key1.key2"])
                if i < len(mini_df) - 2:
                    trigraph_latencies.append(mini_df.iloc[i + 2]["DD.key1.key2"])

            feature_vector["avg_digraph_latency"] = np.mean(digraph_latencies) if digraph_latencies else 0
            feature_vector["std_digraph_latency"] = np.std(digraph_latencies) if digraph_latencies else 0
            feature_vector["avg_trigraph_latency"] = np.mean(trigraph_latencies) if trigraph_latencies else 0
            feature_vector["std_trigraph_latency"] = np.std(trigraph_latencies) if trigraph_latencies else 0

            # Store mini-session as a new session
            session_key = f"{participant}_session{session}_part{split_index+1}"
            formatted_data[session_key] = feature_vector

    return formatted_data

# Convert CSV data to feature vectors
keystroke_features = extract_features_from_csv(df)

# Convert into NumPy arrays for training
def create_pairs(data_dict):
    """
    Generates positive (same user) and negative (different user) pairs for training a Siamese network.
    
    Parameters:
    data_dict (dict): A dictionary where keys are session identifiers and values are feature vectors.

    Returns:
    np.array: Feature vector pairs (X_train).
    np.array: Labels (Y_train).
    """
    pairs = []
    labels = []
    user_ids = list(data_dict.keys())

    for user in user_ids:
        for i in range(len(user_ids) - 1):
            # Convert dictionary features to NumPy array
            vec1 = np.array(list(data_dict[user_ids[i]].values()), dtype=np.float32)
            vec2 = np.array(list(data_dict[user_ids[i + 1]].values()), dtype=np.float32)

            # Positive Pair (Same User, Different Session)
            pairs.append((vec1, vec2))
            labels.append(1)

            # Negative Pair (Different User)
            impostor = np.random.choice(user_ids)
            while impostor == user:
                impostor = np.random.choice(user_ids)

            vec_impostor = np.array(list(data_dict[impostor].values()), dtype=np.float32)
            pairs.append((vec1, vec_impostor))
            labels.append(0)

    return np.array(pairs, dtype=np.float32), np.array(labels, dtype=np.float32)

# Create training pairs
X_train, Y_train = create_pairs(keystroke_features)

# Print dataset stats
print(f"Training Pairs: {X_train.shape[0]}")

# Create training pairs
X_train, Y_train = create_pairs(keystroke_features)

# Print dataset stats
print(f"Training Pairs: {X_train.shape[0]}")
print(X_train[0])
