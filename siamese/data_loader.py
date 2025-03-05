import pandas as pd
import numpy as np
from collections import Counter, defaultdict

import os
file_path = os.path.join(os.path.dirname(__file__), "FreeDB.csv")
df = pd.read_csv(file_path, low_memory=False)  # Do NOT use dtype={"DU.key1.key1": "float"}

print("🔍 Checking missing values in dataset:")
print(df.isna().sum())  # Count missing values per column

df["DU.key1.key1"] = pd.to_numeric(df["DU.key1.key1"], errors="coerce")
df = df.dropna(subset=["DU.key1.key1"])
df.columns = df.columns.str.strip()

print("\n🔍 Column Names in Dataset:")
print(df.columns.tolist())  # Print all column names

print("\n🔍 Dataset Loaded Successfully. Now Processing Feature Extraction...")
print("HEYYYY")  # Keeps track of when extraction begins

# Ensure data is sorted by participant and session
df = df.sort_values(by=["participant", "session"])

def extract_features_from_csv(df, split_factor=2):
    """Extracts keystroke features for authentication, ensuring no NaN values."""
    if df.empty:
        print("⚠️ Dataset is empty. No features extracted.")
        return {}

    formatted_data = {}

    for (participant, session), group in df.groupby(["participant", "session"]):
        # Drop last row of each session to remove missing `key2`
        group = group.iloc[:-1] if group.shape[0] > 1 else group

        # Drop any remaining rows where `key2` is NaN
        group = group.dropna(subset=["key2"])

        # print(f"   ➡️ Number of valid keystrokes after cleaning: {group.shape[0]}")

        split_size = len(group) // split_factor

        for split_index in range(split_factor):
            start_idx = split_index * split_size
            end_idx = start_idx + split_size
            mini_df = group.iloc[start_idx:end_idx]

            if len(mini_df) < 2:
                continue  # Skip empty sessions

            dwell_times = mini_df["DU.key1.key1"].dropna().tolist()
            flight_times = mini_df["DD.key1.key2"].dropna().tolist()
            latencies = mini_df["UD.key1.key2"].dropna().tolist()
            uu_timings = mini_df["UU.key1.key2"].dropna().tolist()

            digraph_latencies = []
            trigraph_latencies = []

            for i in range(len(mini_df) - 1):
                digraph_latencies.append(mini_df.iloc[i + 1]["DD.key1.key2"])
                if i < len(mini_df) - 2:
                    trigraph_latencies.append(mini_df.iloc[i + 2]["DD.key1.key2"])

            # Fix NaN values by replacing them with 0
            feature_vector = {
                "avg_dwell_time": np.nan_to_num(np.mean(dwell_times), nan=0),
                "std_dwell_time": np.nan_to_num(np.std(dwell_times), nan=0),
                "avg_flight_time": np.nan_to_num(np.mean(flight_times), nan=0),
                "std_flight_time": np.nan_to_num(np.std(flight_times), nan=0),
                "avg_latency": np.nan_to_num(np.mean(latencies), nan=0),
                "std_latency": np.nan_to_num(np.std(latencies), nan=0),
                "avg_UU_time": np.nan_to_num(np.mean(uu_timings), nan=0),
                "std_UU_time": np.nan_to_num(np.std(uu_timings), nan=0),
                "avg_digraph_latency": np.nan_to_num(np.mean(digraph_latencies), nan=0),
                "std_digraph_latency": np.nan_to_num(np.std(digraph_latencies), nan=0),
                "avg_trigraph_latency": np.nan_to_num(np.mean(trigraph_latencies), nan=0),
                "std_trigraph_latency": np.nan_to_num(np.std(trigraph_latencies), nan=0),
            }

            session_key = f"{participant}_session{session}"
            formatted_data[session_key] = feature_vector

    return formatted_data

# Convert CSV data to feature vectors
keystroke_features = extract_features_from_csv(df)

# 🔍 **Print Feature Correlation Matrix**
df_features = pd.DataFrame.from_dict(keystroke_features, orient="index")
# print("\n🔍 Feature Correlation Matrix:")
# print(df_features.corr())

# Convert into NumPy arrays for training
def create_pairs(data_dict):
    """Creates positive & negative pairs for training a Siamese network."""
    pairs = []
    labels = []
    user_ids = list(data_dict.keys())

    for user in user_ids:
        for i in range(len(user_ids) - 1):
            vec1 = np.array(list(data_dict[user_ids[i]].values()), dtype=np.float32)
            vec2 = np.array(list(data_dict[user_ids[i + 1]].values()), dtype=np.float32)

            pairs.append((vec1, vec2))
            labels.append(1)

            impostor = np.random.choice(user_ids)
            while impostor == user:
                impostor = np.random.choice(user_ids)

            vec_impostor = np.array(list(data_dict[impostor].values()), dtype=np.float32)
            pairs.append((vec1, vec_impostor))
            labels.append(0)

    return np.array(pairs, dtype=np.float32), np.array(labels, dtype=np.float32)

# Create training pairs
X_train, Y_train = create_pairs(keystroke_features)

# 🔍 **Print Label Distribution**
# print("\n🔍 Checking Label Distribution...")
# print(f"🔍 Label Distribution: {np.bincount(Y_train.astype(int))}")

# # Print dataset stats
# print(f"Training Pairs: {X_train.shape[0]}")
# print(X_train[0])  # Example feature pair
