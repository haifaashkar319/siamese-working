import pandas as pd
import numpy as np
import os
from collections import Counter, defaultdict

# 🔹 Load dataset
file_path = os.path.join(os.path.dirname(__file__), "FreeDB.csv")
df = pd.read_csv(file_path, low_memory=False)

print("🔍 Checking missing values in dataset:")
print(df.isna().sum())  # Count missing values per column

df["DU.key1.key1"] = pd.to_numeric(df["DU.key1.key1"], errors="coerce")
df = df.dropna(subset=["DU.key1.key1"])
df.columns = df.columns.str.strip()

print("\n🔍 Column Names in Dataset:")
print(df.columns.tolist())

print("\n🔍 Dataset Loaded Successfully. Now Processing Feature Extraction...")
print("HEYYYY")  # Keeps track of when extraction begins

# 🔹 Ensure sorting by participant
df = df.sort_values(by=["participant"])

# 🔥 **MERGE RAW DATA BY USER BEFORE FEATURE EXTRACTION**
df = df.drop(columns=["session"], errors="ignore")

def extract_features_from_csv(df):
    """Extracts keystroke features for authentication, ensuring no NaN values."""
    if df.empty:
        print("⚠️ Dataset is empty. No features extracted.")
        return {}

    formatted_data = {}

    for participant, group in df.groupby("participant"):
        group = group.dropna(subset=["key2"])  # Remove invalid keystrokes

        dwell_times = group["DU.key1.key1"].dropna().tolist()
        flight_times = group["DD.key1.key2"].dropna().tolist()
        latencies = group["UD.key1.key2"].dropna().tolist()
        uu_timings = group["UU.key1.key2"].dropna().tolist()

        digraph_latencies = [group.iloc[i + 1]["DD.key1.key2"] for i in range(len(group) - 1)]
        trigraph_latencies = [group.iloc[i + 2]["DD.key1.key2"] for i in range(len(group) - 2)]

        # Replace NaNs with 0
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

        formatted_data[participant] = feature_vector  # Store by user, not session

    return formatted_data

# Convert CSV data to feature vectors
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

def save_user_embeddings(keystroke_features):
    """Saves user embeddings to a file."""
    embedding_path = "user_embeddings.npy"

    if os.path.exists(embedding_path):
        existing_embeddings = np.load(embedding_path, allow_pickle=True).item()
    else:
        existing_embeddings = {}

    existing_embeddings.update(keystroke_features)

    np.save(embedding_path, existing_embeddings)
    print("✅ User embeddings saved to `user_embeddings.npy`.")

# 🔹 Extract features & create training data at the module level
keystroke_features = extract_features_from_csv(df)
X_train, Y_train = create_pairs(keystroke_features)

if __name__ == "__main__":
    # ✅ Save extracted features (embeddings)
    save_user_embeddings(keystroke_features)

    # ✅ Print dataset statistics
    print(f"Training Pairs: {X_train.shape[0]}")
    print(f"🔍 Label Distribution: {np.bincount(Y_train.astype(int))}")
