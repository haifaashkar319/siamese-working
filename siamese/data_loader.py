import pandas as pd
import numpy as np
import os

# 🔹 Load dataset
file_path = os.path.join(os.path.dirname(__file__), "FreeDB.csv")
df = pd.read_csv(file_path, low_memory=False)

# ✅ Strip spaces from column names
df.columns = df.columns.str.strip()

print("🔍 Checking missing values in dataset:")
print(df.isna().sum())  # Count missing values per column

# ✅ Ensure participant column exists
if "participant" not in df.columns:
    raise KeyError("❌ 'participant' column is missing from dataset!")

# ✅ Convert all relevant columns to numeric safely
timing_columns = ["DU.key1.key1", "DD.key1.key2", "UD.key1.key2", "UU.key1.key2"]
for col in timing_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert to float, set invalid to NaN

df = df.dropna(subset=["DU.key1.key1"])  # Drop rows where key timing is missing

print("\n🔍 Column Names in Dataset After Cleanup:", df.columns.tolist())  # DEBUG LINE

print("\n🔍 Dataset Loaded Successfully. Now Processing Feature Extraction...")
print("HEYYYY")  # Keeps track of when extraction begins

# 🔹 Ensure sorting by participant
df = df.sort_values(by=["participant"])

# ✅ Print column names BEFORE dropping session
print("🔍 Column Names in DataFrame Before Grouping:", df.columns.tolist()) 

# 🔥 **MERGE RAW DATA BY USER BEFORE FEATURE EXTRACTION**
df = df.drop(columns=["session"], errors="ignore")

# ✅ Ensure participant column wasn't dropped accidentally
if "participant" not in df.columns:
    raise KeyError("❌ ERROR: 'participant' column is missing after preprocessing!")

def extract_features_from_csv(df):
    """Extracts keystroke features for authentication, ensuring only numeric columns are used."""
    if df.empty:
        print("⚠️ Dataset is empty. No features extracted.")
        return {}

    formatted_data = {}
    total_participants = len(df["participant"].unique())
    processed = 0
    feature_dim = None  # Track feature dimensionality

    for participant, group in df.groupby("participant", dropna=False):
        processed += 1
        print(f"Processing participant {processed}/{total_participants}: {participant}")

        # ✅ Convert all timing columns to numeric before computation
        for col in timing_columns:
            if col in group.columns:
                group[col] = pd.to_numeric(group[col], errors="coerce")  # Convert to float, NaN if error

        # ✅ Print row count before and after dropping missing values
        print(f"🔹 {participant} Before Drop: {len(group)} rows")
        group = group.dropna(subset=["DU.key1.key1", "DD.key1.key2"])
        print(f"🔹 {participant} After Drop: {len(group)} rows")

        # Skip participants with too few keystrokes
        if len(group) < 3:
            print(f"⚠️ Skipping {participant}: not enough valid keystrokes (found {len(group)})")
            continue

        # Extract valid timing features
        dwell_times = group["DU.key1.key1"].dropna().tolist()
        flight_times = group["DD.key1.key2"].dropna().tolist()
        latencies = group["UD.key1.key2"].dropna().tolist() if "UD.key1.key2" in group.columns else []
        uu_timings = group["UU.key1.key2"].dropna().tolist() if "UU.key1.key2" in group.columns else []

        # Extract digraphs and trigraphs safely
        digraph_latencies = []
        trigraph_latencies = []
        
        for i in range(len(group) - 1):
            try:
                if i + 1 < len(group) and not pd.isna(group.iloc[i + 1]["DD.key1.key2"]):
                    digraph_latencies.append(group.iloc[i + 1]["DD.key1.key2"])
            except IndexError:
                continue

        for i in range(len(group) - 2):
            try:
                if i + 2 < len(group) and not pd.isna(group.iloc[i + 2]["DD.key1.key2"]):
                    trigraph_latencies.append(group.iloc[i + 2]["DD.key1.key2"])
            except IndexError:
                continue

        # Compute statistics
        feature_vector = {
            "avg_dwell_time": np.nan_to_num(np.mean(dwell_times), nan=0),
            "std_dwell_time": np.nan_to_num(np.std(dwell_times), nan=0),
            "avg_flight_time": np.nan_to_num(np.mean(flight_times), nan=0),
            "std_flight_time": np.nan_to_num(np.std(flight_times), nan=0),
        }
        
        if latencies:
            feature_vector["avg_latency"] = np.nan_to_num(np.mean(latencies), nan=0)
            feature_vector["std_latency"] = np.nan_to_num(np.std(latencies), nan=0)
        if uu_timings:
            feature_vector["avg_UU_time"] = np.nan_to_num(np.mean(uu_timings), nan=0)
            feature_vector["std_UU_time"] = np.nan_to_num(np.std(uu_timings), nan=0)

        if feature_dim is None:
            feature_dim = len(feature_vector)
        elif feature_dim != len(feature_vector):
            print(f"⚠️ WARNING: Inconsistent feature dimensions! Previous: {feature_dim}, Current: {len(feature_vector)}")

        formatted_data[participant] = feature_vector

    print(f"✅ Final feature dimension: {feature_dim}")
    return formatted_data

# 🔹 Extract features
keystroke_features = extract_features_from_csv(df)


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


# 🔹 Extract features & create training data at the module level
keystroke_features = extract_features_from_csv(df)
X_train, Y_train = create_pairs(keystroke_features)

if __name__ == "__main__":
    # ✅ Save extracted features (embeddings)

    # ✅ Print dataset statistics
    print(f"Training Pairs: {X_train.shape[0]}")
    print(f"🔍 Label Distribution: {np.bincount(Y_train.astype(int))}")
