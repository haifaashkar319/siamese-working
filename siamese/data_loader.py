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



# 🔥 **MERGE RAW DATA BY USER BEFORE FEATURE EXTRACTION**
df = df.drop(columns=["session"], errors="ignore")

def extract_features_from_csv(df):
    """Extracts keystroke features while ensuring only numeric columns are used."""
    
    # ✅ Drop categorical / non-numeric columns BEFORE computation
    df = df.drop(columns=["participants", "session", "key1", "key2"], errors="ignore")

    if df.empty:
        print("⚠️ Dataset is empty. No features extracted.")
        return {}

    formatted_data = {}
    total_participants = len(df['participant'].unique())
    processed = 0

    for participant, group in df.groupby("participant", dropna=False):

        # ✅ Select only numeric columns (avoid categorical contamination)
        numeric_columns = ["DU.key1.key1", "DD.key1.key2", "UD.key1.key2", "UU.key1.key2"]
        
        for col in numeric_columns:
            if col not in group.columns:
                print(f"⚠️ Missing column {col} for participant {participant}. Skipping...")
                continue  # Skip if missing
        
        # ✅ Convert to numeric and drop non-numeric values
        for col in numeric_columns:
            group[col] = pd.to_numeric(group[col], errors="coerce")  # Convert to float, set invalid to NaN
            group = group.dropna(subset=[col])  # Drop rows where timing data is missing

        # ✅ Compute features only on numerical values
        feature_vector = {
            "avg_dwell_time": np.mean(group["DU.key1.key1"]) if not group["DU.key1.key1"].empty else 0,
            "std_dwell_time": np.std(group["DU.key1.key1"]) if not group["DU.key1.key1"].empty else 0,
            "avg_flight_time": np.mean(group["DD.key1.key2"]) if not group["DD.key1.key2"].empty else 0,
            "std_flight_time": np.std(group["DD.key1.key2"]) if not group["DD.key1.key2"].empty else 0,
        }

        formatted_data[participant] = feature_vector  # ✅ Store by user only

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


# 🔹 Extract features & create training data at the module level
keystroke_features = extract_features_from_csv(df)
X_train, Y_train = create_pairs(keystroke_features)

if __name__ == "__main__":
    # ✅ Save extracted features (embeddings)

    # ✅ Print dataset statistics
    print(f"Training Pairs: {X_train.shape[0]}")
    print(f"🔍 Label Distribution: {np.bincount(Y_train.astype(int))}")
