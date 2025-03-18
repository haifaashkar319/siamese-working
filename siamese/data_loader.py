import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

### ----------------------- Data Loading & Preprocessing -----------------------

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

def preprocess_data(df):
    """Convert columns to numeric and apply Min-Max Normalization per user."""
    timing_columns = ["DU.key1.key1", "DD.key1.key2", "UD.key1.key2", "UU.key1.key2"]
    
    for col in timing_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["DU.key1.key1"])  # Ensure required columns have values
    df = df.sort_values(by=["participant", "session"])  
    
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
            
        features = extract_keystroke_features(group)
        session_key = f"{participant}_s{session}"
        features_by_session[session_key] = features

    return features_by_session

def extract_keystroke_features(group):
    """Compute statistical keystroke features for a given participant or session."""
    timing_columns = ["DU.key1.key1", "DD.key1.key2", "UD.key1.key2", "UU.key1.key2"]
    
    for col in timing_columns:
        if col in group.columns:
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

### ----------------------- Creating Training Pairs -----------------------

def create_training_pairs(features_by_session, num_negative_per_positive=5):
    """Create positive and negative training pairs for a Siamese network."""
    pairs = []
    labels = []
    users = list(set([k.split('_s')[0] for k in features_by_session.keys()]))

    total_positive = 0
    total_negative = 0

    for user in users:
        user_sessions = [k for k in features_by_session.keys() if k.startswith(user)]

        for i in range(len(user_sessions)):
            for j in range(i + 1, len(user_sessions)):
                vec1 = np.array(list(features_by_session[user_sessions[i]].values()), dtype=np.float32)
                vec2 = np.array(list(features_by_session[user_sessions[j]].values()), dtype=np.float32)
                
                pairs.append((vec1, vec2))
                labels.append(1)  # Positive pair
                total_positive += 1

                for _ in range(num_negative_per_positive):
                    other_user = np.random.choice([u for u in users if u != user])
                    other_session = np.random.choice([k for k in features_by_session.keys() if k.startswith(other_user)])
                    vec_impostor = np.array(list(features_by_session[other_session].values()), dtype=np.float32)

                    pairs.append((vec1, vec_impostor))
                    labels.append(0)  # Negative pair
                    total_negative += 1

    print(f"\n🔹 Total Positive Pairs: {total_positive}")
    print(f"🔹 Total Negative Pairs: {total_negative} (Each positive has {num_negative_per_positive} negatives)")
    print(f"🔹 Total Training Pairs: {total_positive + total_negative}\n")

    return np.array(pairs), np.array(labels)

### ----------------------- Creating Testing Pairs -----------------------

def create_testing_pairs(features_by_session, test_size=0.2, num_negative_per_positive=5):
    """Create positive and negative testing pairs from unseen users."""
    pairs = []
    labels = []
    
    users = list(set([k.split('_s')[0] for k in features_by_session.keys()]))
    
    train_users, test_users = train_test_split(users, test_size=test_size, random_state=42)

    print(f"🛠 Test Set Users: {len(test_users)} (Unseen during training)")

    total_positive = 0
    total_negative = 0

    for user in test_users:
        user_sessions = [k for k in features_by_session.keys() if k.startswith(user)]

        if len(user_sessions) < 2:
            continue

        for i in range(len(user_sessions)):
            for j in range(i + 1, len(user_sessions)):
                vec1 = np.array(list(features_by_session[user_sessions[i]].values()), dtype=np.float32)
                vec2 = np.array(list(features_by_session[user_sessions[j]].values()), dtype=np.float32)

                pairs.append((vec1, vec2))
                labels.append(1)  # Positive pair
                total_positive += 1

                for _ in range(num_negative_per_positive):
                    other_user = np.random.choice([u for u in test_users if u != user])
                    other_session = np.random.choice([k for k in features_by_session.keys() if k.startswith(other_user)])
                    vec_impostor = np.array(list(features_by_session[other_session].values()), dtype=np.float32)

                    pairs.append((vec1, vec_impostor))
                    labels.append(0)  # Negative pair
                    total_negative += 1

    print(f"\n🔹 Total Positive Test Pairs: {total_positive}")
    print(f"🔹 Total Negative Test Pairs: {total_negative}")
    print(f"🔹 Total Testing Pairs: {total_positive + total_negative}\n")

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

    print("📊 Creating testing pairs...")
    X_test, Y_test = create_testing_pairs(features_by_session)

    print(f"✅ Training Pairs: {len(X_train)}, Testing Pairs: {len(X_test)}")
