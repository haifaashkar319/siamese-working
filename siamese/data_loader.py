import pandas as pd
import numpy as np

# Load the CSV file
file_path = "free-text (1).csv"
df = pd.read_csv(file_path)

# Ensure data is sorted by participant and session
df = df.sort_values(by=["participant", "session"])

def extract_features_from_csv(df, split_factor=2):
    """
    Extracts keystroke features from the CSV file.
    Splits each session into `split_factor` smaller sessions.
    """
    formatted_data = {}

    for (participant, session), group in df.groupby(["participant", "session"]):
        timestamps = group["Timestamp"].values
        split_size = len(timestamps) // split_factor

        # Create `split_factor` mini-sessions
        for split_index in range(split_factor):
            start_idx = split_index * split_size
            end_idx = start_idx + split_size
            mini_timestamps = timestamps[start_idx:end_idx]

            if len(mini_timestamps) < 2:
                continue  # Skip empty sessions

            flight_times = np.diff(mini_timestamps)  # Compute keystroke timing features
            
            feature_vector = {
            "avg_flight_time": np.mean(flight_times) if len(flight_times) > 0 else 0,
            "std_flight_time": np.std(flight_times) if len(flight_times) > 0 else 0,
            "median_flight_time": np.median(flight_times) if len(flight_times) > 0 else 0,
            "min_flight_time": np.min(flight_times) if len(flight_times) > 0 else 0,
            "max_flight_time": np.max(flight_times) if len(flight_times) > 0 else 0,
            "total_keystrokes": len(timestamps),
            "typing_speed": len(timestamps) / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0
        }
            
            # Store mini-session as a new session
            session_key = f"{participant}_session{session}_part{split_index+1}"
            formatted_data[session_key] = feature_vector

    return formatted_data

# Convert CSV data to feature vectors
keystroke_features = extract_features_from_csv(df)

# Convert into NumPy arrays for training
def create_pairs(data_dict):
    """
    Generates positive (same user) and negative (different user) pairs for training.
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
