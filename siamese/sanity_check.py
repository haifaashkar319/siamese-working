import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import tensorflow.keras.backend as K
from sklearn.metrics import roc_curve

# Define proper l1_distance function
@tf.keras.utils.register_keras_serializable()
def l1_distance(vects):
    """Computes L1 (Manhattan) distance between two vectors."""
    x, y = vects
    return K.abs(x - y)

# Load the model and features
try:
    print("📥 Loading model and training features...")
    siamese_model = load_model(
        "models/siamese_model.keras",
        custom_objects={"l1_distance": l1_distance},
        safe_mode=False
    )
    # Load standardized features from CSV; assumes 'session' is the index.
    df_features = pd.read_csv("features.csv", index_col="session")
    training_features = df_features.values
    sessions = df_features.index.tolist()
    print("✅ Model and training features loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model or training features: {e}")
    exit()

def group_sessions_by_user(sessions):
    """
    Group session IDs by user based on the prefix before the underscore.
    Returns a dictionary mapping user ID to a list of session IDs.
    """
    user_sessions = {}
    for sess in sessions:
        user_id = sess.split("_")[0]
        user_sessions.setdefault(user_id, []).append(sess)
    return user_sessions

# New helper function to convert similarity to percentage
def similarity_to_percentage(similarity, base=0.01):
    """
    Translate raw similarity threshold into intuitive percentage.
    A higher raw similarity yields a higher percentage similarity.
    """
    percentage = 100 * (1 - np.power(base, similarity))
    return percentage

def run_similarity_tests(num_tests=100):
    """
    Run num_tests true tests (same user, different sessions) and
    num_tests false tests (different users) to evaluate the model,
    and compute an optimized threshold using ROC analysis.
    """
    user_sessions = group_sessions_by_user(sessions)
    
    # Filter users with at least 2 sessions for true tests.
    valid_users = [user for user, sess_list in user_sessions.items() if len(sess_list) >= 2]
    
    print(f"\n🔎 Running {num_tests} True Tests (Same User - Different Sessions)")
    true_scores = []
    for i in range(num_tests):
        # Choose a random user from valid ones.
        user = np.random.choice(valid_users)
        # Randomly choose two distinct sessions for that user.
        sess_pair = np.random.choice(user_sessions[user], 2, replace=False)
        sample1 = df_features.loc[sess_pair[0]].values.reshape(1, -1)
        sample2 = df_features.loc[sess_pair[1]].values.reshape(1, -1)
        similarity = siamese_model.predict([sample1, sample2], verbose=0)[0][0]
        true_scores.append(similarity)
        print(f"✅ True Test #{i+1} → Sessions: {sess_pair[0]} vs {sess_pair[1]} | Similarity Score: {similarity:.3f}")
    
    print(f"\n🔎 Running {num_tests} False Tests (Different Users)")
    false_scores = []
    all_users = list(user_sessions.keys())
    for i in range(num_tests):
        # Randomly choose two different users.
        users = np.random.choice(all_users, 2, replace=False)
        # Randomly choose one session for each user.
        sess1 = np.random.choice(user_sessions[users[0]])
        sess2 = np.random.choice(user_sessions[users[1]])
        sample1 = df_features.loc[sess1].values.reshape(1, -1)
        sample2 = df_features.loc[sess2].values.reshape(1, -1)
        similarity = siamese_model.predict([sample1, sample2], verbose=0)[0][0]
        false_scores.append(similarity)
        print(f"❌ False Test #{i+1} → Sessions: {sess1} (User: {users[0]}) vs {sess2} (User: {users[1]}) | Similarity Score: {similarity:.3f}")
    
    # Combine scores and assign labels: 1 for same-user, 0 for different-user.
    all_scores = np.concatenate([true_scores, false_scores])
    y_true = np.array([1]*len(true_scores) + [0]*len(false_scores))
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, all_scores)
    # Find threshold where |FPR - (1 - TPR)| is minimized (approx. Equal Error Rate)
    abs_diffs = np.abs(fpr - (1 - tpr))
    optimal_index = np.nanargmin(abs_diffs)
    optimized_threshold = thresholds[optimal_index]
    
    print("\n📊 Similarity Score Summary")
    print(f"Mean True Score (Same User): {np.mean(true_scores):.3f} "
          f"({similarity_to_percentage(np.mean(true_scores)):.2f}%)")
    print(f"Mean False Score (Different Users): {np.mean(false_scores):.3f} "
          f"({similarity_to_percentage(np.mean(false_scores)):.2f}%)")
    print(f"Min/Max True Scores: {min(true_scores):.3f}/{max(true_scores):.3f} "
          f"({similarity_to_percentage(min(true_scores)):.2f}%/{similarity_to_percentage(max(true_scores)):.2f}%)")
    print(f"Min/Max False Scores: {min(false_scores):.3f}/{max(false_scores):.3f} "
          f"({similarity_to_percentage(min(false_scores)):.2f}%/{similarity_to_percentage(max(false_scores)):.2f}%)")
    print(f"Optimized Threshold (EER): {optimized_threshold:.3f} "
          f"({similarity_to_percentage(optimized_threshold):.2f}%)")
    
    return optimized_threshold

def run_sanity_check(optimized_threshold):
    """
    Perform a manual sanity check by comparing feature vectors from chosen sessions.
    You will be prompted to enter session IDs (separated by commas).
    The code will then compare each consecutive pair, display the similarity score,
    and indicate whether it is above or below the optimized threshold.
    """
    input_ids = input("\nEnter session IDs (comma-separated) for sanity check: ")
    session_ids = [s.strip() for s in input_ids.split(",")]
    
    if len(session_ids) < 2:
        print("❌ Please enter at least 2 session IDs.")
        return

    print("\n🔍 Running Manual Sanity Check on Selected Sessions:")
    for i in range(len(session_ids)-1):
        try:
            sample1 = df_features.loc[session_ids[i]].values.reshape(1, -1)
            sample2 = df_features.loc[session_ids[i+1]].values.reshape(1, -1)
        except KeyError as e:
            print(f"❌ Session ID not found: {e}")
            continue

        similarity = siamese_model.predict([sample1, sample2], verbose=0)[0][0]
        similarity_pct = similarity_to_percentage(similarity)
        threshold_pct = similarity_to_percentage(optimized_threshold)
        decision = "Same User" if similarity >= optimized_threshold else "Different Users"
        print(f"Session: {session_ids[i]} vs {session_ids[i+1]} | "
              f"Similarity: {similarity:.3f} ({similarity_pct:.2f}%) | "
              f"Threshold: {optimized_threshold:.3f} ({threshold_pct:.2f}%) → {decision}")

if __name__ == "__main__":
    # Run tests to compute an optimized threshold
    optimized_threshold = run_similarity_tests(num_tests=100)
    print(f"\nYou may consider using the optimized threshold: {optimized_threshold:.3f}\n")
    # Now perform manual sanity check using user-specified session IDs
    run_sanity_check(optimized_threshold)
