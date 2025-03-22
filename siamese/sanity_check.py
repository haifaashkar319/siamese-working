import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import tensorflow.keras.backend as K

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

def run_similarity_tests(num_tests=10):
    """Run 10 true tests (same user) and 10 false tests (different users) to evaluate the model."""
    
    print("\n🔎 Running 10 True Tests (Same User - Different Session)")
    true_scores = []
    # For true tests, assume that consecutive rows belong to the same user.
    for i in range(num_tests):
        # Ensure we don't go out-of-range
        index = np.random.randint(0, len(training_features) - 1)
        sample1 = training_features[index].reshape(1, -1)
        sample2 = training_features[index + 1].reshape(1, -1)
        similarity = siamese_model.predict([sample1, sample2], verbose=0)[0][0]
        true_scores.append(similarity)
        print(f"✅ True Test #{i+1} → Session: {sessions[index]} vs {sessions[index+1]} | Similarity Score: {similarity:.3f}")

    print("\n🔎 Running 10 False Tests (Different Users)")
    false_scores = []
    for i in range(num_tests):
        index1, index2 = np.random.choice(len(training_features), 2, replace=False)
        sample1 = training_features[index1].reshape(1, -1)
        sample2 = training_features[index2].reshape(1, -1)
        similarity = siamese_model.predict([sample1, sample2], verbose=0)[0][0]
        false_scores.append(similarity)
        print(f"❌ False Test #{i+1} → Session: {sessions[index1]} vs {sessions[index2]} | Similarity Score: {similarity:.3f}")

    print("\n📊 Similarity Score Summary")
    print(f"Mean True Score (Same User): {np.mean(true_scores):.3f}")
    print(f"Mean False Score (Different Users): {np.mean(false_scores):.3f}")
    print(f"Min/Max True Scores: {min(true_scores):.3f} / {max(true_scores):.3f}")
    print(f"Min/Max False Scores: {min(false_scores):.3f} / {max(false_scores):.3f}")
    suggested_threshold = (np.mean(true_scores) + np.mean(false_scores)) / 2
    print(f"Suggested Threshold: {suggested_threshold:.3f}")

def run_sanity_check():
    """Perform a sanity check by comparing a feature vector with itself."""
    print("\n🔍 Running Sanity Check: Comparing Two Identical Training Samples")
    # Use the first session as sample
    sample_session = sessions[0]
    sample = training_features[0].reshape(1, -1)
    similarity = siamese_model.predict([sample, sample], verbose=0)[0][0]
    print(f"Session: {sample_session} | Similarity Score for Identical Vectors: {similarity:.3f}")
    if similarity < 0.9:
        print("❌ WARNING: The model does not recognize identical training samples!")
    else:
        print("✅ Sanity Check Passed: The model correctly identifies identical training samples.")

if __name__ == "__main__":
    run_similarity_tests()
    run_sanity_check()
