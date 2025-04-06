import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_custom_objects
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
import seaborn as sns
import matplotlib.pyplot as plt

# Local imports from your siamese module (assumed to be already working)
from siamese import (
    create_base_network, 
    create_head_model, 
    SiameseNetwork
)

### ----------------------- Load Precomputed Feature Vectors -----------------------

def load_feature_vectors(csv_file="features.csv"):
    """
    Load the precomputed standardized session feature vectors from a CSV file.
    Assumes the CSV has a column "session" that is set as the index.
    """
    df = pd.read_csv(csv_file, index_col="session")
    return df

def split_users(df_features, train_ratio=0.7):
    """
    Split users randomly into training and testing sets based on train_ratio
    """
    # Get all unique users
    users = sorted(list(set(k.split('_s')[0] for k in df_features.index)))
    
    # Randomly shuffle users
    np.random.seed(42)  # for reproducibility
    np.random.shuffle(users)
    
    # Split users into train and test sets
    n_train = int(len(users) * train_ratio)
    train_users = users[:n_train]
    test_users = users[n_train:]
    
    # Separate sessions based on the train/test users
    train_sessions = [k for k in df_features.index if k.split('_s')[0] in train_users]
    test_sessions = [k for k in df_features.index if k.split('_s')[0] in test_users]
    
    print(f"Training users: {len(train_users)} users")
    print(f"Testing users: {len(test_users)} users")
    print(f"Training sessions: {len(train_sessions)}")
    print(f"Testing sessions: {len(test_sessions)}")
    
    return train_sessions, test_sessions

def check_vector_validity(vec1, vec2, sess1, sess2):
    """Check vectors for NaN or infinite values"""
    if np.any(np.isnan(vec1)) or np.any(np.isinf(vec1)):
        print(f"Warning: Invalid values in vector from session {sess1}")
        print(f"NaNs: {np.isnan(vec1).sum()}, Infs: {np.isinf(vec1).sum()}")
        return False
    if np.any(np.isnan(vec2)) or np.any(np.isinf(vec2)):
        print(f"Warning: Invalid values in vector from session {sess2}")
        print(f"NaNs: {np.isnan(vec2).sum()}, Infs: {np.isinf(vec2).sum()}")
        return False
    return True

def create_balanced_pairs(df_features, train_sessions, test_sessions, negative_ratio=3):
    """
    Create pairs with negative_ratio times more negative pairs than positive pairs
    This is the main pair creation function we'll use.
    """
    # Clean the features DataFrame first
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    df_features = df_features.fillna(df_features.mean())
    
    features_by_session = df_features.to_dict(orient="index")
    train_pairs = []
    test_pairs = []
    train_labels = []
    test_labels = []

    def generate_pairs(sessions, pairs_list, labels_list):
        positive_pairs = []
        negative_pairs = []
        
        # Group sessions by participant
        sessions_by_participant = {}
        for session in sessions:
            participant = session.split('_s')[0]
            if participant not in sessions_by_participant:
                sessions_by_participant[participant] = []
            sessions_by_participant[participant].append(session)
        
        # Generate positive pairs (same participant)
        for participant, participant_sessions in sessions_by_participant.items():
            for i in range(len(participant_sessions)):
                for j in range(i + 1, len(participant_sessions)):
                    sess1 = participant_sessions[i]
                    sess2 = participant_sessions[j]
                    vec1 = np.array(list(features_by_session[sess1].values()), dtype=np.float32)
                    vec2 = np.array(list(features_by_session[sess2].values()), dtype=np.float32)
                    if check_vector_validity(vec1, vec2, sess1, sess2):
                        positive_pairs.append(((vec1, vec2, sess1, sess2), 1))
        
        # Generate negative pairs (different participants)
        participants = list(sessions_by_participant.keys())
        for i in range(len(participants)):
            for j in range(i + 1, len(participants)):
                participant1_sessions = sessions_by_participant[participants[i]]
                participant2_sessions = sessions_by_participant[participants[j]]
                
                # Generate more negative pairs
                for sess1 in participant1_sessions:
                    for sess2 in participant2_sessions:
                        vec1 = np.array(list(features_by_session[sess1].values()), dtype=np.float32)
                        vec2 = np.array(list(features_by_session[sess2].values()), dtype=np.float32)
                        if check_vector_validity(vec1, vec2, sess1, sess2):
                            negative_pairs.append(((vec1, vec2, sess1, sess2), 0))
        
        # Keep all positive pairs and sample negative pairs to maintain ratio
        num_positives = len(positive_pairs)
        num_negatives = min(len(negative_pairs), num_positives * negative_ratio)
        
        np.random.shuffle(negative_pairs)
        negative_pairs = negative_pairs[:num_negatives]
        
        # Combine and shuffle
        balanced_pairs = positive_pairs + negative_pairs
        np.random.shuffle(balanced_pairs)
        
        # Separate pairs and labels
        for pair, label in balanced_pairs:
            pairs_list.append(pair)
            labels_list.append(label)

    # Generate pairs for training and testing
    generate_pairs(train_sessions, train_pairs, train_labels)
    generate_pairs(test_sessions, test_pairs, test_labels)

    return train_pairs, train_labels, test_pairs, test_labels

### ----------------------- Data Quality Check -----------------------

def check_data_quality(df_features):
    """Check input data quality and feature distributions"""
    print("\n=== Data Quality Report ===")
    
    # Basic info
    print("\nDataset Info:")
    print(f"Number of sessions: {len(df_features)}")
    print(f"Number of features: {df_features.shape[1]}")
    
    # Check for missing values
    missing = df_features.isnull().sum()
    if missing.sum() > 0:
        print("\nWarning: Missing values detected!")
        print(missing[missing > 0])
    else:
        print("\nNo missing values found.")
    
    # Check feature ranges
    print("\nFeature Statistics:")
    stats = df_features.describe()
    print("Min values:\n", stats.loc['min'].min())
    print("Max values:\n", stats.loc['max'].max())
    print("Mean values range:", stats.loc['mean'].min(), "to", stats.loc['mean'].max())
    
    # Check for constant or near-constant features
    variance = df_features.var()
    low_variance_features = variance[variance < 1e-5]
    if len(low_variance_features) > 0:
        print("\nWarning: Low variance features detected:")
        print(low_variance_features.index.tolist())
    
    # Check correlations
    corr_matrix = df_features.corr()
    high_corr = np.where(np.abs(corr_matrix) > 0.95)
    high_corr = [(corr_matrix.index[x], corr_matrix.columns[y], corr_matrix.iloc[x, y])
                 for x, y in zip(*high_corr) if x != y]
    
    if high_corr:
        print("\nWarning: Highly correlated features (>0.95):")
        for feat1, feat2, corr in high_corr[:5]:  # Show first 5 pairs
            print(f"{feat1} - {feat2}: {corr:.3f}")
    
    return True

### ----------------------- Loss Function -----------------------

def get_loss_function():
    """
    Returns the loss function to be used in model compilation.
    In this case, we use binary crossentropy.
    """
    return tf.keras.losses.BinaryCrossentropy()

def modified_bce_loss(y_true, y_pred):
    """Binary crossentropy with increased penalty for wrong predictions"""
    margin = 0.5
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    
    # Increase loss for predictions in wrong direction
    loss = -(y_true * tf.math.log(y_pred) + 
             (1 - y_true) * tf.math.log(1 - y_pred))
    
    # Add margin penalty
    margin_loss = y_true * tf.maximum(0., margin - y_pred) + \
                 (1 - y_true) * tf.maximum(0., y_pred - (1 - margin))
    
    return tf.reduce_mean(loss + margin_loss)

### ----------------------- Pair Validation and Analysis -----------------------

def validate_pairs(pairs, labels, pair_type="Training"):
    """Validate pair consistency and print statistics"""
    print(f"\n=== {pair_type} Pair Validation ===")
    
    # Check pair structure
    n_pairs = len(pairs)
    n_positive = sum(labels)
    n_negative = len(labels) - n_positive
    
    print(f"Total pairs: {n_pairs}")
    print(f"Positive pairs: {n_positive}")
    print(f"Negative pairs: {n_negative}")
    print(f"Positive/Negative ratio: {n_positive/n_negative:.2f}")
    
    # Validate pair consistency
    for i, (vec1, vec2, sess1, sess2) in enumerate(pairs):
        # Check vector shapes and values
        if not (vec1.shape == vec2.shape):
            print(f"Warning: Shape mismatch in pair {i}: {vec1.shape} vs {vec2.shape}")
        
        # Verify participant consistency with label
        p1 = sess1.split('_s')[0]
        p2 = sess2.split('_s')[0]
        expected_label = 1 if p1 == p2 else 0
        if expected_label != labels[i]:
            print(f"Warning: Label mismatch in pair {i}: {sess1} vs {sess2}, label={labels[i]}")
            
    return True

def analyze_pair_distances(pairs, labels, pair_type="Training"):
    """Analyze and visualize distances between pairs"""
    distances = []
    pair_types = []
    valid_pairs = []
    valid_labels = []
    
    for (vec1, vec2, sess1, sess2), label in zip(pairs, labels):
        if check_vector_validity(vec1, vec2, sess1, sess2):
            dist = euclidean(vec1, vec2)
            distances.append(dist)
            pair_types.append("Same Person" if label == 1 else "Different People")
            valid_pairs.append((vec1, vec2, sess1, sess2))
            valid_labels.append(label)
    
    if len(distances) == 0:
        print(f"Warning: No valid pairs found in {pair_type} set!")
        return []
    
    # Create distance distribution plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=pair_types, y=distances)
    plt.title(f"{pair_type} Pair Distances Distribution")
    plt.ylabel("Euclidean Distance")
    plt.savefig(f"{pair_type.lower()}_distances.png")
    plt.close()
    
    print(f"\nValid pairs: {len(valid_pairs)} out of {len(pairs)} total pairs")
    return distances

def check_feature_scaling(X1, X2, name="Training"):
    """Check if features are properly scaled"""
    print(f"\n=== {name} Feature Scaling Check ===")
    
    combined = np.vstack([X1, X2])
    means = np.mean(combined, axis=0)
    stds = np.std(combined, axis=0)
    
    print(f"Feature means range: {means.min():.3f} to {means.max():.3f}")
    print(f"Feature stds range: {stds.min():.3f} to {stds.max():.3f}")
    
    # Check for extreme values
    percentiles = np.percentile(combined, [1, 99])
    n_extreme = np.sum(np.logical_or(combined < percentiles[0], combined > percentiles[1]))
    if n_extreme > 0:
        print(f"Warning: {n_extreme} values outside 1-99 percentile range")
    
    return True

### ----------------------- Training Function -----------------------

def train_model(X1_train, X2_train, Y_train, X1_val, X2_val, Y_val, input_shape, 
                loss_type='binary_crossentropy', similarity_metric='cosine',
                epochs=30, batch_size=32, lr=0.0005):
    """
    Enhanced training function with better regularization and architecture
    """
    # Create base network with more complexity
    base_model = create_base_network(
        input_shape,
        hidden_layers=[128, 64, 32],
        dropout_rate=0.3,
        l2_reg=0.01
    )
    
    # Create head model with enhanced similarity computation
    head_model = create_head_model(
        base_model.output_shape,
        similarity_metric='cosine'  # Keep this simple for now
    )
    
    siamese_network = SiameseNetwork(base_model, head_model, loss_type)
    
    # Add learning rate scheduling
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=1000,
        decay_rate=0.9
    )
    
    # Compile with enhanced metrics
    siamese_network.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    # Add callbacks for better training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]
    
    # Train with callbacks
    history = siamese_network.fit(
        [X1_train, X2_train],
        Y_train,
        batch_size=batch_size,
        validation_data=([X1_val, X2_val], Y_val),
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return siamese_network, history

def calibrate_predictions(model, X1_val, X2_val, Y_val):
    """Calibrate model predictions using validation set"""
    from sklearn.isotonic import IsotonicRegression
    
    raw_predictions = model.predict(x1=X1_val, x2=X2_val)  # Fixed prediction call
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(raw_predictions.ravel(), Y_val)
    
    # Add function to calibrate new predictions
    def calibrated_predict(x1, x2):
        raw_pred = model.predict(x1=x1, x2=x2)
        return calibrator.predict(raw_pred.ravel()).reshape(-1, 1)
    
    model.calibrated_predict = calibrated_predict  # Attach to model
    
    return calibrator

### ----------------------- Main Training Script -----------------------
if __name__ == "__main__":
    # Step 1: Load precomputed standardized features from CSV
    print("Loading precomputed feature vectors from features.csv...")
    df_features = load_feature_vectors("features.csv")
    
    # Add feature validation after loading
    print("\nChecking for invalid values in features...")
    nan_check = df_features.isna().sum()
    inf_check = np.isinf(df_features).sum()
    
    if nan_check.sum() > 0 or inf_check.sum() > 0:
        print("Warning: Found invalid values in features:")
        print("\nNaN counts per feature:")
        print(nan_check[nan_check > 0])
        print("\nInf counts per feature:")
        print(inf_check[inf_check > 0])
        
        # Clean the features
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        df_features = df_features.fillna(df_features.mean())
        print("\nFeatures cleaned by replacing inf with NaN and filling NaN with mean values")
    
    # Check data quality
    check_data_quality(df_features)
    
    # Step 2: Split users into training and testing sessions
    train_sessions, test_sessions = split_users(df_features)
    
    # Step 3: Generate balanced pairs for training
    train_pairs, train_labels, test_pairs, test_labels = create_balanced_pairs(df_features, train_sessions, test_sessions)
    
    # Save pairs to CSV files
    train_pairs_df = pd.DataFrame([
        {
            'session1': pair[2],
            'session2': pair[3],
            'label': label,
            'participant1': pair[2].split('_s')[0],
            'participant2': pair[3].split('_s')[0]
        }
        for pair, label in zip(train_pairs, train_labels)
    ])
    train_pairs_df.to_csv('train_pairs.csv', index=False)

    test_pairs_df = pd.DataFrame([
        {
            'session1': pair[2],
            'session2': pair[3],
            'label': label,
            'participant1': pair[2].split('_s')[0],
            'participant2': pair[3].split('_s')[0]
        }
        for pair, label in zip(test_pairs, test_labels)
    ])
    test_pairs_df.to_csv('test_pairs.csv', index=False)

    print(f"\nSaved {len(train_pairs_df)} training pairs to train_pairs.csv")
    print(f"Saved {len(test_pairs_df)} testing pairs to test_pairs.csv")
    
    # Print detailed vectors for a few sample pairs
    print("\nDetailed Sample Pairs:")
    num_samples = 3  # Number of pairs to show
    for i in range(num_samples):
        vec1, vec2, sess1, sess2 = train_pairs[i]
        label = train_labels[i]
        print(f"\nPair {i+1}:")
        print(f"Session 1 ({sess1}): {vec1}")
        print(f"Session 2 ({sess2}): {vec2}")
        print(f"Label: {label}")
        print("-" * 80)
    
    # Prepare arrays for training (extract only the vector parts)
    X1_train = np.array([pair[0] for pair in train_pairs], dtype=np.float32)
    X2_train = np.array([pair[1] for pair in train_pairs], dtype=np.float32)
    Y_train = np.array(train_labels, dtype=np.float32)

    X1_test = np.array([pair[0] for pair in test_pairs], dtype=np.float32)
    X2_test = np.array([pair[1] for pair in test_pairs], dtype=np.float32)
    Y_test = np.array(test_labels, dtype=np.float32)
    
    # Validate pairs before training
    validate_pairs(train_pairs, train_labels, "Training")
    validate_pairs(test_pairs, test_labels, "Testing")
    
    # Analyze pair distances
    train_distances = analyze_pair_distances(train_pairs, train_labels, "Training")
    test_distances = analyze_pair_distances(test_pairs, test_labels, "Testing")
    
    # Check feature scaling
    check_feature_scaling(X1_train, X2_train, "Training")
    check_feature_scaling(X1_test, X2_test, "Testing")
    
    # Step 4: Split data into training and validation sets
    X1_train, X1_val, X2_train, X2_val, Y_train, Y_val = train_test_split(
        X1_train, X2_train, Y_train, test_size=0.2, random_state=42
    )
    
    # Step 5: Define model input shape based on feature vector size
    input_shape = (df_features.shape[1],)
    
    # Step 6: Train the Siamese network with modified loss function
    print("Training the Siamese network...")
    model, history = train_model(
        X1_train, X2_train, Y_train,
        X1_val, X2_val, Y_val,
        input_shape,
        loss_type=modified_bce_loss,  # Use the modified loss function
        similarity_metric='cosine',
        epochs=30,
        batch_size=32,
        lr=0.0005
    )
    
    # Evaluate model performance on the validation set
    print("\nEvaluating model performance...")
    eval_results = model.evaluate([X1_val, X2_val], Y_val, verbose=1)
    
    # Handle multiple metrics
    val_loss = eval_results[0]
    val_accuracy = eval_results[1]
    val_auc = eval_results[2] if len(eval_results) > 2 else None
    
    print(f"Final validation loss: {val_loss:.4f}")
    print(f"Final validation accuracy: {val_accuracy:.4f}")
    if val_auc is not None:
        print(f"Final validation AUC: {val_auc:.4f}")
    
    # Calibrate predictions
    print("\nCalibrating predictions...")
    calibrator = calibrate_predictions(model, X1_val, X2_val, Y_val)
    print("Calibration complete. Use the calibrator to adjust predictions.")
    
    # Save the trained model
    print("Saving the trained model...")
    model.siamese_model.save("models/siamese_model.keras")

    # Testing model predictions
    print("\nTesting model predictions (with calibration)...")
    
    # Test with identical feature vectors (should give high similarity)
    test_vec = X1_train[0]  # Take first feature vector
    identical_prediction = model.calibrated_predict(
        x1=np.array([test_vec]), 
        x2=np.array([test_vec])
    )
    print(f"\nCalibrated prediction for identical vectors: {identical_prediction[0][0]:.4f}")

    # Test with different participants (should give low similarity)
    # Find a negative pair from test set
    for i, label in enumerate(test_labels):
        if label == 0:  # Found a negative pair
            neg_vec1 = test_pairs[i][0]
            neg_vec2 = test_pairs[i][1]
            neg_sess1 = test_pairs[i][2]
            neg_sess2 = test_pairs[i][3]
            break
    
    different_prediction = model.calibrated_predict(
        x1=np.array([neg_vec1]), 
        x2=np.array([neg_vec2])
    )
    print(f"Calibrated prediction for different participants ({neg_sess1} vs {neg_sess2}): {different_prediction[0][0]:.4f}")

    # Test with more different participants
    count = 0
    for i, label in enumerate(test_labels):
        if label == 0:  # Found a negative pair
            neg_vec1 = test_pairs[i][0]
            neg_vec2 = test_pairs[i][1]
            neg_sess1 = test_pairs[i][2]
            neg_sess2 = test_pairs[i][3]
            different_prediction = model.calibrated_predict(
                x1=np.array([neg_vec1]), 
                x2=np.array([neg_vec2])
            )
            print(f"Calibrated prediction for different participants ({neg_sess1} vs {neg_sess2}): {different_prediction[0][0]:.4f}")
            count += 1
            if count >= 5:  # Test for 5 different pairs
                break

    # Test identical vectors for different users
    print("\nTesting identical vectors for multiple users:")
    for i in range(6):  # Test first 6 users
        test_vec = X1_train[i]
        pred = model.calibrated_predict(
            x1=np.array([test_vec]), 
            x2=np.array([test_vec])
        )
        print(f"User {i} self-similarity: {pred[0][0]:.4f} (should be close to 1)")
    
    # Test different participant pairs
    print("\nTesting different participant pairs:")
    count = 0
    for i, label in enumerate(test_labels):
        if label == 0:  # Found a negative pair
            vec1 = test_pairs[i][0]
            vec2 = test_pairs[i][1]
            sess1 = test_pairs[i][2]
            sess2 = test_pairs[i][3]
            pred = model.calibrated_predict(
                x1=np.array([vec1]), 
                x2=np.array([vec2])
            )
            print(f"{sess1} vs {sess2}: {pred[0][0]:.4f} (should be close to 0)")
            count += 1
            if count >= 5:  # Test 5 different pairs
                break

    # Additional test with same participant, different sessions
    print("\nTesting same participant, different sessions:")
    session_groups = {}
    for pair, label in zip(test_pairs, test_labels):
        participant = pair[2].split('_s')[0]
        if participant not in session_groups:
            session_groups[participant] = []
        session_groups[participant].append(pair[2])
    
    for participant, sessions in session_groups.items():
        if len(sessions) >= 2:
            # Get vectors for first two sessions
            vec1 = np.array([pair[0] for pair in test_pairs if pair[2] == sessions[0]][0])
            vec2 = np.array([pair[0] for pair in test_pairs if pair[2] == sessions[1]][0])
            pred = model.calibrated_predict(
                x1=np.array([vec1]), 
                x2=np.array([vec2])
            )
            print(f"{sessions[0]} vs {sessions[1]}: {pred[0][0]:.4f} (should be close to 1)")
            if len(sessions) > 5:  # Test first 5 participants only
                break


