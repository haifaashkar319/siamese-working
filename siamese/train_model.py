import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_custom_objects
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split

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

### ----------------------- Training Pair Creation -----------------------

def create_training_pairs_from_df(df_features):
    """
    Create training pairs from the standardized features DataFrame.
    Assumes the index of df_features is of the form "participant_s<session>".
    Returns a tuple (pairs, labels) where each pair is a tuple:
       (feature_vector1, feature_vector2, session1, session2)
    Positive pairs (same participant) are labeled as 1, and negative pairs (different participants) as 0.
    """
    features_by_session = df_features.to_dict(orient="index")
    pairs = []
    labels = []
    
    # Generate positive pairs (same participant)
    users = list(set(k.split('_s')[0] for k in features_by_session.keys()))
    for user in users:
        user_sessions = [k for k in features_by_session.keys() if k.startswith(user)]
        for i in range(len(user_sessions)):
            for j in range(i + 1, len(user_sessions)):
                vec1 = np.array(list(features_by_session[user_sessions[i]].values()), dtype=np.float32)
                vec2 = np.array(list(features_by_session[user_sessions[j]].values()), dtype=np.float32)
                pairs.append((vec1, vec2, user_sessions[i], user_sessions[j]))
                labels.append(1)
    
    # Generate negative pairs (different participants)
    # Here, we take one session per participant (the first session when sorted alphabetically)
    first_sessions = {user: sorted([k for k in features_by_session.keys() if k.startswith(user)])[0]
                      for user in users if len([k for k in features_by_session.keys() if k.startswith(user)]) > 0}
    first_sessions = list(first_sessions.values())
    
    # Create negative pairs from these first sessions
    for i in range(len(first_sessions)):
        for j in range(i+1, len(first_sessions)):
            vec1 = np.array(list(features_by_session[first_sessions[i]].values()), dtype=np.float32)
            vec2 = np.array(list(features_by_session[first_sessions[j]].values()), dtype=np.float32)
            pairs.append((vec1, vec2, first_sessions[i], first_sessions[j]))
            labels.append(0)
    
    pairs = np.array(pairs, dtype=object)  # Using dtype=object to keep tuple elements of different types
    labels = np.array(labels)
    return pairs, labels

### ----------------------- Loss Function -----------------------

def get_loss_function():
    """
    Returns the loss function to be used in model compilation.
    In this case, we use binary crossentropy.
    """
    return tf.keras.losses.BinaryCrossentropy()

### ----------------------- Training Function -----------------------

def train_model(X1_train, X2_train, Y_train, X1_val, X2_val, Y_val, input_shape, epochs=30, batch_size=32, lr=0.0005):
    """
    Build, compile, and train the Siamese network.
    Returns the trained model and the training history.
    """
    # Build the base and head models
    base_model = create_base_network(input_shape)
    head_model = create_head_model(base_model.output_shape)
    siamese_network = SiameseNetwork(base_model, head_model)
    
    # Register the L1 distance function
    @tf.keras.utils.register_keras_serializable()
    def l1_distance(vects):
        x, y = vects
        return K.abs(x - y)
    get_custom_objects()["l1_distance"] = l1_distance
    
    # Compile the model with the loss function from get_loss_function
    loss_fn = get_loss_function()
    siamese_network.compile(
        loss=loss_fn,
        optimizer=Adam(learning_rate=lr),
        metrics=['accuracy']
    )
    
    # Train the model
    history = siamese_network.fit(
        [X1_train, X2_train],
        Y_train,
        batch_size=batch_size,
        validation_data=([X1_val, X2_val], Y_val),
        epochs=epochs,
        verbose=1
    )
    
    return siamese_network, history

### ----------------------- Main Training Script -----------------------

if __name__ == "__main__":
    # Step 1: Load precomputed standardized features from CSV
    print("📥 Loading precomputed feature vectors from features.csv...")
    df_features = load_feature_vectors("features.csv")
    
    # Step 2: Create training pairs from the precomputed features
    pairs, labels = create_training_pairs_from_df(df_features)
    # Print a sample of 5 pairs with participant and session info
    print("\n📊 Sample Training Pairs:")
    print ("\n Pairs legth:", len(pairs))
    print ("\n Psotive pairs:")
    for i in range(min(130, len(pairs))):
        vec1, vec2, sess1, sess2 = pairs[i]
        print(f"Pair {i+1}: {sess1} vs {sess2}, label: {labels[i]}")
    
    # Prepare arrays for training (extract only the vector parts)
    X1 = np.array([pair[0] for pair in pairs], dtype=np.float32)
    X2 = np.array([pair[1] for pair in pairs], dtype=np.float32)
    Y = np.array(labels, dtype=np.float32)
    
    # Step 3: Split data into training and validation sets
    X1_train, X1_val, X2_train, X2_val, Y_train, Y_val = train_test_split(
        X1, X2, Y, test_size=0.2, random_state=42
    )
    
    # Step 4: Define model input shape based on feature vector size
    input_shape = (df_features.shape[1],)
    
    # Step 5: Train the Siamese network
    print("🚀 Training the Siamese network...")
    model, history = train_model(
        X1_train, X2_train, Y_train,
        X1_val, X2_val, Y_val,
        input_shape,
        epochs=30,
        batch_size=32,
        lr=0.0005
    )
    
    # Evaluate model performance on the validation set
    print("\n📊 Evaluating model performance...")
    val_loss, val_accuracy = model.evaluate([X1_val, X2_val], Y_val, verbose=1)
    print(f"Final validation accuracy: {val_accuracy:.4f}")
    print(f"Final validation loss: {val_loss:.4f}")
    
    # Save the trained model
    print("💾 Saving the trained model...")
    model.siamese_model.save("models/siamese_model.keras")
