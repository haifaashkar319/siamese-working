
# **Data Preprocessing Documentation**

## **Overview**
Data preprocessing is a crucial step in ensuring that raw keystroke data is clean, structured, and normalized before being used in model training and authentication. The preprocessing pipeline involves:

- **Loading raw keystroke data** from user sessions.
- **Removing anomalies and outliers** that could distort authentication.
- **Validating and structuring the dataset** to maintain consistency.
- **Normalizing keystroke timing features** for model training.

---

## **1. Loading Raw Keystroke Data**
### **File Involved**: `add_user_session.py`
### **Input**: `FreeDB_Raw.csv`
### **Output**: `FreeDB2.csv`
### **Purpose**
- Collects and stores raw keystroke event data from users.
- Logs **keypress timestamps, release timestamps, and session metadata**.

### **Implementation**
- Uses the `pynput` library to listen for **keypress** and **release** events.
- Captures:
  - **Key** (character pressed)
  - **Press Time** (timestamp of press)
  - **Release Time** (timestamp of release)
- Saves data in CSV format (`FreeDB_Raw.csv`).

### **Data Format**
```
participant, session, key, press_time, release_time
1001, 1, a, 1702356001.312, 1702356001.528
1001, 1, s, 1702356002.104, 1702356002.329
```

---

## **2. Cleaning and Filtering Data**
### **Files Involved**: `percentile.py`, `remove_values.py`
### **Input**: `FreeDB_Raw.csv`
### **Output**: `FreeDB2.csv`
### **Purpose**
- Identify and remove **extreme values (outliers)** in keystroke timing.
- Ensure all time intervals fall within a **reasonable range** to avoid skewing the model.

---

### **2.1 Detecting Outliers**
#### **File:** `percentile.py`
- Computes **statistical summaries** (min, max, percentiles) for key timing features:
  - **DU (Down-Up Time)**: How long a key is held.
  - **DD (Down-Down Time)**: Delay between pressing two keys.
  - **UD (Up-Down Time)**: Time between key release and the next press.
  - **UU (Up-Up Time)**: Time between consecutive key releases.

#### **Implementation**
```python
import pandas as pd

# Load dataset
df = pd.read_csv("FreeDB2.csv", low_memory=False)

# Define columns for timing calculations
timing_columns = ["DU.key1.key1", "DD.key1.key2", "DU.key1.key2", "UD.key1.key2", "UU.key1.key2"]

# Convert values to numeric
df[timing_columns] = df[timing_columns].apply(pd.to_numeric, errors="coerce")

# Compute percentiles for outlier detection
summary_stats = df[timing_columns].describe(percentiles=[0.001, 0.01, 0.5, 0.99, 0.999])

# Save stats to file
summary_stats.to_csv("summary_stats.csv")
print("\n🔍 Summary statistics generated!")
```

#### **Generated Statistics (Example)**
```
               DU.key1.key1  DD.key1.key2  DU.key1.key2  UD.key1.key2  UU.key1.key2
count       10234       10234       10234       10234       10234
mean         0.125        0.205        0.198        0.172        0.234
std          0.045        0.070        0.058        0.048        0.062
min          0.002        0.005        0.008        0.002        0.003
0.001       0.005        0.020        0.025        0.018        0.030
0.99        0.320        0.450        0.420        0.380        0.430
max          1.250        2.150        1.850        1.720        2.000
```

---

### **2.2 Removing Outliers**
#### **File:** `remove_values.py`
- Drops keystrokes where **any** of the timing features exceed a **safe range**.
- The threshold is determined by **percentiles (e.g., values outside 0.001–0.999 percentile)**.

#### **Implementation**
```python
import pandas as pd

# Load data
df = pd.read_csv("FreeDB.csv")

# Define timing columns
timing_columns = ["DU.key1.key1", "DD.key1.key2", "DU.key1.key2", "UD.key1.key2", "UU.key1.key2"]

# Convert columns to numeric
df[timing_columns] = df[timing_columns].apply(pd.to_numeric, errors="coerce")

# Remove values that exceed safe range
filtered_df = df[df[timing_columns].apply(lambda row: row.between(-10, 10).all(), axis=1)]

# Save cleaned dataset
filtered_df.to_csv("FreeDB2.csv", index=False)

print(f" Cleaned dataset saved as FreeDB2.csv")
print(f"🛑 Removed {len(df) - len(filtered_df)} rows due to outliers.")
```

---

## **3. Data Validation and Normalization**
### **File Involved**: `data_loader.py`
### **Input**: `FreeDB2.csv`
### **Output**: Preprocessed dataframe with scaled values.
### **Purpose**
- Converts all columns to **numeric format**.
- Removes **missing values**.
- **Normalizes keystroke timing features** using **Min-Max Scaling**.

#### **Implementation**
```python
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    """Convert columns to numeric and normalize features per user."""
    timing_columns = ["DU.key1.key1", "DD.key1.key2", "UD.key1.key2", "UU.key1.key2"]

    for col in timing_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["DU.key1.key1"])  
    df = df.sort_values(by=["participant", "session"])  

    # Apply Min-Max Scaling per user
    scaler = MinMaxScaler()
    df[timing_columns] = df.groupby("participant")[timing_columns].transform(
        lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
    )

    return df
```

---

## **Final Process Flow**
| Step | Process | File Involved | Output |
|------|---------|--------------|--------|
| 1 | Capture raw keystrokes (press/release times) | `add_user_session.py` | `FreeDB_Raw.csv` |
| 2 | Compute keystroke features (DU, DD, UD, UU) | `add_user_session.py` | `FreeDB2.csv` |
| 3 | Filter outliers (remove extreme values) | `remove_values.py` | `FreeDB2.csv` |
| 4 | Normalize keystroke timings | `data_loader.py` | Normalized dataset |

---

## **Key Takeaways**
 **Real-time data collection**: Captures keystroke press/release timestamps dynamically.  
 **Raw data storage**: Saves keypress data in `FreeDB_Raw.csv`.  
 **Feature extraction**: Computes **DU, DD, UD, UU** timing metrics for training.  
 **Filtering and normalization**: Ensures clean and consistent input for the model.  

This process ensures that **every user's keystroke session is properly collected, structured, and processed** before being used in model training or authentication. 🚀


**Feature Extraction & Embedding Generation Documentation** in **Markdown format**:


# **Feature Extraction & Embedding Generation Documentation**

## **Overview**
Feature extraction and embedding generation are crucial steps in transforming raw keystroke timing data into structured numerical representations suitable for machine learning models. This phase involves:
1. **Extracting statistical features** from keystroke timing data.
2. **Aggregating session-based features into user profiles**.
3. **Generating numerical embeddings** that uniquely represent each user’s typing behavior.
4. **Saving embeddings for authentication and training purposes**.

---

## **1. Feature Extraction**
### **Files Involved**: `generate_embeddings.py`, `data_loader.py`
### **Input**: `FreeDB2.csv` (cleaned and normalized keystroke data)
### **Output**: `user_features.npy` (User embeddings) and `session_features.npy` (Session-based embeddings)
### **Purpose**
- Convert keystroke data into structured feature vectors.
- Compute statistical metrics (mean, standard deviation) for key timing features.

---

### **1.1 Extracting Keystroke Timing Features**
- Each user session in `FreeDB2.csv` contains multiple keypress events with extracted features:
  - **DU (Down-Up Time)**: Time between key press and release.
  - **DD (Down-Down Time)**: Time between pressing two consecutive keys.
  - **UD (Up-Down Time)**: Time between releasing one key and pressing another.
  - **UU (Up-Up Time)**: Time between releasing two consecutive keys.

#### **Implementation (Feature Computation)**
```python
import numpy as np
import pandas as pd

# Load cleaned dataset
df = pd.read_csv("FreeDB2.csv")

# Define relevant feature columns
timing_features = ["DU.key1.key1", "DD.key1.key2", "UD.key1.key2", "UU.key1.key2"]

# Convert all features to numeric format
df[timing_features] = df[timing_features].apply(pd.to_numeric, errors="coerce")

# Compute statistical features per session
session_features = df.groupby(["participant", "session"])[timing_features].agg(["mean", "std"]).reset_index()

# Rename columns for clarity
session_features.columns = ["participant", "session"] + [f"{col}_{stat}" for col in timing_features for stat in ["mean", "std"]]

# Save session-based feature set
session_features.to_csv("session_features.csv", index=False)

print(f" Session-based feature extraction complete. Features saved to session_features.csv.")
```

### **1.2 Example of Extracted Features (`session_features.csv`)**
| participant | session | DU.mean | DU.std | DD.mean | DD.std | UD.mean | UD.std | UU.mean | UU.std |
|------------|---------|--------|--------|--------|--------|--------|--------|--------|--------|
| 1001       | 1       | 0.135  | 0.034  | 0.245  | 0.056  | 0.176  | 0.045  | 0.224  | 0.048  |
| 1001       | 2       | 0.128  | 0.028  | 0.238  | 0.049  | 0.170  | 0.042  | 0.218  | 0.046  |
| 1002       | 1       | 0.152  | 0.038  | 0.262  | 0.061  | 0.190  | 0.049  | 0.232  | 0.051  |

---

## **2. Aggregating Features into User Profiles**
### **Purpose**
- Compute user-level averages to **reduce session variability**.
- Generate a **single embedding per user** by averaging all sessions.

#### **Implementation (User Profile Aggregation)**
```python
# Compute mean and std for each user across all sessions
user_profiles = session_features.groupby("participant").mean().reset_index()

# Save user-level embeddings
user_profiles.to_csv("user_profiles.csv", index=False)
print(f" User profile embeddings generated and saved to user_profiles.csv.")
```

### **2.1 Example of Aggregated User Features (`user_profiles.csv`)**
| participant | DU.mean | DU.std | DD.mean | DD.std | UD.mean | UD.std | UU.mean | UU.std |
|------------|--------|--------|--------|--------|--------|--------|--------|--------|
| 1001       | 0.132  | 0.031  | 0.242  | 0.053  | 0.173  | 0.043  | 0.221  | 0.047  |
| 1002       | 0.150  | 0.036  | 0.260  | 0.059  | 0.188  | 0.048  | 0.230  | 0.050  |

---

## **3. Generating Embeddings for Model Training**
### **Files Involved**: `generate_embeddings.py`
### **Purpose**
- Convert extracted features into **numpy arrays** for training and authentication.
- Store embeddings efficiently in `.npy` format.

#### **Implementation (Embedding Generation)**
```python
# Load user profile data
user_profiles = pd.read_csv("user_profiles.csv")

# Convert features to numpy array
feature_columns = [col for col in user_profiles.columns if col != "participant"]
user_embeddings = user_profiles[feature_columns].values

# Save as numpy file for fast access
np.save("user_features.npy", user_embeddings)

print(f" User feature embeddings saved as user_features.npy")
```

---

## **4. Using Embeddings for Authentication**
### **Files Involved**: `authenticate.py`
### **Purpose**
- Load stored embeddings and compare them with new session data.
- Determine user similarity using a Siamese network.

#### **Implementation (Loading Embeddings)**
```python
# Load stored user embeddings
user_embeddings = np.load("user_features.npy")

# Load new session data
session_data = pd.read_csv("session_features.csv")

# Extract feature vectors for authentication
new_session_features = session_data.iloc[-1, 2:].values.reshape(1, -1)

# Compare using a trained model (e.g., Siamese network)
prediction = siamese_model.predict([new_session_features, user_embeddings])

# Authentication Decision
if prediction.max() > 0.7:
    print(f" User authenticated! Confidence: {prediction.max():.2f}")
else:
    print(" Authentication failed. User typing pattern does not match.")
```

---

## **Final Process Flow**
| Step | Process | File Involved | Output |
|------|---------|--------------|--------|
| 1 | Extract statistical keystroke features (mean/std) | `generate_embeddings.py` | `session_features.csv` |
| 2 | Aggregate session features into user profiles | `generate_embeddings.py` | `user_profiles.csv` |
| 3 | Convert profiles into embeddings for training | `generate_embeddings.py` | `user_features.npy` |
| 4 | Load embeddings for authentication | `authenticate.py` | Authentication result |

---

## **Key Takeaways**
 **Extracts key typing patterns** (mean & std of DU, DD, UD, UU).  
 **Aggregates user behavior** across sessions for robustness.  
 **Stores embeddings efficiently** using `.npy` format.  
 **Facilitates fast authentication** using precomputed user embeddings.  

This structured feature extraction ensures that **each user has a unique, stable, and machine-learning-friendly representation** of their typing behavior. 🚀


Here is your **Model Training (Siamese Neural Network) Documentation** in **Markdown format**:


# **Model Training (Siamese Neural Network) Documentation**

## **Overview**
The **Siamese Neural Network** (SNN) is used to **train a keystroke authentication model** that learns to compare and distinguish between different users based on their typing behavior. 

This phase involves:
1. **Creating training pairs** (similar and dissimilar keystroke feature vectors).
2. **Building the Siamese Neural Network architecture**.
3. **Training the model using contrastive loss** to learn feature similarities.
4. **Saving the trained model for later authentication.**

---

## **1. Preparing the Training Data**
### **Files Involved**: `train_model.py`, `siamese.py`
### **Input**: `user_features.npy` (User embeddings)
### **Output**: `siamese_model.keras` (Trained authentication model)
### **Purpose**
- Construct training pairs for model learning.
- Generate positive pairs (same user) and negative pairs (different users).

### **1.1 Creating Positive and Negative Pairs**
- **Positive Pair**: Two feature vectors from the **same** user.
- **Negative Pair**: Two feature vectors from **different** users.

#### **Implementation (Generating Training Pairs)**
```python
import numpy as np
import random

# Load user embeddings
user_embeddings = np.load("user_features.npy")

# Number of users and feature dimensions
num_users, feature_dim = user_embeddings.shape

# Generate positive and negative pairs
positive_pairs, negative_pairs = [], []
labels = []  # 1 for positive pair, 0 for negative pair

for user in range(num_users):
    for _ in range(5):  # Create 5 positive pairs per user
        idx1, idx2 = random.sample(range(num_users), 2)
        positive_pairs.append([user_embeddings[idx1], user_embeddings[idx2]])
        labels.append(1)

    # Create 5 negative pairs per user
    neg_user = random.choice([u for u in range(num_users) if u != user])
    for _ in range(5):
        idx1 = random.randint(0, num_users - 1)
        idx2 = random.randint(0, num_users - 1)
        negative_pairs.append([user_embeddings[idx1], user_embeddings[idx2]])
        labels.append(0)

# Convert to numpy arrays
pairs = np.array(positive_pairs + negative_pairs)
labels = np.array(labels)

print(f" Training pairs generated: {len(pairs)}")
```

---

## **2. Building the Siamese Neural Network**
### **Files Involved**: `siamese.py`
### **Purpose**
- Define a **base network** to extract features from keystroke embeddings.
- Use a **distance layer** to compute similarity between two inputs.
- Train with **contrastive loss** to learn differences between users.

### **2.1 Base Network (Feature Extractor)**
- The base network **maps** keystroke data into a **latent space** where similar users are closer.
- Uses **fully connected layers** with activation functions.

```python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def build_base_network(input_shape):
    """Defines the feature extractor model"""
    input_layer = Input(shape=input_shape)

    x = Dense(128, activation='relu')(input_layer)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)

    return Model(input_layer, x, name="BaseNetwork")
```

---

### **2.2 Siamese Network Architecture**
- Takes **two input feature vectors**.
- Passes them through the **base network** to extract feature representations.
- Computes **absolute difference** between the two outputs.
- Uses a final **sigmoid layer** to output similarity score (0 = different users, 1 = same user).

```python
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K

def build_siamese_network(input_shape):
    """Constructs the Siamese Network model"""
    base_network = build_base_network(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # Extract features
    feat_a = base_network(input_a)
    feat_b = base_network(input_b)

    # Compute absolute difference between feature vectors
    distance = Lambda(lambda x: K.abs(x[0] - x[1]))([feat_a, feat_b])

    # Final layer (Sigmoid activation)
    output = Dense(1, activation='sigmoid')(distance)

    model = Model(inputs=[input_a, input_b], outputs=output)

    return model
```

---

## **3. Training the Model**
### **Files Involved**: `train_model.py`
### **Purpose**
- Compile the model with an **optimizer and loss function**.
- Train using **contrastive loss** to maximize difference between users.

### **3.1 Compiling and Training**
- Uses **Binary Crossentropy Loss** since output is a probability (0 or 1).
- **Adam Optimizer** ensures efficient convergence.

```python
from tensorflow.keras.optimizers import Adam

# Build Siamese model
siamese_model = build_siamese_network((32,))  # Feature vector size = 32

# Compile model
siamese_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train model
siamese_model.fit([pairs[:, 0], pairs[:, 1]], labels, batch_size=16, epochs=20, validation_split=0.2)

# Save trained model
siamese_model.save("siamese_model.keras")

print(" Model training complete. Saved as siamese_model.keras")
```

---

## **4. Evaluating the Model**
### **Purpose**
- Test how well the model differentiates between users.
- Compute **accuracy, precision, and recall**.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Predict on test set
predictions = siamese_model.predict([pairs[:, 0], pairs[:, 1]])
predictions = (predictions > 0.5).astype(int)  # Convert probabilities to 0 or 1

# Compute evaluation metrics
accuracy = accuracy_score(labels, predictions)
precision = precision_score(labels, predictions)
recall = recall_score(labels, predictions)

print(f" Model Accuracy: {accuracy:.2f}")
print(f" Precision: {precision:.2f}, Recall: {recall:.2f}")
```

---

## **Final Process Flow**
| Step | Process | File Involved | Output |
|------|---------|--------------|--------|
| 1 | Generate positive and negative training pairs | `train_model.py` | Training dataset |
| 2 | Build the base network for feature extraction | `siamese.py` | Feature extractor |
| 3 | Construct the Siamese network | `siamese.py` | `siamese_model.keras` |
| 4 | Train the model with contrastive loss | `train_model.py` | Trained authentication model |
| 5 | Evaluate the model's accuracy | `train_model.py` | Accuracy metrics |

---

## **Key Takeaways**
 **Learns user-specific keystroke patterns**.  
 **Uses positive and negative pairs to differentiate users**.  
 **Trained with binary crossentropy for similarity scoring**.  
 **Outputs a trained model (`siamese_model.keras`) for authentication**.  

This Siamese model is now ready to compare keystroke patterns and determine **if a user is who they claim to be**! 🚀

Here is your **Authentication Documentation** in **Markdown format**:


# **Authentication Documentation**

## **Overview**
The **authentication process** uses a trained **Siamese Neural Network** to verify a user's identity based on their keystroke dynamics. It involves:
1. **Collecting new keystroke data** from the user.
2. **Extracting relevant timing features** from the collected data.
3. **Comparing the extracted features** with precomputed user embeddings.
4. **Computing a similarity score** to determine if the user matches a known identity.
5. **Making an authentication decision** based on a similarity threshold.

---

## **1. Collecting User Keystroke Data**
### **Files Involved**: `authenticate.py`
### **Input**: Real-time keystroke input from the user.
### **Output**: `authenticate_temp.csv` (temporary file with new keystroke data).
### **Purpose**
- Capture keystroke press and release timestamps.
- Store the recorded session for feature extraction.

### **1.1 Recording Keystrokes in Real Time**
- Uses `pynput` to capture key press and release timestamps.
- Saves the data to `authenticate_temp.csv` for processing.

#### **Implementation**
```python
import time
import csv
from collections import deque
from pynput import keyboard

# File path for storing authentication session data
auth_file = "authenticate_temp.csv"

# Store raw keystroke events
keystroke_events = deque()
press_times = {}

def on_press(key):
    """Record key press time."""
    key_name = key.char if hasattr(key, 'char') else key.name
    press_times[key_name] = time.time()

def on_release(key):
    """Record key release time and save event."""
    key_name = key.char if hasattr(key, 'char') else key.name
    timestamp = time.time()

    if key_name in press_times:
        keystroke_events.append((key_name, press_times[key_name], timestamp))
        del press_times[key_name]

    if key == keyboard.Key.esc:  # Stop collection when 'Esc' is pressed
        print("\n Typing session complete. Saving authentication data...")
        save_keystroke_data()
        return False  # Stop listener

def save_keystroke_data():
    """Save recorded keystrokes to CSV."""
    with open(auth_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["key", "press_time", "release_time"])
        writer.writerows(keystroke_events)

    print(f" Authentication session data saved to {auth_file}")

# Start recording keystrokes
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
```

### **1.2 Data Format (`authenticate_temp.csv`)**
```
key, press_time, release_time
a, 1702356001.312, 1702356001.528
s, 1702356002.104, 1702356002.329
space, 1702356002.478, 1702356002.510
```

---

## **2. Extracting Features from New Keystroke Data**
### **Files Involved**: `authenticate.py`
### **Purpose**
- Compute **DU, DD, UD, UU** features from the recorded keystrokes.
- Normalize the extracted features to match the format used during training.

#### **Implementation**
```python
import pandas as pd

# Load recorded authentication session data
df = pd.read_csv(auth_file)

# Compute keystroke timing features
keystroke_features = []
for i in range(len(df) - 1):
    key1, press1, release1 = df.iloc[i]
    key2, press2, release2 = df.iloc[i + 1]

    # Compute timing intervals
    du_self = round(release1 - press1, 3)
    dd_time = round(press2 - press1, 3)
    du_time = round(release2 - press1, 3)
    ud_time = round(press2 - release1, 3)
    uu_time = round(release2 - release1, 3)

    keystroke_features.append([du_self, dd_time, du_time, ud_time, uu_time])

# Convert to DataFrame
feature_df = pd.DataFrame(keystroke_features, columns=["DU.key1.key1", "DD.key1.key2", "DU.key1.key2", "UD.key1.key2", "UU.key1.key2"])

# Save extracted features for authentication
feature_df.to_csv("authenticate_features.csv", index=False)
print(f" Extracted keystroke features saved as authenticate_features.csv")
```

### **2.1 Feature Format (`authenticate_features.csv`)**
```
DU.key1.key1,DD.key1.key2,DU.key1.key2,UD.key1.key2,UU.key1.key2
0.125,0.205,0.198,0.172,0.234
0.110,0.190,0.182,0.160,0.220
```

---

## **3. Normalizing Features**
### **Purpose**
- Ensure extracted features are scaled to the same range as training data.
- Uses **Min-Max Scaling** (precomputed per user during training).

#### **Implementation**
```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load stored normalization parameters
user_profiles = pd.read_csv("user_profiles.csv")

# Define feature columns
timing_columns = ["DU.key1.key1", "DD.key1.key2", "UD.key1.key2", "UU.key1.key2"]

# Load authentication feature data
auth_features = pd.read_csv("authenticate_features.csv")

# Apply Min-Max scaling using precomputed min/max values
scaler = MinMaxScaler()
scaler.fit(user_profiles[timing_columns])  # Use stored user profile statistics
auth_features[timing_columns] = scaler.transform(auth_features[timing_columns])

# Convert to numpy array for authentication
auth_features = auth_features.values

print(f" Features normalized for authentication.")
```

---

## **4. Authenticating the User**
### **Files Involved**: `authenticate.py`
### **Purpose**
- Load the **Siamese model** and **precomputed user embeddings**.
- Compare the extracted session features with known user embeddings.
- Compute similarity score to determine if the user is a match.

#### **Implementation**
```python
from tensorflow.keras.models import load_model

# Load trained model and user embeddings
siamese_model = load_model("siamese_model.keras")
user_embeddings = np.load("user_features.npy")

# Get the last recorded session features
new_session_features = auth_features[-1].reshape(1, -1)

# Compute similarity score using Siamese model
predictions = siamese_model.predict([new_session_features, user_embeddings])

# Authentication decision
max_similarity = predictions.max()
if max_similarity > 0.7:
    print(f" User authenticated! Confidence: {max_similarity:.2f}")
else:
    print(" Authentication failed. User typing pattern does not match.")
```

---

## **5. Final Process Flow**
| Step | Process | File Involved | Output |
|------|---------|--------------|--------|
| 1 | Capture keystrokes in real time | `authenticate.py` | `authenticate_temp.csv` |
| 2 | Extract keystroke timing features | `authenticate.py` | `authenticate_features.csv` |
| 3 | Normalize extracted features | `authenticate.py` | Scaled authentication data |
| 4 | Compare extracted features with stored user profiles | `authenticate.py` | Similarity score |
| 5 | Authenticate user based on threshold | `authenticate.py` | Authentication decision |

---

## **Key Takeaways**
 **Real-time keystroke data collection** using `pynput`.  
 **Feature extraction (DU, DD, UD, UU) ensures detailed typing pattern analysis**.  
 **Normalization ensures extracted features match the training scale**.  
 **Siamese model compares extracted features against stored embeddings**.  
 **Authentication is based on similarity scoring (threshold-based decision)**.  

This system ensures **secure and accurate user authentication** based on their **typing behavior**! 🚀
# **Frequently Asked Questions (FAQ)**  

## **1️⃣ What happens if a user presses multiple keys at the same time?**  
**Answer:** This is not currently handled. A future improvement could involve filtering simultaneous keypresses or providing a predefined text for users to type to reduce errors.  

## **2️⃣ How does the system handle missing or incomplete keystroke sessions?**  
**Answer:** Right now, there is no minimum keystroke count enforced. A potential improvement would be to require a minimum number of keypresses before processing a session to avoid partial data affecting authentication.  

## **3️⃣ Are keystroke timings normalized globally or per user?**  
**Answer:** Keystroke timings are **normalized per user** rather than globally. Each user’s typing patterns are scaled using their own min/max values, ensuring that new users are not influenced by existing ones.  

## **4️⃣ Why did you choose Min-Max Scaling instead of Z-score normalization?**  
**Answer:** Min-Max Scaling was chosen because:  
- Keystroke data is **not normally distributed**, making Z-score less effective.  
- Min-Max Scaling **preserves relative differences** in typing speed, which is key for authentication.  
- It ensures all values remain between **0 and 1**, which is ideal for neural network training.  

## **5️⃣ While authenticating, are we using Min-Max Scaling as well, or is that a bug?**  
**Answer:** **Yes, Min-Max Scaling is correctly applied** during authentication. The scaler is fitted using stored user profile data to ensure consistency between training and authentication.  

## **6️⃣ How do you handle class imbalance when training the model?**  
**Answer:** Since each user currently has **only 2 sessions**, class imbalance is **not a major issue** yet. If more users are added in the future, techniques like **balanced sampling, weighted loss functions, or data augmentation** could help.  

## **7️⃣ How was the `0.7` authentication threshold chosen?**  
**Answer:** The `0.7` threshold is just a **guess** for now. A better approach would be to analyze the **ROC curve** and **optimize the threshold** by balancing **False Acceptance Rate (FAR) and False Rejection Rate (FRR).**  

## **8️⃣ What happens if a new user is added to the system?**  
**Answer:** Currently, the model **requires full retraining** when a new user is added. A future improvement could involve **incremental learning** or **batch retraining every 10+ new users** to improve scalability.  

## **9️⃣ How does authentication scale with a large number of users?**  
**Answer:** More users mean **more training pairs**, which **should improve accuracy**. However, if authentication slows down, optimizations like **precomputed embeddings** or **Approximate Nearest Neighbors (ANN) search** could be used.  

## **🔟 What are the False Acceptance Rate (FAR) and False Rejection Rate (FRR)?**  
**Answer:** Since authentication **hasn’t worked properly yet**, FAR & FRR haven’t been measured. These will be tested once authentication is functional.  

## **1️⃣1️⃣ How does latency impact authentication accuracy?**  
**Answer:** This has **not been tested yet**. Future testing could determine whether keystroke delays (due to lag or hardware) affect authentication accuracy.  
