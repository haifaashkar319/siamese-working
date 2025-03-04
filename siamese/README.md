# **Keystroke & Mouse Dynamics Authentication - README**

## **📌 Overview**
This project is a **Siamese Neural Network** for **keystroke & mouse dynamics authentication**. The model learns unique user typing and mouse movement behaviors to verify identity.

---

## **📌 Setup & Installation**
### ✅ **1. Clone the Repository**
```sh
git clone https://github.com/haifaashkar319/siamese-working.git
cd siamese-git-rep/siamese
```

### ✅ **2. Create a Virtual Environment**
```sh
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate    # Windows
```

### ✅ **3. Install Dependencies**
```sh
pip install -r requirements.txt
```

### ✅ **4. Install Additional Libraries (If Needed)**
```sh
pip install tensorflow pandas numpy matplotlib keyboard pynput
```

---

## **📌 Data Collection** **(UNDER CONSTRUCTION)**
### **1. Collect Keystroke & Mouse Data**
Run the script to record **keystroke and mouse movement dynamics**:
```sh
python data_collection.py
```
- **User enters their ID (e.g., `p101`).**
- **User types a passage (~200 words).**
- **Mouse movements & clicks are tracked.**
- **Data is saved in `free-text (1).csv`.**

---

## **📌 Data Preprocessing**
To extract features from raw typing & mouse data, run:
```sh
python data_loader.py
```
- Converts raw **timestamps** into **feature vectors**.
- Generates **training pairs** for the Siamese network.
- Stores processed data for model training.

---

## **📌 Training the Siamese Network**
Once data preprocessing is complete, train the model:
```sh
python mnist_siamese_example.py
```
- Splits **train/test data (80/20)**.
- Trains for **20 epochs**.
- Saves the final model as `models/siamese_model.h5`.

---

## **📌 Running Authentication Tests**
To test authentication by comparing live keystrokes & mouse dynamics:
```sh
python test_authentication.py
```
- **User enters a test phrase**.
- **The model compares the new sample to stored sessions**.
- **Outputs a similarity score (0-1) and authentication decision**.

---

## **📌 Customizing the Model**
### **Modify Training Parameters**
Edit `mnist_siamese_example.py`:
```python
siamese_network.fit(X_train_data, Y_train_data,
                    batch_size=32, epochs=20,
                    validation_data=(X_val_data, Y_val_data))
```
- Increase `epochs` for better accuracy.
- Adjust `batch_size` for performance tuning.

### **Change Decision Threshold for Authentication**
Modify `test_authentication.py`:
```python
threshold = 0.75  # Adjust to balance false positives & false negatives
if similarity_score > threshold:
    print("✅ User authenticated.")
else:
    print("❌ Authentication failed.")
```

---

## **📌 Future Improvements**
🔹 **Enhance Feature Extraction:** Add **dwell time, click speed, and trajectory analysis**.
🔹 **Optimize Model Performance:** Tune **dropout, batch size, and learning rate decay**.
🔹 **Deploy as an API:** Create a **Flask/FastAPI-based authentication system**.

---

## **📌 Troubleshooting**
### **Common Issues & Fixes**
| **Issue** | **Possible Cause** | **Solution** |
|---------|------------------|------------|
| Model not training well | Small dataset | Collect more sessions per user |
| Low accuracy | Threshold too strict/loose | Test different thresholds (0.5 - 0.9) |
| `ModuleNotFoundError` | Missing dependencies | Run `pip install -r requirements.txt` |

---

## **📌 Contact & Contribution**
💡 **Found a bug or have suggestions?** Open an issue or contribute via **pull requests**!

🚀 **Happy coding!**

