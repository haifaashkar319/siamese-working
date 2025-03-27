import pandas as pd
import os

### ----------------------- Step 1: Data Loading & Initial Validation -----------------------

def load_raw_data(file_path="FreeDB2.csv"):
    """Load the raw keystroke dataset."""
    file_path = os.path.join(os.path.dirname(__file__), file_path)
    df = pd.read_csv(file_path, low_memory=False)
    df.columns = df.columns.str.strip()  # Clean column names
    return df

def validate_raw_data(df):
    """Ensure required columns exist and perform basic checks before processing."""
    required_columns = {"participant", "session", "DU.key1.key1", "DD.key1.key2"}
    
    if not required_columns.issubset(df.columns):
        missing_cols = required_columns - set(df.columns)
        raise KeyError(f" Missing required columns: {missing_cols}")

    #  Debug: Check database statistics before preprocessing
    print("\n Raw Database Statistics Before Cleaning:")
    print(df.describe())

    #  Check for negative DD and DU values
    negative_dd = (df["DD.key1.key2"] < 0).sum()
    negative_du = (df["DU.key1.key2"] < 0).sum()
    
    print(f"\n Negative DD values: {negative_dd}")
    print(f" Negative DU values: {negative_du}")

    return df

### ----------------------- Step 2: Cleaning & Overwriting -----------------------

def clean_and_overwrite_data(df, file_path="FreeDB2.csv"):
    """Filter the dataset and overwrite the original file."""
    
    timing_columns = ["DU.key1.key1", "DD.key1.key2", "UD.key1.key2", "UU.key1.key2"]
    
    # Convert to numeric
    for col in timing_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop NaN values in required columns
    before_dropna = len(df)
    df = df.dropna(subset=timing_columns)
    after_dropna = len(df)
    print(f"\n Removed {before_dropna - after_dropna} rows with missing values.")

    # Apply filtering based on acceptable ranges
    before_filter = len(df)
    
    df = df[
        (df["DU.key1.key1"].between(0.03, 3.0)) &  # Key hold duration
        (df["DD.key1.key2"].between(-0.3, 5.0)) &  # Key transition time
        (df["UD.key1.key2"].between(-0.5, 4.0)) &  # Key latency (Up-Down)
        (df["UU.key1.key2"].between(0.0, 5.0))     # Key release delay
    ]
    
    after_filter = len(df)
    print(f"\n Applied range filters. Removed {before_filter - after_filter} rows.")

    #  Debug: Print filtered data stats
    print("\n Feature Statistics After Cleaning:")
    print(df[timing_columns].describe())

    # Overwrite the original file
    df.to_csv(file_path, index=False)
    print(f"\n Overwritten original dataset at `{file_path}` with cleaned data.")

    return df

### ----------------------- Execution Flow -----------------------

if __name__ == "__main__":
    print(" Loading raw data...")
    df = load_raw_data()
    
    print(" Validating raw data...")
    df = validate_raw_data(df)
    
    print(" Cleaning dataset and overwriting...")
    clean_and_overwrite_data(df)
