import pandas as pd

# 🔹 Define the path to your dataset
input_file = "FreeDB.csv"  # Change this to your actual file name
output_file = "FreeDB2.csv"  # File to save after filtering

# 🔹 Load the dataset
df = pd.read_csv(input_file)

# 🔹 Specify the columns to filter
timing_columns = ["DU.key1.key1", "DD.key1.key2", "DU.key1.key2", "UD.key1.key2", "UU.key1.key2"]

# 🔹 Convert timing columns to numeric (handle mixed types)
df[timing_columns] = df[timing_columns].apply(pd.to_numeric, errors="coerce")

# 🔹 Apply filtering: Drop rows where **any** value is outside -5 < x < 5
filtered_df = df[df[timing_columns].apply(lambda row: row.between(-10, 10).all(), axis=1)]

# 🔹 Save the cleaned dataset
filtered_df.to_csv(output_file, index=False, header=True)

print(f"✅ Cleaned dataset saved as: {output_file}")
print(f"🛑 Removed {len(df) - len(filtered_df)} rows where at least one column had values outside -5 < x < 5")
