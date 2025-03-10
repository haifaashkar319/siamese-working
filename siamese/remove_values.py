import pandas as pd

# 🔹 Define the path to your dataset
input_file = "FreeDB.csv"  # Change this to your actual file name
output_file = "FreeDB2.csv"  # File to save after filtering

# 🔹 Load the dataset
df = pd.read_csv(input_file)

# 🔹 Convert timing columns to numeric (to avoid issues with string values)
timing_columns = ["DU.key1.key1", "DD.key1.key2", "DU.key1.key2", "UD.key1.key2", "UU.key1.key2"]
df[timing_columns] = df[timing_columns].apply(pd.to_numeric, errors="coerce")

# 🔹 Filter: Remove rows where UD.key1.key2 > 10s
filtered_df = df[df["UD.key1.key2"] <= 5]

# 🔹 Save the cleaned dataset
filtered_df.to_csv(output_file, index=False)

print(f"✅ Cleaned dataset saved as: {output_file}")
print(f"🛑 Removed {len(df) - len(filtered_df)} rows with UD.key1.key2 > 10s")
