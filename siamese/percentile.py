import pandas as pd

# Load dataset
input_file = "FreeDB2.csv"
df = pd.read_csv(input_file, low_memory=False)

# Define timing columns
timing_columns = ["DU.key1.key1", "DD.key1.key2", "DU.key1.key2", "UD.key1.key2", "UU.key1.key2"]

# Ensure numeric conversion
df[timing_columns] = df[timing_columns].apply(pd.to_numeric, errors="coerce")

# Generate summary statistics
summary_stats = df[timing_columns].describe(percentiles=[0.001, 0.01, 0.5, 0.99, 0.999])
print("\n🔍 Summary Statistics:")
print(summary_stats)

# Save log to file
summary_stats.to_csv("summary_stats.csv")
