import pandas as pd

def calculate_percentiles(df, columns, percentiles=[25, 50, 75]):
    """
    Calculate and print the specified percentiles for each column.
    """
    for col in columns:
        # Convert to numeric in case there are non-numeric values.
        df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"Percentiles for column {col}:")
        for p in percentiles:
            value = df[col].quantile(p / 100)
            print(f"  {p}th percentile: {value}")
        print()

def save_out_of_range_entries(df, output_file="out_of_range_entries.csv"):
    """
    Identify and save rows that have column values outside these “normal” intervals:
      DU.key1.key1: [0.0045, 0.1845]
      DD.key1.key2: [-0.1055, 0.4985]
      DU.key1.key2: [-0.0215, 0.6145]
      UD.key1.key2: [-0.211, 0.405]
      UU.key1.key2: [-0.108, 0.5]

    Any row with a value outside its corresponding interval gets flagged. 
    The output CSV includes a new column ("Problematic_Cols") listing which columns are out of range.
    """
    # Define the thresholds for each column
    INTERVALS = {
        "DU.key1.key1": (0.0045, 0.1845),
        "DD.key1.key2": (-0.1055, 0.4985),
        "DU.key1.key2": (-0.0215, 0.6145),
        "UD.key1.key2": (-0.211, 0.405),
        "UU.key1.key2": (-0.108, 0.5)
    }
    
    # Ensure the relevant columns are numeric
    for col in INTERVALS.keys():
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    def find_problematic_cols(row):
        problematic = []
        for col, (low, high) in INTERVALS.items():
            val = row[col]
            if pd.notna(val) and (val < low or val > high):
                problematic.append(col)
        return ", ".join(problematic)

    # Create a new column indicating which columns are out of range for each row
    df["Problematic_Cols"] = df.apply(find_problematic_cols, axis=1)
    
    # Filter and save the rows that have any out-of-range values
    out_of_range_df = df[df["Problematic_Cols"] != ""].copy()
    out_of_range_df.to_csv(output_file, index=False)

    print(f" Found {len(out_of_range_df)} row(s) with out-of-range values.")
    print(f" Saved these rows (with Problematic_Cols) to: {output_file}")

def parse_cols(problematic_str):
    """
    Convert a string like "DD.key1.key2, DU.key1.key2" 
    into a Python set: {"DD.key1.key2", "DU.key1.key2"}.
    """
    if pd.isna(problematic_str) or problematic_str.strip() == "":
        return set()
    # Split by commas and remove extra whitespace
    return set(x.strip() for x in problematic_str.split(","))

def count_exact_combinations(df, col_name="Problematic_Cols"):
    """
    Parse the 'Problematic_Cols' column into a set of flags for each row.
    For each exact combination in the list, count how many rows match exactly.
    Print the count and fraction (percentage) of total rows for each combination.
    """
    # List of (label, set_of_flags) tuples for exact combinations
    combinations_list = [
        ("DD.key1.key2", {"DD.key1.key2"}),
        ("DD.key1.key2, DU.key1.key2", {"DD.key1.key2", "DU.key1.key2"}),
        ("DD.key1.key2, DU.key1.key2, UD.key1.key2", {"DD.key1.key2", "DU.key1.key2", "UD.key1.key2"}),
        ("DD.key1.key2, DU.key1.key2, UD.key1.key2, UU.key1.key2", {"DD.key1.key2", "DU.key1.key2", "UD.key1.key2", "UU.key1.key2"}),
        ("DD.key1.key2, DU.key1.key2, UU.key1.key2", {"DD.key1.key2", "DU.key1.key2", "UU.key1.key2"}),
        ("DD.key1.key2, UD.key1.key2", {"DD.key1.key2", "UD.key1.key2"}),
        ("DD.key1.key2, UD.key1.key2, UU.key1.key2", {"DD.key1.key2", "UD.key1.key2", "UU.key1.key2"}),
        ("DU.key1.key1", {"DU.key1.key1"}),
        ("DU.key1.key1, DD.key1.key2", {"DU.key1.key1", "DD.key1.key2"}),
        ("DU.key1.key1, DD.key1.key2, DU.key1.key2", {"DU.key1.key1", "DD.key1.key2", "DU.key1.key2"}),
        ("DU.key1.key1, DD.key1.key2, DU.key1.key2, UD.key1.key2", {"DU.key1.key1", "DD.key1.key2", "DU.key1.key2", "UD.key1.key2"}),
        ("DU.key1.key1, DD.key1.key2, DU.key1.key2, UD.key1.key2, UU.key1.key2", {"DU.key1.key1", "DD.key1.key2", "DU.key1.key2", "UD.key1.key2", "UU.key1.key2"}),
        ("DU.key1.key1, DD.key1.key2, DU.key1.key2, UU.key1.key2", {"DU.key1.key1", "DD.key1.key2", "DU.key1.key2", "UU.key1.key2"}),
        ("DU.key1.key1, DU.key1.key2", {"DU.key1.key1", "DU.key1.key2"}),
        ("DU.key1.key1, DU.key1.key2, UU.key1.key2", {"DU.key1.key1", "DU.key1.key2", "UU.key1.key2"}),
        ("DU.key1.key1, UU.key1.key2", {"DU.key1.key1", "UU.key1.key2"}),
        ("DU.key1.key2", {"DU.key1.key2"}),
        ("DU.key1.key2, UD.key1.key2, UU.key1.key2", {"DU.key1.key2", "UD.key1.key2", "UU.key1.key2"}),
        ("DU.key1.key2, UU.key1.key2", {"DU.key1.key2", "UU.key1.key2"}),
        ("UD.key1.key2", {"UD.key1.key2"}),
        ("UD.key1.key2, UU.key1.key2", {"UD.key1.key2", "UU.key1.key2"}),
        ("UU.key1.key2", {"UU.key1.key2"}),
    ]

    # Parse each row's Problematic_Cols into a set
    df["Parsed_Cols"] = df[col_name].apply(parse_cols)

    total_rows = len(df)
    results = []
    
    # For each combination, count rows that match exactly
    for combo_label, combo_set in combinations_list:
        match_count = sum(df["Parsed_Cols"] == combo_set)
        fraction = match_count / total_rows if total_rows else 0
        results.append((combo_label, match_count, fraction))

    # Print the results
    for combo_label, match_count, fraction in results:
        print(f"{combo_label} -> Count: {match_count}, Fraction: {fraction:.4f}")
    
    return results

# Main execution
if __name__ == "__main__":
    # Update the file path to the location of your FreeDB2.csv file
    file_path = "FreeDB2.csv"
    df = pd.read_csv(file_path)
    
    # Define the columns for which we want to calculate percentiles
    columns_to_check = [
        'DU.key1.key1', 'DD.key1.key2', 
        'DU.key1.key2', 'UD.key1.key2', 
        'UU.key1.key2'
    ]
    
    # Print percentiles for each column
    calculate_percentiles(df, columns_to_check)
    
    # Save rows with out-of-range values based on provided thresholds
    save_out_of_range_entries(df, output_file="out_of_range_entries.csv")
    
    # Count exact combinations of out-of-range flags (non-overlapping)
    print("\nExact combination counts and fractions:")
    count_exact_combinations(df, col_name="Problematic_Cols")
