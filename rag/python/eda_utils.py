import pandas as pd
from IPython.display import display

def summarize_data(df):
    """Displays the first few rows and the schema information."""
    print("--- First 3 Rows ---")
    display(df.head(3))
    
    print("\n--- DataFrame Info ---")
    df.info()

def get_missing_report(df):
    """
    Returns a DataFrame showing the count and percentage of missing values 
    for columns that have at least one missing value.
    """
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    
    report = pd.DataFrame({
        'Missing Count': missing,
        'Missing Percentage (%)': missing_percent
    }).sort_values(by='Missing Percentage (%)', ascending=False)
    
    # Filter to show only columns that actually have missing data
    return report[report['Missing Count'] > 0]

def view_unique_values(df, column_name):
    """Quickly check the unique values and their frequencies for a specific column."""
    print(f"--- Value Counts for '{column_name}' ---")
    display(df[column_name].value_counts(dropna=False))

def get_low_cardinality_columns(df, max_unique=6):
    """
    Finds columns with a small number of unique values.
    Prints the unique values and returns them as a dictionary.
    
    Parameters:
    - df: The Pandas DataFrame.
    - max_unique: The maximum number of unique values to qualify.
    """
    print(f"--- Columns with {max_unique} or fewer unique values ---\n")
    
    # Dictionary to store the results (Column Name -> List of Values)
    low_card_dict = {}
    
    for col in df.columns:
        # Get the number of unique values (ignoring empty/NaN cells)
        n_unique = df[col].nunique()
        
        # Check if it fits our criteria
        if 0 < n_unique <= max_unique:
            # Extract the actual unique values as a Python list
            # We use dropna() so 'nan' doesn't show up in your clean lists
            unique_vals = df[col].dropna().unique().tolist()
            
            # Save it to our dictionary
            low_card_dict[col] = unique_vals
            
            # Print it out nicely formatted
            print(f"🔹 '{col}' ({n_unique} values):")
            print(f"   {unique_vals}\n")
            
    if not low_card_dict:
        print("No columns matched this criteria.")
        
    return low_card_dict

def analyze_single_value_columns(df):
    print("--- Analyzing 1-Value Columns ---")
    single_val_cols = [col for col in df.columns if df[col].nunique(dropna=True) == 1]

    for col in single_val_cols:
        val = df[col].dropna().unique()[0]
        nan_count = df[col].isna().sum()
        total_rows = len(df)
        nan_percent = (nan_count / total_rows) * 100
        
        print(f"🔹 '{col}':")
        print(f"   Value: [{val}]")
        print(f"   Empty (NaN): {nan_count} rows ({nan_percent:.1f}%)\n")

def preview_sparse_columns(df, threshold=85):
    """
    Shows the unique values of columns that have a missing percentage >= threshold.
    This helps you see what will be compressed before running the consolidation.
    """
    # Calculate missing percentage
    missing_percent = (df.isnull().sum() / len(df)) * 100
    
    # Filter columns
    sparse_cols = missing_percent[missing_percent >= threshold].index.tolist()
    
    # Ignore 'Other' if it already exists from a previous run
    if 'Other' in sparse_cols:
        sparse_cols.remove('Other')
        
    print(f"--- Previewing {len(sparse_cols)} columns with >= {threshold}% missing data ---\n")
    
    if not sparse_cols:
        print("No columns met the threshold.")
        return {}
        
    sparse_dict = {}
    for col in sparse_cols:
        pct = missing_percent[col]
        # Get unique non-null values and convert to list
        unique_vals = df[col].dropna().unique().tolist()
        
        # Save to dictionary
        sparse_dict[col] = unique_vals
        
        # Print the results nicely
        print(f"🔹 '{col}' ({pct:.1f}% missing | {len(unique_vals)} unique values):")
        print(f"   {unique_vals}\n")
        
    return sparse_dict