import pandas as pd
import re

def load_data(file_path):
    """Loads the CSV file into a Pandas DataFrame."""
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        print(f"Success! Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def drop_empty_columns(df):
    """Drops columns that contain 100% NaN/missing values and prints them paginated."""
    
    # Quick helper function to format lists into rows of 10
    def paginate_list(lst, chunk_size=10):
        # Slice the list into chunks of 10
        chunks = [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
        # Join items in a chunk with commas, then join chunks with newlines
        return "\n".join([", ".join(f"'{item}'" for item in chunk) for chunk in chunks])

    # 1. Save the original columns
    original_columns = df.columns.tolist()
    initial_cols = len(original_columns)
    
    # 2. Drop the empty columns
    df_cleaned = df.dropna(axis=1, how='all')
    remaining_columns = df_cleaned.columns.tolist()
    
    # 3. Find exactly which columns were removed
    dropped_column_names = [col for col in original_columns if col not in remaining_columns]
    
    # Print the results using our paginator
    print(f"Original column count: {initial_cols}")
    print(f"--- Original Column Order ---\n{paginate_list(original_columns)}\n")
    
    print(f"Dropped {len(dropped_column_names)} completely empty columns.")
    if dropped_column_names:
        print(f"--- Dropped Columns ---\n{paginate_list(dropped_column_names)}\n")
    else:
        print("--- Dropped Columns ---\nNone\n")
    
    print(f"Remaining columns: {len(remaining_columns)}")
    
    return df_cleaned

def consolidate_sparse_columns(df, threshold=85):
    """
    Consolidates columns with a missing percentage >= threshold into a single 'Other' column.
    Format: 'col1=value, col2=value2'
    """
    # 1. Calculate the percentage of missing values for each column
    missing_percent = (df.isnull().sum() / len(df)) * 100
    
    # 2. Identify columns that meet or exceed the missing threshold
    # Note: We exclude 'Other' just in case you run this multiple times
    sparse_cols = missing_percent[missing_percent >= threshold].index.tolist()
    if 'Other' in sparse_cols:
        sparse_cols.remove('Other')
        
    print(f"Found {len(sparse_cols)} columns with >= {threshold}% missing data.")
    
    if not sparse_cols:
        print("No columns met the threshold to consolidate.")
        return df

    # 3. Define a helper function to build the 'col=val' string for each row
    def build_other_string(row):
        items = []
        for col in sparse_cols:
            val = row[col]
            # Check if the value is not null and not an empty string
            if pd.notna(val) and val != "":
                items.append(f"{col}={val}")
        
        # Return the joined string, or None if the list is empty
        return ", ".join(items) if items else None

    # 4. Apply the function to create the new 'Other' column
    # If 'Other' already exists, we could append to it, but here we assume a fresh creation
    if 'Other' in df.columns:
        # Merge with existing 'Other' column just in case
        temp_other = df.apply(build_other_string, axis=1)
        df['Other'] = df['Other'].astype(str) + ", " + temp_other.astype(str)
        # Clean up weird trailing/leading commas if one was None
        df['Other'] = df['Other'].str.replace('None, |, None', '', regex=True)
    else:
        df['Other'] = df.apply(build_other_string, axis=1)
    
    # 5. Drop the original sparse columns
    df_cleaned = df.drop(columns=sparse_cols)
    
    print(f"Successfully consolidated {len(sparse_cols)} columns into 'Other'.")
    print(f"Remaining columns: {df_cleaned.shape[1]}")
    
    return df_cleaned

def clean_column_names(df):
    """Removes ANY messy metafield tags from column names."""
    new_names = {}
    for col in df.columns:
        # Stronger Rule: Looks for any parentheses containing the word 'metafield'
        clean_name = re.sub(r'\s*\([^)]*metafield[^)]*\)', '', col)
        
        # Also clean up any extra spaces at the beginning or end just in case
        new_names[col] = clean_name.strip()
        
    df_cleaned = df.rename(columns=new_names)
    print("\n--- Cleaned Column Names! ---")
    
    # Optional: Print out a few examples of what changed so you can verify
    changed_cols = [f"'{old}' -> '{new}'" for old, new in new_names.items() if old != new]
    if changed_cols:
        print(f"Fixed {len(changed_cols)} columns. Examples:")
        for c in changed_cols[:5]:
            print(f"  {c}")
            
    return df_cleaned

def aggregate_products(df):
    """
    Groups the DataFrame by 'Handle' (the unique product identifier).
    Compresses variant rows into a single product row.
    """
    print(f"\n--- Aggregating {len(df)} rows into unique Products... ---")
    
    # This is the secret sauce: our custom logic for combining rows
    def compress_data(series):
        # 1. Drop the NaNs and find all unique values in this group
        uniques = series.dropna().unique()
        
        # 2. Decide what to do based on how many unique values we found
        if len(uniques) == 0:
            return pd.NA                  # It was completely empty
        elif len(uniques) == 1:
            return uniques[0]             # It's a standard Parent column (e.g., Title)
        else:
            # It's a Variant column with multiple values (e.g., Prices, Sizes)
            # We join them together with a ' | ' separator
            return " | ".join([str(x) for x in uniques])

    # Group everything by the 'Handle' column and apply our compression logic
    df_agg = df.groupby('Handle', as_index=False).agg(compress_data)
    
    print(f"Success! Compressed down to {len(df_agg)} unique products.")
    return df_agg