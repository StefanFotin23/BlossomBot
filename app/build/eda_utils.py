from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import json
import uuid
import re
import os
from app.logger import get_logger

logger = get_logger(__name__)

def load_data(file_path):
    logger.info(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Success! Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def clean_column_names(df):
    new_names = {}
    for col in df.columns:
        clean_name = re.sub(r'\s*\([^)]*metafield[^)]*\)', '', col)
        new_names[col] = clean_name.strip()
    return df.rename(columns=new_names)

def drop_empty_columns(df):
    initial_cols = df.shape[1]
    df_cleaned = df.dropna(axis=1, how='all')
    logger.info(f"--- Dropped {initial_cols - df_cleaned.shape[1]} completely empty columns ---")
    return df_cleaned

def get_high_missing_columns(df, threshold=0.8):
    """
    Returns a list of columns that have a missing value percentage strictly greater than the threshold.
    Threshold should be a float between 0 and 1 (e.g., 0.8 means 80% missing).
    """
    # Calculate the fraction of missing values per column
    missing_fractions = df.isnull().mean()

    # Filter columns that exceed the threshold
    sparse_cols = missing_fractions[missing_fractions > threshold].index.tolist()
    return sparse_cols

def display_missing_info(df, min_threshold=0.9, max_threshold=1.0):
    """
    Afișează detalii despre coloanele care au un procent de valori lipsă 
    situat STRICT MAI MARE decât min_threshold și PÂNĂ LA max_threshold.
    """
    # Calculăm procentajul de valori lipsă pentru TOATE coloanele
    missing_fractions = df.isnull().mean()

    # Creăm o "mască" (filtru) pentru intervalul dorit
    # Exemplu: > 0.50 și <= 0.90
    mask = (missing_fractions >= min_threshold) & (missing_fractions <= max_threshold)

    # Aplicăm filtrul și sortăm descrescător din start
    target_fractions = missing_fractions[mask].sort_values(ascending=False)

    logger.info(f"--- Coloane cu valori lipsă între {min_threshold*100}% și {max_threshold*100}% ({len(target_fractions)} coloane) ---")

    if target_fractions.empty:
        logger.info("Nu s-au găsit coloane în acest interval.")
        return

    # Iterăm prin rezultate
    for col, fraction in target_fractions.items():
        pct_missing = fraction * 100

        # Extragem datele coloanei (protecție contra coloanelor duplicate)
        col_data = df[col]
        if isinstance(col_data, pd.DataFrame):
            col_data = col_data.iloc[:, 0]

        # Obținem numărul de valori unice și valorile în sine (fără NaN)
        n_unique = col_data.nunique(dropna=True)
        unique_vals = col_data.dropna().unique().tolist()

        # Afișăm pe un singur rând
        logger.info(f"{col}: {pct_missing:.2f}% missing | {n_unique} unique: {unique_vals}")

def aggregate_sparse_columns_to_other(df, threshold=0.9):
    """
    Identifies sparse columns, concatenates their non-null values into a new 'Other' column 
    in the format 'Col1=val1, Col2=val2', and drops the original sparse columns.
    """
    sparse_cols = get_high_missing_columns(df, threshold)

    if not sparse_cols:
        logger.info("--- No columns exceeded the threshold. 'Other' column will be empty. ---")
        df['Other'] = np.nan
        return df

    logger.info(f"--- Aggregating {len(sparse_cols)} columns into 'Other' ---")

    # Helper function to process row by row
    def build_other_string(row):
        # row.items() gives us (column_name, value) pairs
        # We only keep items that are not null and not empty strings
        items = [f"{col}={val}" for col, val in row.items() if pd.notna(val) and str(val).strip() != ""]
        return ", ".join(items) if items else np.nan

    # Apply the helper function only to the sparse columns portion of the dataframe
    df['Other'] = df[sparse_cols].apply(build_other_string, axis=1)

    # Drop the original sparse columns
    df_cleaned = df.drop(columns=sparse_cols)

    logger.info(f"Success! Dropped {len(sparse_cols)} columns. Dataframe now has {df_cleaned.shape[1]} columns.")
    return df_cleaned

def display_low_cardinality_info(df, threshold=5):
    """
    Finds columns with fewer unique values than the threshold.
    Displays the column name, the number of unique values, and a list of those values
    along with their percentage of appearance (excluding NaNs).
    """
    logger.info(f"--- Columns with <= {threshold} unique values ---")
    count = 0

    for col in df.columns:
        # Get the count of unique values, dropping NaNs so they don't count as a unique value
        n_unique = df[col].nunique(dropna=True)

        # Check if the number of unique values is between 1 and the threshold
        if 0 < n_unique <= threshold:
            # Get the frequencies as percentages
            val_percentages = df[col].value_counts(normalize=True) * 100

            # Create a clean, formatted list of strings like: "Value (45.2%)"
            formatted_vals = []
            for val, pct in val_percentages.items():
                formatted_vals.append(f"'{val}' ({pct:.1f}%)")

            # Join them together for a clean print output
            vals_string = ", ".join(formatted_vals)
            logger.info(f"- {col} ({n_unique} unique): [{vals_string}]")
            count += 1

    if count == 0:
        logger.info("No columns found below this threshold.")

def drop_low_cardinality_columns(df, threshold=1):
    """
    Drops columns that have fewer unique values than the specified threshold.
    By default (threshold=2), it drops constant columns (exactly 1 unique value).
    """
    cols_to_drop = []
    dropped_details = {}

    for col in df.columns:
        # Count unique values, ignoring NaNs
        n_unique = df[col].nunique(dropna=True)

        # If the unique count is strictly less than the threshold
        if 0 < n_unique <= threshold:
            cols_to_drop.append(col)
            # Save the unique values just so we can print them out
            dropped_details[col] = df[col].dropna().unique().tolist()

    # Perform the drop
    if cols_to_drop:
        logger.info(f"--- Dropping {len(cols_to_drop)} columns with < {threshold} unique values ---")
        for col in cols_to_drop:
            logger.info(f" ❌ Dropped [{col}] - Contained: {dropped_details[col]}")

        df_cleaned = df.drop(columns=cols_to_drop)
        logger.info(f"Dataframe reduced from {df.shape[1]} to {df_cleaned.shape[1]} columns.")
    else:
        logger.info(f"--- No columns found with < {threshold} unique values. None dropped. ---")
        df_cleaned = df

    return df_cleaned

def analyze_column(df, col_name, top_n=10):
    """
    Provides a detailed analysis of a single column, including:
    - Missing values (count and percentage)
    - Total number of unique values
    - Distribution of the most frequent values (counts and percentages)
    """
    if col_name not in df.columns:
        logger.error(f"❌ Error: Column '{col_name}' not found in the dataframe.")
        return

    logger.info(f"\n" + "="*50)
    logger.info(f"🔍 ANALYSIS FOR: [{col_name}]")
    logger.info("="*50)

    # 1. Missing Values
    total_rows = len(df)
    missing_count = df[col_name].isnull().sum()
    missing_pct = (missing_count / total_rows) * 100
    logger.info(f"• Missing Values:  {missing_count} rows ({missing_pct:.2f}%)")

    # 2. Unique Values
    n_unique = df[col_name].nunique(dropna=True)
    logger.info(f"• Unique Values:   {n_unique}")
    logger.info("-" * 50)

    if n_unique == 0:
        logger.info("This column has no data (100% empty).")
        return

    # 3. Value Distribution
    logger.info(f"📊 Top {min(top_n, n_unique)} Value Distribution:")
    counts = df[col_name].value_counts(dropna=True)
    percentages = df[col_name].value_counts(normalize=True, dropna=True) * 100

    # Loop through the top N items and print them nicely
    display_limit = min(top_n, n_unique)
    for i, (val, count) in enumerate(counts.head(display_limit).items()):
        pct = percentages.iloc[i]
        logger.info(f"  {i+1:2d}. {count:4d} rows | {pct:5.1f}% | '{val}'")

    # If there are more values than we displayed, show a summary note
    if n_unique > top_n:
        remaining = n_unique - top_n
        logger.info(f"  ... plus {remaining} other unique values.")
    logger.info("="*50 + "\n")

def clean_html_column(df, column_name='Body (HTML)'):
    """
    Parcurge o coloană cu format HTML și extrage doar textul simplu.
    """
    if column_name not in df.columns:
        logger.error(f"❌ Eroare: Coloana '{column_name}' nu a fost găsită în dataframe.")
        return df

    logger.info(f"--- Extragere text din HTML pentru coloana: [{column_name}] ---")

    def extract_text(html_content):
        # Dacă celula este goală (NaN), o lăsăm așa
        if pd.isna(html_content) or str(html_content).strip() == "":
            return np.nan

        # Folosim BeautifulSoup pentru a parsa HTML-ul
        soup = BeautifulSoup(str(html_content), "html.parser")

        # Extragem textul. separator=" " previne lipirea cuvintelor între paragrafe
        # strip=True șterge spațiile goale de la început și sfârșit
        clean_text = soup.get_text(separator=" ", strip=True)

        return clean_text

    # Aplicăm funcția pe toată coloana
    df[column_name] = df[column_name].apply(extract_text)

    logger.info(" ✅ HTML transformat cu succes în text simplu.")
    return df

def aggregate_by_handle(df, separator=" ; "):
    """
    Grupează dataframe-ul după 'Handle' pentru a unifica produsele.
    Dacă o coloană are aceeași valoare pe toate rândurile, o păstrează o singură dată.
    Dacă are valori diferite (ex: SKU-uri, imagini, prețuri), le concatenează.
    """
    if 'Handle' not in df.columns:
        logger.error("❌ Eroare: Coloana 'Handle' nu există în DataFrame.")
        return df

    logger.info(f"--- Agregare produse după 'Handle' ---")
    logger.info(f"Rânduri inițiale: {len(df)}")

    def smart_agg(series):
        # Eliminăm valorile nule (NaN) și transformăm totul în text
        valid_items = series.dropna().astype(str)

        # Extragem doar valorile unice, păstrând ordinea în care au apărut
        unique_items = valid_items.unique()

        if len(unique_items) == 0:
            return np.nan
        elif len(unique_items) == 1:
            # Dacă e o singură valoare unică (ex: același Titlu pe toate 4 rândurile), o lăsăm simplă
            return unique_items[0]
        else:
            # Dacă sunt valori diferite (ex: imagini sau prețuri), le lipim cu separatorul dorit
            return separator.join(unique_items)

    # Grupăm după Handle și aplicăm funcția inteligentă pe restul coloanelor
    df_agg = df.groupby('Handle', as_index=False).agg(smart_agg)

    logger.info(f"✅ Succes! Datele au fost comprimate la {len(df_agg)} rânduri (produse unice).")
    return df_agg

def compress_variations(df, column_name='Variations'):
    """
    Găsește prefixul comun pentru elementele separate prin ';' dintr-o celulă
    și le rescrie într-un format mult mai scurt și mai lizibil.
    """
    if column_name not in df.columns:
        logger.error(f"❌ Eroare: Coloana '{column_name}' nu a fost găsită în DataFrame.")
        return df

    logger.info(f"--- Optimizare text pentru coloana: [{column_name}] ---")

    def process_cell(text):
        if pd.isna(text) or str(text).strip() == "":
            return np.nan

        # 1. Separăm textul după ';' și curățăm eventualele spații libere
        items = [item.strip() for item in str(text).split(';') if item.strip()]

        # Dacă există doar o singură variantă, nu avem ce comprima
        if len(items) <= 1:
            return text

        # 2. Găsim prefixul comun tuturor variantelor
        prefix = os.path.commonprefix(items)

        # Dacă avem un prefix relevant (mai mare de 3 caractere ca să nu fie o coincidență)
        if len(prefix) > 3:
            # 3. Extragem restul de text (sufixele) tăind prefixul din fiecare element
            suffixes = [item[len(prefix):] for item in items]

            # Curățăm prefixul (eliminăm cratima '-' de la final dacă există)
            clean_prefix = prefix.rstrip('- _')

            # 4. Lipim la loc sub noul format: "Prefix: varianta1; varianta2"
            # Am adăugat două puncte (": ") pentru a fi super clar la citire
            return f"{clean_prefix}: " + "; ".join(suffixes)
        else:
            # Dacă nu există nicio parte comună, returnăm textul original
            return text

    # Aplicăm logica pe toată coloana
    df[column_name] = df[column_name].apply(process_cell)

    logger.info(" ✅ Variantele au fost comprimate cu succes!")
    return df

def export_for_qdrant(df_clean, output_json='qdrant_payload.json'):
    logger.info("📦 Pregătim documentele pentru Qdrant...")
    qdrant_documents = []

    # Helper 1: Transformă textul în listă
    def parse_list(val):
        if pd.isna(val) or str(val).strip() == "": return []
        return [item.strip() for item in str(val).replace(';', ',').split(',') if item.strip()]

    # Helper 2: Extrage "Prețul de pornire" (primul preț din listă)
    def extract_base_price(price_val):
        if pd.isna(price_val) or str(price_val).strip() == "":
            return 0.0
        try:
            # Desparte string-ul după ';' și ia primul element (ex: din '500 ; 550' ia '500')
            first_price_str = str(price_val).split(';')[0].strip()
            return float(first_price_str)
        except ValueError:
            return 0.0 # Măsură de siguranță dacă textul nu e număr deloc

    for index, row in df_clean.iterrows():
        # 1. TEXTUL SEMANTIC
        semantic_text = f"Produs: {row['Title']}.\n"
        if pd.notna(row.get('Type')): semantic_text += f"Tip: {row['Type']}.\n"
        if pd.notna(row.get('Flowers Type')): semantic_text += f"Flori: {row['Flowers Type']}.\n"
        if pd.notna(row.get('Colors')): semantic_text += f"Culori: {row['Colors']}.\n"
        if pd.notna(row.get('Dimensiune')): semantic_text += f"Dimensiune: {row['Dimensiune']}.\n"
        if pd.notna(row.get('Event Type')): semantic_text += f"Evenimente: {row['Event Type']}.\n"
        if pd.notna(row.get('Other_Specs')): semantic_text += f"Extra: {row['Other_Specs']}.\n"
        if pd.notna(row.get('Body (HTML)')): semantic_text += f"Descriere: {row['Body (HTML)']}"

        # 2. PAYLOAD-UL
        payload = {
            "title": str(row['Title']),
            "handle": str(row.get('Handle', f'produs-{index}')),
            # Folosim noua funcție aici:
            "price": extract_base_price(row.get('Variant Price')),
            "category": str(row['Type']) if pd.notna(row.get('Type')) else "",
            "dimensiune": str(row['Dimensiune']) if pd.notna(row.get('Dimensiune')) else "",
            "event_type": parse_list(row.get('Event Type')),
            "tags": parse_list(row.get('All_Tags')),
            "page_content": semantic_text 
        }

        qdrant_documents.append({
            "id": str(uuid.uuid4()),
            "payload": payload,
            "text_to_embed": semantic_text 
        })

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(qdrant_documents, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ Gata! Am exportat {len(qdrant_documents)} documente fără erori.")
    return qdrant_documents