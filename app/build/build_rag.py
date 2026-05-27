import os
import pandas as pd
import numpy as np
import app.build.eda_utils as eda
import app.rag_utils as rag
from app.config import Config
from app.logger import get_logger

logger = get_logger(__name__)

def main():
    logger.info("🚀 Starting RAG Build Process (Production Mode)...")
    
    # 1. Load data
    df = eda.load_data(Config.DATASET_PATH)
    if df is None:
        logger.error(f"❌ Failed to load data from {Config.DATASET_PATH}. Exiting.")
        return

    # 2. Clean data
    logger.info("🧹 Cleaning data...")
    df = eda.clean_column_names(df)
    df = eda.drop_empty_columns(df)
    
    # Specific notebook drops
    cols_to_drop = [
        'Published', 'Option1 Name', 'Option2 Name', 'Variant Taxable', 
        'Complementary Heading', 'Product rating count', 'Dietary preferences', 
        'Plant characteristics', 'Status', 'Image Position', 'Variant SKU',
        'Cost per item', 'Shape', 'Variant Image', 'Option2 Value', 
        'Variant Compare At Price', 'Google Shopping / Google Product Category', 
        'Custom Label #3'
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Notebook renames and transformations
    if 'Option1 Value' in df.columns:
        df['Option1 Value'] = df['Option1 Value'].replace('Default Title', np.nan)
        
    df = df.rename(columns={
        'Option1 Value': 'Details',
        'Badges': 'Tags2',
        'Custom Label #1': 'Sezon',
        'Custom Label #2': 'Parfum',
        'Custom Label #3': 'Polen',
        'Custom Label #4': 'Cadou',
        'Custom Label #5': 'Dimensiune'
    })
    
    if 'Dimensiune' in df.columns:
        df['Dimensiune'] = df['Dimensiune'].str.replace('Dimensiune ', '', regex=False).str.strip()

    df = eda.clean_html_column(df, column_name='Body (HTML)')
    df = eda.aggregate_by_handle(df)
    
    # 3. Build and Index RAG
    logger.info("📦 Building and Indexing RAG...")
    # Using the template directory for payload output
    output_payload = os.path.join(Config.TEMPLATE_DIR, "qdrant_payload.json")
    
    rag.build_and_index_rag(
        df, 
        output_json=output_payload, 
        qdrant_url=Config.QDRANT_URL, 
        qdrant_api_key=Config.QDRANT_API_KEY
    )
    
    logger.info(f"✅ RAG Build Process Completed! Payload saved to {output_payload}")

if __name__ == "__main__":
    main()
