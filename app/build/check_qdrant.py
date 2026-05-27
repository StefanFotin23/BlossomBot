import sys
import logging
from qdrant_client import QdrantClient
from app.config import Config
from app.build import build_rag
from app.logger import get_logger

# We use the standard app logger
logger = get_logger(__name__)

def ensure_data_exists():
    client = QdrantClient(host=Config.QDRANT_HOST, port=Config.QDRANT_PORT)
    collection_name = "city_flowers_catalog"
    
    try:
        logger.info(f"🔍 Checking if Qdrant collection '{collection_name}' exists...")
        collections = client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)
        
        if exists:
            # Verificăm și dacă are puncte (nu e doar creată și goală)
            info = client.get_collection(collection_name)
            if info.points_count > 0:
                logger.info(f"✅ Collection found with {info.points_count} products. Ready to go!")
                return True
            else:
                logger.warning("⚠️ Collection exists but it is empty.")

        # Dacă am ajuns aici, fie nu există, fie e goală -> Pornim Indexarea
        logger.info("🚀 Data missing or incomplete. Starting AUTO-INDEXING flow...")
        build_rag.main()
        
        # Verificare finală după indexare
        info_final = client.get_collection(collection_name)
        if info_final.points_count > 0:
            logger.info(f"✨ Auto-indexing successful! {info_final.points_count} products indexed.")
            return True
        else:
            logger.error("❌ Auto-indexing finished but collection is still empty.")
            return False

    except Exception as e:
        logger.error(f"💥 Critical error during data check/indexing: {e}")
        return False

if __name__ == "__main__":
    success = ensure_data_exists()
    if success:
        sys.exit(0)
    else:
        sys.exit(1)
