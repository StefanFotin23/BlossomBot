import os
from sentence_transformers import SentenceTransformer, CrossEncoder

# Definim căile locale unde vor fi salvate modelele
MODEL_DIR = "/app/models"
EMBEDDING_PATH = os.path.join(MODEL_DIR, "embedding_model")
RERANKER_PATH = os.path.join(MODEL_DIR, "reranker_model")

# Numele modelelor de pe Hub
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

def download():
    print(f"📥 Downloading Embedding model to {EMBEDDING_PATH}...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embed_model.save(EMBEDDING_PATH)
    
    print(f"📥 Downloading Reranker model to {RERANKER_PATH}...")
    reranker_model = CrossEncoder(RERANKER_MODEL_NAME)
    reranker_model.save(RERANKER_PATH)
    
    print("✅ All models saved locally for offline use!")

if __name__ == "__main__":
    download()
