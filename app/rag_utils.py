import os
import json
import uuid
import pandas as pd
from typing import Dict, List, Any

# Librării NLP
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue, Range

from app.config import Config

# ==========================================
# CONFIGURĂRI GLOBALE
# ==========================================
COLLECTION_NAME = "city_flowers_catalog"

DEFAULT_QDRANT_URL = Config.QDRANT_URL
DEFAULT_QDRANT_API_KEY = Config.QDRANT_API_KEY

EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
VECTOR_SIZE = 384

# SOLUȚIA: Utilizăm un model standard de Cross-Encoder care nu necesită cod custom
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

_embedding_model = None
_reranker_model = None
_qdrant_client = None

def format_context_for_llm(results: List[Dict]) -> str:
    if not results: return "Nu am găsit produse potrivite."
    out = "PRODUSE RECOMANDATE (din baza de date):\n"
    for i, r in enumerate(results, 1):
        out += f"{i}. {r['title']} | Preț: {r['price']} RON\nDetalii: {r['context']}\n---\n"
    return out

def get_qdrant_client(url=None, api_key=None):
    global _qdrant_client
    final_url = url if url is not None else DEFAULT_QDRANT_URL
    final_key = api_key if api_key is not None else DEFAULT_QDRANT_API_KEY
    
    print(f"🔌 Ne conectăm la Qdrant ({final_url})...")
    _qdrant_client = QdrantClient(url=final_url, api_key=final_key)
    return _qdrant_client

def get_models():
    """Încarcă modelele standard."""
    global _embedding_model, _reranker_model
    
    if _embedding_model is None:
        print(f"🧠 Încărcăm modelul de Embedding: {EMBEDDING_MODEL_NAME}...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
    if _reranker_model is None:
        print(f"🎯 Încărcăm modelul SOTA de Reranking: {RERANKER_MODEL_NAME}...")
        
        # Oprim fast tokenizer pentru a evita bug-ul de serializare din Windows/Rust
        _reranker_model = CrossEncoder(
            RERANKER_MODEL_NAME, 
            tokenizer_args={"use_fast": False}
        )
        
    return _embedding_model, _reranker_model

# ==========================================
# 1. CREARE ȘI INDEXARE BAZĂ DE DATE (MODEL BUILDING)
# ==========================================
def build_and_index_rag(df_clean: pd.DataFrame, output_json: str = 'qdrant_payload.json', qdrant_url: str = None, qdrant_api_key: str = None):
    print("📦 Construim Payload-ul și Vectorii...")
    qdrant_documents = []
    
    def parse_list(val):
        if pd.isna(val) or str(val).strip() == "": return []
        return [item.strip() for item in str(val).replace(';', ',').split(',') if item.strip()]

    def extract_base_price(price_val):
        if pd.isna(price_val) or str(price_val).strip() == "": return 0.0
        try: return float(str(price_val).split(';')[0].strip())
        except ValueError: return 0.0

    for index, row in df_clean.iterrows():
        semantic_text = f"Produs: {row.get('Title', '')}.\n"
        for col in ['Type', 'Flowers Type', 'Colors', 'Dimensiune', 'Event Type', 'Body (HTML)']:
            if pd.notna(row.get(col)):
                semantic_text += f"{col}: {row[col]}\n"
        
        payload = {
            "title": str(row.get('Title', '')),
            "handle": str(row.get('Handle', f'produs-{index}')),
            "price": extract_base_price(row.get('Variant Price')),
            "category": str(row.get('Type', '')) if pd.notna(row.get('Type')) else "",
            "dimensiune": str(row.get('Dimensiune', '')) if pd.notna(row.get('Dimensiune')) else "",
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
    print(f"💾 Backup salvat în {output_json}.")

    embed_model, _ = get_models()
    client = get_qdrant_client(url=qdrant_url, api_key=qdrant_api_key)

    print(f"📁 Pregătim colecția '{COLLECTION_NAME}'...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )

    print("⚙️ Generăm vectorii (Embedding)...")
    texts_to_embed = [doc['text_to_embed'] for doc in qdrant_documents]
    embeddings = embed_model.encode(texts_to_embed, show_progress_bar=True)

    points = [
        PointStruct(id=doc['id'], vector=embeddings[i].tolist(), payload=doc['payload'])
        for i, doc in enumerate(qdrant_documents)
    ]

    print("🚀 Încărcăm în Qdrant...")
    client.upload_points(collection_name=COLLECTION_NAME, points=points, batch_size=100)
    print("✅ Baza de date a fost populată cu succes!")

# ==========================================
# 2. SOTA HYBRID SEARCH (CU RERANKING)
# ==========================================
def sota_hybrid_search(state_json: Dict[str, Any], final_top_k: int = 3, qdrant_url: str = None, qdrant_api_key: str = None) -> List[Dict]:
    client = get_qdrant_client(url=qdrant_url, api_key=qdrant_api_key)
    embed_model, reranker_model = get_models()
    
    filters = state_json.get("filters", {})
    semantic_query = state_json.get("semantic_query", "")
    
    if not semantic_query.strip():
        return []

    # 1. Construim filtrele Qdrant
    must_conditions = []
    if filters.get("price_max"):
        must_conditions.append(FieldCondition(key="price", range=Range(lte=float(filters["price_max"]))))
    if filters.get("dimensiune"):
        must_conditions.append(FieldCondition(key="dimensiune", match=MatchValue(value=filters["dimensiune"])))
    if filters.get("category"):
        must_conditions.append(FieldCondition(key="category", match=MatchValue(value=filters["category"])))
    
    qdrant_filter = Filter(must=must_conditions) if must_conditions else None

    # 2. Căutare vectorială inițială
    pool_size = final_top_k * 5 
    query_vector = embed_model.encode(semantic_query).tolist()
    
    try:
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=pool_size
        )
        initial_results = response.points 
    except Exception as e:
        print(f"❌ Eroare la interogarea Qdrant: {e}")
        return []

    if not initial_results:
        return []

    # 3. Reranking cu Cross-Encoder
    rerank_pairs = [[semantic_query, hit.payload.get("page_content", "")] for hit in initial_results]
    cross_scores = reranker_model.predict(rerank_pairs)

    # 4. Extragem datele în dicționare standard Python (ocolim restricțiile Pydantic)
    combined_results = []
    for i, hit in enumerate(initial_results):
        combined_results.append({
            "score": round(float(cross_scores[i]) if not isinstance(cross_scores[i], (list, tuple)) and getattr(cross_scores[i], "size", 1) == 1 else float(max(cross_scores[i])), 4),
            "title": hit.payload.get("title"),
            "price": hit.payload.get("price"),
            "handle": hit.payload.get("handle"),
            "context": hit.payload.get("page_content") 
        })
        
    # 5. Sortăm descrescător după precizia Cross-Encoder-ului
    reranked_results = sorted(combined_results, key=lambda x: x["score"], reverse=True)
        
    # 6. Returnăm doar Top K
    return reranked_results[:final_top_k]