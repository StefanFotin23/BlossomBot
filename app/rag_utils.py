import os
import json
import uuid
import logging
import pandas as pd
from typing import Dict, List, Any

# Librării NLP
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue, Range

from app.config import Config
from app.logger import get_logger

# Standard logger name
logger = get_logger(__name__)

# ==========================================
# CONFIGURĂRI GLOBALE
# ==========================================
COLLECTION_NAME = "city_flowers_catalog"

DEFAULT_QDRANT_URL = Config.QDRANT_URL
DEFAULT_QDRANT_API_KEY = Config.QDRANT_API_KEY

# Căi locale către modele (descărcate la build)
EMBEDDING_MODEL_PATH = '/app/models/embedding_model'
RERANKER_MODEL_PATH = '/app/models/reranker_model'
VECTOR_SIZE = 384

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
    
    logger.info(f"🔌 Ne conectăm la Qdrant ({final_url})...")
    _qdrant_client = QdrantClient(url=final_url, api_key=final_key)
    return _qdrant_client

import threading
import asyncio

_model_lock = threading.Lock()
# STRICT MEMORY GOVERNOR: Limit to 2 concurrent NLP tasks
_nlp_semaphore = asyncio.Semaphore(2) 

def get_models():
    """Încarcă modelele direct de pe disc (Strict Offline) - Thread Safe."""
    global _embedding_model, _reranker_model
    
    with _model_lock:
        if _embedding_model is None:
            logger.info(f"🧠 Încărcăm modelul de Embedding de la: {EMBEDDING_MODEL_PATH}...")
            _embedding_model = SentenceTransformer(
                EMBEDDING_MODEL_PATH, 
                device='cpu',
                local_files_only=True
            )
            
        if _reranker_model is None:
            logger.info(f"🎯 Încărcăm modelul SOTA de Reranking de la: {RERANKER_MODEL_PATH}...")
            _reranker_model = CrossEncoder(
                RERANKER_MODEL_PATH, 
                tokenizer_args={"use_fast": False},
                device='cpu',
                local_files_only=True
            )
            
    return _embedding_model, _reranker_model

# ==========================================
# 1. CREARE ȘI INDEXARE BAZĂ DE DATE (MODEL BUILDING)
# ==========================================
def generate_schema(df_clean: pd.DataFrame, schema_output_path: str):
    """Generează schema de valori unice pentru Gemini fără a interoga Qdrant."""
    logger.info("📂 Generăm schema de filtre din dataset...")
    
    def parse_list(val):
        if pd.isna(val) or str(val).strip() == "": return []
        cleaned = str(val).replace('\n', ',').replace(';', ',')
        return sorted(list(set([item.strip() for item in cleaned.split(',') if item.strip()])))

    unique_values = {
        "categories": sorted(list(set(df_clean['Type'].dropna().astype(str).unique()))),
        "dimensiuni": sorted(list(set(df_clean['Dimensiune'].dropna().astype(str).unique()))),
        "event_types": set()
    }
    
    for _, row in df_clean.iterrows():
        et_list = parse_list(row.get('Event Type'))
        for et in et_list:
            unique_values["event_types"].add(et)
            
    unique_values["event_types"] = sorted(list(unique_values["event_types"]))
    
    with open(schema_output_path, 'w', encoding='utf-8') as f:
        json.dump(unique_values, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ Schema a fost salvată în {schema_output_path}")

def generate_files_only(df_clean: pd.DataFrame, output_json: str):
    """Generează doar fișierele JSON (Payload și Schema) fără a folosi AI sau Qdrant."""
    logger.info("📦 Procesăm datele pentru fișierele locale...")
    qdrant_documents = []
    
    unique_values = {
        "categories": set(),
        "dimensiuni": set(),
        "event_types": set(),
        "colors": set(),
        "flowers_types": set(),
        "sezoane": set()
    }
    
    def parse_list(val):
        if pd.isna(val) or str(val).strip() == "": return []
        cleaned = str(val).replace('\n', ',').replace(';', ',')
        return sorted(list(set([item.strip() for item in cleaned.split(',') if item.strip()])))

    def extract_base_price(price_val):
        if pd.isna(price_val) or str(price_val).strip() == "": return 0.0
        try:
            val = str(price_val).split(';')[0].strip()
            return float(val)
        except (ValueError, IndexError): return 0.0

    for index, row in df_clean.iterrows():
        semantic_text = f"Produs: {row.get('Title', '')}.\n"
        relevant_cols = {
            'Type': 'Categorie', 
            'Flowers Type': 'Flori', 
            'Colors': 'Culori', 
            'Dimensiune': 'Dimensiune', 
            'Event Type': 'Eveniment', 
            'Sezon': 'Sezon',
            'Body (HTML)': 'Descriere'
        }
        
        for col, label in relevant_cols.items():
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                semantic_text += f"{label}: {val}\n"
        
        category = str(row.get('Type', '')) if pd.notna(row.get('Type')) else ""
        dimensiune = str(row.get('Dimensiune', '')) if pd.notna(row.get('Dimensiune')) else ""
        event_types = parse_list(row.get('Event Type'))
        colors = parse_list(row.get('Colors'))
        flowers_types = parse_list(row.get('Flowers Type'))
        sezoane = parse_list(row.get('Sezon'))
        
        if category: unique_values["categories"].add(category)
        if dimensiune: unique_values["dimensiuni"].add(dimensiune)
        for et in event_types: unique_values["event_types"].add(et)
        for c in colors: unique_values["colors"].add(c)
        for ft in flowers_types: unique_values["flowers_types"].add(ft)
        for s in sezoane: unique_values["sezoane"].add(s)

        payload = {
            "title": str(row.get('Title', '')),
            "price": extract_base_price(row.get('Variant Price')),
            "category": category,
            "dimensiune": dimensiune,
            "event_type": event_types,
            "colors": colors,
            "flowers_type": flowers_types,
            "sezon": sezoane,
            "page_content": semantic_text 
        }

        qdrant_documents.append({
            "id": str(uuid.uuid4()),
            "payload": payload,
            "text_to_embed": semantic_text 
        })
        
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(qdrant_documents, f, ensure_ascii=False, indent=2)
    logger.info(f"💾 Payload salvat în {output_json}.")

    schema_path = os.path.join(os.path.dirname(output_json), "schema.json")
    schema_data = {k: sorted(list(v)) for k, v in unique_values.items()}
    with open(schema_path, 'w', encoding='utf-8') as f:
        json.dump(schema_data, f, ensure_ascii=False, indent=2)
    logger.info(f"📂 Schema salvată în {schema_path}.")

def build_and_index_rag(df_clean: pd.DataFrame, output_json: str = 'qdrant_payload.json', qdrant_url: str = None, qdrant_api_key: str = None):
    generate_files_only(df_clean, output_json)
    try:
        logger.info(f"🔍 Încercăm conexiunea la Qdrant ({qdrant_url or DEFAULT_QDRANT_URL})...")
        client = get_qdrant_client(url=qdrant_url, api_key=qdrant_api_key)
        client.get_collections()
        embed_model, _ = get_models()
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        with open(output_json, 'r', encoding='utf-8') as f:
            qdrant_documents = json.load(f)
        texts_to_embed = [doc['text_to_embed'] for doc in qdrant_documents]
        embeddings = embed_model.encode(texts_to_embed, show_progress_bar=True)
        points = [
            PointStruct(id=doc['id'], vector=embeddings[i].tolist(), payload=doc['payload'])
            for i, doc in enumerate(qdrant_documents)
        ]
        client.upload_points(collection_name=COLLECTION_NAME, points=points, batch_size=100)
    except Exception as e:
        logger.warning(f"⚠️ Nu s-a putut face indexarea în Qdrant: {e}")

# ==========================================
# 2. SOTA HYBRID SEARCH (CU RERANKING) - ASYNC FRIENDLY
# ==========================================
async def sota_hybrid_search(state_json: Dict[str, Any], filter_scores: Dict[str, int] = None, final_top_k: int = 3, qdrant_url: str = None, qdrant_api_key: str = None) -> (List[Dict], Dict[str, Any]):
    client = get_qdrant_client(url=qdrant_url, api_key=qdrant_api_key)
    embed_model, reranker_model = await asyncio.to_thread(get_models)
    
    filters = state_json.get("filters") or {}
    semantic_query = state_json.get("semantic_query", "")
    
    conditions_map = {}
    negative_map = {}

    def build_qdrant_condition(key, val, is_negative=False):
        if val is None or str(val).strip() == "" or str(val).lower() == "nan": return None
        try:
            if key == "price_max" and not is_negative: return FieldCondition(key="price", range=Range(lte=float(val)))
            if key == "price_min" and not is_negative: return FieldCondition(key="price", range=Range(gte=float(val)))
            if key == "category": return FieldCondition(key="category", match=MatchValue(value=str(val)))
            if key == "dimensiune": return FieldCondition(key="dimensiune", match=MatchValue(value=str(val)))
            if key == "event_type": return FieldCondition(key="event_type", match=MatchValue(value=str(val)))
            if key == "colors": return FieldCondition(key="colors", match=MatchValue(value=str(val)))
            if key == "flowers_type": return FieldCondition(key="flowers_type", match=MatchValue(value=str(val)))
        except: return None
        return None

    for k, v in filters.items():
        cond = build_qdrant_condition(k, v)
        if cond: conditions_map[k] = cond

    negative_filters = state_json.get("negative_filters") or {}
    for k, v in negative_filters.items():
        cond = build_qdrant_condition(k, v, is_negative=True)
        if cond: negative_map[f"no_{k}"] = cond

    # Throttle Embedding
    async with _nlp_semaphore:
        query_vector = await asyncio.to_thread(embed_model.encode, semantic_query)
    query_vector = query_vector.tolist()

    all_active_keys = list(conditions_map.keys()) + list(negative_map.keys())
    # Critical: Keep Category and Flowers Type at the end of the fallback list
    fallback_order = ["price_max", "price_min", "colors", "dimensiune", "event_type", "flowers_type", "category"]

    def get_sort_score(key):
        score = filter_scores.get(key, 0) if filter_scores else 0
        base_key = key.replace("no_", "")
        base_priority = fallback_order.index(base_key) if base_key in fallback_order else 99
        return (score, base_priority)

    # Sort so that items with lower priority (index in fallback_order) are removed first
    sorted_keys = sorted(all_active_keys, key=get_sort_score)
    search_strategies = [{"name": "ALL", "pos": list(conditions_map.keys()), "neg": list(negative_map.keys())}]
    current_pos = list(conditions_map.keys())
    current_neg = list(negative_map.keys())

    for key_to_remove in sorted_keys:
        # HARD CONSTRAINT: Prețul nu se relaxează niciodată
        if key_to_remove in ["price_max", "price_min"]: continue
        
        if key_to_remove.startswith("no_"):
            if key_to_remove in current_neg: current_neg.remove(key_to_remove)
        else:
            if key_to_remove in current_pos: current_pos.remove(key_to_remove)
        search_strategies.append({"name": f"RELAXED_{key_to_remove}", "pos": list(current_pos), "neg": list(current_neg)})

    initial_results = []
    failed_filters = []
    used_strategy = "SEMANTIC"

    try:
        for strategy in search_strategies:
            must_cond = [conditions_map[k] for k in strategy["pos"]]
            must_not_cond = [negative_map[k] for k in strategy["neg"]]
            current_filter = Filter(must=must_cond if must_cond else None, must_not=must_not_cond if must_not_cond else None)
            response = client.query_points(collection_name=COLLECTION_NAME, query=query_vector, query_filter=current_filter if (must_cond or must_not_cond) else None, limit=final_top_k * 5)
            if response.points:
                initial_results = response.points
                used_strategy = strategy["name"]
                failed_filters = [k for k in all_active_keys if k not in (strategy["pos"] + strategy["neg"])]
                break
    except Exception as e:
        logger.error(f"❌ Qdrant error: {e}")
        return [], {"error": str(e)}

    if not initial_results: return [], {"strategy": "NONE", "failed_filters": all_active_keys}

    # Throttle Reranking
    rerank_pairs = [[semantic_query, hit.payload.get("page_content", "")] for hit in initial_results]
    async with _nlp_semaphore:
        cross_scores = await asyncio.to_thread(reranker_model.predict, rerank_pairs)

    combined_results = []
    seen_titles = set(); seen_categories = set()
    all_hits = sorted(enumerate(initial_results), key=lambda x: cross_scores[x[0]], reverse=True)
    
    if len(failed_filters) >= 2 or used_strategy == "SEMANTIC":
        import random
        if len(all_hits) > 1:
            top_hit = all_hits[0]; rest_hits = all_hits[1:10]; random.shuffle(rest_hits)
            all_hits = [top_hit] + rest_hits + all_hits[10:]
            
    for i, hit in all_hits:
        title = hit.payload.get("title"); category = hit.payload.get("category")
        if title in seen_titles: continue
        if "SEMANTIC" in used_strategy and category in seen_categories and len(combined_results) < final_top_k: continue
        score_val = cross_scores[i]
        if hasattr(score_val, "item"): score_val = score_val.item()
        elif isinstance(score_val, (list, tuple)): score_val = score_val[0]
        combined_results.append({
            "score": round(float(score_val), 4),
            "title": title,
            "price": hit.payload.get("price"),
            "handle": hit.payload.get("handle"),
            "context": hit.payload.get("page_content") 
        })
        seen_titles.add(title); seen_categories.add(category)
        if len(combined_results) >= final_top_k: break
            
    reranked_results = sorted(combined_results, key=lambda x: x["score"], reverse=True)
    return reranked_results[:final_top_k], {"strategy": used_strategy, "failed_filters": failed_filters, "total_found": len(reranked_results)}
