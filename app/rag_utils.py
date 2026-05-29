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

def get_models():
    """Încarcă modelele direct de pe disc (Strict Offline)."""
    global _embedding_model, _reranker_model
    
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
    
    # Colectăm valori unice pentru schema (pentru a ajuta Gemini)
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
    # 1. Generăm fișierele (Payload + Schema) - Acest pas se execută MEREU
    generate_files_only(df_clean, output_json)

    # 2. Încercăm indexarea în Qdrant (Doar dacă suntem în cluster/producție)
    try:
        logger.info(f"🔍 Încercăm conexiunea la Qdrant ({qdrant_url or DEFAULT_QDRANT_URL})...")
        client = get_qdrant_client(url=qdrant_url, api_key=qdrant_api_key)
        
        # Verificăm conexiunea efectivă
        client.get_collections()
        
        logger.info("🧠 Încărcăm modelele AI pentru embedding...")
        embed_model, _ = get_models()

        logger.info(f"📁 Pregătim colecția '{COLLECTION_NAME}' în Qdrant...")
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )

        # Citim datele din payload-ul proaspăt generat
        with open(output_json, 'r', encoding='utf-8') as f:
            qdrant_documents = json.load(f)

        logger.info("⚙️ Generăm vectorii (Embedding)...")
        texts_to_embed = [doc['text_to_embed'] for doc in qdrant_documents]
        embeddings = embed_model.encode(texts_to_embed, show_progress_bar=True)

        points = [
            PointStruct(id=doc['id'], vector=embeddings[i].tolist(), payload=doc['payload'])
            for i, doc in enumerate(qdrant_documents)
        ]

        logger.info("🚀 Încărcăm în Qdrant...")
        client.upload_points(collection_name=COLLECTION_NAME, points=points, batch_size=100)
        logger.info("✅ Indexarea în Qdrant a fost finalizată cu succes!")

    except Exception as e:
        logger.warning(f"⚠️ Nu s-a putut face indexarea în Qdrant: {e}")
        logger.info("💡 Probabil rulezi local. Fișierele JSON au fost generate, dar upload-ul a fost sărit.")

# ==========================================
# 2. SOTA HYBRID SEARCH (CU RERANKING)
# ==========================================
def sota_hybrid_search(state_json: Dict[str, Any], final_top_k: int = 3, qdrant_url: str = None, qdrant_api_key: str = None) -> List[Dict]:
    client = get_qdrant_client(url=qdrant_url, api_key=qdrant_api_key)
    embed_model, reranker_model = get_models()
    
    filters = state_json.get("filters") or {}
    semantic_query = state_json.get("semantic_query", "")
    
    # Conversie ultra-robustă în string (Gemini poate returna listă sau dict uneori)
    if not isinstance(semantic_query, str):
        if isinstance(semantic_query, list):
            semantic_query = " ".join([str(s) for s in semantic_query])
        elif isinstance(semantic_query, dict):
            # Dacă e dict, concatenăm valorile pentru a extrage keywords
            semantic_query = " ".join([str(v) for v in semantic_query.values()])
        else:
            semantic_query = str(semantic_query)
    
    if not semantic_query.strip():
        return []

    # 1. Construim condițiile de filtrare
    conditions = {}
    if filters.get("price_max"):
        conditions["price"] = FieldCondition(key="price", range=Range(lte=float(filters["price_max"])))
    if filters.get("dimensiune"):
        conditions["dimensiune"] = FieldCondition(key="dimensiune", match=MatchValue(value=filters["dimensiune"]))
    if filters.get("category"):
        conditions["category"] = FieldCondition(key="category", match=MatchValue(value=filters["category"]))
    if filters.get("event_type"):
        # MatchValue pe o listă (event_type) în Qdrant funcționează ca "conține"
        conditions["event_type"] = FieldCondition(key="event_type", match=MatchValue(value=filters["event_type"]))

    # 2. Căutare vectorială inițială cu RELAXARE GRADUALĂ
    pool_size = final_top_k * 5 
    query_vector = embed_model.encode(semantic_query).tolist()
    
    # Definim ordinea de relaxare:
    # Pas 1: Toate filtrele active
    # Pas 2: Fără preț (adesea cel mai restrictiv)
    # Pas 3: Fără dimensiune
    # Pas 4: Doar categoria sau event_type (trăsăturile structurale de bază)
    # Pas 5: Fără niciun filtru (căutare pur semantică)
    
    search_strategies = [
        list(conditions.values()), # TOATE
        [v for k, v in conditions.items() if k != "price"], # Fără preț
        [v for k, v in conditions.items() if k not in ["price", "dimensiune"]], # Fără preț și dimensiune
        [v for k, v in conditions.items() if k in ["category", "event_type"]], # Doar categoria și tipul de eveniment
        [] # Nimic (Doar semantic)
    ]

    initial_results = []
    strategy_names = ["TOATE FILTRELE", "FĂRĂ PREȚ", "FĂRĂ PREȚ ȘI DIMENSIUNE", "DOAR CATEGORIA/EVENIMENT", "SEMANTIC ONLY"]
    
    try:
        for i, must_cond in enumerate(search_strategies):
            # Dacă strategia curentă are aceleași filtre ca precedenta (ex: nu aveam preț de la început), o sărim
            if i > 0 and must_cond == search_strategies[i-1]:
                continue
                
            current_filter = Filter(must=must_cond) if must_cond else None
            
            response = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                query_filter=current_filter,
                limit=pool_size
            )
            
            if response.points:
                initial_results = response.points
                if i > 0:
                    logger.info(f"💡 Rezultate găsite folosind strategia de fallback: {strategy_names[i]}")
                break
            else:
                if must_cond: # Logăm doar dacă chiar am avut filtre care au eșuat
                    logger.warning(f"🔍 Niciun rezultat cu strategia: {strategy_names[i]}. Încercăm următoarea...")

    except Exception as e:
        logger.error(f"❌ Eroare la interogarea Qdrant: {e}")
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
