import re
import json
import os
import asyncio
from app import llm_agent as llm
from app import rag_utils as rag
from app.config import Config
from app.logger import get_logger

import redis
import json

logger = get_logger(__name__)

# --- Redis & State Management ---
redis_client = None
try:
    redis_client = redis.Redis(
        host=Config.REDIS_HOST,
        port=Config.REDIS_PORT,
        db=Config.REDIS_DB,
        password=Config.REDIS_PASSWORD,
        decode_responses=True,
        socket_timeout=2
    )
    # Test connection
    redis_client.ping()
    logger.info("✅ Connected to Redis for state management.")
except Exception as e:
    logger.warning(f"⚠️ Redis not available ({e}). Falling back to in-memory storage.")
    redis_client = None

# In-memory storage (Fallback)
_local_user_states = {}

async def get_user_state(user_id):
    """Obține starea utilizatorului din Redis sau memorie locală."""
    default_state = {"state": "GREETING", "history": [], "filter_scores": {}}
    
    if redis_client:
        try:
            data = redis_client.get(f"user:{user_id}")
            if data:
                return json.loads(data)
            return default_state
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            
    return _local_user_states.get(user_id, default_state)

async def set_user_state(user_id, state_data):
    """Salvează starea utilizatorului în Redis sau memorie locală."""
    if redis_client:
        try:
            redis_client.set(f"user:{user_id}", json.dumps(state_data), ex=3600*24) # TTL 24h
            return
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            
    _local_user_states[user_id] = state_data

async def process_recommendations(user_id, user_query, history=None):
    """Execută flow-ul de RAG cu gestionarea scorurilor de importanță ale filtrelor."""
    user_data = await get_user_state(user_id)
    filter_scores = user_data.get("filter_scores", {})
    
    logger.debug(f"[User: {user_id}] ➡️ Starting search. Query: '{user_query}'")
    logger.debug(f"[User: {user_id}] ➡️ Current Filter Scores: {filter_scores}")

    # 1. Extracție filtre (și detecție mențiuni noi/insistențe)
    extracted_data = llm.extract_filters(
        user_query, 
        history=history,
        api_key=Config.GEMINI_API_KEY
    )

    current_filters = extracted_data.get("filters", {})
    mentioned_now = extracted_data.get("mentioned_now", [])
    insisted = extracted_data.get("insisted", [])
    
    logger.debug(f"[User: {user_id}] 🧠 LLM Extracted Filters: {current_filters}")
    logger.debug(f"[User: {user_id}] 🧠 LLM Mentioned Now: {mentioned_now} | Insisted: {insisted}")

    # 2. Update Scoruri (Logica solicitată: New = +2, Insisted = +2, etc.)
    for f_key in current_filters.keys():
        if f_key not in filter_scores:
            filter_scores[f_key] = 2 # Prima apariție (nou)
        else:
            if f_key in mentioned_now:
                filter_scores[f_key] += 2 # Menționat ulterior (dublu ca număr de apariții/boost)
            if f_key in insisted:
                filter_scores[f_key] += 2 # Insistență (boost extra)

    user_data["filter_scores"] = filter_scores
    await set_user_state(user_id, user_data)
    
    logger.debug(f"[User: {user_id}] 📊 Updated Filter Scores: {filter_scores}")

    # 3. Căutare Qdrant (Trimitem scorurile pentru relaxare dinamică)
    logger.debug(f"[User: {user_id}] 🔍 Searching Qdrant...")
    results, metadata = rag.sota_hybrid_search(
        extracted_data, 
        filter_scores=filter_scores,
        qdrant_url=Config.QDRANT_URL
    )
    
    logger.debug(f"[User: {user_id}] 🏁 Search Results: Found {metadata.get('total_found', 0)} items.")
    logger.debug(f"[User: {user_id}] 🏁 Search Metadata (Relaxation): {metadata}")

    if not results:
        logger.debug(f"[User: {user_id}] ❌ No results found even after relaxation.")
        return "Îmi pare rău, nu am găsit flori care să se potrivească descrierii tale. Poți încerca să modifici puțin cerințele?", []

    # 3. Generare feedback despre filtre eșuate
    failed = metadata.get("failed_filters", [])
    feedback_msg = ""
    if failed:
        filter_map = {
            "price_max": "sub prețul dorit",
            "category": "din categoria cerută",
            "dimensiune": "de mărimea dorită",
            "event_type": "pentru acest tip de eveniment",
            "colors": "în culorile solicitate",
            "flowers_type": "cu tipul de flori specificat"
        }
        failed_texts = [filter_map.get(f, f) for f in failed]
        
        if len(failed_texts) > 1:
            joined_failed = ", ".join(failed_texts[:-1]) + " sau " + failed_texts[-1]
        else:
            joined_failed = failed_texts[0]
            
        feedback_msg = f"⚠️ Momentan nu avem opțiuni {joined_failed}, dar iată cele mai apropiate alternative pe care ți le pot recomanda:\n\n"

    # 4. Generare răspuns hibrid
    response_text = feedback_msg + llm.generate_hybrid_response(results)
    return response_text, results


async def retry_recommendations(user_id, channel_id, user_query, history, client):
    """Background task care reîncearcă procesarea până când Gemini este disponibil."""
    max_retries = 10
    retry_delay = 5 # secunde
    
    logger.info(f"🔄 Starting background retry for user {user_id}")
    
    for i in range(max_retries):
        try:
            await asyncio.sleep(retry_delay)
            logger.info(f"🔄 Retry attempt {i+1} for user {user_id}")
            
            response_text, results = await process_recommendations(user_id, user_query, history=history)
            
            # Trimitem răspunsul final
            await client.chat_postMessage(channel=channel_id, text=response_text)
            
            # Actualizăm starea
            user_data = await get_user_state(user_id)
            user_data["last_results"] = results
            user_data["state"] = "SEARCH_PRODUCTS"
            await set_user_state(user_id, user_data)
            
            logger.info(f"✅ Success on retry {i+1} for user {user_id}")
            return
            
        except llm.GeminiOverloadedError:
            logger.warning(f"⚠️ Retry {i+1} failed: Gemini still busy.")
        except Exception as e:
            logger.error(f"❌ Error during retry: {e}")
            break
            
    await client.chat_postMessage(
        channel=channel_id, 
        text="Îmi cer scuze, dar serviciul conversational este extrem de solicitat acum. Te rog să încerci din nou peste câteva minute."
    )

async def handle_message(body, say, client, publish_purchase_callback=None):
    """EntryPoint pentru logica bot-ului. Acționează ca un State Machine robust."""
    event = body["event"]
    user_id = event["user"]
    channel_id = event["channel"]
    text = event["text"]
    
    user_data = await get_user_state(user_id)
    current_state = user_data.get("state")
    
    # 1. Gemini Routing (Determine Intent)
    new_state = llm.classify_intent(text, current_state, api_key=Config.GEMINI_API_KEY)
    logger.info(f"User: {user_id} | State: {current_state} -> {new_state}")

    if new_state == "RESTART":
        user_data = {"state": "GREETING", "history": []}
        await set_user_state(user_id, user_data)
        await say("Nicio problemă, am resetat conversația. Cu ce te pot ajuta?")
        return

    # LOGICA SPECIALĂ PENTRU CONFIRMARE (Așteptăm DA/NU după selecție)
    if current_state == "AWAITING_CONFIRMATION":
        text_clean = text.lower()
        if any(word in text_clean for word in ["da", "confirm", "ok", "sigur", "vrea"]):
            if publish_purchase_callback:
                selected_item = user_data.get("selected_item", "Produs selectat")
                selected_price = user_data.get("selected_price", "N/A")
                purchase_data = {
                    "user_id": user_id,
                    "items": [selected_item],
                    "price": selected_price,
                    "timestamp": event.get("event_ts")
                }
                success = await publish_purchase_callback(purchase_data)
                if success:
                    await say(f"Excelent! Am trimis comanda ta pentru *{selected_item}*. 🎉 O zi frumoasă!")
                    user_data["state"] = "GREETING"
                else:
                    await say("Îmi cer scuze, a apărut o problemă tehnică la procesarea comenzii. Te rog să încerci din nou.")
            else:
                await say("Comanda a fost confirmată intern! (Modul demo)")
                user_data["state"] = "GREETING"
        else:
            await say("Am înțeles, am anulat comanda. Dorești să cauți altceva?")
            user_data["state"] = "SEARCH_PRODUCTS"
        
        await set_user_state(user_id, user_data)
        return

    if new_state == "GREETING":
        await say(f"Bună! Sunt BlossomBot 🌸 asistentul tău virtual. Spune-mi ce fel de flori cauți?")
        user_data["state"] = "SEARCH_PRODUCTS" 
    
    elif new_state == "GENERAL_INQUIRY":
        response_text = llm.generate_general_response(text, api_key=Config.GEMINI_API_KEY)
        await say(response_text)
        
    elif new_state == "PRODUCT_DETAILS":
        last_results = user_data.get("last_results", [])
        selected_index = -1
        
        text_lower = text.lower()
        if "prim" in text_lower or " 1" in text_lower: selected_index = 0
        elif "doua" in text_lower or " 2" in text_lower: selected_index = 1
        elif "treia" in text_lower or " 3" in text_lower: selected_index = 2
        
        if selected_index >= 0 and selected_index < len(last_results):
            product = last_results[selected_index]
            details_text = llm.generate_detailed_product_info(product)
            await say(details_text)
        else:
            if not last_results:
                await say("Caută întâi niște produse, iar apoi te pot ajuta cu detalii despre ele.")
            else:
                await say("Despre care produs dorești mai multe detalii? (ex: 'despre primul')")
        
    elif new_state == "SEARCH_PRODUCTS":
        await say("Analizez cererea ta și verific catalogul nostru...")
        try:
            response_text, results = await process_recommendations(user_id, text, history=user_data.get("history"))
            await say(response_text)
            # Salvăm obiectele complete pentru a putea extrage prețul mai târziu
            user_data["last_results"] = results 
            user_data["state"] = "SEARCH_PRODUCTS"
        except llm.GeminiOverloadedError:
            logger.warning(f"🚨 Gemini Overloaded for user {user_id}. Starting retry task.")
            await say("🌸 Sunt puțin ocupat cu multe cereri acum, dar nu te-am uitat! Revin în câteva secunde cu cele mai bune recomandări pentru tine...")
            # PORNIM TASK-UL ÎN FUNDAL (RAM)
            asyncio.create_task(retry_recommendations(user_id, channel_id, text, user_data.get("history"), client))
            return # Ieșim din handler, mesajul va fi trimis ulterior de task-ul de fundal

    elif new_state == "PURCHASE_INTENT":
        # Încercăm să identificăm ce produs a ales
        last_results = user_data.get("last_results", [])
        selected_index = -1
        
        text_lower = text.lower()
        if "prim" in text_lower or " 1" in text_lower: selected_index = 0
        elif "doua" in text_lower or " 2" in text_lower: selected_index = 1
        elif "treia" in text_lower or " 3" in text_lower: selected_index = 2
        
        if selected_index >= 0 and selected_index < len(last_results):
            product = last_results[selected_index]
            p_name = product.get('title')
            p_price = product.get('price')
            
            user_data["selected_item"] = p_name
            user_data["selected_price"] = p_price
            user_data["state"] = "AWAITING_CONFIRMATION"
            
            await say(f"Vrei să adăugăm *{p_name}* în comandă pentru *{p_price}* RON? (Spune 'Da' pentru confirmare)")
        else:
            if not last_results:
                await say("Spune-mi întâi ce flori cauți, ca să te pot ajuta cu o comandă.")
                user_data["state"] = "SEARCH_PRODUCTS"
            else:
                await say("Nu am înțeles exact pe care dorești să îl cumperi. Îmi poți spune, de exemplu, 'îl vreau pe primul'?")

    # Save State
    user_data["history"].append({"user": text, "state": new_state})
    await set_user_state(user_id, user_data)

