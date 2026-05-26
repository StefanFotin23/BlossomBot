import re
import json
import os
from app import llm_agent as llm
from app import rag_utils as rag
from app.config import Config
from app.logger import get_logger

logger = get_logger(__name__)

# State configurations
VALID_STATES = ["GREETING", "SEARCH_PRODUCTS", "PURCHASE_INTENT", "RESTART", "GENERAL_INQUIRY"]

# In-memory storage for MVP (consider Redis for multi-instance production)
user_states = {}

async def get_user_state(user_id):
    return user_states.get(user_id, {"state": "GREETING", "history": []})

async def set_user_state(user_id, state_data):
    user_states[user_id] = state_data

async def process_recommendations(user_query):
    """Execută flow-ul de RAG: Extracție -> Căutare -> Generare Răspuns."""
    # 1. Extracție filtre
    logger.info(f"Extracting filters for query: {user_query}")
    extracted_data = llm.extract_filters(
        user_query, 
        ollama_url=Config.OLLAMA_URL, 
        model_name=Config.MODEL_NAME, 
        api_key=Config.GEMINI_API_KEY
    )
    
    # 2. Căutare Qdrant
    logger.info(f"Searching Qdrant with filters: {extracted_data}")
    results = rag.sota_hybrid_search(
        extracted_data, 
        qdrant_url=Config.QDRANT_URL
    )
    
    if not results:
        logger.warning(f"No results found for query: {user_query}")
        return "Îmi pare rău, nu am găsit flori care să se potrivească exact descrierii tale. Poți încerca să modifici puțin cerințele?", []

    # 3. Generare răspuns natural
    logger.info(f"Generating conversational response for {len(results)} results")
    rag_context = rag.format_context_for_llm(results)
    response_text = llm.generate_conversational_response(
        user_query, 
        rag_context, 
        ollama_url=Config.OLLAMA_URL, 
        model_name=Config.MODEL_NAME, 
        api_key=Config.GEMINI_API_KEY
    )
    
    return response_text, results

async def handle_message(body, say, publish_purchase_callback=None):
    """EntryPoint pentru logica bot-ului. Acționează ca un State Machine robust."""
    event = body["event"]
    user_id = event["user"]
    text = event["text"]
    
    # Load State
    user_data = await get_user_state(user_id)
    current_state = user_data.get("state")
    
    # Gemini Routing (Determine Intent)
    new_state = llm.classify_intent(text, current_state, api_key=Config.GEMINI_API_KEY)
    logger.info(f"User: {user_id} | State: {current_state} -> {new_state} | Message: {text}")

    # State Transitions & Actions
    if new_state == "RESTART":
        user_data = {"state": "GREETING", "history": []}
        await set_user_state(user_id, user_data)
        logger.info(f"Resetting state for user: {user_id}")
        await say("Nicio problemă, am resetat conversația. Cu ce te pot ajuta astăzi?")
        return

    if new_state == "GREETING":
        await say(f"Bună! Sunt BlossomBot 🌸 asistentul tău virtual de la City Flowers. Spune-mi ce fel de flori cauți sau pentru ce ocazie?")
        user_data["state"] = "SEARCH_PRODUCTS" 
    
    elif new_state == "SEARCH_PRODUCTS":
        await say("Caut cele mai potrivite flori în catalogul nostru...")
        response_text, results = await process_recommendations(text)
        await say(response_text)
        user_data["last_results"] = [r['title'] for r in results]
        user_data["state"] = "SEARCH_PRODUCTS"

    elif new_state == "PURCHASE_INTENT":
        if publish_purchase_callback:
            purchase_data = {
                "user_id": user_id,
                "items": user_data.get("last_results", ["Produs selectat"]),
                "timestamp": event.get("event_ts")
            }
            logger.info(f"Publishing purchase for user: {user_id}")
            success = await publish_purchase_callback(purchase_data)
            if success:
                logger.info(f"Purchase published successfully for user: {user_id}")
                await say("Excelent! Am trimis comanda ta către echipa noastră. Vei fi contactat în scurt timp pentru livrare. 🎉")
                user_data["state"] = "GREETING" 
            else:
                logger.error(f"Failed to publish purchase for user: {user_id}")
                await say("Îmi cer scuze, a apărut o problemă la procesarea comenzii. Te rog să încerci din nou.")
        else:
            await say("Vrei să finalizezi comanda? Spune-mi 'da' pentru a trimite solicitarea către colegii mei.")

    elif new_state == "GENERAL_INQUIRY":
        logger.info(f"Handling general inquiry for user: {user_id}")
        response_text = llm.generate_conversational_response(
            text, "Informații generale florărie: Livrăm zilnic 09-21, locație București Str. Florilor 10.", 
            ollama_url=Config.OLLAMA_URL, model_name=Config.MODEL_NAME, api_key=Config.GEMINI_API_KEY
        )
        await say(response_text)

    # Save State
    user_data["history"].append({"user": text, "state": new_state})
    await set_user_state(user_id, user_data)
