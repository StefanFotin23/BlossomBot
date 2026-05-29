import re
import json
import os
import asyncio
from app import llm_agent as llm
from app import rag_utils as rag
from app.config import Config
from app.logger import get_logger

logger = get_logger(__name__)

# In-memory storage
user_states = {}

async def get_user_state(user_id):
    return user_states.get(user_id, {"state": "GREETING", "history": []})

async def set_user_state(user_id, state_data):
    user_states[user_id] = state_data

async def process_recommendations(user_query):
    """Execută flow-ul de RAG."""
    # 1. Extracție filtre (aruncă GeminiOverloadedError dacă e 503)
    extracted_data = llm.extract_filters(
        user_query, 
        api_key=Config.GEMINI_API_KEY
    )
    
    # 2. Căutare Qdrant
    results = rag.sota_hybrid_search(
        extracted_data, 
        qdrant_url=Config.QDRANT_URL
    )
    
    if not results:
        return "Îmi pare rău, nu am găsit flori care să se potrivească exact descrierii tale. Poți încerca să modifici puțin cerințele?", []

    # 3. Generare răspuns hibrid (instant)
    response_text = llm.generate_hybrid_response(results)
    return response_text, results

async def retry_recommendations(user_id, channel_id, user_query, client):
    """Background task care reîncearcă procesarea până când Gemini este disponibil."""
    max_retries = 10
    retry_delay = 5 # secunde
    
    logger.info(f"🔄 Starting background retry for user {user_id}")
    
    for i in range(max_retries):
        try:
            await asyncio.sleep(retry_delay)
            logger.info(f"🔄 Retry attempt {i+1} for user {user_id}")
            
            response_text, results = await process_recommendations(user_query)
            
            # Trimitem răspunsul final
            await client.chat_postMessage(channel=channel_id, text=response_text)
            
            # Actualizăm starea
            user_data = await get_user_state(user_id)
            user_data["last_results"] = [r['title'] for r in results]
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

    if new_state == "GREETING":
        await say(f"Bună! Sunt BlossomBot 🌸 asistentul tău virtual. Spune-mi ce fel de flori cauți?")
        user_data["state"] = "SEARCH_PRODUCTS" 
    
    elif new_state == "SEARCH_PRODUCTS":
        await say("Analizez cererea ta și verific catalogul nostru...")
        try:
            response_text, results = await process_recommendations(text)
            await say(response_text)
            user_data["last_results"] = [r['title'] for r in results]
        except llm.GeminiOverloadedError:
            logger.warning(f"🚨 Gemini Overloaded for user {user_id}. Starting retry task.")
            await say("🌸 Sunt puțin ocupat cu multe cereri acum, dar nu te-am uitat! Revin în câteva secunde cu cele mai bune recomandări pentru tine...")
            # PORNIM TASK-UL ÎN FUNDAL (RAM)
            asyncio.create_task(retry_recommendations(user_id, channel_id, text, client))
            return # Ieșim din handler, mesajul va fi trimis ulterior de task-ul de fundal

    elif new_state == "PURCHASE_INTENT":
        if publish_purchase_callback:
            purchase_data = {
                "user_id": user_id,
                "items": user_data.get("last_results", ["Produs selectat"]),
                "timestamp": event.get("event_ts")
            }
            success = await publish_purchase_callback(purchase_data)
            if success:
                await say("Excelent! Am trimis comanda ta. 🎉")
                user_data["state"] = "GREETING" 
            else:
                await say("Îmi cer scuze, a apărut o problemă. Te rog să încerci din nou.")
        else:
            await say("Vrei să finalizezi comanda? Spune-mi 'da'.")

    # Save State
    user_data["history"].append({"user": text, "state": new_state})
    await set_user_state(user_id, user_data)
