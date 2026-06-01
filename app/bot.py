import re
import json
import os
import asyncio
from app import llm_agent as llm
from app import rag_utils as rag
from app.config import Config
from app.logger import get_logger

import redis

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
    default_state = {"state": "GREETING", "history": [], "filter_scores": {}, "cart": []}
    
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
    
    logger.debug(f"\n[User: {user_id}]\n➡️ Starting search. Query: '{user_query}'\n➡️ Current Filter Scores: {json.dumps(filter_scores, indent=2, ensure_ascii=False)}")

    # 1. Extracție filtre
    extracted_data = llm.extract_filters(
        user_query, 
        history=history,
        api_key=Config.GEMINI_API_KEY
    )

    current_filters = extracted_data.get("filters", {})
    mentioned_now = extracted_data.get("mentioned_now", [])
    insisted = extracted_data.get("insisted", [])
    
    logger.debug(f"\n[User: {user_id}]\n🧠 LLM Extracted Filters:\n{json.dumps(current_filters, indent=2, ensure_ascii=False)}\n🧠 Mentioned Now: {mentioned_now} | Insisted: {insisted}")

    # 2. Update Scoruri
    for f_key in current_filters.keys():
        if f_key not in filter_scores:
            filter_scores[f_key] = 2 
        else:
            if f_key in mentioned_now:
                filter_scores[f_key] += 2 
            if f_key in insisted:
                filter_scores[f_key] += 2 

    user_data["filter_scores"] = filter_scores
    await set_user_state(user_id, user_data)
    
    logger.debug(f"\n[User: {user_id}]\n📊 Updated Filter Scores:\n{json.dumps(filter_scores, indent=2, ensure_ascii=False)}")

    # 3. Căutare Qdrant
    results, metadata = rag.sota_hybrid_search(
        extracted_data, 
        filter_scores=filter_scores,
        qdrant_url=Config.QDRANT_URL
    )
    
    logger.debug(f"\n[User: {user_id}]\n🏁 Search Results: Found {metadata.get('total_found', 0)} items.\n🏁 Search Metadata (Relaxation):\n{json.dumps(metadata, indent=2, ensure_ascii=False)}\n")
    
    if not results:
        return "Îmi pare rău, nu am găsit flori care să se potrivească descrierii tale. Poți încerca să modifici puțin cerințele?", []

    # 4. Feedback filtre eșuate (DOAR DACĂ AU FOST CERUTE ACUM SAU REPETATE)
    failed = metadata.get("failed_filters", [])
    feedback_msg = ""
    
    user_requested_keys = set(mentioned_now + insisted)
    real_failed = [f for f in failed if f in user_requested_keys or (f == "price" and "price_max" in user_requested_keys)]

    if real_failed:
        filter_map = {
            "price_max": "sub prețul dorit",
            "price": "sub prețul dorit",
            "category": "din categoria cerută",
            "dimensiune": "de mărimea dorită",
            "event_type": "pentru acest tip de eveniment",
            "colors": "în culorile solicitate",
            "flowers_type": "cu tipul de flori specificat"
        }
        failed_texts = [filter_map.get(f, f) for f in real_failed]
        
        if len(failed_texts) > 1:
            joined_failed = ", ".join(failed_texts[:-1]) + " sau " + failed_texts[-1]
        else:
            joined_failed = failed_texts[0]
            
        if ("price_max" in real_failed or "price" in real_failed) and results:
            min_price = min([r.get("price", 0) for r in results])
            feedback_msg = f"💡 *Am ajustat puțin căutarea.* Nu am găsit exact potriviri {joined_failed}. Cele mai apropiate opțiuni pornesc de la {min_price} RON. Iată ce am găsit:\n\n"
        else:
            feedback_msg = f"💡 *Am ajustat puțin căutarea.* Nu am găsit exact potriviri {joined_failed}, dar iată cele mai frumoase alternative pe care ți le recomand:\n\n"

    # 5. Generare răspuns hibrid
    response_text = feedback_msg + llm.generate_hybrid_response(results)
    return response_text, results


async def retry_recommendations(user_id, channel_id, user_query, history, client):
    """Background task care reîncearcă procesarea până când Gemini este disponibil."""
    max_retries = 10
    retry_delay = 5
    
    for i in range(max_retries):
        try:
            await asyncio.sleep(retry_delay)
            response_text, results = await process_recommendations(user_id, user_query, history=history)
            await client.chat_postMessage(channel=channel_id, text=response_text)
            
            user_data = await get_user_state(user_id)
            user_data["last_results"] = results
            user_data["state"] = "SEARCH_PRODUCTS"
            await set_user_state(user_id, user_data)
            return
            
        except llm.GeminiOverloadedError:
            logger.warning(f"⚠️ Retry {i+1} failed: Gemini busy.")
        except Exception as e:
            logger.error(f"❌ Error retry: {e}")
            break
            
    await client.chat_postMessage(channel=channel_id, text="Îmi cer scuze, serviciul este foarte solicitat. Revino te rog mai târziu.")

async def handle_message(body, say, client, publish_purchase_callback=None):
    """EntryPoint pentru logica bot-ului. Acționează ca un State Machine robust."""
    event = body["event"]
    user_id = event["user"]
    channel_id = event["channel"]
    text = event["text"]
    
    user_data = await get_user_state(user_id)
    current_state = user_data.get("state")
    
    # 1. Routing
    new_state = llm.classify_intent(text, current_state, api_key=Config.GEMINI_API_KEY)
    logger.info(f"User: {user_id} | State: {current_state} -> {new_state}")

    if new_state == "RESTART":
        user_data = {"state": "GREETING", "history": [], "filter_scores": {}, "cart": []}
        await set_user_state(user_id, user_data)
        
        restart_msg = (
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "🔄 **Conversația a fost resetată.**\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Nicio problemă, am anulat totul și am golit coșul. Cu ce te pot ajuta?"
        )
        await say(restart_msg)
        return

    # LOGICA CONFIRMARE & ADĂUGARE ÎN COȘ
    if current_state == "AWAITING_CONFIRMATION":
        conf = llm.parse_confirmation(text, api_key=Config.GEMINI_API_KEY)
        is_confirmed = conf.get("confirmed", False)
        
        if is_confirmed:
            pending = user_data.get("pending_item")
            if pending:
                if conf.get("quantity") is not None:
                    pending["qty"] = conf["quantity"]
                
                cart = user_data.get("cart", [])
                cart.append(pending)
                user_data["cart"] = cart
                
                await say(f"✅ Am adăugat {pending['qty']} x *{pending['name']}* în coș! 🛒 (Total produse diferite în coș: {len(cart)})\n_Dorești să mai cauți altceva sau finalizăm comanda?_")
                user_data["state"] = "SEARCH_PRODUCTS" # Setăm pe search pentru a aștepta următoarea comandă (sau finalizează)
            else:
                await say("Nu am găsit produsul în așteptare.")
                user_data["state"] = "SEARCH_PRODUCTS"
        else:
            await say("Am anulat adăugarea în coș. Ce ai vrea să cauți în schimb?")
            user_data["state"] = "SEARCH_PRODUCTS"
            
        await set_user_state(user_id, user_data)
        return

    # LOGICA STATE MACHINE
    if new_state == "GREETING":
        # Greeting funcționează și ca un mini-restart dacă dorim un nou context
        # user_data = {"state": "SEARCH_PRODUCTS", "history": [], "filter_scores": {}, "cart": []}
        await say(f"Bună! Sunt BlossomBot 🌸 asistentul tău virtual. Spune-mi ce fel de flori cauți?")
        user_data["state"] = "SEARCH_PRODUCTS" 
    
    elif new_state == "GENERAL_INQUIRY":
        response_text = llm.generate_general_response(text, api_key=Config.GEMINI_API_KEY)
        await say(response_text)

    elif new_state == "CATALOG_DISCOVERY":
        await say("Sigur! Avem o gamă variată de produse. Iată câteva recomandări din oferta noastră:")
        response_text, results = await process_recommendations(user_id, "produse populare flori cadouri", history=[])
        await say(response_text)
        user_data["last_results"] = results
        user_data["state"] = "SEARCH_PRODUCTS"

    elif new_state == "PRODUCT_DETAILS":
        last_results = user_data.get("last_results", [])
        selection = llm.extract_purchase_selection(text, api_key=Config.GEMINI_API_KEY)
        selected_index = selection.get("index", 0)
        
        if selected_index >= 0 and selected_index < len(last_results):
            product = last_results[selected_index]
            details_text = llm.generate_detailed_product_info(product)
            await say(details_text)
        else:
            if not last_results:
                await say("Caută întâi niște produse, apoi te pot ajuta cu detalii.")
            else:
                await say("Despre care produs dorești detalii? (ex: 'primul')")

    elif new_state == "VIEW_CART":
        cart = user_data.get("cart", [])
        if not cart:
            await say("🛒 Coșul tău este momentan gol. Hai să căutăm niște flori frumoase!")
            user_data["state"] = "SEARCH_PRODUCTS"
        else:
            cart_msg = "🛒 *Coșul tău de cumpărături:*\n\n"
            total_price = 0
            for i, item in enumerate(cart, 1):
                qty = item['qty']
                name = item['name']
                price = item['price']
                item_total = qty * price
                total_price += item_total
                cart_msg += f"{i}. {qty}x {name} - {item_total} RON ({price} lei/buc)\n"
            
            cart_msg += f"\n💰 **Total comandă: {total_price} RON**\n"
            cart_msg += "_Spune 'finalizează' pentru a trimite comanda sau poți continua cumpărăturile._"
            await say(cart_msg)
            # Rămânem în SEARCH_PRODUCTS ca să poată continua sau să dea finalizare
            user_data["state"] = "SEARCH_PRODUCTS"

    elif new_state == "SEARCH_PRODUCTS":
        await say("Analizez cererea ta și verific catalogul nostru...")
        try:
            response_text, results = await process_recommendations(user_id, text, history=user_data.get("history"))
            await say(response_text)
            user_data["last_results"] = results 
            user_data["state"] = "SEARCH_PRODUCTS"
        except llm.GeminiOverloadedError:
            await say("🌸 Sunt puțin ocupat, revin în câteva secunde cu recomandări...")
            asyncio.create_task(retry_recommendations(user_id, channel_id, text, user_data.get("history"), client))
            return

    elif new_state == "PURCHASE_INTENT":
        last_results = user_data.get("last_results", [])
        if not last_results:
            await say("Spune-mi întâi ce cauți.")
            user_data["state"] = "SEARCH_PRODUCTS"
        else:
            selection = llm.extract_purchase_selection(text, api_key=Config.GEMINI_API_KEY)
            idx = selection.get("index", 0)
            qty = selection.get("quantity", 1)
            
            if idx >= 0 and idx < len(last_results):
                product = last_results[idx]
                p_name = product.get('title')
                p_price = product.get('price')
                
                user_data["pending_item"] = {
                    "name": p_name,
                    "price": p_price,
                    "qty": qty
                }
                user_data["state"] = "AWAITING_CONFIRMATION"
                
                total = p_price * qty
                await say(f"Vrei să adăugăm *{qty} x {p_name}* în coș (Total: {total} RON)? (Spune 'Da' pentru a confirma)")
            else:
                await say("Nu am înțeles exact care produs îl dorești (ex: 'pe primul').")

    elif new_state == "FINISH_ORDER":
        cart = user_data.get("cart", [])
        if not cart:
            await say("Coșul tău este gol! Nu avem ce să finalizăm. Ce ai dori să comanzi?")
            user_data["state"] = "SEARCH_PRODUCTS"
        else:
            if publish_purchase_callback:
                items_str = [f"{item['qty']}x {item['name']}" for item in cart]
                total_price = sum(item['price'] * item['qty'] for item in cart)
                
                purchase_data = {
                    "user_id": user_id,
                    "items": items_str,
                    "price": total_price,
                    "timestamp": event.get("event_ts")
                }
                success = await publish_purchase_callback(purchase_data)
                if success:
                    success_msg = (
                        f"🎉 Excelent! Comanda ta (Total: {total_price} RON) a fost trimisă spre procesare!\n\n"
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        "Îți mulțumim că ai folosit **BlossomBot**, asistentul virtual de la City Flowers, care îți este alături 24/7!\n"
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    )
                    await say(success_msg)
                    # HARD RESET PENTRU A SIMULA O CONVERSAȚIE NOUĂ
                    user_data = {"state": "GREETING", "history": [], "filter_scores": {}, "cart": []}
                else:
                    await say("Îmi cer scuze, a apărut o problemă tehnică la trimiterea comenzii. Te rog spune 'finalizează' pentru a reîncerca.")
                    user_data["state"] = "SEARCH_PRODUCTS"
            else:
                demo_msg = (
                    "Comanda a fost trimisă intern! (Modul Demo)\n\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "Îți mulțumim că ai folosit **BlossomBot**, asistentul virtual de la City Flowers, care îți este alături 24/7!\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                )
                await say(demo_msg)
                user_data = {"state": "GREETING", "history": [], "filter_scores": {}, "cart": []}

    # Save History & State
    user_data["history"].append({"user": text, "state": new_state})
    await set_user_state(user_id, user_data)
