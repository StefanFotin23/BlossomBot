import re
import json
import os
import asyncio
import time
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
    redis_client.ping()
    logger.info("✅ Connected to Redis for state management.")
except Exception as e:
    logger.warning(f"⚠️ Redis not available ({e}). Falling back to in-memory storage.")
    redis_client = None

_local_user_states = {}

async def get_user_state(user_id):
    default_state = {"state": "GREETING", "history": [], "filter_scores": {}, "cart": []}
    if redis_client:
        try:
            data = redis_client.get(f"user:{user_id}")
            if data: return json.loads(data)
            return default_state
        except Exception as e:
            logger.error(f"Redis get error: {e}")
    return _local_user_states.get(user_id, default_state)

async def set_user_state(user_id, state_data):
    if redis_client:
        try:
            redis_client.set(f"user:{user_id}", json.dumps(state_data), ex=3600*24)
            return
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    _local_user_states[user_id] = state_data

# --- ATOMIC SEQUENCING (Cross-Pod / Intra-Pod User Lock) ---
async def is_user_busy(user_id):
    """Verifică dacă un user are deja sarcini în coadă (pentru a păstra ordinea)."""
    if redis_client:
        return redis_client.exists(f"busy:{user_id}")
    return False # Fallback

def set_user_busy(user_id, busy=True):
    if redis_client:
        if busy: redis_client.set(f"busy:{user_id}", "1", ex=300) # Lock 5 min max
        else: redis_client.delete(f"busy:{user_id}")

# --- VITAL AI TASK QUEUE (QoS System) ---
vital_task_queue = asyncio.Queue()

async def background_task_worker(client, publish_purchase_callback):
    """Procesează SECVENȚIAL sarcinile globale, dar respectă ordinea pe USER."""
    logger.info("⚙️ Atomic Task Worker started.")
    while True:
        task = await vital_task_queue.get()
        user_id = task["user_id"]
        channel_id = task["channel_id"]
        body = task["body"]
        
        logger.info(f"🔄 [QUEUE] Processing task for user {user_id}")
        
        try:
            # Delay anti-429
            await asyncio.sleep(4) 
            
            async def queue_say(message):
                if task.get("is_mock"):
                    logger.info(f"📝 [MOCK QUEUE] {user_id}: {message[:60]}...")
                else:
                    await client.chat_postMessage(channel=channel_id, text=message)

            # Executăm logica. handle_message va salva starea la final.
            await handle_message(
                body=body, 
                say=queue_say, 
                client=client, 
                publish_purchase_callback=publish_purchase_callback,
                is_mock=task.get("is_mock", False),
                is_retry=True 
            )
            
        except Exception as e:
            logger.error(f"❌ [QUEUE] Error for {user_id}: {e}")
        finally:
            # Eliberăm lock-ul imediat ce am terminat această sarcină
            set_user_busy(user_id, False)
            vital_task_queue.task_done()

# --- RECOMANDĂRI ---
async def process_recommendations(user_id, user_query, history=None):
    user_data = await get_user_state(user_id)
    filter_scores = user_data.get("filter_scores", {})
    
    extracted_data = llm.extract_filters(user_query, history=history, api_key=Config.GEMINI_API_KEY)
    current_filters = extracted_data.get("filters", {})
    negative_filters = extracted_data.get("negative_filters", {})
    mentioned_now = extracted_data.get("mentioned_now", [])
    insisted = extracted_data.get("insisted", [])
    
    all_keys = set(list(current_filters.keys()) + list(negative_filters.keys()))
    for f_key in all_keys:
        score_key = f_key if f_key in current_filters else f"no_{f_key}"
        if score_key not in filter_scores:
            filter_scores[score_key] = 2 
        else:
            if f_key in mentioned_now: filter_scores[score_key] += 2 
            if f_key in insisted: filter_scores[score_key] += 2 

    user_data["filter_scores"] = filter_scores
    await set_user_state(user_id, user_data)
    
    results, metadata = rag.sota_hybrid_search(extracted_data, filter_scores=filter_scores, qdrant_url=Config.QDRANT_URL)
    
    if not results:
        return "Îmi pare rău, nu am găsit ceva potrivit. Poți modifica cerințele?", []

    failed = metadata.get("failed_filters", [])
    feedback_msg = ""
    user_requested_keys = set(mentioned_now + insisted)
    real_failed = [f for f in failed if f in user_requested_keys or (f == "price" and "price_max" in user_requested_keys)]

    if real_failed:
        filter_map = {"price_max": "sub prețul dorit", "price": "sub prețul dorit", "category": "din categoria cerută", "dimensiune": "de mărimea dorită", "event_type": "pentru acest tip de eveniment", "colors": "în culorile solicitate", "flowers_type": "cu tipul de flori specificat"}
        failed_texts = [filter_map.get(f, f) for f in real_failed]
        joined_failed = ", ".join(failed_texts[:-1]) + " sau " + failed_texts[-1] if len(failed_texts) > 1 else failed_texts[0]
        
        if ("price_max" in real_failed or "price" in real_failed) and results:
            min_price = min([r.get("price", 0) for r in results])
            feedback_msg = f"💡 *Am ajustat căutarea.* Cele mai apropiate opțiuni pornesc de la {min_price} RON:\n\n"
        else:
            feedback_msg = f"💡 *Am ajustat căutarea.* Iată ce am găsit:\n\n"

    response_text = feedback_msg + llm.generate_hybrid_response(results)
    return response_text, results

# --- MAIN HANDLER ---
async def handle_message(body, say, client, publish_purchase_callback=None, is_mock=False, is_retry=False):
    event = body["event"]
    user_id = event["user"]
    channel_id = event.get("channel", "direct")
    text = event["text"]
    text_clean = text.lower().strip()
    mock_responses = []
    
    async def smart_say(message):
        if is_mock: mock_responses.append(message)
        else: await say(message)

    # --- 1. SEQUENCING CHECK ---
    if not is_retry and await is_user_busy(user_id):
        queue_size = vital_task_queue.qsize()
        est = (queue_size + 1) * 5
        await smart_say(f"⏳ Procesez încă cererea ta anterioară. Am adăugat și acest mesaj în coadă.\nEstimare: {est} secunde.")
        await vital_task_queue.put({"user_id": user_id, "channel_id": channel_id, "body": body, "is_mock": is_mock})
        return mock_responses if is_mock else None

    user_data = await get_user_state(user_id)
    current_state = user_data.get("state", "GREETING")

    # 2. Logică Locală (Confirmare / Restart)
    if any(w in text_clean for w in ["restart", "anulează tot"]):
        user_data = {"state": "GREETING", "history": [], "filter_scores": {}, "cart": []}
        await set_user_state(user_id, user_data)
        await smart_say("🔄 Conversația a fost resetată.")
        return mock_responses if is_mock else None

    if current_state == "AWAITING_CONFIRMATION":
        is_yes = any(w in text_clean for w in ["da", "yes", "ok", "k", "yep", "confirm", "il vreau"])
        is_no = any(w in text_clean for w in ["nu", "no", "anuleaza"])
        if is_yes or is_no:
            if is_yes:
                pending = user_data.get("pending_item")
                if pending:
                    match_qty = re.search(r'(\d+)', text_clean)
                    if match_qty: pending["qty"] = int(match_qty.group(1))
                    cart = user_data.get("cart", [])
                    cart.append(pending)
                    user_data["cart"] = cart
                    user_data["pending_item"] = None
                    await smart_say(f"✅ Adăugat {pending['qty']}x {pending['name']} în coș!")
                    user_data["state"] = "SEARCH_PRODUCTS"
            else:
                await smart_say("Anulat.")
                user_data["state"] = "SEARCH_PRODUCTS"
            await set_user_state(user_id, user_data)
            return mock_responses if is_mock else None

    # 3. AI Execution
    try:
        new_state = llm.classify_intent(text, current_state, api_key=Config.GEMINI_API_KEY)
        
        if new_state == "GREETING":
            await smart_say("Bună! Cu ce te pot ajuta?")
            user_data["state"] = "SEARCH_PRODUCTS" 
        
        elif new_state == "GENERAL_INQUIRY":
            await smart_say(llm.generate_general_response(text, api_key=Config.GEMINI_API_KEY))

        elif new_state == "CATALOG_DISCOVERY":
            await smart_say("Iată ce avem în catalog:")
            resp, res = await process_recommendations(user_id, "produse populare", history=[])
            await smart_say(resp)
            user_data["last_results"] = res
            user_data["state"] = "SEARCH_PRODUCTS"

        elif new_state == "PRODUCT_DETAILS":
            last_results = user_data.get("last_results", [])
            selection = llm.extract_purchase_selection(text, api_key=Config.GEMINI_API_KEY)
            idx = selection.get("index", 0)
            if idx >= 0 and idx < len(last_results):
                await smart_say(llm.generate_detailed_product_info(last_results[idx]))

        elif new_state == "VIEW_CART":
            cart = user_data.get("cart", [])
            if not cart: await smart_say("🛒 Coșul e gol.")
            else: await smart_say("🛒 Coș: " + ", ".join([f"{i['qty']}x {i['name']} ({i['price']*i['qty']} RON)" for i in cart]))

        elif new_state == "SEARCH_PRODUCTS":
            await smart_say("Caut în catalog...")
            resp, res = await process_recommendations(user_id, text, history=user_data.get("history"))
            await smart_say(resp)
            user_data["last_results"] = res 
            user_data["state"] = "SEARCH_PRODUCTS"

        elif new_state == "PURCHASE_INTENT":
            last_results = user_data.get("last_results", [])
            if not last_results: await smart_say("Caută ceva întâi.")
            else:
                selection = llm.extract_purchase_selection(text, api_key=Config.GEMINI_API_KEY)
                idx = selection.get("index", 0)
                qty = selection.get("quantity", 1)
                if idx >= 0 and idx < len(last_results):
                    p = last_results[idx]
                    user_data["pending_item"] = {"name": p.get('title'), "price": p.get('price'), "qty": qty}
                    user_data["state"] = "AWAITING_CONFIRMATION"
                    await smart_say(f"Vrei să adăugăm *{qty}x {p.get('title')}*? ({p.get('price')*qty} RON)")

        elif new_state == "FINISH_ORDER":
            cart = user_data.get("cart", [])
            if not cart: await smart_say("Coș gol.")
            else:
                items_str = [f"{i['qty']}x {i['name']}" for i in cart]
                total = sum(i['price'] * i['qty'] for i in cart)
                
                # RUTA CĂTRE COADA DE STRESS TEST SAU PROD
                target_q = Config.STRESS_TEST_QUEUE if is_mock else Config.PURCHASE_QUEUE
                
                if await publish_purchase_callback({"user_id": user_id, "items": items_str, "price": total, "timestamp": str(time.time())}, queue_name=target_q):
                    await smart_say(f"🎉 Comandă trimisă ({total} RON)!")
                    user_data = {"state": "GREETING", "history": [], "filter_scores": {}, "cart": []}

    except llm.GeminiOverloadedError:
        if is_retry: 
            await smart_say("⚠️ Server ocupat, reîncearcă mai târziu.")
        else:
            set_user_busy(user_id, True)
            queue_size = vital_task_queue.qsize()
            await smart_say(f"🌸 Suntem ocupați. Cererea ta e în coadă (poziția {queue_size + 1}).")
            await vital_task_queue.put({"user_id": user_id, "channel_id": channel_id, "body": body, "is_mock": is_mock})
            return mock_responses if is_mock else None

    user_data["history"].append({"user": text, "state": new_state})
    await set_user_state(user_id, user_data)
    return mock_responses if is_mock else None
