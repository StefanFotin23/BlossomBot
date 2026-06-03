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
    default_state = {"state": "GREETING", "history": [], "filter_scores": {}, "cart": [], "deferred_tasks": []}
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

# --- ATOMIC SEQUENCING ---
async def is_user_busy(user_id):
    if redis_client: return redis_client.exists(f"busy:{user_id}")
    return False

def set_user_busy(user_id, busy=True):
    if redis_client:
        if busy: redis_client.set(f"busy:{user_id}", "1", ex=300)
        else: redis_client.delete(f"busy:{user_id}")

# --- VITAL AI TASK QUEUE ---
vital_task_queue = asyncio.Queue()

async def background_task_worker(client, publish_purchase_callback):
    logger.info("⚙️ Atomic Task Worker started.")
    while True:
        task = await vital_task_queue.get()
        user_id, ch_id, body = task["user_id"], task["channel_id"], task["body"]
        try:
            await asyncio.sleep(1.5) 
            async def q_say(msg):
                if task.get("is_mock"): logger.info(f"📝 [MOCK QUEUE] {user_id}: {msg[:60]}...")
                else: await client.chat_postMessage(channel=ch_id, text=msg)

            await handle_message(body, q_say, client, publish_purchase_callback, is_mock=task.get("is_mock", False), is_retry=True)
        except Exception as e: logger.error(f"❌ [QUEUE] Error: {e}")
        finally:
            set_user_busy(user_id, False)
            vital_task_queue.task_done()

# --- RECOMANDĂRI ---
async def process_recommendations(user_id, user_query, history=None):
    user_data = await get_user_state(user_id)
    filter_scores = user_data.get("filter_scores", {})
    extracted_data = llm.extract_filters(user_query, history=history, api_key=Config.GEMINI_API_KEY)
    
    for f_key in set(list(extracted_data.get("filters", {}).keys()) + list(extracted_data.get("negative_filters", {}).keys())):
        score_key = f_key if f_key in extracted_data.get("filters", {}) else f"no_{f_key}"
        filter_scores[score_key] = filter_scores.get(score_key, 0) + 2

    user_data["filter_scores"] = filter_scores
    await set_user_state(user_id, user_data)
    
    results, metadata = rag.sota_hybrid_search(extracted_data, filter_scores=filter_scores, qdrant_url=Config.QDRANT_URL)
    if not results: return "Nu am găsit nimic potrivit. Modifică cerințele?", []

    feedback = ""
    if metadata.get("failed_filters"):
        min_p = min([r.get("price", 0) for r in results])
        feedback = f"💡 *Am ajustat căutarea.* Cele mai apropiate opțiuni pornesc de la {min_p} RON:\n\n"

    return feedback + llm.generate_hybrid_response(results), results

# --- MAIN HANDLER ---
async def handle_message(body, say, client, publish_purchase_callback=None, is_mock=False, is_retry=False):
    event = body["event"]
    user_id, channel_id, full_text = event["user"], event.get("channel", "direct"), event["text"]
    text_cl = full_text.lower().strip()
    words = text_cl.split()
    mock_responses = []
    
    async def smart_say(m):
        if is_mock: mock_responses.append(m)
        else: await say(m)

    user_data = await get_user_state(user_id)
    current_state = user_data.get("state", "GREETING")

    # DETERMINISTIC FAST-PATHS
    K_YES = ["da", "dap", "daaa", "yes", "ok", "k", "kk", "yep", "yup", "y", "sigur", "confirm", "perfect", "vreau", "baga", "hai", "corect", "exact", "asa", "așa"]
    K_NO = ["nu", "no", "nope", "n", "stai", "anuleaza", "anulează", "ba nu", "pas", "renunt", "renunț"]
    K_RESTART = ["restart", "reset", "de la capat", "anuleaza tot", "anulează tot", "sterge tot", "șterge tot"]
    K_GREET = ["salut", "buna", "bună", "hei", "hello", "hi", "hey", "salutare", "ciao"]
    K_IDENTITY = ["cine esti", "cine ești", "ce ești", "cine sunteti", "ce faci", "cum te numești", "cine ești?", "cine esti?"]

    if any(w in K_RESTART for w in [text_cl] + words):
        user_data = {"state": "GREETING", "history": [], "filter_scores": {}, "cart": [], "deferred_tasks": []}
        await set_user_state(user_id, user_data); await smart_say("🔄 Conversație resetată."); return mock_responses if is_mock else None

    # Deterministic Greeting / Identity (Strict)
    is_exact_greet = text_cl in K_GREET or (len(words) == 1 and words[0] in K_GREET)
    is_exact_ident = text_cl in K_IDENTITY or any(p in text_cl for p in ["cine esti", "cine ești"])
    
    if (is_exact_greet or is_exact_ident) and current_state != "AWAITING_CONFIRMATION":
        if is_exact_greet: await smart_say("Bună! Cu ce te pot ajuta?")
        else: await smart_say("Sunt BlossomBot 🌸, asistentul tău virtual pentru flori și cadouri superbe!")
        user_data["state"] = "SEARCH_PRODUCTS"
        if user_data.get("deferred_tasks"):
            next_t = user_data["deferred_tasks"].pop(0); await set_user_state(user_id, user_data)
            await handle_message({"event": {"user": user_id, "text": next_t["text"], "channel": channel_id}, "is_deferred": True, "intent": next_t["intent"]}, say, client, publish_purchase_callback, is_mock)
        else:
            await set_user_state(user_id, user_data)
        return mock_responses if is_mock else None

    if not is_retry and await is_user_busy(user_id):
        queue_size = vital_task_queue.qsize()
        est_wait = (queue_size + 1) * 1.5
        msg = f"⏳ Procesez încă cererea ta anterioară. Mesajul tău a fost pus în coadă (Poz: {queue_size+1})."
        if est_wait > 5: msg += f"\n⏳ Timp estimat: ~{int(est_wait)}s"
        await smart_say(msg)
        await vital_task_queue.put({"user_id": user_id, "channel_id": channel_id, "body": body, "is_mock": is_mock})
        return mock_responses if is_mock else None

    if current_state == "AWAITING_CONFIRMATION":
        is_y, is_n = any(w in K_YES for w in [text_cl] + words), any(w in K_NO for w in [text_cl] + words)
        m_q = re.search(r'^(\d+)$', text_cl) # Doar un număr (cantitate)
        
        if is_y or is_n or m_q:
            if is_y or m_q:
                p = user_data.get("pending_item")
                if p:
                    if m_q: p["qty"] = int(m_q.group(1))
                    else:
                        m_inner_q = re.search(r'(\d+)', text_cl)
                        if m_inner_q: p["qty"] = int(m_inner_q.group(1))
                    
                    user_data["cart"] = user_data.get("cart", []) + [p]
                    user_data["pending_item"] = None; 
                    await smart_say(f"✅ Adăugat {p['qty']}x {p['name']} în coș!")
                    user_data["state"] = "SEARCH_PRODUCTS"
            else:
                await smart_say("Anulat."); user_data["state"] = "SEARCH_PRODUCTS"
            await set_user_state(user_id, user_data)
            if user_data.get("deferred_tasks"):
                next_t = user_data["deferred_tasks"].pop(0); await set_user_state(user_id, user_data)
                await handle_message({"event": {"user": user_id, "text": next_t["text"], "channel": channel_id}, "is_deferred": True, "intent": next_t["intent"]}, say, client, publish_purchase_callback, is_mock)
            return mock_responses if is_mock else None

    # MULTI-INTENT & COMMAND FAST-PATH
    forced_intent = body.get("intent")
    
    if not is_retry and not body.get("is_deferred"):
        # COMMAND FAST-PATH (Bypass LLM for core actions)
        if any(w in K_YES for w in [text_cl] + words) and current_state in ["SEARCH_PRODUCTS", "PRODUCT_DETAILS", "PURCHASE_INTENT"]:
            forced_intent = "PURCHASE_INTENT"
        elif any(w in ["primul", "prima", "doilea", "treilea", "varianta"] for w in words) or re.search(r'^(\d+)$', text_cl):
            forced_intent = "PURCHASE_INTENT"
        elif any(w in ["recomanda", "recomanda-mi", "recomandari", "recomandări", "oferta", "ofertă"] for w in words):
            forced_intent = "CATALOG_DISCOVERY"
        elif any(p in text_cl for p in ["arata-mi cosul", "arată-mi coșul", "cosul meu", "coșul meu"]) or text_cl in ["cos", "coș", "cart"]:
            forced_intent = "VIEW_CART"

        if forced_intent:
            current_task = {"text": full_text, "intent": forced_intent}
        else:
            # Send thinking indicator if it's an LLM request
            is_fast = is_exact_greet or is_exact_ident or any(w in K_RESTART for w in [text_cl] + words)
            if not is_fast and not is_mock:
                await smart_say("🌸 Analizez cererea ta...")
            
            tasks = llm.decompose_intents(full_text, current_state, api_key=Config.GEMINI_API_KEY)
            if len(tasks) > 1:
                user_data["deferred_tasks"] = tasks[1:]; await set_user_state(user_id, user_data)
            current_task = tasks[0]
    else:
        current_task = {"text": full_text, "intent": body.get("intent", "UNKNOWN")}

    text_t = current_task["text"]
    new_state = current_task.get("intent")
    if not new_state or new_state == "UNKNOWN":
        new_state = llm.classify_intent(text_t, current_state, api_key=Config.GEMINI_API_KEY)

    try:
        should_c = False

        if new_state == "GREETING":
            await smart_say("Bună! Cu ce te pot ajuta?"); user_data["state"] = "SEARCH_PRODUCTS"; should_c = True
        elif new_state == "CATALOG_DISCOVERY":
            resp, res = await process_recommendations(user_id, "produse populare", history=[])
            await smart_say(resp)
            user_data["last_results"] = res; user_data["state"] = "SEARCH_PRODUCTS"; should_c = True
        elif new_state == "PRODUCT_DETAILS":
            last_res = user_data.get("last_results", [])
            if not last_res:
                await smart_say("Spune-mi întâi ce flori cauți, ca să îți pot oferi detalii!"); should_c = True
            else:
                # FAST-PATH: Numeric selection
                m_idx = re.search(r'^(\d+)$', text_t)
                idx = int(m_idx.group(1)) - 1 if m_idx else None
                
                if idx is None:
                    titles = [r.get('title') for r in last_res]
                    sel = llm.extract_purchase_selection(text_t, titles=titles, api_key=Config.GEMINI_API_KEY)
                    idx = sel.get("index") if sel.get("index") is not None else 0
                
                if isinstance(idx, int) and 0 <= idx < len(last_res): 
                    p = last_res[idx]
                    await smart_say(llm.generate_detailed_product_info(p))
                    # Implicit transition to confirmation state for the next 'da'
                    user_data["pending_item"] = {"name": p.get('title'), "price": p.get('price'), "qty": 1}
                    user_data["state"] = "AWAITING_CONFIRMATION"
                    should_c = False # STOP multi-intent loop, wait for confirmation
                else:
                    should_c = True
        elif new_state == "VIEW_CART":
            cart = user_data.get("cart", [])
            if not cart: await smart_say("🛒 Coșul e gol.")
            else:
                items = "\n".join([f"• {i['qty']} x {i['name']} ({i['price']*i['qty']} RON)" for i in cart])
                await smart_say(f"🛒 *Coșul tău:*\n{items}\n━━━━━━━━━━━━━━━━━━━━━\n💰 *Total: {sum(i['price']*i['qty'] for i in cart)} RON*")
            should_c = True
        elif new_state == "SEARCH_PRODUCTS":
            user_data["pending_item"] = None # Clear any pending item if user starts searching again
            resp, res = await process_recommendations(user_id, text_t, history=user_data.get("history"))
            await smart_say(resp); user_data["last_results"] = res; user_data["state"] = "SEARCH_PRODUCTS"; should_c = True
        elif new_state == "PURCHASE_INTENT":
            last_res = user_data.get("last_results", [])
            if not last_res:
                await smart_say("Alege întâi o floare din listă sau spune-mi ce cauți!"); should_c = True
            else:
                # FAST-PATH: Numeric selection
                m_idx = re.search(r'^(\d+)$', text_t)
                idx = int(m_idx.group(1)) - 1 if m_idx else None
                qty = 1

                if idx is None:
                    titles = [r.get('title') for r in last_res]
                    sel = llm.extract_purchase_selection(text_t, titles=titles, api_key=Config.GEMINI_API_KEY)
                    idx_raw, qty = sel.get("index"), sel.get("quantity", 1)
                    idx = idx_raw if idx_raw is not None else 0
                
                if isinstance(idx, int) and 0 <= idx < len(last_res):
                    p = last_res[idx]
                    price = p.get('price') or 0.0
                    user_data["pending_item"] = {"name": p.get('title'), "price": price, "qty": qty}
                    user_data["state"] = "AWAITING_CONFIRMATION"
                    await smart_say(f"Vrei să adăugăm *{qty}x {p.get('title')}*? ({price * qty} RON)")
                    should_c = False # STOP multi-intent loop, wait for confirmation
                else:
                    should_c = True
        elif new_state == "FINISH_ORDER":
            cart = user_data.get("cart", [])
            if not cart:
                await smart_say("Coșul este gol. Spune-mi ce flori cauți pentru a începe o comandă!"); user_data["state"] = "SEARCH_PRODUCTS"
            else:
                total = sum(i['price'] * i['qty'] for i in cart); items_s = "\n".join([f"• {i['qty']} x {i['name']} ({i['price']*i['qty']} RON)" for i in cart])
                target_q = Config.STRESS_TEST_QUEUE if is_mock else Config.PURCHASE_QUEUE
                if await publish_purchase_callback({"user_id": user_id, "items": items_s, "price": total, "timestamp": str(time.time())}, queue_name=target_q):
                    await smart_say(f"🎉 *Comandă trimisă!*\n{items_s}\n━━━━━━━━━━━━━━━━━━━━━\n💰 *TOTAL: {total} RON*\nMulțumim! 🌸")
                    user_data = {"state": "GREETING", "history": [], "filter_scores": {}, "cart": [], "deferred_tasks": []}
        else:
            await smart_say("Interesant! Spune-mi mai multe sau întreabă-mă despre florile noastre."); should_c = True

    except llm.GeminiOverloadedError:
        if is_retry: await smart_say("⚠️ Server ocupat.")
        else:
            set_user_busy(user_id, True); q_size = vital_task_queue.qsize()
            est_wait = (q_size + 1) * 1.5
            msg = f"🌸 Ocupat. Poziția {q_size + 1} în coadă."
            if est_wait > 5: msg += f" (Timp estimat: ~{int(est_wait)}s)"
            await smart_say(msg); await vital_task_queue.put({"user_id": user_id, "channel_id": channel_id, "body": body, "is_mock": is_mock}); return mock_responses if is_mock else None

    user_data["history"].append({"user": text_t, "state": new_state}); await set_user_state(user_id, user_data)
    if should_c and user_data.get("deferred_tasks"):
        next_t = user_data["deferred_tasks"].pop(0); await set_user_state(user_id, user_data)
        recursive_resp = await handle_message({"event": {"user": user_id, "text": next_t["text"], "channel": channel_id}, "is_deferred": True, "intent": next_t["intent"]}, say, client, publish_purchase_callback, is_mock)
        if is_mock and recursive_resp: mock_responses.extend(recursive_resp)

    return mock_responses if is_mock else None
