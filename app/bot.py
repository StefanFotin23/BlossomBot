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
    default_state = {"state": "GREETING", "history": [], "filter_scores": {}, "cart": [], "deferred_tasks": [], "task_buffer": None, "last_results": []}
    if redis_client:
        try:
            data = redis_client.get(f"user:{user_id}")
            if data:
                state = json.loads(data)
                # STRUCTURE REPAIR
                for k in ["last_results", "cart", "deferred_tasks", "history"]:
                    if k not in state or state[k] is None: state[k] = []
                if "filter_scores" not in state or state["filter_scores"] is None: state["filter_scores"] = {}
                return state
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

# --- ATOMIC LOCKING & SERIALIZATION ---
async def acquire_user_lock(user_id):
    if not redis_client: return True
    return redis_client.set(f"busy:{user_id}", "1", nx=True, ex=30)

def release_user_lock(user_id):
    if redis_client: redis_client.delete(f"busy:{user_id}")

async def is_user_busy(user_id):
    if redis_client: return redis_client.exists(f"busy:{user_id}")
    return False

def increment_pending(user_id):
    if redis_client: return redis_client.incr(f"pending:{user_id}")
    return 1

def decrement_pending(user_id):
    if redis_client: 
        val = redis_client.decr(f"pending:{user_id}")
        if val < 0: redis_client.set(f"pending:{user_id}", "0")
        return val
    return 0

# --- SHADOW TASK BUFFERING ---
async def fill_task_buffer(user_id, task):
    try:
        user_data = await get_user_state(user_id)
        if not user_data: return
        text, intent = task.get("text", ""), task.get("intent", "UNKNOWN")
        if not text: return
        buffered_data = {"text": text, "intent": intent, "response": None, "results": None}

        if intent == "SEARCH_PRODUCTS":
            resp, res = await process_recommendations(user_id, text, history=user_data.get("history"))
            buffered_data["response"], buffered_data["results"] = resp, res
        elif intent == "CATALOG_DISCOVERY":
            resp, res = await process_recommendations(user_id, "produse populare", history=[])
            buffered_data["response"], buffered_data["results"] = resp, res

        if buffered_data["response"] and redis_client:
            redis_client.set(f"buffer:{user_id}", json.dumps(buffered_data), ex=300)
    except Exception as e:
        logger.error(f"❌ [SHADOW] Error: {e}")

async def get_buffered_task(user_id, text_t):
    if redis_client:
        try:
            data = redis_client.get(f"buffer:{user_id}")
            if data:
                buf = json.loads(data)
                if buf.get("text") == text_t:
                    redis_client.delete(f"buffer:{user_id}")
                    return buf
        except: pass
    return None

# --- TASK QUEUE & PARALLEL WORKERS ---
vital_task_queue = asyncio.Queue()

async def background_task_worker(worker_id, client, publish_purchase_callback):
    logger.info(f"⚙️ Atomic Task Worker {worker_id} started.")
    while True:
        task = await vital_task_queue.get()
        user_id, ch_id, body = task["user_id"], task["channel_id"], task["body"]
        
        try:
            # BUSY-WAIT LOCK
            attempts = 0
            while attempts < 60:
                if await acquire_user_lock(user_id): break
                await asyncio.sleep(0.5); attempts += 1
            
            async def q_say(msg):
                if task.get("is_mock"): logger.info(f"📝 [W{worker_id}] {user_id}: {msg[:60]}...")
                else: await client.chat_postMessage(channel=ch_id, text=msg)
            
            # IN-PLACE RETRY
            retries = 0
            while retries < 3:
                try:
                    await handle_message(body, q_say, client, publish_purchase_callback, is_mock=task.get("is_mock", False), is_retry=True)
                    break
                except llm.GeminiOverloadedError:
                    retries += 1
                    if retries < 3:
                        wait_s = 3 * retries
                        logger.warning(f"⏳ [W{worker_id}] Gemini Overloaded. Retry {retries}/3 in {wait_s}s for {user_id}")
                        await asyncio.sleep(wait_s)
                    else: logger.error(f"❌ [W{worker_id}] Final Gemini retry failed for {user_id}")
                except Exception as e:
                    logger.error(f"❌ [W{worker_id}] Logic Error for {user_id}: {e}")
                    break
        finally:
            decrement_pending(user_id)
            release_user_lock(user_id)
            vital_task_queue.task_done()

async def start_task_workers(count, client, publish_purchase_callback):
    for i in range(count):
        asyncio.create_task(background_task_worker(i, client, publish_purchase_callback))

# --- RECOMANDĂRI ---
async def process_recommendations(user_id, user_query, history=None):
    user_data = await get_user_state(user_id)
    filter_scores = user_data.get("filter_scores", {})
    extracted_data = llm.extract_filters(user_query, history=history, api_key=Config.GEMINI_API_KEY)
    
    for f_key in extracted_data.get("mentioned_now", []):
        filter_scores[f_key] = filter_scores.get(f_key, 0) + 2

    user_data["filter_scores"] = filter_scores
    await set_user_state(user_id, user_data)
    
    results, metadata = rag.sota_hybrid_search(extracted_data, filter_scores=filter_scores, qdrant_url=Config.QDRANT_URL)
    
    # UX FALLBACK
    if not results:
        results, _ = rag.sota_hybrid_search({"semantic_query": "produse populare flori"}, filter_scores={}, qdrant_url=Config.QDRANT_URL)
        if not results: return "Nu am găsit nimic momentan. Te rog să încerci o altă căutare!", []
        feedback = "Nu am găsit exact ce căutai, dar iată cele mai populare alegeri ale noastre:\n\n"
        return feedback + llm.generate_hybrid_response(results), results

    feedback = ""
    if metadata.get("failed_filters"):
        min_p = min([r.get("price", 0) for r in results])
        feedback = f"💡 *Am ajustat căutarea.* Cele mai apropiate opțiuni pornesc de la {min_p} RON:\n\n"

    return feedback + llm.generate_hybrid_response(results), results

# --- MAIN HANDLER ---
async def handle_message(body, say, client, publish_purchase_callback=None, is_mock=False, is_retry=False):
    event = body.get("event", {})
    client_msg_id = event.get("client_msg_id") or body.get("event_id") or event.get("event_ts")
    user_id = event.get("user")
    if not user_id: return None
    
    # 1. IDEMPOTENCY
    if not is_retry and not body.get("is_deferred") and client_msg_id:
        if redis_client and redis_client.set(f"evt:{client_msg_id}", "1", nx=True, ex=600) is None:
            logger.info(f"⏭️ Skipping retry: {client_msg_id}")
            return None

    channel_id = event.get("channel", "direct")
    full_text = event.get("text", "")
    if not full_text: return None

    text_cl = re.sub(r'[!?.,]+$', '', full_text.lower().strip())
    words = text_cl.split()
    mock_responses = []
    
    async def smart_say(m):
        if is_mock: mock_responses.append(m)
        else: await say(m)

    # 2. ATOMIC SERIALIZATION
    lock_owned = False
    is_deferred = body.get("is_deferred", False)
    if not is_retry and not is_deferred:
        p_count = increment_pending(user_id)
        if p_count > 1 or not await acquire_user_lock(user_id):
            await vital_task_queue.put({"user_id": user_id, "channel_id": channel_id, "body": body, "is_mock": is_mock})
            if p_count == 2: await smart_say("🌸 Ocupat. Poziția în coadă...")
            return mock_responses if is_mock else None
        lock_owned = True

    try:
        user_data = await get_user_state(user_id)
        current_state = user_data.get("state", "GREETING")

        K_GREET = ["salut", "buna", "bună", "hei", "hello", "hi", "hey", "salutare", "ciao", "servus"]
        K_YES = ["da", "dap", "daaa", "yes", "ok", "k", "sigur", "confirm", "confirmare", "perfect", "vreau", "baga", "hai", "corect", "exact", "asa", "așa", "il vreau", "îl vreau", "asta", "ăsta", "afirmativ", "da te rog"]
        K_RESTART = ["restart", "reset", "de la capat", "anuleaza tot", "sterge tot"]

        # 3. COMMAND FAST-PATHS
        if any(w in K_RESTART for w in [text_cl] + words):
            user_data = {"state": "GREETING", "history": [], "filter_scores": {}, "cart": [], "deferred_tasks": [], "task_buffer": None, "last_results": []}
            await set_user_state(user_id, user_data); await smart_say("🔄 Conversație resetată. Cu ce te pot ajuta?"); return mock_responses if is_mock else None

        # 4. STATE MACHINE (CONFIRMATION)
        if current_state == "AWAITING_CONFIRMATION":
            is_y, is_n = any(w in K_YES for w in [text_cl] + words), any(w in ["nu", "no", "nope", "stai", "anuleaza", "negativ"] for w in words)
            m_q = re.search(r'^(\d+)$', text_cl)
            
            if not is_y and not is_n and not m_q:
                ai_conf = llm.parse_confirmation(full_text, api_key=Config.GEMINI_API_KEY)
                is_y, is_n = ai_conf.get("confirmed", False), not ai_conf.get("confirmed", True)
            
            if is_y or is_n or m_q:
                if is_y or m_q:
                    p = user_data.get("pending_item")
                    if p:
                        if m_q: p["qty"] = int(m_q.group(1))
                        user_data["cart"] = user_data.get("cart", []) + [p]
                        user_data["pending_item"] = None
                        await smart_say(f"✅ Adăugat {p['qty']}x {p['name']} în coș!")
                        user_data["state"] = "SEARCH_PRODUCTS"
                else:
                    await smart_say("Am anulat. Mai dorești altceva?"); user_data["state"] = "SEARCH_PRODUCTS"
                await set_user_state(user_id, user_data)
                
                if user_data.get("deferred_tasks"):
                    next_t = user_data["deferred_tasks"].pop(0); await set_user_state(user_id, user_data)
                    await smart_say(f"Acum să ne ocupăm de: \"{next_t.get('text', '')}\"...")
                    recursive_resp = await handle_message({"event": {"user": user_id, "text": next_t.get('text', ''), "channel": channel_id}, "is_deferred": True, "intent": next_t.get('intent')}, say, client, publish_purchase_callback, is_mock, is_retry=True)
                    if is_mock and recursive_resp: mock_responses.extend(recursive_resp)
                return mock_responses if is_mock else None

        # 5. TASK DECOMPOSITION
        if not is_retry and not is_deferred:
            tasks = llm.manual_fallback(full_text, current_state)
            if tasks[0]["intent"] in ["RESTART", "FINISH_ORDER", "VIEW_CART", "CATALOG_DISCOVERY"]:
                current_task = tasks[0]; user_data["deferred_tasks"] = tasks[1:]
            else:
                tasks = llm.decompose_intents(full_text, current_state, api_key=Config.GEMINI_API_KEY)
                user_data["deferred_tasks"] = tasks[1:]; current_task = tasks[0]
            await set_user_state(user_id, user_data)
        else:
            current_task = {"text": full_text, "intent": body.get("intent", "SEARCH_PRODUCTS")}

        new_state = current_task.get("intent", "SEARCH_PRODUCTS")
        text_t = current_task.get("text", full_text)
        shadow_buf = await get_buffered_task(user_id, text_t)
        buffered_resp, buffered_res = (shadow_buf["response"], shadow_buf["results"]) if shadow_buf else (None, None)

        # 6. EXECUTION
        should_c = False
        if new_state == "GREETING":
            if any(w in text_cl for w in ["cine esti", "cine ești", "cine sunteți", "cine sunteti"]):
                await smart_say("Sunt BlossomBot 🌸, asistentul tău virtual pentru flori și cadouri superbe! Cu ce te pot ajuta?")
            elif any(w in text_cl for w in K_GREET): await smart_say("Bună! Cu ce te pot ajuta astăzi?")
            else: await smart_say("Sunt BlossomBot 🌸, asistentul tău virtual. Spune-mi ce cauți!")
            user_data["state"] = "SEARCH_PRODUCTS"; should_c = True
        elif new_state == "CATALOG_DISCOVERY":
            resp, res = (buffered_resp, buffered_res) if buffered_resp else await process_recommendations(user_id, "produse populare", history=[])
            await smart_say(resp); user_data["last_results"] = res; user_data["state"] = "SEARCH_PRODUCTS"
            should_c = not bool(user_data.get("deferred_tasks"))
        elif new_state == "SEARCH_PRODUCTS":
            user_data["pending_item"] = None
            resp, res = (buffered_resp, buffered_res) if buffered_resp else await process_recommendations(user_id, text_t, history=user_data.get("history"))
            await smart_say(resp); user_data["last_results"] = res; user_data["state"] = "SEARCH_PRODUCTS"
            should_c = not bool(user_data.get("deferred_tasks"))
        elif new_state == "PRODUCT_DETAILS":
            last_res = user_data.get("last_results", [])
            if not last_res:
                resp, res = await process_recommendations(user_id, text_t, history=user_data.get("history"))
                await smart_say(resp); user_data["last_results"] = res; user_data["state"] = "SEARCH_PRODUCTS"
            else:
                sel = llm.extract_purchase_selection(text_t, titles=[r.get('title') for r in last_res], api_key=Config.GEMINI_API_KEY)
                idx = sel.get("index", 0)
                if 0 <= idx < len(last_res):
                    await smart_say(llm.generate_detailed_product_info(last_res[idx]))
                else: 
                    resp, res = await process_recommendations(user_id, text_t, history=user_data.get("history"))
                    await smart_say(resp); user_data["last_results"] = res
                user_data["state"] = "SEARCH_PRODUCTS"
            should_c = True
        elif new_state == "PURCHASE_INTENT":
            last_res = user_data.get("last_results", [])
            K_SEARCH_TRIGGER = ["caut", "vreau", "buchet", "flori", "lalele", "trandafiri", "cadou", "martisoare", "vin", "ciocolata", "mimoza", "mimosa"]
            has_search_jargon = any(w in text_t.lower() for w in K_SEARCH_TRIGGER) and not any(w in ["primul", "prima", "doilea", "treilea", "varianta"] for w in text_t.lower().split())
            
            if (has_search_jargon or not last_res) and "da" not in text_t.lower():
                resp, res = await process_recommendations(user_id, text_t, history=user_data.get("history"))
                await smart_say(resp); user_data["last_results"] = res; user_data["state"] = "SEARCH_PRODUCTS"
                should_c = True
            else:
                # REFINED REGEX: Ignore numbers followed by quantity keywords
                m_idx = re.search(r'\b([1-9])\b(?!\s*(buc|fire|item|prod|bucati|bucăți))', text_cl)
                idx = int(m_idx.group(1)) - 1 if m_idx else None
                
                sel = llm.extract_purchase_selection(text_t, titles=[r.get('title') for r in last_res], api_key=Config.GEMINI_API_KEY)
                if idx is None or idx >= len(last_res):
                    idx = sel.get("index", 0)
                qty = sel.get("quantity", 1)
                
                if isinstance(idx, int) and 0 <= idx < len(last_res):
                    p = last_res[idx]; price = p.get('price', 0.0)
                    user_data["pending_item"] = {"name": p.get('title'), "price": price, "qty": qty}
                    user_data["state"] = "AWAITING_CONFIRMATION"
                    await smart_say(f"Vrei să adăugăm *{qty}x {p.get('title')}*? ({price * qty} RON)")
                    should_c = False
                else: 
                    # If 'da' was said but no index found, ask instead of vague search
                    if "da" in text_t.lower() and not last_res:
                        await smart_say("Spune-mi te rog ce produs dorești să adaugi în coș?")
                    else:
                        resp, res = await process_recommendations(user_id, text_t, history=user_data.get("history"))
                        await smart_say(resp); user_data["last_results"] = res; user_data["state"] = "SEARCH_PRODUCTS"
                    should_c = True
        elif new_state == "VIEW_CART":
            cart = user_data.get("cart", [])
            if not cart: await smart_say("🛒 Coșul tău este gol momentan. Spune-mi ce flori cauți?")
            else:
                items_s = "\n".join([f"• {i['qty']} x {i['name']} ({i['price']*i['qty']} RON)" for i in cart])
                await smart_say(f"🛒 *Coșul tău:*\n{items_s}\n━━━━━━━━━━━━━━━━━━━━━\n💰 *Total: {sum(i['price']*i['qty'] for i in cart)} RON*")
            should_c = True
        elif new_state == "FINISH_ORDER":
            cart = user_data.get("cart", [])
            if not cart: await smart_say("Coșul tău este gol. Adaugă ceva frumos înainte de a finaliza!"); user_data["state"] = "SEARCH_PRODUCTS"; should_c = True
            else:
                total = sum(i['price'] * i['qty'] for i in cart); items_s = "\n".join([f"• {i['qty']} x {i['name']} ({i['price']*i['qty']} RON)" for i in cart])
                if await publish_purchase_callback({"user_id": user_id, "items": items_s, "price": total, "timestamp": str(time.time())}, queue_name=Config.PURCHASE_QUEUE):
                    await smart_say(f"🎉 *Comandă trimisă!*\n{items_s}\n━━━━━━━━━━━━━━━━━━━━━\n💰 *TOTAL: {total} RON*\nMulțumim! 🌸")
                    user_data = {"state": "GREETING", "history": [], "filter_scores": {}, "cart": [], "deferred_tasks": [], "last_results": []}
                    should_c = False
        elif new_state == "CONTINUE": should_c = True
        else:
            resp, res = await process_recommendations(user_id, text_t, history=user_data.get("history"))
            await smart_say(resp); user_data["last_results"] = res; user_data["state"] = "SEARCH_PRODUCTS"
            should_c = True

        if not should_c and user_data.get("deferred_tasks"):
            asyncio.create_task(fill_task_buffer(user_id, user_data["deferred_tasks"][0]))

        user_data["history"].append({"user": text_cl, "state": new_state}); await set_user_state(user_id, user_data)
        
        if should_c and user_data.get("deferred_tasks"):
            next_t = user_data["deferred_tasks"].pop(0); await set_user_state(user_id, user_data)
            await smart_say(f"Trecem la: \"{next_t.get('text', '')}\"...")
            recursive_resp = await handle_message({"event": {"user": user_id, "text": next_t.get('text', ''), "channel": channel_id}, "is_deferred": True, "intent": next_t.get('intent')}, say, client, publish_purchase_callback, is_mock, is_retry=is_retry)
            if is_mock and recursive_resp: mock_responses.extend(recursive_resp)

    except llm.GeminiOverloadedError:
        if not is_retry:
            await smart_say("🌸 Ocupat. Poziția în coadă...")
            await vital_task_queue.put({"user_id": user_id, "channel_id": channel_id, "body": body, "is_mock": is_mock})
        else: raise
    except Exception as e: 
        logger.error(f"❌ Error in handle_message: {e}")
        if not is_retry: await smart_say("Ups, am întâmpinat o mică eroare. Putem să mai încercăm o dată?")
    finally:
        if lock_owned:
            decrement_pending(user_id)
            release_user_lock(user_id)
    return mock_responses if is_mock else None
