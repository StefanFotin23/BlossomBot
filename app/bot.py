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
                for k in ["last_results", "cart", "deferred_tasks", "history"]:
                    if k not in state or not isinstance(state[k], list): state[k] = []
                if "filter_scores" not in state or not isinstance(state["filter_scores"], dict): state["filter_scores"] = {}
                return state
            return default_state
        except Exception as e:
            logger.error(f"Redis get error: {e}")
    return _local_user_states.get(user_id, default_state)

async def set_user_state(user_id, state_data):
    if redis_client:
        try:
            redis_client.set(f"user:{user_id}", json.dumps(state_data), ex=3600*24)
            await asyncio.sleep(0.02) # Propagation safety
            return
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    _local_user_states[user_id] = state_data

# --- CONTEXTUAL MATCHING LOGIC ---
def normalize_ro_word(word):
    w = word.lower().strip()
    if len(w) <= 3: return w
    suffixes = ['ele', 'elor', 'ilor', 'lor', 'ul', 'ului', 'uri', 'ate', 'i', 'e']
    for s in sorted(suffixes, key=len, reverse=True):
        if w.endswith(s): return w[:-len(s)]
    return w

def find_best_contextual_match(user_text, products):
    if not products: return None
    text_clean = re.sub(r'[^\w\s]', ' ', user_text.lower()).strip()
    words = text_clean.split()
    
    # MATCH PE ORDINALE (Prioritate maximă)
    ordinal_map = {"primul": 0, "prima": 0, "unu": 0, "1": 0, "doilea": 1, "doua": 1, "două": 1, "doi": 1, "2": 1, "treilea": 2, "treia": 2, "trei": 2, "3": 2, "ultimul": len(products)-1, "ultima": len(products)-1}
    for w in words:
        # Ignorăm numerele care sunt probabil filtre de preț (ex: > 10)
        if w.isdigit() and int(w) > 10: continue
        if w in ordinal_map:
            idx = ordinal_map[w]
            if 0 <= idx < len(products): return idx

    # MATCH PE ATRIBUTE
    GLOBAL_STOPWORDS = {"un", "o", "de", "cu", "din", "la", "pe", "pentru", "si", "și", "vrea", "vreau", "caut", "arata", "imi", "îmi", "place", "detalii", "despre", "multumesc", "mulțumesc"}
    user_keywords = set([normalize_ro_word(w) for w in words if w not in GLOBAL_STOPWORDS and len(w) > 2])
    if not user_keywords: return None

    word_freq = {}; product_features = []
    for p in products:
        raw_feat = f"{p.get('title', '')} {p.get('category', '')} {p.get('context', '')}"
        feat_words = set([normalize_ro_word(w) for w in re.sub(r'[^\w\s]', ' ', raw_feat.lower()).split() if len(w) > 2])
        product_features.append(feat_words)
        for w in feat_words: word_freq[w] = word_freq.get(w, 0) + 1

    scores = []
    for feat_words in product_features:
        match_score = sum([(2 if word_freq.get(w, 0) == 1 else 1) for w in user_keywords.intersection(feat_words)])
        scores.append(match_score)

    if scores and max(scores) > 0 and scores.count(max(scores)) == 1:
        return scores.index(max(scores))
    return None

# --- ATOMIC LOCKING ---
async def acquire_user_lock(user_id):
    if not redis_client: return True
    return redis_client.set(f"busy:{user_id}", "1", nx=True, ex=30)

def release_user_lock(user_id):
    if redis_client: redis_client.delete(f"busy:{user_id}")

def increment_pending(user_id):
    if redis_client: return redis_client.incr(f"pending:{user_id}")
    return 1

def decrement_pending(user_id):
    if redis_client: 
        val = redis_client.decr(f"pending:{user_id}")
        if val < 0: redis_client.set(f"pending:{user_id}", "0")
        return val
    return 0

# --- TASK QUEUE ---
vital_task_queue = asyncio.Queue()

async def background_task_worker(worker_id, client, publish_purchase_callback):
    while True:
        task = await vital_task_queue.get()
        user_id, ch_id, body = task["user_id"], task["channel_id"], task["body"]
        try:
            attempts = 0
            while attempts < 120:
                if await acquire_user_lock(user_id): break
                await asyncio.sleep(0.5); attempts += 1
            async def q_say(msg):
                if task.get("is_mock"): logger.info(f"📝 [W{worker_id}] {user_id}: {msg[:60]}...")
                else: await client.chat_postMessage(channel=ch_id, text=msg)
            retries = 0
            while retries < 3:
                try:
                    await handle_message(body, q_say, client, publish_purchase_callback, is_mock=task.get("is_mock", False), is_retry=True)
                    break
                except llm.GeminiOverloadedError:
                    retries += 1; await asyncio.sleep(2 * retries)
                except Exception as e:
                    logger.error(f"❌ [W{worker_id}] Error: {e}"); break
        finally:
            decrement_pending(user_id); release_user_lock(user_id); vital_task_queue.task_done()

async def start_task_workers(count, client, publish_purchase_callback):
    actual_count = min(count, 5)
    for i in range(actual_count):
        asyncio.create_task(background_task_worker(i, client, publish_purchase_callback))

# --- RECOMANDĂRI ---
async def process_recommendations(user_data, user_query, force_manual: bool = False):
    filter_scores = user_data.get("filter_scores", {})
    history = user_data.get("history", [])
    extracted_data = await llm.extract_filters(user_query, history=history, api_key=Config.GEMINI_API_KEY, force_manual=force_manual)
    for f_key in extracted_data.get("mentioned_now", []):
        filter_scores[f_key] = filter_scores.get(f_key, 0) + 2
    user_data["filter_scores"] = filter_scores
    results, metadata = await rag.sota_hybrid_search(extracted_data, filter_scores=filter_scores, qdrant_url=Config.QDRANT_URL)
    if not results:
        results, _ = await rag.sota_hybrid_search({"semantic_query": "produse populare flori"}, filter_scores={}, qdrant_url=Config.QDRANT_URL)
        if not results: return "Nu am găsit nimic momentan. Te rog să încerci o altă căutare!", []
        return "Nu am găsit exact ce căutai, dar iată cele mai populare alegeri:\n\n" + llm.generate_hybrid_response(results), results
    feedback = ""
    if metadata.get("failed_filters"):
        min_p = min([r.get("price", 0) for r in results]) if results else 0
        feedback = f"💡 *Am ajustat căutarea.* Cele mai apropiate opțiuni pornesc de la {min_p} RON:\n\n"
    return feedback + llm.generate_hybrid_response(results), results

# --- MAIN HANDLER ---
async def handle_message(body, say, client, publish_purchase_callback=None, is_mock=False, is_retry=False):
    event = body.get("event", {})
    client_msg_id = event.get("client_msg_id") or body.get("event_id") or event.get("event_ts")
    user_id = event.get("user")
    if not user_id: return None
    if not is_retry and not body.get("is_deferred") and client_msg_id and not str(client_msg_id).startswith("stress"):
        if redis_client and redis_client.set(f"evt:{client_msg_id}", "1", nx=True, ex=600) is None: return None

    channel_id = event.get("channel", "direct"); full_text = event.get("text", "")
    if not full_text: return None
    text_cl = re.sub(r'[!?.,]+$', '', full_text.lower().strip())
    words = text_cl.split()
    
    mock_responses = []
    async def smart_say(m):
        if is_mock: mock_responses.append(m)
        else: await say(m)

    lock_owned = False
    if not is_retry and not body.get("is_deferred"):
        p_count = increment_pending(user_id)
        if p_count > 1 or not await acquire_user_lock(user_id):
            await vital_task_queue.put({"user_id": user_id, "channel_id": channel_id, "body": body, "is_mock": is_mock})
            if p_count == 2: await smart_say("🌸 Ocupat. Poziția în coadă...")
            return mock_responses if is_mock else None
        lock_owned = True

    try:
        user_data = await get_user_state(user_id)
        last_results = user_data.get("last_results", [])
        current_state = user_data.get("state", "GREETING")

        # --- 1. DETERMINISTIC FAST PATHS (Zero-AI latency) ---
        K_RESTART = ["restart", "reset", "de la capat", "sterge tot"]; K_GREET = ["salut", "buna", "bună", "hei", "hello"]
        K_FINISH = ["gata", "final", "finalizeaza", "finalizează", "finalizez", "trimite", "atat", "atât", "am terminat", "comanda", "comandă"]
        K_YES = ["da", "dap", "ok", "k", "perfect", "baga", "da te rog"]
        K_NO = ["nu", "anuleaza", "anulează", "stai", "negativ"]
        
        # Determină contextul deterministic
        context_idx = find_best_contextual_match(full_text, last_results)
        is_asking_details = any(w in text_cl for w in ["detalii", "info", "spune-mi", "arata-mi", "arată-mi", "mai multe"])
        
        forced_intent = None
        if any(w in K_RESTART for w in [text_cl] + words):
            user_data = {"state": "GREETING", "history": [], "filter_scores": {}, "cart": [], "deferred_tasks": [], "task_buffer": None, "last_results": []}
            await set_user_state(user_id, user_data); await smart_say("🔄 Conversație resetată. Cu ce te pot ajuta?"); return mock_responses if is_mock else None

        if text_cl in K_GREET and len(words) <= 2:
            await smart_say("Bună! Cu ce te pot ajuta astăzi?"); user_data["state"] = "SEARCH_PRODUCTS"; await set_user_state(user_id, user_data)
            return mock_responses if is_mock else None

        # LOGICĂ DE FORȚARE A INTENȚIEI
        if is_asking_details and current_state == "AWAITING_CONFIRMATION" and context_idx is None and len(words) <= 6:
            # Dacă userul cere detalii în timp ce confirmă un produs, îi dăm detaliile acelui produs
            # DAR verificăm să nu fie o cerere de "altceva"
            if not any(w in text_cl for w in ["altceva", "alt", "altele", "altă", "nou"]):
                p_pending = user_data.get("pending_item")
                if p_pending and last_results:
                    for i, r in enumerate(last_results):
                        if r.get('title') == p_pending.get('name'):
                            context_idx = i; break
                forced_intent = "PRODUCT_DETAILS"

        if not forced_intent:
            if any(w in K_FINISH for w in words) and len(words) <= 3: forced_intent = "FINISH_ORDER"
            elif any(w in ["cos", "coș", "cosul", "coșul"] for w in words) and len(words) <= 3: forced_intent = "VIEW_CART"
            elif context_idx is not None: forced_intent = "PRODUCT_DETAILS" if is_asking_details else "PURCHASE_INTENT"
            elif text_cl in K_YES and last_results and current_state != "AWAITING_CONFIRMATION": 
                forced_intent = "PURCHASE_INTENT"; context_idx = 0
            elif text_cl in K_NO and current_state == "AWAITING_CONFIRMATION": 
                pass # Handled by Section 2

        # --- 2. STATE MACHINE (CONFIRMATION) ---
        if current_state == "AWAITING_CONFIRMATION":
            is_n = any(w in K_NO for w in words)
            is_y = any(w in K_YES for w in words)
            is_finish = any(w in K_FINISH for w in words)
            m_num = re.search(r'^(\d+)$', text_cl)
            is_finish_add = is_finish and not is_n
            
            is_view_cart = (forced_intent == "VIEW_CART")
            is_shift = (not is_view_cart and (any(k in text_cl for k in ["pune", "caut", "scoate", "șterge", "sterge", "elimin"]) or (context_idx is not None and not is_y and not is_finish_add)))

            if is_n and not is_shift: 
                await smart_say("Am anulat. Mai dorești altceva?"); user_data["state"] = "SEARCH_PRODUCTS"; user_data["pending_item"] = None; await set_user_state(user_id, user_data)
                return mock_responses if is_mock else None
            
            if (is_y or m_num or is_finish_add or is_shift):
                p = user_data.get("pending_item")
                if p and not is_n and (is_y or m_num or is_finish_add):
                    if m_num: p["qty"] = int(m_num.group(1))
                    user_data["cart"] = user_data.get("cart", []) + [p]
                    user_data["pending_item"] = None; user_data["state"] = "SEARCH_PRODUCTS"
                    await set_user_state(user_id, user_data)
                    await smart_say(f"✅ Adăugat {p['qty']}x {p['name']} în coș!")
                    if is_finish: current_state = "SEARCH_PRODUCTS"; forced_intent = "FINISH_ORDER"
                    else: return mock_responses if is_mock else None
                
                if is_shift and not is_finish_add: 
                    user_data["pending_item"] = None; user_data["state"] = "SEARCH_PRODUCTS"; await set_user_state(user_id, user_data)
                
                if not is_shift and not is_finish_add and user_data.get("deferred_tasks"):
                    next_t = user_data["deferred_tasks"].pop(0); await set_user_state(user_id, user_data)
                    return await handle_message({"event": {"user": user_id, "text": next_t.get('text', ''), "channel": channel_id}, "is_deferred": True, "intent": next_t.get('intent')}, say, client, publish_purchase_callback, is_mock, is_retry=True)
                
                if not is_finish: return mock_responses if is_mock else None

        # --- 3. TASK DECOMPOSITION (Skip for deterministic paths) ---
        if forced_intent:
            current_task = {"text": full_text, "intent": forced_intent}
        elif not is_retry and not body.get("is_deferred"):
            tasks = await llm.decompose_intents(full_text, current_state, api_key=Config.GEMINI_API_KEY)
            user_data["deferred_tasks"] = tasks[1:]; current_task = tasks[0]; await set_user_state(user_id, user_data)
        else:
            current_task = {"text": full_text, "intent": body.get("intent", "SEARCH_PRODUCTS")}

        new_intent = current_task.get("intent", "SEARCH_PRODUCTS")
        text_t = current_task.get("text", full_text)

        # --- 4. EXECUTION ---
        if new_intent == "SEARCH_PRODUCTS":
            resp, res = await process_recommendations(user_data, text_t)
            user_data["last_results"] = res; user_data["state"] = "SEARCH_PRODUCTS"; await set_user_state(user_id, user_data)
            await smart_say(resp)
            
        elif new_intent == "PRODUCT_DETAILS":
            idx = context_idx
            if idx is None and last_results:
                sel = await llm.extract_purchase_selection(text_t, titles=[r.get('title') for r in last_results], api_key=Config.GEMINI_API_KEY)
                idx = sel.get("index")
            if isinstance(idx, int) and 0 <= idx < len(last_results):
                await smart_say(llm.generate_detailed_product_info(last_results[idx]))
            else:
                resp, res = await process_recommendations(user_data, text_t)
                user_data["last_results"] = res; await set_user_state(user_id, user_data); await smart_say(resp)
            user_data["state"] = "SEARCH_PRODUCTS"; await set_user_state(user_id, user_data)
            
        elif new_intent == "PURCHASE_INTENT":
            if not last_results:
                await asyncio.sleep(0.05); user_data = await get_user_state(user_id); last_results = user_data.get("last_results", [])
            idx = context_idx
            qty = 1
            if idx is None and last_results:
                sel = await llm.extract_purchase_selection(text_t, titles=[r.get('title') for r in last_results], api_key=Config.GEMINI_API_KEY)
                idx = sel.get("index"); qty = sel.get("quantity", 1)
            
            if isinstance(idx, int) and 0 <= idx < len(last_results):
                p = last_results[idx]
                user_data["pending_item"] = {"name": p.get('title'), "price": p.get('price', 0.0), "qty": qty}
                user_data["state"] = "AWAITING_CONFIRMATION"; await set_user_state(user_id, user_data)
                await smart_say(f"Vrei să adăugăm *{qty}x {p.get('title')}*? ({p.get('price', 0.0) * qty} RON)")
            else:
                if text_cl in K_YES: await smart_say("Spune-mi te rog numele produsului sau varianta dorită (1, 2 sau 3).")
                else:
                    resp, res = await process_recommendations(user_data, text_t)
                    user_data["last_results"] = res; await set_user_state(user_id, user_data); await smart_say(resp)

        elif new_intent == "REMOVE_FROM_CART":
            cart = user_data.get("cart", [])
            if not cart: await smart_say("Coșul tău este deja gol.")
            else:
                if any(w in text_t.lower() for w in ["ultimul", "ultima", "tot"]):
                    removed = cart.pop() if "tot" not in text_t.lower() else None
                    if "tot" in text_t.lower(): cart = []; await smart_say("✅ Am golit coșul de cumpărături.")
                    else: await smart_say(f"✅ Am scos {removed['name']} din coș.")
                else:
                    found_idx = -1
                    for i, item in enumerate(cart):
                        if any(normalize_ro_word(w) in normalize_ro_word(item['name']) for w in words if len(w) > 3):
                            found_idx = i; break
                    if found_idx != -1:
                        removed = cart.pop(found_idx); await smart_say(f"✅ Am scos {removed['name']} din coș.")
                    else: await smart_say("Nu am găsit produsul în coș. Spune-mi numele lui sau 'ultimul'.")
            user_data["cart"] = cart; user_data["state"] = "SEARCH_PRODUCTS"; await set_user_state(user_id, user_data)

        elif new_intent == "VIEW_CART":
            await asyncio.sleep(0.1); user_data = await get_user_state(user_id); cart = user_data.get("cart", [])
            if not cart: await smart_say("🛒 Coșul tău este gol momentan.")
            else:
                items_s = "\n".join([f"• {i['qty']} x {i['name']} ({i['price']*i['qty']} RON)" for i in cart])
                await smart_say(f"🛒 *Coșul tău:*\n{items_s}\n━━━━━━━━━━━━━━━━━━━━━\n💰 *Total: {sum(i['price']*i['qty'] for i in cart)} RON*")

        elif new_intent == "FINISH_ORDER":
            await asyncio.sleep(0.2); user_data = await get_user_state(user_id); cart = user_data.get("cart", [])
            if not cart: await smart_say("Coșul tău este gol. Adaugă ceva frumos înainte de a finaliza!")
            else:
                total = sum(i['price'] * i['qty'] for i in cart); items_s = "\n".join([f"• {i['qty']} x {i['name']} ({i['price']*i['qty']} RON)" for i in cart])
                if await publish_purchase_callback({"user_id": user_id, "items": items_s, "price": total, "timestamp": str(time.time())}, queue_name=Config.PURCHASE_QUEUE):
                    await smart_say(f"🎉 *Comandă trimisă!*\n{items_s}\n━━━━━━━━━━━━━━━━━━━━━\n💰 *TOTAL: {total} RON*\nMulțumim! 🌸")
                    user_data = {"state": "GREETING", "history": [], "filter_scores": {}, "cart": [], "deferred_tasks": [], "last_results": []}; await set_user_state(user_id, user_data)

        user_data["history"].append({"user": text_cl, "state": new_intent}); await set_user_state(user_id, user_data)
        if user_data.get("deferred_tasks"):
            next_t = user_data["deferred_tasks"].pop(0); await set_user_state(user_id, user_data)
            await handle_message({"event": {"user": user_id, "text": next_t.get('text', ''), "channel": channel_id}, "is_deferred": True, "intent": next_t.get('intent')}, say, client, publish_purchase_callback, is_mock, is_retry=True)

    except Exception as e: 
        logger.error(f"❌ Error: {e}")
        if not is_retry: await smart_say("Ups, am întâmpinat o mică eroare. Putem să mai încercăm o dată?")
    finally:
        if lock_owned: release_user_lock(user_id); decrement_pending(user_id)
    return mock_responses if is_mock else None
