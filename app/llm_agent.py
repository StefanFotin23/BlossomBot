import os
import requests
import random
import json
import re
import logging
from google import genai
from typing import Dict, Any, List
from app.config import Config

# Get logger
logger = logging.getLogger(__name__)

GEMINI_MODEL_NAME = Config.GEMINI_MODEL_NAME

class GeminiOverloadedError(Exception):
    """Excepție custom pentru când Gemini API dă 503 sau 429."""
    pass

def getApiClient(api_key: str = None):
    if api_key:
        return genai.Client(api_key=api_key)
    return None

def safe_json_parse(text: str, default_val: Any = None) -> Any:
    """Parsează JSON-ul din textul AI-ului, curățând markdown-ul și validând rezultatul."""
    if not text: return default_val
    try:
        # Curățare tag-uri code blocks și whitespace
        clean_text = re.sub(r"```json|```", "", text).strip()
        # Uneori Gemini returnează "true" sau "false" literal sau în ghilimele
        if clean_text.lower() in ["true", '"true"']: return True
        if clean_text.lower() in ["false", '"false"']: return False
        
        data = json.loads(clean_text)
        return data
    except:
        return default_val

# ==========================================
# 1. CONVERSATION ROUTER (State Classification)
# ==========================================
def manual_fallback(msg: str, current: str) -> List[Dict[str, str]]:
    msg_cl = re.sub(r'[!?.,]+$', '', msg.lower().strip())
    words = msg_cl.split()

    K_GREET = ["salut", "buna", "bună", "hei", "hello", "hi", "hey", "salutare", "ciao"]
    K_IDENTITY = ["cine esti", "cine ești", "ce ești", "ce faci", "cine sunteti", "cine sunteți"]
    K_SEARCH = ["caut", "vreau", "buchet", "flori", "recomand", "lalele", "trandafiri", "cadou", "ceva", "aranjament", "plante", "arata-mi", "arată-mi", "mai ieftin", "mai scump", "altceva", "vin", "prosecco", "ciocolata", "dulce", "martisoare", "mărțișoare", "mimosa", "mimoza"]
    K_DETAILS = ["detalii", "info", "informatii", "informații", "descriere", "mai mult", "povesteste", "despre"]
    K_PURCHASE = ["iau", "cumpar", "cumpăr", "adauga", "adaugă", "il vreau", "îl vreau", "baga", "bagă", "pune"]
    K_SELECTION = ["primul", "prima", "doilea", "treilea", "varianta", "v1", "v2", "v3", "asta", "aceasta"]
    K_FINISH = ["gata", "final", "finalizeaza", "finalizează", "finalizez", "trimite", "atat", "atât", "am terminat", "stop", "plateste", "plătește", "checkout", "comanda", "comandă"]
    K_RESTART = ["restart", "reset", "de la capat", "anuleaza tot", "anulează tot", "sterge tot", "șterge tot"]
    K_CATALOG = ["produse", "catalog", "aveti", "aveți", "oferta", "oferiti", "recomanda", "recomandări", "recomandari", "recomandati-mi", "ce-mi recomanzi", "ce-mi recomandați"]
    K_CONFIRM = ["da", "yes", "ok", "k", "sigur", "perfect", "confirm", "corect", "exact", "afirmativ", "da te rog"]
    K_REMOVE = ["scoate", "scoatem", "sterge", "șterge", "elimina", "elimină", "delete", "remove", "fara", "fără"]
    K_CART = ["cos", "coș", "cosul", "coșul", "cosul meu", "coșul meu", "cart"]

    if any(w in K_RESTART for w in [msg_cl] + words): return [{"text": msg, "intent": "RESTART"}]
    if any(w in K_FINISH for w in [msg_cl] + words): return [{"text": msg, "intent": "FINISH_ORDER"}]
    if any(w in K_REMOVE for w in words) and any(w in K_CART + ["articol", "produs", "ultimul", "prima"] for w in words):
        return [{"text": msg, "intent": "REMOVE_FROM_CART"}]
    
    # Priority 1: Purchase/Selection
    is_selection_digit = re.search(r'\b[1-9]\b', msg_cl) or re.search(r'v[1-9]', msg_cl)
    has_sel_word = any(w in K_SELECTION for w in words)
    has_pur_word = any(w in K_PURCHASE for w in words)
    
    if has_pur_word or has_sel_word or (len(words) <= 3 and is_selection_digit):
        return [{"text": msg, "intent": "PURCHASE_INTENT"}]
    
    # Priority 2: Cart View
    if any(p in msg_cl for p in ["arata-mi cosul", "arată-mi coșul"]) or any(w in K_CART for w in words):
        if not any(w in ["baga", "pune", "adauga", "adaugă"] for w in words):
            return [{"text": msg, "intent": "VIEW_CART"}]

    if any(w in K_CATALOG for w in words): return [{"text": msg, "intent": "CATALOG_DISCOVERY"}]
    if any(w in K_SEARCH for w in words): return [{"text": msg, "intent": "SEARCH_PRODUCTS"}]

    # Priority 4: Confirmations
    if any(w in K_CONFIRM for w in [msg_cl] + words):
            if current in ["SEARCH_PRODUCTS", "PRODUCT_DETAILS", "PURCHASE_INTENT", "AWAITING_CONFIRMATION"]:
                return [{"text": msg, "intent": "PURCHASE_INTENT"}]
            return [{"text": msg, "intent": "CONTINUE"}]

    if any(w in K_DETAILS for w in words): return [{"text": msg, "intent": "PRODUCT_DETAILS"}]
    if msg_cl in K_GREET or any(p in msg_cl for p in K_IDENTITY) or (words and words[0] in K_GREET and len(words) <= 3):
        return [{"text": msg, "intent": "GREETING"}]
    
    return [{"text": msg, "intent": "SEARCH_PRODUCTS"}]

async def decompose_intents(user_message: str, current_state: str = "GREETING", api_key: str = None) -> List[Dict[str, str]]:
    client = getApiClient(api_key)
    if not client: return manual_fallback(user_message, current_state)

    states_desc = "GREETING, SEARCH_PRODUCTS, CATALOG_DISCOVERY, PRODUCT_DETAILS, PURCHASE_INTENT, VIEW_CART, FINISH_ORDER, RESTART, CONTINUE, REMOVE_FROM_CART"
    system_prompt = (
        "Ești un expert în descompunerea mesajelor pentru un chatbot de florărie. "
        "Descompune mesajul utilizatorului în sarcini atomice.\n"
        "REGULI CRITICE:\n"
        "1. PURCHASE_INTENT: Când alege un produs (ex: 'v1', 'varianta 2', 'primul', 'da', 'pune-l în coș') SAU când specifică cantitatea (ex: 'vreau 5 bucăți').\n"
        "2. REMOVE_FROM_CART: Când vrea să scoată ceva din coș (ex: 'scoate ultimul articol').\n"
        "3. Dacă mesajul are mai multe intenții, desparte-le CLAR. NU repeta textul primei intenții în a doua.\n"
        "Răspunde EXCLUSIV cu un JSON format dintr-o LISTĂ de OBIECTE.\n"
        f"Intenții posibile: {states_desc}\n"
    )

    try:
        # WRAP SYNC CALL IN THREAD
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=GEMINI_MODEL_NAME,
            contents=f"{system_prompt}\n\nStare actuală: {current_state}\nMesaj: {user_message}",
            config={"temperature": 0.0}
        )
        tasks = safe_json_parse(response.text)
        if not isinstance(tasks, list):
            if isinstance(tasks, dict): tasks = [tasks]
            else: return manual_fallback(user_message, current_state)
        
        validated = []
        for t in tasks:
            if isinstance(t, dict) and "intent" in t:
                if "text" not in t or not t["text"]: t["text"] = user_message
                validated.append(t)
        return validated if validated else manual_fallback(user_message, current_state)
    except Exception as e:
        if any(code in str(e) for code in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"]):
            raise GeminiOverloadedError(f"Gemini overloaded: {e}")
        return manual_fallback(user_message, current_state)

# ==========================================
# 2. RAG ROUTER (Text -> JSON pentru RAG)
# ==========================================
def manual_extract_filters(msg: str) -> Dict[str, Any]:
    """Extracție bazată pe Regex pentru reziliență (GOD MODE)."""
    msg_cl = msg.lower()
    filters = {}
    mentioned = []
    
    p_match = re.search(r'(\d+)\s*(lei|ron)', msg_cl)
    if p_match:
        filters["price_max"] = float(p_match.group(1))
        mentioned.append("price_max")
    
    flowers_map = {"Trandafiri": ["trandafir", "rose"], "Lalele": ["lalea", "lalele", "tulip"], "Hortensii": ["hortensi", "hydrange"], "Frezii": ["frezii", "freesia"], "Mimosa": ["mimosa", "mimoza"], "Bujori": ["bujor", "peony"]}
    for f_name, keys in flowers_map.items():
        if any(k in msg_cl for k in keys):
            filters["flowers_type"] = f_name
            mentioned.append("flowers_type")
            break
            
    if any(k in msg_cl for k in ["vin", "prosecco", "bautura", "ciocolata", "praline", "dulce"]):
        filters["category"] = "Cadouri"
        mentioned.append("category")
    elif any(k in msg_cl for k in ["martisor", "mărțișor"]):
        filters["event_type"] = "1 Martie"
        mentioned.append("event_type")

    return {"filters": filters, "negative_filters": {}, "semantic_query": msg, "mentioned_now": mentioned, "insisted": []}

async def extract_filters(user_message: str, history: List[Dict] = None, api_key: str = None, force_manual: bool = False) -> Dict[str, Any]:
    if force_manual: return manual_extract_filters(user_message)
    
    client = getApiClient(api_key)
    schema_path = os.path.join(Config.TEMPLATE_DIR, "schema.json")
    schema_info = ""
    if os.path.exists(schema_path):
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
            schema_info = f"\nValori VALIDE: Categorie: {schema.get('categories', [])}, Flori: {schema.get('flowers_types', [])}, Eveniment: {schema.get('event_types', [])}"

    history_context = ""
    if history:
        recent_history = history[-3:]
        history_context = "\nContext recent:\n" + "\n".join([f"Utilizator: {h.get('user', '')}" for h in recent_history])

    system_prompt = (
        "Ești un expert în extragerea de informații din text. "
        "Extrage JSON cu: 'filters', 'negative_filters', 'semantic_query', 'mentioned_now'.\n"
        f"{schema_info}\n"
        "Regulă: Mărțișoarele -> 'event_type': '1 Martie'."
    )

    if client:
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=GEMINI_MODEL_NAME,
                contents=f"{system_prompt}\n{history_context}\nMesaj: {user_message}",
                config={"temperature": 0.0}
            )
            parsed = safe_json_parse(response.text)
            if not isinstance(parsed, dict): return manual_extract_filters(user_message)
            if "filters" not in parsed: parsed["filters"] = {}
            if "semantic_query" not in parsed: parsed["semantic_query"] = user_message
            return parsed
        except Exception as e:
            if any(code in str(e) for code in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"]):
                raise GeminiOverloadedError(f"Gemini overloaded: {e}")
            return manual_extract_filters(user_message)
    return manual_extract_filters(user_message)

# ==========================================
# 3. HYBRID RESPONSE
# ==========================================
def generate_hybrid_response(rag_results):
    intros = ["Bună! Am găsit câteva opțiuni superbe:", "Iată selecția noastră de flori:", "O alegere minunată! Iată opțiunile din stoc:"]
    selected_intro = random.choice(intros)
    items_text = ""
    for i, res in enumerate(rag_results, 1):
        title = res.get('title', 'Produs necunoscut'); price = res.get('price', 'N/A')
        context_str = res.get('context', '')
        description = ""
        for tag in ["Descriere:", "Body (HTML):", "Description:"]:
            if tag in context_str: description = context_str.split(tag)[-1].strip(); break
        if not description: description = "Vezi detaliile complete în catalog."
        short_desc = description[:350] + "..." if len(description) > 350 else description
        items_text += f"{i}. *{title}* - Preț: {price} lei\n_{short_desc}_\n\n"
    return f"{selected_intro}\n\n{items_text}"

# ==========================================
# 4. DETAILED PRODUCT INFO
# ==========================================
def generate_detailed_product_info(product: Dict[str, Any]) -> str:
    title = product.get('title', 'Produs'); price = product.get('price', 'N/A')
    label_map = {"category": "Categorie", "dimensiune": "Dimensiune", "event_type": "Ideal pentru", "colors": "Culori", "flowers_type": "Tipuri de flori"}
    header = f"🔍 *Detalii complete pentru: {title}*\n💰 *Preț:* {price} lei\n"
    details = []
    for key, label in label_map.items():
        val = product.get(key)
        if val: details.append(f"• *{label}:* {val}")
    context_str = product.get('context', '')
    description = ""
    for tag in ["Descriere:", "Body (HTML):", "Description:"]:
        if tag in context_str: description = context_str.split(tag)[-1].strip(); break
    full_text = header + "\n".join(details)
    if description: full_text += f"\n\n📖 *Descriere:* \n_{description}_"
    full_text += "\n\n_Dorești să adaugi acest produs în comandă? Spune-mi 'Da' sau 'Îl vreau pe acesta'._"
    return full_text

# ==========================================
# 6. PURCHASE INTENT (Index & Quantity)
# ==========================================
async def extract_purchase_selection(user_message: str, titles: List[str] = None, api_key: str = None) -> Dict[str, Any]:
    client = getApiClient(api_key)
    default = {"index": 0, "quantity": 1}
    msg_cl = user_message.lower().strip()
    
    ordinal_map = {r'\bv1\b': 0, r'varianta 1': 0, r'primul': 0, r'prima': 0, r'\bv2\b': 1, r'varianta 2': 1, r'doilea': 1, r'doua': 1, r'\bv3\b': 2, r'varianta 3': 2, r'treilea': 2, r'treia': 2}
    for pattern, idx in ordinal_map.items():
        if re.search(pattern, msg_cl): return {"index": idx, "quantity": 1}

    if not client: return default
    context_titles = ""
    if titles: context_titles = "\nProduse afișate anterior:\n" + "\n".join([f"{i}. {t}" for i, t in enumerate(titles)])
    system_prompt = f"Ești un asistent de vânzări. Identifică produsul ales de utilizator dintr-o listă.\n{context_titles}\nExtrage JSON: 'index' (0-based) și 'quantity' (default 1)."
    
    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=GEMINI_MODEL_NAME,
            contents=f"{system_prompt}\n\nMesaj: {user_message}",
            config={"temperature": 0.0, "max_output_tokens": 50}
        )
        parsed = safe_json_parse(response.text)
        if isinstance(parsed, list): parsed = parsed[0]
        if not isinstance(parsed, dict): return default
        return parsed
    except: return default

# ==========================================
# 7. CONFIRMATION PARSER
# ==========================================
async def parse_confirmation(user_message: str, api_key: str = None) -> Dict[str, Any]:
    client = getApiClient(api_key); msg_cl = user_message.lower().strip()
    if msg_cl in ["da", "yes", "ok", "k", "yep", "sigur", "confirm", "perfect", "il vreau"]: return {"confirmed": True, "quantity": None}
    if msg_cl in ["nu", "no", "nope", "anuleaza", "anulează", "stai", "negativ"]: return {"confirmed": False, "quantity": None}
    if not client: return {"confirmed": False, "quantity": None}
    system_prompt = "Analizează răspunsul utilizatorului la confirmarea adăugării în coș.\nExtrage JSON: 'confirmed' (boolean) și 'quantity' (dacă specifică o NOUĂ cantitate, ex: 'da, 5 bucăți' -> 5, altfel null)."
    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=GEMINI_MODEL_NAME,
            contents=f"{system_prompt}\n\nMesaj: {user_message}",
            config={"temperature": 0.0, "max_output_tokens": 50}
        )
        parsed = safe_json_parse(response.text)
        if not isinstance(parsed, dict): return {"confirmed": False, "quantity": None}
        return parsed
    except: return {"confirmed": False, "quantity": None}
