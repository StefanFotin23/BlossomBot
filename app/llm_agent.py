import os
import requests
import random
import json
import re
import logging
import asyncio
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
from app.config import Config

# Get logger
logger = logging.getLogger(__name__)

# Custom exception for Gemini rate-limits/overloads
class GeminiOverloadedError(Exception):
    """Excepție custom pentru când Gemini API dă 503 sau 429."""
    pass

# ==========================================
# LOCAL INTENT CLASSIFIER (GRU Model)
# ==========================================
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        out, h_n = self.gru(embedded)
        logits = self.fc(h_n.squeeze(0))
        return logits

# Global variables for model
INTENTS = [
    "CATALOG_DISCOVERY", "CONTINUE", "FINISH_ORDER", "GREETING", 
    "PRODUCT_DETAILS", "PURCHASE_INTENT", "REMOVE_FROM_CART", 
    "RESTART", "SEARCH_PRODUCTS", "VIEW_CART"
]
MAX_LEN = 10

gru_model = None
word_to_idx = {}

def load_local_classifier():
    global gru_model, word_to_idx
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vocab_path = os.path.join(base_dir, "models", "vocab.json")
    weights_path = os.path.join(base_dir, "models", "intent_gru.pt")
    
    if os.path.exists(vocab_path) and os.path.exists(weights_path):
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                word_to_idx = json.load(f)
            vocab_size = len(word_to_idx)
            gru_model = GRUClassifier(vocab_size, 64, 32, len(INTENTS))
            gru_model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
            gru_model.eval()
            logger.info("Successfully loaded local GRU Intent Classifier model.")
        except Exception as e:
            logger.error(f"Error loading local GRU model: {e}")

load_local_classifier()

def get_top_intents(user_message: str, k: int = 3) -> List[Dict[str, Any]]:
    """Returns top k predicted intents with their probabilities from the local GRU model."""
    if gru_model is None or not word_to_idx:
        val = 1.0 / len(INTENTS)
        return [{"intent": intent, "probability": val} for intent in INTENTS[:k]]
        
    # Clean text using same rules as training
    text_clean = user_message.lower().strip()
    text_clean = re.sub(r'[^\w\s]', ' ', text_clean)
    words = text_clean.split()
    
    # Text to sequence
    seq = [word_to_idx.get(w, 1) for w in words]
    if len(seq) < MAX_LEN:
        seq = seq + [0] * (MAX_LEN - len(seq))
    else:
        seq = seq[:MAX_LEN]
        
    try:
        seq_tensor = torch.tensor([seq], dtype=torch.long)
        with torch.no_grad():
            logits = gru_model(seq_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
            
        results = []
        for i, p in enumerate(probs):
            results.append({"intent": INTENTS[i], "probability": float(p)})
        results.sort(key=lambda x: x["probability"], reverse=True)
        return results[:k]
    except Exception as e:
        logger.error(f"Error running local GRU inference: {e}")
        val = 1.0 / len(INTENTS)
        return [{"intent": intent, "probability": val} for intent in INTENTS[:k]]

GEMINI_MODEL_NAME = Config.GEMINI_MODEL_NAME

def getApiClient(api_key: str = None):
    # Stub for compatibility
    return None

# ==========================================
# GEMINI HTTP QUERY METHOD (GenAI SDK replacement)
# ==========================================
def query_gemini_http(system_prompt: str, user_message: str, api_key: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": f"{system_prompt}\n\n{user_message}"}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0
        }
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        if response.status_code == 429:
            raise GeminiOverloadedError("Gemini rate limit exceeded (429)")
        if response.status_code != 200:
            raise Exception(f"Gemini API returned status {response.status_code}: {response.text}")
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            raise GeminiOverloadedError(f"Gemini rate limit exceeded: {e}")
        raise e

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

# Keyword lists for semantic validation and fallback routing
K_GREET = ["salut", "buna", "bună", "hei", "hello", "hi", "hey", "salutare", "ciao", "servus", "neatza", "neata"]
K_IDENTITY = ["cine esti", "cine ești", "ce ești", "ce faci", "cine sunteti", "cine sunteți"]
K_SEARCH = ["caut", "vreau", "buchet", "flori", "recomand", "lalele", "trandafiri", "cadou", "ceva", "aranjament", "plante", "arata-mi", "arată-mi", "mai ieftin", "mai scump", "altceva", "vin", "prosecco", "ciocolata", "dulce", "martisoare", "mărțișoare", "mimosa", "mimoza"]
K_DETAILS = ["detalii", "info", "informatii", "informații", "descriere", "mai mult", "povesteste", "despre"]
K_PURCHASE = ["iau", "cumpar", "cumpăr", "adauga", "adaugă", "il vreau", "îl vreau", "baga", "bagă", "pune"]
K_SELECTION = ["primul", "prima", "doilea", "treilea", "varianta", "v1", "v2", "v3", "asta", "aceasta"]
K_FINISH = ["gata", "final", "finalizeaza", "finalizează", "finalizez", "trimite", "atat", "atât", "am terminat", "stop", "plateste", "plătește", "checkout", "comanda", "comandă", "plata"]
K_RESTART = ["restart", "reset", "de la capat", "de la capăt", "anuleaza tot", "anulează tot", "sterge tot", "șterge tot"]
K_CATALOG = ["produse", "catalog", "aveti", "aveți", "oferta", "oferiti", "recomanda", "recomandări", "recomandari", "recomandati-mi", "ce-mi recomanzi", "ce-mi recomandați"]
K_CONFIRM = ["da", "yes", "ok", "k", "sigur", "perfect", "confirm", "corect", "exact", "afirmativ", "da te rog"]
K_REMOVE = ["scoate", "scoatem", "sterge", "șterge", "elimina", "elimină", "delete", "remove", "fara", "fără"]
K_CART = ["cos", "coș", "cosul", "coșul", "cosul meu", "coșul meu", "cart"]

# ==========================================
# 1. CONVERSATION ROUTER (State Classification)
# ==========================================
def manual_fallback(msg: str, current: str) -> List[Dict[str, str]]:
    msg_cl = re.sub(r'[!?.,]+$', '', msg.lower().strip())
    words = msg_cl.split()

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
    # Tier 1: Local GRU Classifier + Contextual Bayesian Update
    if gru_model is not None and word_to_idx:
        try:
            top_candidates = get_top_intents(user_message, k=3)
            raw_probs = {c["intent"]: c["probability"] for c in top_candidates}
            
            # Context-guided state priors (Bayesian-style context tuning)
            priors = {intent: 0.01 for intent in INTENTS}
            
            if current_state == "PRODUCT_DETAILS":
                priors["PRODUCT_DETAILS"] = 0.50
                priors["PURCHASE_INTENT"] = 0.35
                priors["CONTINUE"] = 0.08
            elif current_state == "SEARCH_PRODUCTS":
                priors["SEARCH_PRODUCTS"] = 0.45
                priors["PRODUCT_DETAILS"] = 0.25
                priors["CATALOG_DISCOVERY"] = 0.15
                priors["PURCHASE_INTENT"] = 0.10
            elif current_state == "AWAITING_CONFIRMATION":
                priors["CONTINUE"] = 0.45
                priors["PURCHASE_INTENT"] = 0.35
                priors["SEARCH_PRODUCTS"] = 0.10
            elif current_state == "VIEW_CART":
                priors["VIEW_CART"] = 0.40
                priors["FINISH_ORDER"] = 0.35
                priors["REMOVE_FROM_CART"] = 0.15
            elif current_state == "FINISH_ORDER":
                priors["FINISH_ORDER"] = 0.50
                priors["CONTINUE"] = 0.30
                priors["RESTART"] = 0.10
            else: # GREETING, etc.
                priors["GREETING"] = 0.40
                priors["SEARCH_PRODUCTS"] = 0.40
                priors["CATALOG_DISCOVERY"] = 0.10
                
            # Compute posterior probabilities
            posteriors = {}
            for intent in INTENTS:
                likelihood = raw_probs.get(intent, 0.001)
                posteriors[intent] = priors[intent] * likelihood
                
            # Normalize posteriors
            total_post = sum(posteriors.values())
            if total_post > 0:
                for intent in posteriors:
                    posteriors[intent] /= total_post
                    
            # Find the top two posteriors
            sorted_posteriors = sorted(posteriors.items(), key=lambda x: x[1], reverse=True)
            best_intent, best_prob = sorted_posteriors[0]
            second_intent, second_prob = sorted_posteriors[1]
            
            cleaned_lower = user_message.lower().strip()
            conjunctions = ["si", "și", "apoi", "iar"]
            words = cleaned_lower.split()
            has_conjunction = any(w in words for w in conjunctions)
            
            is_very_short = len(words) <= 2
            local_route_success = False
            
            # Check decision boundary for single intent local routing
            if not has_conjunction:
                if is_very_short and best_prob > 0.65:
                    local_route_success = True
                elif best_prob > 0.82 and (best_prob - second_prob) > 0.35:
                    local_route_success = True
            
            # Semantic keyword guardrails on local GRU routing to prevent false positives
            is_valid = True
            words_lower = [re.sub(r'[!?.,]+$', '', w.lower()) for w in words]
            if best_intent == "FINISH_ORDER":
                is_valid = any(w in K_FINISH for w in words_lower)
            elif best_intent == "REMOVE_FROM_CART":
                is_valid = any(w in K_REMOVE for w in words_lower)
            elif best_intent == "VIEW_CART":
                is_valid = any(w in K_CART for w in words_lower)
            elif best_intent == "RESTART":
                is_valid = any(w in K_RESTART for w in words_lower)
            elif best_intent == "GREETING":
                is_valid = any(w in K_GREET for w in words_lower)
                    
            if local_route_success and is_valid:
                logger.info(f"⚡ Local GRU Route: '{best_intent}' (Posterior: {best_prob:.4f}, Raw: {raw_probs.get(best_intent, 0.0):.4f})")
                return [{"text": user_message, "intent": best_intent}]
                
            logger.info(f"🤖 Query '{user_message}' too complex/ambiguous (best posterior {best_prob:.2f}). Delegating to Gemini...")
        except Exception as e:
            logger.error(f"Error in local GRU routing path: {e}")

    # Tier 2: Fallback to Gemini API
    if not api_key: return manual_fallback(user_message, current_state)

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
        response_text = await asyncio.to_thread(
            query_gemini_http,
            system_prompt,
            f"Stare actuală: {current_state}\nMesaj: {user_message}",
            api_key
        )
        tasks = safe_json_parse(response_text)
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
        logger.warning(f"⚠️ Gemini API failed (decompose_intents): {e}. Falling back to manual_fallback.")
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
        val = float(p_match.group(1))
        if any(w in msg_cl for w in ["peste", "mai mult", "minim", "maxim", "cel putin", "scump"]):
            filters["price_min"] = val
            mentioned.append("price_min")
        else:
            filters["price_max"] = val
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
    if force_manual or not api_key: return manual_extract_filters(user_message)

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
        "REGULI FILTRE:\n"
        "1. Preț: Dacă cere 'sub X lei' sau ieftin -> filters['price_max'] = X. Dacă cere 'peste X lei', minim sau scump -> filters['price_min'] = X. ATENȚIE, NU LE ÎNCURCA!\n"
        f"{schema_info}\n"
        "Regulă: Mărțișoarele -> 'event_type': '1 Martie'."
    )

    try:
        response_text = await asyncio.to_thread(
            query_gemini_http,
            system_prompt,
            f"{history_context}\nMesaj: {user_message}",
            api_key
        )
        parsed = safe_json_parse(response_text)
        if not isinstance(parsed, dict): return manual_extract_filters(user_message)
        
        # Type validation for defense-in-depth robustness
        if "filters" not in parsed or not isinstance(parsed["filters"], dict):
            parsed["filters"] = {}
        if "negative_filters" not in parsed or not isinstance(parsed["negative_filters"], dict):
            parsed["negative_filters"] = {}
        if "mentioned_now" not in parsed or not isinstance(parsed["mentioned_now"], list):
            parsed["mentioned_now"] = []
        if "semantic_query" not in parsed or not isinstance(parsed["semantic_query"], str):
            parsed["semantic_query"] = user_message
        if "insisted" not in parsed or not isinstance(parsed["insisted"], list):
            parsed["insisted"] = []
            
        return parsed
    except Exception as e:
        logger.warning(f"⚠️ Gemini API failed (extract_filters): {e}. Falling back to manual_extract_filters.")
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
    default = {"index": None, "quantity": 1}
    msg_cl = user_message.lower().strip()
    
    qty_match = re.search(r'(\d+)\s*(buc|bucati|bucăți|x)', msg_cl)
    qty = int(qty_match.group(1)) if qty_match else 1
    
    ordinal_map = {r'\bv1\b': 0, r'varianta 1': 0, r'primul': 0, r'prima': 0, r'\bv2\b': 1, r'varianta 2': 1, r'doilea': 1, r'doua': 1, r'\bv3\b': 2, r'varianta 3': 2, r'treilea': 2, r'treia': 2}
    for pattern, idx in ordinal_map.items():
        if re.search(pattern, msg_cl): return {"index": idx, "quantity": qty}

    if not api_key: return default
    context_titles = ""
    if titles: context_titles = "\nProduse afișate anterior:\n" + "\n".join([f"{i}. {t}" for i, t in enumerate(titles)])
    system_prompt = f"Ești un asistent de vânzări. Identifică produsul ales de utilizator dintr-o listă. Dacă nu ești sigur sau mesajul e ambiguu, pune index null.\n{context_titles}\nExtrage JSON: 'index' (0-based) și 'quantity' (default 1)."
    
    try:
        response_text = await asyncio.to_thread(
            query_gemini_http,
            system_prompt,
            f"Mesaj: {user_message}",
            api_key
        )
        parsed = safe_json_parse(response_text)
        if isinstance(parsed, list): parsed = parsed[0]
        if not isinstance(parsed, dict): return default
        return parsed
    except Exception as e:
        logger.warning(f"⚠️ Gemini API failed (extract_purchase_selection): {e}. Falling back to default.")
        return default

# ==========================================
# 7. CONFIRMATION PARSER
# ==========================================
async def parse_confirmation(user_message: str, api_key: str = None) -> Dict[str, Any]:
    msg_cl = user_message.lower().strip()
    if msg_cl in ["da", "yes", "ok", "k", "yep", "sigur", "confirm", "perfect", "il vreau"]: return {"confirmed": True, "quantity": None}
    if msg_cl in ["nu", "no", "nope", "anuleaza", "anulează", "stai", "negativ"]: return {"confirmed": False, "quantity": None}
    if not api_key: return {"confirmed": False, "quantity": None}
    system_prompt = "Analizează răspunsul utilizatorului la confirmarea adăugării în coș.\nExtrage JSON: 'confirmed' (boolean) și 'quantity' (dacă specifică o NOUĂ cantitate, ex: 'da, 5 bucăți' -> 5, altfel null)."
    try:
        response_text = await asyncio.to_thread(
            query_gemini_http,
            system_prompt,
            f"Mesaj: {user_message}",
            api_key
        )
        parsed = safe_json_parse(response_text)
        if not isinstance(parsed, dict): return {"confirmed": False, "quantity": None}
        return parsed
    except Exception as e:
        logger.warning(f"⚠️ Gemini API failed (parse_confirmation): {e}. Falling back to default.")
        return {"confirmed": False, "quantity": None}
