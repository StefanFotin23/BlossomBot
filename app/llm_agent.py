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

# ==========================================
# 1. CONVERSATION ROUTER (State Classification)
# ==========================================
def decompose_intents(user_message: str, current_state: str = "GREETING", api_key: str = None) -> List[Dict[str, str]]:
    """Descompune un mesaj complex în mai multe acțiuni atomice."""
    client = getApiClient(api_key)
    
    def manual_fallback(msg, current):
        # Fallback ultra-robust cu jargon extins
        msg_cl = msg.lower().strip()
        words = msg_cl.split()

        # Categorii de Keywords
        K_GREET = ["salut", "buna", "bună", "hei", "hello", "hi", "hey", "salutare", "ciao"]
        K_SEARCH = ["caut", "vreau", "buchet", "flori", "recomand", "lalele", "trandafiri", "cadou", "ceva", "aranjament"]
        K_DETAILS = ["detalii", "info", "informatii", "informații", "descriere", "mai mult", "povesteste", "despre"]
        K_PURCHASE = ["da", "dap", "yes", "ok", "k", "kk", "yep", "yup", "y", "sigur", "confirm", "perfect", "vreau", "baga", "hai", "primul", "doilea", "treilea", "1", "2", "3", "iau", "cumpar"]
        K_FINISH = ["gata", "final", "finalizeaza", "finalizează", "trimite", "atat", "atât", "am terminat", "stop", "plateste", "plătește", "checkout"]
        K_RESTART = ["restart", "reset", "de la capat", "anuleaza tot", "anulează tot", "sterge tot", "șterge tot"]

        intent = "SEARCH_PRODUCTS" # Default
        if any(w in K_RESTART for w in [msg_cl] + words): intent = "RESTART"
        elif any(w in K_GREET for w in [msg_cl] + words): intent = "GREETING"
        elif any(w in K_FINISH for w in [msg_cl] + words): intent = "FINISH_ORDER"
        elif any(w in K_DETAILS for w in [msg_cl] + words): intent = "PRODUCT_DETAILS"
        elif any(w in K_PURCHASE for w in [msg_cl] + words): intent = "PURCHASE_INTENT"

        return [{"text": msg, "intent": intent}]


    if not client: return manual_fallback(user_message, current_state)

    states_desc = "GREETING, SEARCH_PRODUCTS, CATALOG_DISCOVERY, PRODUCT_DETAILS, PURCHASE_INTENT, VIEW_CART, FINISH_ORDER, RESTART"

    system_prompt = (
        "Ești un expert în descompunerea mesajelor pentru un chatbot de florărie.\n"
        "Analizează dacă mesajul are mai multe intenții și împarte-l în sarcini atomice.\n"
        "Exemplu: 'al doilea e ok și mai vreau praline' -> \n"
        "[{\"text\": \"al doilea e ok\", \"intent\": \"PURCHASE_INTENT\"}, {\"text\": \"mai vreau praline\", \"intent\": \"SEARCH_PRODUCTS\"}]\n"
        f"Intenții posibile: {states_desc}\n"
        "Răspunde DOAR cu un JSON (listă de obiecte)."
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=f"{system_prompt}\n\nStare actuală: {current_state}\nMesaj: {user_message}",
            config={"temperature": 0.0}
        )
        out = re.sub(r"```json|```", "", response.text).strip()
        tasks = json.loads(out)
        return tasks if isinstance(tasks, list) else [tasks]
    except Exception as e:
        logger.warning(f"Intent Decomposition failed ({e}), using fallback.")
        return manual_fallback(user_message, current_state)

# ==========================================
# 2. RAG ROUTER (Text -> JSON pentru RAG)
# ==========================================
def extract_filters(user_message: str, history: List[Dict] = None, api_key: str = None) -> Dict[str, Any]:
    client = getApiClient(api_key)
    
    # Încărcăm schema de valori unice pentru a ghida Gemini
    schema_path = os.path.join(Config.TEMPLATE_DIR, "schema.json")
    schema_info = ""
    if os.path.exists(schema_path):
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
            schema_info = (
                f"\nValori VALIDE pentru filtre (folosește EXACT aceste denumiri):\n"
                f"- Categorie: {schema.get('categories', [])}\n"
                f"- Dimensiune: {schema.get('dimensiuni', [])}\n"
                f"- Tip Eveniment: {schema.get('event_types', [])}\n"
                f"- Culori: {schema.get('colors', [])}\n"
                f"- Tip Flori: {schema.get('flowers_types', [])}\n"
                f"- Sezon: {schema.get('sezoane', [])}\n"
            )

    history_context = ""
    if history:
        # Luăm ultimele 3 mesaje pentru context
        recent_history = history[-3:]
        history_context = "\nContext recent al conversației:\n"
        for h in recent_history:
            history_context += f"Utilizator: {h.get('user', '')}\n"

    system_prompt = (
        "Ești un expert în extragerea de informații din text pentru o florărie. "
        "Extrage un JSON care să conțină:\n"
        "1. 'filters': un obiect cu 'price_max' (număr), 'category' (string), 'dimensiune' (string), 'event_type' (string), 'colors' (string), 'flowers_type' (string), 'sezon' (string).\n"
        "2. 'negative_filters': un obiect cu aceleași chei ca 'filters', dar pentru lucruri pe care utilizatorul NU le dorește (ex: 'fără trandafiri' -> 'flowers_type': 'Trandafiri').\n"
        "3. 'semantic_query': keywords pentru căutare (string).\n"
        "4. 'mentioned_now': o listă cu cheile filtrelor (pozitive sau negative) menționate EXPLICIT acum.\n"
        "5. 'insisted': o listă cu cheile filtrelor asupra cărora utilizatorul INSISTĂ.\n"
        f"{schema_info}\n"
        "Reguli:\n"
        "- 'filters' și 'negative_filters' trebuie să conțină valorile cumulate.\n"
        "- Răspunde DOAR cu obiectul JSON."
    )

    
    if client:
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=f"{system_prompt}\n{history_context}\nMesaj actual client: {user_message}",
                config={"temperature": 0.0}
            )
            llm_output = response.text
            llm_output = re.sub(r"```json|```", "", llm_output).strip()
            parsed = json.loads(llm_output)
            
            # Curățăm obiectele de valori null/None care pot strica RAG-ul
            def clean_nulls(d):
                if not isinstance(d, dict): return d
                return {k: clean_nulls(v) for k, v in d.items() if v is not None}
            
            cleaned = clean_nulls(parsed)
            # Ne asigurăm că cheile obligatorii de meta-date există (chiar dacă sunt liste goale)
            if "mentioned_now" not in cleaned: cleaned["mentioned_now"] = []
            if "insisted" not in cleaned: cleaned["insisted"] = []
            if "filters" not in cleaned: cleaned["filters"] = {}
            if "negative_filters" not in cleaned: cleaned["negative_filters"] = {}
            if "semantic_query" not in cleaned: cleaned["semantic_query"] = user_message
            
            return cleaned
        except Exception as e:
            error_str = str(e)
            if any(code in error_str for code in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"]):
                raise GeminiOverloadedError(f"Gemini is busy or quota exceeded: {error_str}")
            logger.error(f"⚠️ Eroare neașteptată Gemini Filter Extraction: {e}")
            return {"filters": {}, "semantic_query": user_message, "mentioned_now": [], "insisted": []}
    return {"filters": {}, "semantic_query": user_message, "mentioned_now": [], "insisted": []}


# ==========================================
# 3. HYBRID RESPONSE (Instant Template)
# ==========================================
def generate_hybrid_response(rag_results):
    intros = [
        "Bună! Am găsit câteva opțiuni superbe care s-ar putea să îți placă:",
        "Vă mulțumim pentru solicitare! Iată selecția noastră de flori elegante:",
        "O alegere minunată! Iată cele mai proaspete opțiuni din stoc:"
    ]
    
    selected_intro = random.choice(intros)
    items_text = ""
    
    for i, res in enumerate(rag_results, 1):
        title = res.get('title', 'Produs necunoscut')
        price = res.get('price', 'Preț indisponibil')
        
        # Extragem descrierea
        context_str = res.get('context', '')
        description = ""
        for tag in ["Descriere:", "Body (HTML):", "Description:"]:
            if tag in context_str:
                description = context_str.split(tag)[-1].strip()
                break
        
        if not description:
            lines = context_str.split("\n")
            tags_to_skip = ["Produs:", "Type:", "Flowers Type:", "Colors:", "Categorie:", "Flori:", "Culori:", "Dimensiune:", "Eveniment:", "Sezon:", "Body (HTML):"]
            clean_lines = [line for line in lines if not any(tag in line for tag in tags_to_skip) and ":" not in line[:15]]
            description = " ".join(clean_lines).strip()
            
        if not description:
            description = "Vezi detaliile complete în catalogul nostru online."
            
        limit = 350
        short_desc = description[:limit] + "..." if len(description) > limit else description
        items_text += f"{i}. *{title}* - Preț: {price} lei\n_{short_desc}_\n\n"
    
    return f"{selected_intro}\n\n{items_text}"

# ==========================================
# 4. DETAILED PRODUCT INFO (Readable Metadata)
# ==========================================
def generate_detailed_product_info(product: Dict[str, Any]) -> str:
    """Formatează toate metadatele unui produs într-un mod citibil și frumos."""
    title = product.get('title', 'Produs')
    price = product.get('price', 'N/A')
    
    label_map = {
        "category": "Categorie",
        "dimensiune": "Dimensiune",
        "event_type": "Ideal pentru",
        "colors": "Culori disponibile",
        "flowers_type": "Tipuri de flori",
        "sezon": "Sezon"
    }
    
    header = f"🔍 *Detalii complete pentru: {title}*\n💰 *Preț:* {price} lei\n"
    
    details = []
    for key, label in label_map.items():
        val = product.get(key)
        if val:
            if isinstance(val, list):
                val_str = ", ".join([str(v) for v in val if v])
            else:
                val_str = str(val)
                
            if val_str and val_str.lower() != "nan" and val_str.lower() != "none":
                details.append(f"• *{label}:* {val_str}")
    
    # Descrierea completă
    context_str = product.get('context', '')
    description = ""
    for tag in ["Descriere:", "Body (HTML):", "Description:"]:
        if tag in context_str:
            description = context_str.split(tag)[-1].strip()
            break
            
    if not description:
        lines = context_str.split("\n")
        tags_to_skip = ["Produs:", "Type:", "Flowers Type:", "Colors:", "Categorie:", "Flori:", "Culori:", "Dimensiune:", "Eveniment:", "Sezon:", "Body (HTML):"]
        clean_lines = [line for line in lines if not any(t in line for t in tags_to_skip) and ":" not in line[:15]]
        description = " ".join(clean_lines).strip()

    full_text = header + "\n".join(details)
    if description:
        full_text += f"\n\n📖 *Descriere:* \n_{description}_"
        
    full_text += "\n\n_Dorești să adaugi acest produs în comandă? Spune-mi 'Da' sau 'Îl vreau pe acesta'._"
    
    return full_text

# ==========================================
# 5. GENERAL RESPONSE (Deterministic & Fast)
# ==========================================
def generate_general_response(user_message: str, api_key: str = None) -> str:
    """Clasifică și returnează un răspuns instantaneu pentru întrebări comune."""
    client = getApiClient(api_key)
    
    responses = {
        "GREETING": "Bună! Sunt BlossomBot 🌸 asistentul tău. Cu ce te pot ajuta astăzi?",
        "IDENTITY": "Sunt BlossomBot, un asistent virtual creat pentru a te ajuta să alegi cele mai frumoase flori și cadouri.",
        "AVAILABILITY": "Sunt aici și te ascult! Spune-mi ce cauți.",
        "HUMOR": "Sunt un bot serios, dar pot spune că florile mele nu se ofilesc niciodată în catalog! 😂",
        "UNKNOWN": "Sunt aici să te ajut cu drag! Ce anume te interesează din oferta noastră?"
    }

    if not client:
        return responses["UNKNOWN"]

    system_prompt = (
        "Clasifică mesajul utilizatorului în una din următoarele categorii:\n"
        "- GREETING: saluturi (salut, bună, hei)\n"
        "- IDENTITY: cine ești, ce faci\n"
        "- AVAILABILITY: ești acolo, răspunzi, ești activ\n"
        "- HUMOR: glume, bancuri\n"
        "- UNKNOWN: orice altceva\n"
        "Răspunde DOAR cu numele categoriei."
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=f"{system_prompt}\n\nUtilizator: {user_message}",
            config={
                "temperature": 0.0,
                "max_output_tokens": 10
            }
        )
        category = response.text.strip().upper()
        return responses.get(category, responses["UNKNOWN"])
    except:
        return responses["UNKNOWN"]

# ==========================================
# 6. PURCHASE INTENT (Index & Quantity)
# ==========================================
def extract_purchase_selection(user_message: str, titles: List[str] = None, api_key: str = None) -> Dict[str, Any]:
    """Extrage indexul produsului și cantitatea, folosind și titlurile pentru context."""
    client = getApiClient(api_key)
    default = {"index": 0, "quantity": 1}
    
    if not client:
        return default

    context_titles = ""
    if titles:
        context_titles = "\nProduse afișate anterior:\n" + "\n".join([f"{i}. {t}" for i, t in enumerate(titles)])

    system_prompt = (
        "Ești un asistent de vânzări. Analizează mesajul utilizatorului care alege un produs dintr-o listă.\n"
        f"{context_titles}\n"
        "Extrage JSON:\n"
        "1. 'index': numărul de ordine (0-based). Dacă numește un produs, identifică-i indexul din listă.\n"
        "2. 'quantity': numărul de bucăți. Default 1.\n"
        "Răspunde DOAR cu JSON."
    )
    
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=f"{system_prompt}\n\nMesaj client: {user_message}",
            config={"temperature": 0.0, "max_output_tokens": 50}
        )
        out = re.sub(r"```json|```", "", response.text).strip()
        parsed = json.loads(out)
        return parsed[0] if isinstance(parsed, list) else parsed
    except:
        return default

# ==========================================
# 7. CONFIRMATION PARSER (Yes/No & Qty)
# ==========================================
def parse_confirmation(user_message: str, api_key: str = None) -> Dict[str, Any]:
    """Returnează dacă a confirmat și dacă a specificat o cantitate nouă."""
    client = getApiClient(api_key)
    
    msg_clean = user_message.lower().strip()
    if msg_clean in ["da", "yes", "ok", "k", "yep", "daaa", "sigur", "confirm", "perfect", "il vreau"]:
        return {"confirmed": True, "quantity": None}
    if msg_clean in ["nu", "no", "nope", "anuleaza", "anulează", "stai"]:
        return {"confirmed": False, "quantity": None}
        
    if not client:
        return {"confirmed": False, "quantity": None}

    system_prompt = (
        "Analizează răspunsul utilizatorului la confirmarea adăugării în coș.\n"
        "Extrage JSON cu:\n"
        "1. 'confirmed': true (da, ok, k, yes, vreau, adauga) sau false (nu, anuleaza, nu vreau).\n"
        "2. 'quantity': dacă specifică O NOUĂ cantitate (ex: 'da, 3 bucati' -> 3), pune numărul. Altfel null.\n"
        "Răspunde DOAR cu JSON."
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=f"{system_prompt}\n\nMesaj: {user_message}",
            config={"temperature": 0.0, "max_output_tokens": 50}
        )
        out = re.sub(r"```json|```", "", response.text).strip()
        return json.loads(out)
    except:
        return {"confirmed": False, "quantity": None}
