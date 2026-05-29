import os
import requests
import random
import json
import re
import logging
from google import genai
from typing import Dict, Any
from app.config import Config

# Get logger
logger = logging.getLogger(__name__)

OLLAMA_OPTIMIZED_OPTIONS = {
    "num_ctx": 2048,
    "num_gpu": 99,
    "temperature": 0.1,
    "num_predict": 150,
    "low_vram": True
}

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
def classify_intent(user_message: str, current_state: str = "GREETING", api_key: str = None) -> str:
    """Identifică intenția utilizatorului folosind Gemini."""
    client = getApiClient(api_key)
    
    def manual_fallback(msg, current):
        msg_clean = msg.lower()
        if any(word in msg_clean for word in ["salut", "buna", "bună", "hello", "hi"]):
            return "GREETING"
        if any(word in msg_clean for word in ["caut", "vreau", "buchet", "flori", "recomand", "lalele", "trandafiri"]):
            return "SEARCH_PRODUCTS"
        if any(word in msg_clean for word in ["cumpăr", "comand", "iau", "da", "confirm", "finaliz", "place", "prima", "doua", "treia", "ultim"]):
            return "PURCHASE_INTENT"
        if "restart" in msg_clean:
            return "RESTART"
        return current

    if not client:
        return manual_fallback(user_message, current_state)

    states_desc = """
    - GREETING: Utilizatorul salută sau începe conversația.
    - SEARCH_PRODUCTS: Utilizatorul caută flori, cere recomandări sau descrie o ocazie.
    - PURCHASE_INTENT: Utilizatorul confirmă că vrea să cumpere, alege un produs din listă (ex: "o vreau pe prima", "imi place varianta 2") sau întreabă cum comandă.
    - RESTART: Utilizatorul vrea să o ia de la capăt.
    - GENERAL_INQUIRY: Utilizatorul pune întrebări generale.
    """

    system_prompt = (
        f"Ești un router de conversație pentru o florărie. "
        f"Starea actuală este: {current_state}. "
        f"Analizează mesajul utilizatorului și alege următoarea stare din această listă:\n{states_desc}\n"
        "Regulă importantă: Dacă utilizatorul se referă la un produs afișat anterior (ex: 'prima varianta', 'o aleg pe a doua', 'imi place prima'), clasifică drept PURCHASE_INTENT.\n"
        "Răspunde DOAR cu numele stării (ex: SEARCH_PRODUCTS)."
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=f"{system_prompt}\n\nMesaj utilizator: {user_message}"
        )
        intent = response.text.strip().upper()
        valid_states = ["GREETING", "SEARCH_PRODUCTS", "PURCHASE_INTENT", "RESTART", "GENERAL_INQUIRY"]
        for s in valid_states:
            if s in intent: return s
        return current_state
    except Exception as e:
        error_str = str(e)
        if any(code in error_str for code in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"]):
            logger.warning(f"Gemini is overloaded or quota exceeded during classification ({error_str}), using manual fallback.")
            return manual_fallback(user_message, current_state)
        logger.error(f"⚠️ Eroare neașteptată Gemini Intent Router: {e}")
        return current_state

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
        "2. 'semantic_query': keywords pentru căutare (string).\n"
        "3. 'mentioned_now': o listă cu cheile filtrelor menționate EXPLICIT în mesajul actual (ex: ['price_max', 'colors']).\n"
        "4. 'insisted': o listă cu cheile filtrelor asupra cărora utilizatorul INSISTĂ sau se plânge (ex: 'prea scump' -> 'price_max').\n"
        f"{schema_info}\n"
        "Reguli:\n"
        "- 'filters' trebuie să conțină valorile cumulate (din mesajul actual + context dacă sunt relevante).\n"
        "- 'mentioned_now' conține doar ce a zis acum.\n"
        "- 'insisted' conține filtrele pe care le repetă pentru că nu a fost mulțumit anterior.\n"
        "- Răspunde DOAR cu obiectul JSON."
    )
    
    if client:
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=f"{system_prompt}\n{history_context}\nMesaj actual client: {user_message}"
            )
            llm_output = response.text
            llm_output = re.sub(r"```json|```", "", llm_output).strip()
            return json.loads(llm_output)
        except Exception as e:
            error_str = str(e)
            if any(code in error_str for code in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"]):
                raise GeminiOverloadedError(f"Gemini is busy or quota exceeded: {error_str}")
            logger.error(f"⚠️ Eroare neașteptată Gemini Filter Extraction: {e}")
            return {"filters": {}, "semantic_query": user_message}
    return {"filters": {}, "semantic_query": user_message}


# ==========================================
# 3. HYBRID RESPONSE (Instant Template)
# ==========================================
def generate_hybrid_response(rag_results):
    intros = [
        "Bună! Am găsit câteva opțiuni superbe care se încadrează perfect în dorințele tale:",
        "Vă mulțumim pentru solicitare! Iată selecția noastră de flori elegante:",
        "O alegere minunată! Iată cele mai proaspete flori din stoc pentru tine:"
    ]
    
    selected_intro = random.choice(intros)
    items_text = ""
    
    for i, res in enumerate(rag_results, 1):
        title = res.get('title', 'Produs necunoscut')
        price = res.get('price', 'Preț indisponibil')
        
        # Extragem descrierea curată (fără etichete Produs/Categorie etc.)
        context_str = res.get('context', '')
        description = ""
        if "Descriere: " in context_str:
            description = context_str.split("Descriere: ")[-1].strip()
        else:
            # Fallback dacă nu avem tagul Descriere
            description = context_str.split("\n")[-1].strip()
            
        short_desc = description[:150] + "..." if len(description) > 150 else description
        
        # Formatare nouă: Titlu + Preț pe primul rând, Descriere pe rândul 2, /n între produse
        items_text += f"{i}. *{title}* - Preț: {price} lei\n_{short_desc}_\n\n"
    
    return f"{selected_intro}\n\n{items_text}"

