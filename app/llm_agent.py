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
        if any(word in msg_clean for word in ["caut", "vreau", "buchet", "flori", "recomand", "lalele", "trandafiri", "cadou"]):
            return "SEARCH_PRODUCTS"
        if any(word in msg_clean for word in ["detalii", "info", "mai mult", "descriere"]):
            return "PRODUCT_DETAILS"
        if any(word in msg_clean for word in ["cumpăr", "comand", "iau", "da", "confirm", "finaliz", "place", "prima", "doua", "treia", "ultim"]):
            return "PURCHASE_INTENT"
        if "restart" in msg_clean:
            return "RESTART"
        return current

    if not client:
        return manual_fallback(user_message, current_state)

    states_desc = """
    - GREETING: Utilizatorul salută sau începe conversația.
    - SEARCH_PRODUCTS: Utilizatorul caută flori, CADOURI, ciocolată, vin sau cere recomandări.
    - PRODUCT_DETAILS: Utilizatorul cere mai multe informații/detalii despre un produs afișat (ex: 'vreau mai multe detalii despre primul', 'spune-mi mai multe despre buchetul de trandafiri').
    - PURCHASE_INTENT: Utilizatorul confirmă că vrea să cumpere, alege un produs din listă.
    - RESTART: Utilizatorul vrea să o ia de la capăt.
    - GENERAL_INQUIRY: Utilizatorul pune întrebări despre bot, program sau alte discuții libere.
    """

    system_prompt = (
        f"Ești un router de conversație pentru o florărie online. "
        f"Starea actuală este: {current_state}. "
        f"Analizează mesajul utilizatorului și alege următoarea stare din listă:\n{states_desc}\n"
        "Regulă importantă 1: Orice cerere de produs (cadou, pachet, ceva de băut) este SEARCH_PRODUCTS.\n"
        "Regulă importantă 2: Dacă cere detalii suplimentare, specificații sau 'mai multe info' despre ceva afișat deja, este PRODUCT_DETAILS.\n"
        "Regulă importantă 3: Dacă se referă la un produs pentru a-l cumpăra, este PURCHASE_INTENT.\n"
        "Răspunde DOAR cu numele stării."
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=f"{system_prompt}\n\nMesaj utilizator: {user_message}",
            config={"temperature": 0.0}
        )
        intent = response.text.strip().upper()
        valid_states = ["GREETING", "SEARCH_PRODUCTS", "PRODUCT_DETAILS", "PURCHASE_INTENT", "RESTART", "GENERAL_INQUIRY"]
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
                contents=f"{system_prompt}\n{history_context}\nMesaj actual client: {user_message}",
                config={"temperature": 0.0}
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
        "Bună! Am găsit câteva opțiuni superbe care s-ar putea să îți placă:",
        "Vă mulțumim pentru solicitare! Iată selecția noastră de flori elegante:",
        "O alegere minunată! Iată cele mai proaspete opțiuni din stoc:"
    ]
    
    selected_intro = random.choice(intros)
    items_text = ""
    
    for i, res in enumerate(rag_results, 1):
        title = res.get('title', 'Produs necunoscut')
        price = res.get('price', 'Preț indisponibil')
        
        # Extragem descrierea (încercăm multiple tag-uri, altfel curățăm contextul)
        context_str = res.get('context', '')
        description = ""
        
        # Căutăm tag-ul de început al descrierii reale
        for tag in ["Descriere:", "Body (HTML):", "Description:"]:
            if tag in context_str:
                description = context_str.split(tag)[-1].strip()
                break
        
        if not description:
            # Fallback: eliminăm toate liniile care conțin "Label: "
            lines = context_str.split("\n")
            tags_to_skip = ["Produs:", "Type:", "Flowers Type:", "Colors:", "Categorie:", "Flori:", "Culori:", "Dimensiune:", "Eveniment:", "Sezon:", "Body (HTML):"]
            clean_lines = []
            for line in lines:
                # Sărim liniile care sunt doar etichete
                if not any(t in line for t in tags_to_skip) and ":" not in line[:15]:
                    clean_lines.append(line)
            description = " ".join(clean_lines).strip()
            
        if not description:
            description = "Vezi detaliile complete în catalogul nostru online."
            
        # Limităm la 350 caractere pentru o descriere decentă
        limit = 350
        short_desc = description[:limit] + "..." if len(description) > limit else description
        
        # Formatare finală: Titlu + Preț, apoi Descrierea pe rând nou
        items_text += f"{i}. *{title}* - Preț: {price} lei\n_{short_desc}_\n\n"
    
    return f"{selected_intro}\n\n{items_text}"

# ==========================================
# 4. DETAILED PRODUCT INFO (Readable Metadata)
# ==========================================
def generate_detailed_product_info(product: Dict[str, Any]) -> str:
    """Formatează toate metadatele unui produs într-un mod citibil și frumos."""
    title = product.get('title', 'Produs')
    price = product.get('price', 'N/A')
    
    # Mapare etichete tehnice -> Română prietenoasă
    label_map = {
        "category": "Categorie",
        "dimensiune": "Dimensiune",
        "event_type": "Ideal pentru",
        "colors": "Culori disponibile",
        "flowers_type": "Tipuri de flori",
        "sezon": "Sezon"
    }
    
    header = f"🔍 *Detalii complete pentru: {title}*\n💰 *Preț:* {price} lei\n\n"
    
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
    
    # Adăugăm și descrierea completă
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
