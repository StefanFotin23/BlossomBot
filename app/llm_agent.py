import os
import requests
import random
import json
import re
from google import genai
from typing import Dict, Any

OLLAMA_OPTIMIZED_OPTIONS = {
    "num_ctx": 2048,        # Limitează contextul pentru a nu umple VRAM-ul
    "num_gpu": 99,          # Forțează toate straturile pe GPU
    "temperature": 0.1,     # Determinism pentru extracție și stări
    "num_predict": 150,     # Limitează lungimea output-ului pentru latență mică
    "low_vram": True        # Hint pentru Ollama să fie conservator
}

GEMINI_MODEL_NAME = "gemini-2.5-flash"

def getApiClient(api_key: str = None):
    """Inițializează și returnează clientul GenAI dacă api_key este prezent."""
    if api_key:
        return genai.Client(api_key=api_key)
    return None

# ==========================================
# 1. CONVERSATION ROUTER (State Classification)
# ==========================================
def classify_intent(user_message: str, current_state: str = "GREETING", api_key: str = None) -> str:
    """Identifică intenția utilizatorului folosind Gemini pentru a ghida starea conversației."""
    client = getApiClient(api_key)
    if not client:
        # Fallback simplu dacă nu avem API Key
        if any(word in user_message.lower() for word in ["caut", "vreau", "buchet", "flori", "recomand"]):
            return "SEARCH_PRODUCTS"
        if any(word in user_message.lower() for word in ["cumpăr", "comand", "iau", "da"]):
            return "PURCHASE_INTENT"
        return current_state

    states_desc = """
    - GREETING: Utilizatorul salută sau începe conversația.
    - SEARCH_PRODUCTS: Utilizatorul caută flori, cere recomandări sau descrie o ocazie.
    - PURCHASE_INTENT: Utilizatorul confirmă că vrea să cumpere sau întreabă cum comandă.
    - RESTART: Utilizatorul vrea să o ia de la capăt.
    - GENERAL_INQUIRY: Utilizatorul pune întrebări generale (livrare, program, locație).
    """

    system_prompt = (
        f"Ești un router de conversație pentru o florărie. "
        f"Starea actuală este: {current_state}. "
        f"Analizează mesajul utilizatorului și alege următoarea stare din această listă:\n{states_desc}\n"
        "Răspunde DOAR cu numele stării (ex: SEARCH_PRODUCTS)."
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=f"{system_prompt}\n\nMesaj utilizator: {user_message}"
        )
        intent = response.text.strip().upper()
        # Validare rapidă
        valid_states = ["GREETING", "SEARCH_PRODUCTS", "PURCHASE_INTENT", "RESTART", "GENERAL_INQUIRY"]
        for s in valid_states:
            if s in intent: return s
        return current_state
    except Exception as e:
        print(f"⚠️ Eroare la Gemini Intent Router: {e}")
        return current_state

# ==========================================
# 2. RAG ROUTER (Text -> JSON pentru RAG)
# ==========================================
def _build_router_prompt(template_dir: str = None) -> str:
    """Metodă internă care asamblează promptul dinamic pe baza fișierelor."""
    t_dir = template_dir if template_dir is not None else "app/template"
    schema_path = os.path.join(t_dir, "query_template.json")
    prompt_path = os.path.join(t_dir, "prompt_router.txt")
    
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_json = f.read()
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
            
        return prompt_template.replace("{json_schema}", schema_json)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"❌ Lipsesc fișierele template în {t_dir}! Detalii: {e}")

def extract_filters(user_message: str, ollama_url: str = None, model_name: str = None, template_dir: str = None, api_key: str = None) -> Dict[str, Any]:
    """Interoghează LLM-ul cu temperatura 0 și forțează output JSON pentru căutarea Qdrant."""
    client = getApiClient(api_key)

    system_prompt = _build_router_prompt(template_dir)
    full_prompt = f"{system_prompt}\n\nMesaj client: {user_message}\nOutput JSON:"
    
    llm_output = ""
    if client:
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=full_prompt
            )
            llm_output = response.text
        except Exception as e:
            print(f"⚠️ Eroare la Gemini Router: {e}")
            return {"filters": {}, "semantic_query": user_message}
    else:
        agent_options = OLLAMA_OPTIMIZED_OPTIONS.copy()
        payload = {
            "model": model_name,
            "prompt": full_prompt,
            "stream": False,
            "format": "json", # Obligăm modelul să emită doar JSON
            "options": agent_options
        }
        
        try:
            response = requests.post(ollama_url, json=payload)
            response.raise_for_status()
            llm_output = response.json().get("response", "")
        except Exception as e:
            print(f"⚠️ Eroare la Router Ollama: {e}")
            return {"filters": {}, "semantic_query": user_message}

    try:
        llm_output = re.sub(r"```json|```", "", llm_output).strip()
        parsed_json = json.loads(llm_output)
        
        # Eliminăm cheile null din filtre
        if "filters" in parsed_json and isinstance(parsed_json["filters"], dict):
            parsed_json["filters"] = {k: v for k, v in parsed_json["filters"].items() if v is not None}
            
        return parsed_json
    except Exception as e:
        print(f"⚠️ Eroare la procesare output JSON: {e}")
        return {"filters": {}, "semantic_query": user_message}


# ==========================================
# 2. SALES AGENT (Context -> Răspuns Natural)
# ==========================================
def generate_conversational_response(user_message: str, rag_context: str, ollama_url: str = None, model_name: str = None, api_key: str = None) -> str:
    client = getApiClient(api_key)

    system_prompt = (
        "Ești un asistent amabil și profesionist al florăriei City Flowers. "
        "Sarcina ta este să prezinți clientului opțiunile găsite într-un mod atractiv și util. "
        "Reguli: \n"
        "1. Salută clientul și confirmă că ai găsit câteva variante superbe care se potrivesc cererii lui. \n"
        "2. Prezintă produsele într-o listă numerotată, menționând Numele, Prețul și o scurtă descriere pentru fiecare. \n"
        "3. Folosește STRICT informațiile din contextul PRODUSE furnizat mai jos. \n"
        "4. Încheie oferindu-te să ajuți cu alte detalii sau cu plasarea comenzii."
    )
    
    full_prompt = f"{system_prompt}\n\nPRODUSE:\n{rag_context}\n\nCLIENT: {user_message}\nASISTENT:"

    if client:
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=full_prompt
            )
            return response.text.strip()
        except Exception as e:
            return f"Îmi cer scuze, întâmpinăm o problemă tehnică cu Gemini. (Detalii eroare: {e})"
    else:
        agent_options = OLLAMA_OPTIMIZED_OPTIONS.copy()
        agent_options["temperature"] = 0.2 # Puțin mai relaxat pentru vorbire

        payload = {
            "model": model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": agent_options
        }
        
        try:
            response = requests.post(ollama_url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            return f"Îmi cer scuze, întâmpinăm o problemă tehnică de comunicare. Vă rugăm să reîncercați. (Detalii eroare: {e})"

def generate_hybrid_response(rag_results):
    intros = [
        "Bună! Am găsit câteva opțiuni superbe care se încadrează perfect în dorințele tale:",
        "Vă mulțumim pentru solicitare! Iată selecția noastră de flori elegante pentru această ocazie:",
        "O alegere minunată! Am selectat pentru dumneavoastră cele mai proaspete flori din stoc:",
        "Iată propunerile noastre care respectă bugetul și preferințele menționate:",
        "Ne bucurăm să vă ajutăm! Am pregătit o listă cu cele mai rafinate variante pentru dumneavoastră:",
        "Specialiștii noștri au ales câteva buchete deosebite care s-ar potrivi de minune cu cererea dumneavoastră:",
        "Bună ziua! Iată ce v-am pregătit pentru a transforma această ocazie într-un moment memorabil:",
        "Am analizat cu atenție preferințele dumneavoastră și iată cele mai bune recomandări:",
        "Vă propunem câteva aranjamente florale de excepție care respectă întocmai criteriile selectate:",
        "Căutarea a luat sfârșit! Iată cele mai frumoase flori care se potrivesc bugetului dumneavoastră:",
        "Salutări! Am selectat special pentru dumneavoastră aceste variante pline de prospețime și stil:",
        "Pentru o ocazie atât de specială, am ales câteva buchete care emană eleganță și rafinament:",
        "Iată propunerile noastre florale, create pentru a impresiona prin culoare și parfum:",
        "Ne-am uitat în atelierul nostru și am găsit aceste opțiuni perfecte pentru ceea ce căutați:",
        "Vă invităm să descoperiți selecția noastră de flori, atent alese pentru a se încadra în bugetul stabilit:",
        "Am găsit câteva combinații florale unice care vor aduce cu siguranță un zâmbet persoanei dragi:",
        "Echipa noastră a pregătit următoarele recomandări premium pentru solicitarea dumneavoastră:",
        "Suntem încântați să vă prezentăm aceste variante florale care îmbină calitatea cu eleganța:",
        "Iată selecția noastră curatoriată, gândită special pentru a răspunde nevoilor dumneavoastră:",
        "Vă prezentăm cele mai apreciate buchete din colecția noastră care se potrivesc cerințelor actuale:"
    ]
    
    selected_intro = random.choice(intros)
    items_text = ""
    
    for i, res in enumerate(rag_results, 1):
        # 1. Extragem datele de bază
        title = res.get('title', 'Produs necunoscut')
        price = res.get('price', 'Preț indisponibil')
        imageUrl = res.get('Image Src', '')
        
        # 3. Extragem o scurtă descriere din string-ul 'context'
        # Ne folosim de faptul că descrierea lungă începe după "Body (HTML):"
        context_str = res.get('context', '')
        if "Body (HTML):" in context_str:
            # Luăm doar textul de după "Body (HTML):" și curățăm spațiile
            full_desc = context_str.split("Body (HTML):")[1].strip()
            # Tăiem la primele 150 de caractere ca să nu inundăm clientul cu text
            short_desc = full_desc[:150] + "..." if len(full_desc) > 150 else full_desc
        else:
            short_desc = "Detalii suplimentare direct pe pagina produsului."
        
        # 4. Asamblăm textul produsului
        items_text += f"{i}. **{title}** - Preț: {price} lei\n   Detalii: {short_desc}\n   Image: {imageUrl}\n\n"
    
    footer = "Doriți să adăugăm și o felicitare personalizată la acest buchet?"
    
    return f"{selected_intro}\n\n{items_text}{footer}"
