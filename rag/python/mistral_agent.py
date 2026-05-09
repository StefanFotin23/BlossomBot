import os
import requests
import random
import json
import re
from typing import Dict, Any

# ==========================================
# CONFIGURĂRI GLOBALE / MACROS (Fallback)
# ==========================================
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
DEFAULT_MODEL_NAME = os.getenv("MISTRAL_MODEL", "romistral")
DEFAULT_TEMPLATE_DIR = os.getenv("TEMPLATE_DIR", "./")

# ==========================================
# 1. MISTRAL ROUTER (Text -> JSON pentru RAG)
# ==========================================
def _build_router_prompt(template_dir: str = None) -> str:
    """Metodă internă care asamblează promptul dinamic pe baza fișierelor."""
    t_dir = template_dir if template_dir is not None else DEFAULT_TEMPLATE_DIR
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

def extract_filters(user_message: str, ollama_url: str = None, model_name: str = None, template_dir: str = None) -> Dict[str, Any]:
    """Interoghează LLM-ul cu temperatura 0 și forțează output JSON pentru căutarea Qdrant."""
    final_url = ollama_url if ollama_url is not None else DEFAULT_OLLAMA_URL
    final_model = model_name if model_name is not None else DEFAULT_MODEL_NAME
    
    system_prompt = _build_router_prompt(template_dir)
    full_prompt = f"{system_prompt}\n\nMesaj client: {user_message}\nOutput JSON:"
    
    payload = {
        "model": final_model,
        "prompt": full_prompt,
        "stream": False,
        "format": "json", # Obligăm modelul să emită doar JSON
        "options": {
            "temperature": 0.0 # Vrem precizie absolută
        }
    }
    
    try:
        response = requests.post(final_url, json=payload)
        response.raise_for_status()
        
        llm_output = response.json().get("response", "")
        llm_output = re.sub(r"```json|```", "", llm_output).strip()
        parsed_json = json.loads(llm_output)
        
        # Eliminăm cheile null din filtre
        if "filters" in parsed_json and isinstance(parsed_json["filters"], dict):
            parsed_json["filters"] = {k: v for k, v in parsed_json["filters"].items() if v is not None}
            
        return parsed_json

    except Exception as e:
        print(f"⚠️ Eroare la Mistral Router: {e}")
        # Returnăm un fallback util care să nu blocheze aplicația
        return {"filters": {}, "semantic_query": user_message}


# ==========================================
# 2. MISTRAL SALES AGENT (Context -> Răspuns Natural)
# ==========================================
def generate_conversational_response(user_message: str, rag_context: str, ollama_url: str = None, model_name: str = None) -> str:
    final_url = ollama_url if ollama_url is not None else DEFAULT_OLLAMA_URL
    final_model = model_name if model_name is not None else DEFAULT_MODEL_NAME
    
    # Prompt scurt care forțează un format fix
    system_prompt = (
        "Ești un asistent concis pentru florăria City Flowers. "
        "Sarcina ta: Salută clientul, confirmă că ai găsit opțiuni pentru cererea lui și prezintă-le sub formă de listă numerotată. "
        "Reguli: Maxim 3 propoziții de introducere. Folosește STRICT datele din context. "
        "Format: [Introducere scurtă] \n\nIată recomandările noastre:\n[Lista numerotată cu Nume, Preț și Link]"
    )
    
    full_prompt = f"{system_prompt}\n\nPRODUSE:\n{rag_context}\n\nCLIENT: {user_message}\nASISTENT:"

    payload = {
        "model": final_model,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,  # Viteză maximă, zero creativitate inutilă
            "num_predict": 250,  # Limităm lungimea răspunsului ca să nu o ia pe arătură
            "top_k": 20,
            "top_p": 0.9
        }
    }
    
    try:
        response = requests.post(final_url, json=payload)
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