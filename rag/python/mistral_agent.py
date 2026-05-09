import os
import requests
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
    """Preia lista de produse din Qdrant și formulează un răspuns elegant pentru utilizator."""
    final_url = ollama_url if ollama_url is not None else DEFAULT_OLLAMA_URL
    final_model = model_name if model_name is not None else DEFAULT_MODEL_NAME
    
    system_prompt = (
        "Ești un asistent de vânzări politicos, empatic și profesionist pentru o florărie online premium. "
        "Sarcina ta este să recomanzi clientului florile potrivite BAZÂNDU-TE STRICT pe 'PRODUSE RECOMANDATE' de mai jos. "
        "Formatează răspunsul frumos, folosește diacritice, și fii persuasiv. "
        "Nu inventa produse, link-uri sau prețuri care nu apar în textul furnizat. "
        "Dacă nu ai primit produse potrivite în context, cere-ți scuze politicos și roagă clientul să modifice preferințele."
    )
    
    full_prompt = f"{system_prompt}\n\n[PRODUSE RECOMANDATE DIN BAZA DE DATE]\n{rag_context}\n\nMesaj Client: {user_message}\nRăspunsul Tău:"
    
    payload = {
        "model": final_model,
        "prompt": full_prompt,
        "stream": False,
        # Aici NU mai avem format='json', vrem text normal
        "options": {
            "temperature": 0.4 # Oferim puțină creativitate pentru un discurs mai prietenos și uman
        }
    }
    
    try:
        response = requests.post(final_url, json=payload)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"Îmi cer scuze, întâmpinăm o problemă tehnică de comunicare. Vă rugăm să reîncercați. (Detalii eroare: {e})"