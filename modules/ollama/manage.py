import requests
import json
import os
from modules.database import get_setting, save_setting

DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Standard Models Catalog
MODELS_CATALOG = {
    "llama3": {"size": "4.7 GB", "description": "Meta Llama 3 8B model - high capability generalist"},
    "gemma3": {"size": "5.5 GB", "description": "Google Gemma 2 9B model - premium reasoning accuracy"},
    "qwen3": {"size": "4.5 GB", "description": "Alibaba Qwen 2.5 7B model - strong multilingual logic"},
    "phi4": {"size": "8.2 GB", "description": "Microsoft Phi-4 14B model - state-of-the-art compact logic"}
}

def get_ollama_endpoint():
    return get_setting("ollama_endpoint", DEFAULT_OLLAMA_URL).rstrip("/")

def list_installed_models():
    """Queries local Ollama for installed models. Falls back to simulated catalog."""
    if not (os.environ.get("STREAMLIT_SHARING_MODE") or os.environ.get("SPACE_ID")):
        url = f"{get_ollama_endpoint()}/api/tags"
        try:
            res = requests.get(url, timeout=1.0)
            if res.status_code == 200:
                data = res.json()
                models = []
                for m in data.get("models", []):
                    size_gb = round(m.get("size", 0) / (1024**3), 2)
                    models.append({
                        "name": m.get("name"),
                        "size": f"{size_gb} GB",
                        "status": "Installed"
                    })
                # Add other catalog items as available
                installed_names = [x["name"].split(":")[0] for x in models]
                for c_name, c_info in MODELS_CATALOG.items():
                    if c_name not in installed_names:
                        models.append({
                            "name": c_name,
                            "size": c_info["size"],
                            "status": "Available"
                        })
                return models
        except Exception:
            pass
        
    # Return simulated mock models catalog if Ollama is offline or not running
    models = []
    downloaded = json.loads(get_setting("simulated_downloaded_models", '["llama3"]'))
    for c_name, c_info in MODELS_CATALOG.items():
        models.append({
            "name": c_name,
            "size": c_info["size"],
            "status": "Installed" if c_name in downloaded else "Available"
        })
    return models

def download_model(model_name):
    """Triggers pulling/downloading the model from Ollama library."""
    url = f"{get_ollama_endpoint()}/api/pull"
    try:
        res = requests.post(url, json={"name": model_name, "stream": False}, timeout=5.0)
        if res.status_code == 200:
            return True, f"Successfully downloaded model: {model_name}"
    except Exception:
        pass
        
    # Simulated Mock Download
    downloaded = json.loads(get_setting("simulated_downloaded_models", '["llama3"]'))
    if model_name not in downloaded:
        downloaded.append(model_name)
        save_setting("simulated_downloaded_models", json.dumps(downloaded))
    return True, f"Simulated download of model '{model_name}' completed."

def delete_model(model_name):
    """Triggers deleting/removing model from Ollama."""
    url = f"{get_ollama_endpoint()}/api/delete"
    try:
        res = requests.delete(url, json={"name": model_name}, timeout=2.0)
        if res.status_code == 200:
            return True, f"Successfully deleted model: {model_name}"
    except Exception:
        pass
        
    # Simulated Mock Delete
    downloaded = json.loads(get_setting("simulated_downloaded_models", '["llama3"]'))
    if model_name in downloaded:
        downloaded.remove(model_name)
        save_setting("simulated_downloaded_models", json.dumps(downloaded))
    return True, f"Simulated deletion of model '{model_name}' completed."
