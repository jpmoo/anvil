"""Configuration and directory management"""

from pathlib import Path
import os

BASE_DIR = Path(__file__).parent.parent

# Legacy paths (for backward compatibility during migration)
TRAINING_DIR = BASE_DIR / "training"
CONTEXT_DIR = TRAINING_DIR / "context"
LOGS_DIR = TRAINING_DIR / "logs"
DATA_DIR = TRAINING_DIR / "data"

MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

def get_model_training_dir(model_name: str) -> Path:
    """Get the training directory for a specific model"""
    return MODELS_DIR / model_name / "training"

def get_model_context_dir(model_name: str) -> Path:
    """Get the context directory for a specific model"""
    return get_model_training_dir(model_name) / "context"

def get_model_logs_dir(model_name: str) -> Path:
    """Get the logs directory for a specific model"""
    return get_model_training_dir(model_name) / "logs"

def get_model_data_dir(model_name: str) -> Path:
    """Get the data directory for a specific model"""
    return get_model_training_dir(model_name) / "data"

def get_model_queue_dir(model_name: str) -> Path:
    """Get the queue directory for JSON files to be uploaded to Axolotl"""
    return get_model_training_dir(model_name) / "queue"

def get_model_behavioral_path(model_name: str) -> Path:
    """Get the behavioral.json path for a specific model"""
    return get_model_training_dir(model_name) / "behavioral.json"

def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        MODELS_DIR,
        ASSETS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def ensure_model_directories(model_name: str):
    """Create training directories for a specific model"""
    directories = [
        get_model_training_dir(model_name),
        get_model_context_dir(model_name),
        get_model_logs_dir(model_name),
        get_model_data_dir(model_name)
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# App configuration file path
APP_CONFIG_FILE = BASE_DIR / "app_config.json"

def get_app_config() -> dict:
    """Get app configuration (API keys, etc.)"""
    if APP_CONFIG_FILE.exists():
        try:
            import json
            with open(APP_CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_app_config(config: dict):
    """Save app configuration"""
    import json
    APP_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(APP_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def get_vast_api_key() -> str:
    """Get Vast.ai API key from config or environment"""
    import os
    # First check environment variable
    env_key = os.getenv("VAST_API_KEY", "")
    if env_key:
        return env_key
    
    # Then check config file
    config = get_app_config()
    return config.get("vast_api_key", "")

def save_vast_api_key(api_key: str):
    """Save Vast.ai API key to config file"""
    config = get_app_config()
    config["vast_api_key"] = api_key
    save_app_config(config)

def delete_vast_api_key():
    """Delete Vast.ai API key from config file"""
    config = get_app_config()
    if "vast_api_key" in config:
        del config["vast_api_key"]
        save_app_config(config)

def get_hf_token() -> str:
    """Get Hugging Face token from config or environment"""
    import os
    # First check environment variable
    env_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN", "")
    if env_token:
        return env_token
    
    # Then check config file
    config = get_app_config()
    return config.get("hf_token", "")

def save_hf_token(token: str):
    """Save Hugging Face token to config file"""
    config = get_app_config()
    config["hf_token"] = token
    save_app_config(config)

def delete_hf_token():
    """Delete Hugging Face token from config file"""
    config = get_app_config()
    if "hf_token" in config:
        del config["hf_token"]
        save_app_config(config)

def get_model_preferences_file(model_name: str) -> Path:
    """Get the preferences file path for a specific model"""
    return MODELS_DIR / model_name / "preferences.json"

def get_model_preferences(model_name: str) -> dict:
    """Get user preferences for a specific model (prepend text, summary setting, etc.)"""
    prefs_file = get_model_preferences_file(model_name)
    if prefs_file.exists():
        try:
            import json
            with open(prefs_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_model_preferences(model_name: str, preferences: dict):
    """Save user preferences for a specific model"""
    import json
    prefs_file = get_model_preferences_file(model_name)
    prefs_file.parent.mkdir(parents=True, exist_ok=True)
    with open(prefs_file, 'w') as f:
        json.dump(preferences, f, indent=2)


