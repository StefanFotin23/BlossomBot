import os
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

class Config:
    # Slack
    SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
    SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")

    # Gemini
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-3.1-flash-lite")

    # Qdrant
    QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
    QDRANT_URL = os.environ.get("QDRANT_URL", f"http://{QDRANT_HOST}:{QDRANT_PORT}")
    QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

    # Ollama (Fallback/Local LLM)
    OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "localhost")
    OLLAMA_PORT = os.environ.get("OLLAMA_PORT", "11434")
    OLLAMA_URL = os.environ.get("OLLAMA_URL", f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate")
    MODEL_NAME = os.environ.get("MODEL_NAME", "qwen2.5:0.5b")

    # RabbitMQ
    RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "localhost")
    # K8s sometimes sets RABBITMQ_PORT to a URL like 'tcp://...'. We want the numeric port.
    _rmq_port = os.environ.get("RABBITMQ_PORT_NUMBER") or os.environ.get("RABBITMQ_PORT", "5672")
    if "://" in _rmq_port:
        RABBITMQ_PORT = 5672 # Fallback to default if it's a K8s service URL
    else:
        RABBITMQ_PORT = int(_rmq_port)
    PURCHASE_QUEUE = os.environ.get("PURCHASE_QUEUE", "flower_purchases")

    # Redis
    REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
    _redis_port = os.environ.get("REDIS_PORT_NUMBER") or os.environ.get("REDIS_PORT", "6379")
    if "://" in _redis_port:
        REDIS_PORT = 6379
    else:
        REDIS_PORT = int(_redis_port)
    REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
    REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", None)

    # App Settings
    DEBUG = os.environ.get("DEBUG", "False").lower() in ("true", "1", "t")
    TEMPLATE_DIR = os.environ.get("TEMPLATE_DIR", "app/template")
    DATASET_PATH = os.environ.get("DATASET_PATH", "app/build/dataset/products.csv")
    
    # AI/ML Settings
    TRANSFORMERS_OFFLINE = os.environ.get("TRANSFORMERS_OFFLINE", "1")
    HF_HUB_OFFLINE = os.environ.get("HF_HUB_OFFLINE", "1")

    @classmethod
    def validate(cls):
        """Simple validation to ensure critical variables are set."""
        missing = []
        if not cls.SLACK_BOT_TOKEN: missing.append("SLACK_BOT_TOKEN")
        if not cls.SLACK_SIGNING_SECRET: missing.append("SLACK_SIGNING_SECRET")
        if not cls.GEMINI_API_KEY:
            print("⚠️ Warning: GEMINI_API_KEY is not set. Application will fallback to Ollama if available.")
        
        if missing:
            print(f"❌ Missing critical environment variables: {', '.join(missing)}")
            return False
        return True
