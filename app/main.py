import httpx
import pika
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request, HTTPException
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
from slack_bolt.async_app import AsyncApp
from qdrant_client import QdrantClient

from app import bot
from app.config import Config
from app.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Validate configuration at startup
Config.validate()

# --- Slack App Configuration ---
slack_app = AsyncApp(
    token=Config.SLACK_BOT_TOKEN,
    signing_secret=Config.SLACK_SIGNING_SECRET
)

# --- FastAPI App Configuration ---
app = FastAPI(title="BlossomBot Gateway")
handler = AsyncSlackRequestHandler(slack_app)

# Thread pool for blocking operations (like pika)
executor = ThreadPoolExecutor(max_workers=5)

# --- RabbitMQ Client ---
def _blocking_publish_purchase(purchase_data):
    """Surgical RabbitMQ publish (blocking)."""
    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=Config.RABBITMQ_HOST, port=Config.RABBITMQ_PORT)
        )
        channel = connection.channel()
        channel.queue_declare(queue=Config.PURCHASE_QUEUE, durable=True)
        
        channel.basic_publish(
            exchange='',
            routing_key=Config.PURCHASE_QUEUE,
            body=json.dumps(purchase_data),
            properties=pika.BasicProperties(
                delivery_mode=2,  # make message persistent
            )
        )
        connection.close()
        logger.info(f"Purchase event published to RabbitMQ for user: {purchase_data['user_id']}")
        return True
    except Exception as e:
        logger.error(f"Failed to publish purchase event to RabbitMQ: {e}")
        return False

async def publish_purchase(purchase_data):
    """Async wrapper for RabbitMQ publishing."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, _blocking_publish_purchase, purchase_data)

# --- Register Slack Events via Gateway ---
@slack_app.event("message")
async def handle_message_events(body, say):
    logger.debug(f"Received message event: {body}")
    await bot.handle_message(body, say, publish_purchase_callback=publish_purchase)

@slack_app.event("app_mention")
async def handle_app_mention_events(body, say):
    user = body['event']['user']
    logger.info(f"App mentioned by user: {user}")
    await say(f"Salutare <@{user}>! Te rog trimite-mi un mesaj direct (DM) ca să te pot ajuta cu alegerea florilor.")

# --- Dependency Clients ---
qdrant_client = QdrantClient(host=Config.QDRANT_HOST, port=Config.QDRANT_PORT)
http_client = httpx.AsyncClient()

# --- Health Check Functions ---
def _blocking_check_rabbitmq_health():
    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=Config.RABBITMQ_HOST, port=Config.RABBITMQ_PORT, heartbeat=10)
        )
        connection.close()
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"RabbitMQ health check failed: {e}")
        return {"status": "error", "message": str(e)}

async def check_rabbitmq_health():
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, _blocking_check_rabbitmq_health)

async def check_qdrant_health():
    try:
        qdrant_client.get_collections()
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")
        return {"status": "error", "message": str(e)}

async def check_ollama_health():
    try:
        ollama_url = f"http://{Config.OLLAMA_HOST}:{Config.OLLAMA_PORT}/api/tags"
        response = await http_client.get(ollama_url, timeout=5)
        response.raise_for_status()
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Ollama health check failed: {e}")
        return {"status": "error", "message": str(e)}

# --- API Endpoints ---
@app.post("/slack/events")
async def slack_events(req: Request):
    return await handler.handle(req)

@app.get("/")
async def root():
    return {"message": "BlossomBot Gateway is active!"}

@app.get("/healthz")
async def healthz():
    results = await asyncio.gather(
        check_rabbitmq_health(),
        check_qdrant_health(),
        check_ollama_health(),
        return_exceptions=True
    )
    
    details = {
        "rabbitmq": results[0],
        "qdrant": results[1],
        "ollama": results[2]
    }
    
    overall_status = "ok"
    for s in details.values():
        if isinstance(s, dict) and s.get("status") == "error":
            overall_status = "error"
            break
            
    if overall_status == "error":
        logger.warning(f"Health check failed with details: {details}")
        raise HTTPException(status_code=500, detail=details)
    return {"status": "ok", "details": details}

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down...")
    await http_client.aclose()
    executor.shutdown(wait=True)
