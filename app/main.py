import os
import httpx
import pika
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request, HTTPException
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
from slack_bolt.async_app import AsyncApp
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from app.bot import register_events

# Load environment variables from .env file
load_dotenv()

# Create a Slack App
slack_app = AsyncApp(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)

# Register event handlers
register_events(slack_app)

# Create a FastAPI app
app = FastAPI()
handler = AsyncSlackRequestHandler(slack_app)

# Thread pool for blocking operations (like pika)
executor = ThreadPoolExecutor(max_workers=1)

# --- Dependency Clients ---
# Qdrant Client
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = os.environ.get("QDRANT_PORT", "6333")
qdrant_client = QdrantClient(host=QDRANT_HOST, port=int(QDRANT_PORT))

# HTTP Client for Ollama
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "localhost")
OLLAMA_PORT = os.environ.get("OLLAMA_PORT", "11434")
http_client = httpx.AsyncClient()


# --- Health Check Functions ---
def _blocking_check_rabbitmq_health():
    RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "localhost")
    RABBITMQ_PORT = os.environ.get("RABBITMQ_PORT", "5672")
    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=RABBITMQ_HOST, port=int(RABBITMQ_PORT), heartbeat=10)
        )
        connection.close()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": f"RabbitMQ connection failed: {e}"}

async def check_rabbitmq_health():
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, _blocking_check_rabbitmq_health)

async def check_qdrant_health():
    try:
        # Attempt to list collections as a health check
        await qdrant_client.get_collections()
        return {"status": "ok"}
    except UnexpectedResponse as e:
        # This might happen if Qdrant is up but has no collections, or other API errors
        return {"status": "warning", "message": f"Qdrant API responded with warning/error: {e}"}
    except Exception as e:
        return {"status": "error", "message": f"Qdrant connection failed: {e}"}

async def check_ollama_health():
    try:
        ollama_url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/tags" # A simple Ollama API endpoint
        response = await http_client.get(ollama_url, timeout=5)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        return {"status": "ok"}
    except httpx.RequestError as e:
        return {"status": "error", "message": f"Ollama request failed: {e}"}
    except httpx.HTTPStatusError as e:
        return {"status": "error", "message": f"Ollama returned HTTP error: {e.response.status_code}"}
    except Exception as e:
        return {"status": "error", "message": f"Ollama check failed: {e}"}


@app.post("/slack/events")
async def slack_events(req: Request):
    return await handler.handle(req)

@app.get("/")
async def root():
    return {"message": "BlossomBot is running!"}

@app.get("/healthz")
async def healthz():
    overall_status = "ok"
    details = {}

    # Run health checks concurrently
    results = await asyncio.gather(
        check_rabbitmq_health(),
        check_qdrant_health(),
        check_ollama_health(),
        return_exceptions=True  # Ensure all tasks complete even if one fails
    )

    details["rabbitmq"] = results[0]
    details["qdrant"] = results[1]
    details["ollama"] = results[2]

    for service, status_info in details.items():
        # Handle exceptions returned by asyncio.gather
        if isinstance(status_info, Exception):
            details[service] = {"status": "error", "message": f"Health check failed unexpectedly: {status_info}"}
            overall_status = "error"
            continue

        if status_info["status"] == "error":
            overall_status = "error"
            break
        if status_info["status"] == "warning" and overall_status == "ok":
            overall_status = "warning"
            
    if overall_status == "error":
        raise HTTPException(status_code=500, detail=details)
    elif overall_status == "warning":
        return {"status": "warning", "details": details}
    else:
        return {"status": "ok", "details": details}

# Close HTTP client on app shutdown
@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()
    executor.shutdown(wait=True) # Shutdown the thread pool executor
