import os
from fastapi import FastAPI, Request
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
from slack_bolt.async_app import AsyncApp
from dotenv import load_dotenv

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

@app.post("/slack/events")
async def slack_events(req: Request):
    return await handler.handle(req)

@app.get("/")
async def root():
    return {"message": "BlossomBot is running!"}
