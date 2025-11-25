import re
import json
from slack_bolt.async_app import AsyncApp
from app.redis_client import get_redis_client

USER_STATE_KEY = "user_state:{user_id}"
USER_CONVERSATION_KEY = "user_conversation:{user_id}"
RECOMMENDATION_MESSAGE_IDS_KEY = "recommendation_messages:{user_id}"

async def get_user_state(user_id):
    redis = get_redis_client()
    state_json = await redis.get(USER_STATE_KEY.format(user_id=user_id))
    if state_json:
        return json.loads(state_json)
    return {"state": "GREETING"}

async def set_user_state(user_id, state):
    redis = get_redis_client()
    await redis.set(USER_STATE_KEY.format(user_id=user_id), json.dumps(state))

async def add_to_conversation(user_id, message_type, text):
    redis = get_redis_client()
    message = json.dumps({"type": message_type, "text": text})
    await redis.rpush(USER_CONVERSATION_KEY.format(user_id=user_id), message)

async def send_recommendations(user_id, say):
    # Mock recommendations
    recommendations = [
        {"name": "Sunny Day Bouquet", "description": "A beautiful bouquet of yellow and white flowers.", "price": "$25"},
        {"name": "Ocean Breeze Arrangement", "description": "A mix of blue and white flowers, reminiscent of a day at the beach.", "price": "$35"},
        {"name": "Romantic Red Roses", "description": "A classic bouquet of a dozen red roses.", "price": "$45"}
    ]

    message_ids = []
    for i, rec in enumerate(recommendations):
        response = await say(f"Recommendation {i+1}: *{rec['name']}*\n{rec['description']}\nPrice: {rec['price']}")
        if response and response.get('ts'):
            message_ids.append(response.get('ts'))

    if message_ids:
        redis = get_redis_client()
        await redis.set(RECOMMENDATION_MESSAGE_IDS_KEY.format(user_id=user_id), json.dumps(message_ids))


def register_events(app: AsyncApp):
    @app.event("message")
    async def handle_message_events(body, say):
        event = body["event"]
        
        if event.get("channel_type") != "im":
            return

        user_id = event["user"]
        text = event["text"]

        if re.search(r'\brestart\b', text, re.IGNORECASE):
            await set_user_state(user_id, {"state": "GREETING"})
            await say("Let's start over. How can I help you today?")
            return

        user_state = await get_user_state(user_id)
        state = user_state.get("state")
        
        await add_to_conversation(user_id, "user", text)

        if state == "GREETING":
            await say(f"Hi there <@{user_id}>! I'm BlossomBot. To help me find the perfect flowers for you, please describe the occasion and what you're looking for.")
            await set_user_state(user_id, {"state": "GATHERING_DESCRIPTION"})
            await add_to_conversation(user_id, "bot", "Hi there! I'm BlossomBot...")

        elif state == "GATHERING_DESCRIPTION":
            if len(text.split()) < 10:
                response_text = "That's a start! Could you tell me a little more? For example, what's the occasion, who are the flowers for, and are there any favorite colors or types of flowers?"
                await say(response_text)
                await add_to_conversation(user_id, "bot", response_text)
            else:
                await set_user_state(user_id, {"state": "CONFIRMING_DESCRIPTION", "description": text})
                response_text = f"Thanks! Just to confirm, you're looking for: \"{text}\". Is that correct? (yes/no)"
                await say(response_text)
                await add_to_conversation(user_id, "bot", response_text)

        elif state == "CONFIRMING_DESCRIPTION":
            if re.search(r'\byes\b', text, re.IGNORECASE):
                await say("Great! I'll find some recommendations for you shortly.")
                await set_user_state(user_id, {"state": "RECOMMENDING"})
                await send_recommendations(user_id, say)

            elif re.search(r'\bno\b', text, re.IGNORECASE):
                response_text = "My apologies. Let's try again. Please describe what you are looking for."
                await say(response_text)
                await add_to_conversation(user_id, "bot", response_text)
                await set_user_state(user_id, {"state": "GATHERING_DESCRIPTION"})
            else:
                response_text = "Sorry, I didn't understand. Please respond with 'yes' or 'no'."
                await say(response_text)
                await add_to_conversation(user_id, "bot", response_text)

        elif state == "PURCHASE_CONFIRMATION":
            if re.search(r'\byes\b', text, re.IGNORECASE):
                response_text = "Excellent! Your order has been placed. Thank you for shopping with BlossomBot!"
                await say(response_text)
                await add_to_conversation(user_id, "bot", response_text)
                await set_user_state(user_id, {"state": "CLOSED"})
            elif re.search(r'\bno\b', text, re.IGNORECASE):
                response_text = "No problem. Would you like to see more options?"
                await say(response_text)
                await add_to_conversation(user_id, "bot", response_text)
                await set_user_state(user_id, {"state": "RECOMMENDING"})
            else:
                response_text = "Sorry, I didn't understand. Please respond with 'yes' or 'no'."
                await say(response_text)
                await add_to_conversation(user_id, "bot", response_text)

        else:
            await say("I'm not sure how to help with that right now. You can start over by saying 'restart'.")

    @app.event("reaction_added")
    async def handle_reaction_added_events(event, say, client):
        user_id = event["user"]
        item_user = event["item_user"]
        
        # Check if the reaction is on a message from the bot
        if user_id != item_user:
            return

        reaction = event["reaction"]
        
        # For simplicity, let's assume any positive reaction is an intent to purchase
        if reaction in ["thumbsup", "heart"]:
            # Check if the reacted message is one of the recommendations
            redis = get_redis_client()
            message_ids_json = await redis.get(RECOMMENDATION_MESSAGE_IDS_KEY.format(user_id=user_id))
            if message_ids_json:
                message_ids = json.loads(message_ids_json)
                if event["item"]["ts"] in message_ids:
                    # Fetch the message content
                    try:
                        result = await client.conversations_history(
                            channel=event["item"]["channel"],
                            latest=event["item"]["ts"],
                            inclusive=True,
                            limit=1
                        )
                        message_text = result["messages"][0]["text"]
                        
                        await set_user_state(user_id, {"state": "PURCHASE_CONFIRMATION", "selected_item": message_text})
                        response_text = f"It looks like you liked this option:\n>{message_text}\n\nWould you like to purchase it? (yes/no)"
                        await say(response_text)
                        await add_to_conversation(user_id, "bot", response_text)

                    except Exception as e:
                        print(f"Error fetching message: {e}")


    @app.event("app_mention")
    async def handle_app_mention_events(body, say):
        user = body['event']['user']
        await say(f"Hi there <@{user}>! Please send me a direct message to get started.")
