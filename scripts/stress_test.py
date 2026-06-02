import asyncio
import httpx
import uuid
import random
import time

# Configuration
BASE_URL = "http://localhost:45080/slack/events" # Change this if your ingress port is different
NUM_USERS = 5

SCENARIOS = [
    ["Salut", "Vreau niste trandafiri rosii", "prima varianta", "da", "gata"],
    ["Buna!", "Caut un cadou sub 150 lei", "o vreau pe a doua", "da", "finalizeaza comanda"],
    ["Hei", "Ce produse aveti?", "mai multe detalii despre primul", "il vreau pe acesta", "gata"],
    ["Salutare", "Vreau lalele mov", "prima", "da", "gata"],
    ["Buna", "Vreau martisoare galbene", "a doua", "da", "gata"]
]

async def simulate_user(user_id, scenario):
    print(f"🚀 [User {user_id}] Starting scenario...")
    async with httpx.AsyncClient() as client:
        for message in scenario:
            payload = {
                "token": "verification_token",
                "team_id": "T09US2AV9UJ",
                "api_app_id": "A09V52KNSE5",
                "event": {
                    "type": "message",
                    "user": user_id,
                    "text": message,
                    "ts": str(time.time()),
                    "channel": f"C_{user_id}",
                    "event_ts": str(time.time()),
                    "channel_type": "im"
                },
                "type": "event_callback",
                "event_id": str(uuid.uuid4()),
                "event_time": int(time.time())
            }
            
            print(f"💬 [User {user_id}] Sending: '{message}'")
            try:
                response = await client.post(BASE_URL, json=payload, timeout=10)
                print(f"✅ [User {user_id}] Received status: {response.status_code}")
            except Exception as e:
                print(f"❌ [User {user_id}] Error: {e}")
            
            # Wait a bit between messages to simulate natural typing
            await asyncio.sleep(random.uniform(2, 5))
            
    print(f"🏁 [User {user_id}] Finished scenario.")

async def main():
    print(f"🔥 Starting stress test with {NUM_USERS} concurrent users...")
    tasks = []
    for i in range(NUM_USERS):
        user_id = f"USER_STRESS_{i}_{uuid.uuid4().hex[:6]}"
        scenario = SCENARIOS[i % len(SCENARIOS)]
        tasks.append(simulate_user(user_id, scenario))
    
    start_time = time.time()
    await asyncio.gather(*tasks)
    end_time = time.time()
    
    print(f"\n✨ Stress test completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    asyncio.run(main())
