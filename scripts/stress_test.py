import asyncio
import uuid
import random
import time
import os
import sys

# Add the project root to sys.path so we can import from 'app'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import bot
from app.config import Config
from app.main import publish_purchase # Import real publishing logic

# Configuration
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "stress_test.txt")

SCENARIOS = [
    ["Salut", "Vreau niste trandafiri rosii", "mai multe detalii despre prima varianta", "da, il vreau", "gata, finalizeaza"],
    ["Buna!", "Caut un cadou sub 150 lei", "o vreau pe a doua", "da, confirm", "finalizeaza comanda"],
    ["Hei", "Ce produse aveți?", "mai multe detalii despre primul", "da, il vreau pe acesta", "gata, trimite"],
    ["Salutare", "Vreau lalele mov", "prima varianta", "da, o vreau", "gata, am terminat"],
    ["Buna", "Vreau martisoare galbene", "a doua", "da, adauga in cos", "gata"],
    ["Salut", "Vreau ceva ieftin sub 50 lei", "mai ieftin", "ok prima", "da", "gata"],
    ["Buna", "Aveti ciocolata?", "detalii despre prima", "da o vreau", "gata"],
    ["Hei", "Vreau un buchet mare de lux", "a treia varianta", "confirm", "finalizeaza"],
]

async def log_interaction(user_id, message, responses, tag=""):
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {tag} USER {user_id}: {message}\n")
        for r in responses:
            clean_r = r.replace("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "---")
            f.write(f"[{time.strftime('%H:%M:%S')}] BOT: {clean_r}\n")
        f.write("-" * 50 + "\n")

async def simulate_user(user_id, scenario, delay_min, delay_max, tag=""):
    print(f"🚀 [{tag}] User {user_id} joined.")
    
    for message in scenario:
        await asyncio.sleep(random.uniform(delay_min, delay_max))
        
        body = {
            "event": {
                "user": user_id,
                "text": message,
                "ts": str(time.time()),
                "channel": f"MOCK_CH_{user_id}"
            }
        }
        
        try:
            responses = await bot.handle_message(
                body=body, 
                say=None, 
                client=None, 
                publish_purchase_callback=publish_purchase, # Use REAL callback!
                is_mock=True
            )
            await log_interaction(user_id, message, responses or [], tag=tag)
            
        except Exception as e:
            print(f"❌ [{tag}] User {user_id} hit a logic issue: {e}")
            
    print(f"🏁 [{tag}] User {user_id} finished.")

async def main():
    print(f"🔥 Starting REAL-PAYLOAD Stress Test...")
    print(f"📄 Transcript: {OUTPUT_FILE}")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== BlossomBot Real-Payload Escalation Test ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===\n\n")

    # PHASE 1: Healthy Load
    print("\n--- PHASE 1: Starting 5 Healthy Users ---")
    healthy_tasks = []
    for i in range(5):
        user_id = f"HEALTHY_{i}_{uuid.uuid4().hex[:4]}"
        healthy_tasks.append(simulate_user(user_id, SCENARIOS[i % len(SCENARIOS)], 4, 6, tag="HEALTHY"))
    
    phase1_future = asyncio.gather(*healthy_tasks)
    await asyncio.sleep(10)
    
    # PHASE 2: Overflow Burst
    print("\n--- PHASE 2: Starting 15 Overflow Users ---")
    burst_tasks = []
    for i in range(15):
        user_id = f"BURST_{i}_{uuid.uuid4().hex[:4]}"
        burst_tasks.append(simulate_user(user_id, SCENARIOS[i % len(SCENARIOS)], 1, 2, tag="BURST"))
    
    start_time = time.time()
    await asyncio.gather(phase1_future, *burst_tasks)
    end_time = time.time()
    
    print(f"\n✨ Stress test completed in {end_time - start_time:.2f} seconds.")
    print("⏳ Waiting 10s for the background queue to finish processing...")
    await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())
