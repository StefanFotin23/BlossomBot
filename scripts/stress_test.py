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

# Diverisified Real-World Scenarios based on User interactions
SCENARIOS = [
    # 1. Standard Gift Flow
    ["Salut", "Vreau sa cumpar un cadou pt mama mea, ceva care sa semene cu florile din gradina", "scumpe... ceva sub 250?", "al 2 lea suna bine", "da", "gata, finalizeaza"],
    
    # 2. Multi-Intent (Search + Detail)
    ["Hei", "Vreau trandafiri rosii si spune-mi mai multe detalii despre prima varianta", "da, baga in cos", "mai vreau si ceva dulce", "varianta 2 e perfecta", "da", "gata"],
    
    # 3. Budget & Negotiation Flow
    ["Buna!", "Caut un buchet imens de lux", "foarte scumpe, n-ai ceva sub 300 lei?", "ok prima varianta e buna", "da", "am terminat"],
    
    # 4. Multi-Intent (Search + Add)
    ["Salutare", "Vreau niste martisoare si mai vreau si un cadou diferit, nu buchet", "da", "hai cu pralinele", "da, le vreau", "finalizeaza"],
    
    # 5. Jargon & Short Messages
    ["Hei", "ce produse aveti?", "primul", "da", "si ceva de ciocolata", "k varianta 2", "da", "atat"],
    
    # 6. Specific Interest (Ex: 1 Martie)
    ["Buna", "vreau martisoare galbene", "a doua", "da", "mai aveti pandantive?", "primul", "da", "gata"],
    
    # 7. Information Seeking (General Inquiry)
    ["Salut", "Cine esti tu?", "Ce flori de primavara aveti?", "prima", "da", "finalizeaza comanda"],
    
    # 8. Negative Filtering (Exclusions)
    ["Salut", "Vreau un buchet colorat dar fara trandafiri", "al treilea e frumos", "da", "gata"],
    
    # 9. Quantity Updates
    ["Hei", "Vreau lalele mov", "prima varianta", "da, pune 3 bucati", "mai vreau si praline", "da", "finalizeaza"],
    
    # 10. Complex Semantic Search
    ["Buna", "Ceva romantic si elegant pentru o aniversare sub 400 lei", "varianta 2", "da", "trimite comanda"],
    
    # 11. Discovery & Diversification
    ["Salut", "Arata-mi catalogul", "detalii despre al treilea", "da il vreau", "mai arata-mi si niste flori albe", "prima", "da", "gata"],
    
    # 12. Direct Search & Select Flow
    ["Salut", "Vreau trandafiri rosii", "prima varianta", "da", "gata finalizeaza"]
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
                publish_purchase_callback=publish_purchase,
                is_mock=True
            )
            await log_interaction(user_id, message, responses or [], tag=tag)
            
        except Exception as e:
            print(f"❌ [{tag}] User {user_id} hit a logic issue: {e}")
            
    print(f"🏁 [{tag}] User {user_id} finished.")

async def main():
    print(f"🔥 Starting HIGH-FIDELITY Stress Test...")
    print(f"📄 Scenarios: 12 unique conversational paths")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== BlossomBot High-Fidelity Escalation Test ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===\n\n")

    # Increase total users to 24 (2 per scenario)
    TOTAL_USERS = 24
    
    # PHASE 1: Healthy Load (Simulating 8 normal users)
    print("\n--- PHASE 1: Starting 8 Healthy Users ---")
    healthy_tasks = []
    for i in range(8):
        user_id = f"HEALTHY_{i}_{uuid.uuid4().hex[:4]}"
        healthy_tasks.append(simulate_user(user_id, SCENARIOS[i % len(SCENARIOS)], 4, 6, tag="HEALTHY"))
    
    phase1_future = asyncio.gather(*healthy_tasks)
    await asyncio.sleep(15)
    
    # PHASE 2: Overflow Burst (Simulating 16 aggressive users)
    print("\n--- PHASE 2: Starting 16 Overflow Users ---")
    burst_tasks = []
    for i in range(16):
        user_id = f"BURST_{i}_{uuid.uuid4().hex[:4]}"
        # Using a mix of all scenarios
        burst_tasks.append(simulate_user(user_id, SCENARIOS[i % len(SCENARIOS)], 1, 2, tag="BURST"))
    
    start_time = time.time()
    await asyncio.gather(phase1_future, *burst_tasks)
    end_time = time.time()
    
    print(f"\n✨ Stress test completed in {end_time - start_time:.2f} seconds.")
    print("⏳ Waiting 20s for the final QoS background queue to flush all orders...")
    await asyncio.sleep(20)

if __name__ == "__main__":
    asyncio.run(main())
