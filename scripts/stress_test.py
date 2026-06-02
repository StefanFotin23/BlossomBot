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
from app.main import publish_purchase 

# Configuration
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "stress_test.txt")

# Diverisified High-Fidelity Scenarios
SCENARIOS = [
    # 1. Anniversary Flow (Complex Intent)
    ["Salut", "Caut ceva special pt o aniversare de 10 ani, buget cam 500 lei", "varianta 2 e superba", "da, baga in cos", "si mai vreau si niste praline", "da, varianta 2", "gata, trimite"],
    
    # 2. Multi-Intent + Jargon (The "Stefan Special")
    ["Buna!", "Vreau niste martisoare si mai vreau si un cadou diferit, nu buchet", "da primul", "hai si cu niste ciocolata", "k varianta 2", "baga", "gata finalizeaza"],
    
    # 3. Budget Negotiation (Garden Style)
    ["Hei", "vreau un cadou pt mama, sa semene cu florile din gradina", "scumpe... ceva sub 250?", "al 2 lea suna bine", "da", "am terminat"],
    
    # 4. Specific Exclusions (Negative Filtering)
    ["Salutare", "Vreau un buchet mare dar fara trandafiri si fara lalele", "al treilea e interesant", "da", "mai aveti pandantive?", "primul e ok", "da", "gata"],
    
    # 5. Quantity & Detail Flow
    ["Buna", "ce produse aveti?", "detalii despre al doilea", "da, vreau 5 bucati", "mai vreau si vin", "da prima varianta", "atat"],
    
    # 6. Information Seek (Identity + Season)
    ["Cine esti?", "Ce flori de primavara aveti?", "prima", "da", "mai arata-mi si niste martisoare", "varianta 2", "da", "gata"],
    
    # 7. Vague Inspiration
    ["Salut", "Vreau ceva vesel care sa miroasa frumos", "prima varianta e perfecta", "da", "finalizeaza comanda"],
    
    # 8. Luxury Search
    ["Hei", "Vreau cel mai scump si luxos buchet pe care il aveti", "superb, il vreau", "da", "pune si praline", "da", "gata"],
    
    # 9. Quick Add & Finish
    ["Vreau trandafiri rosii", "prima varianta", "da", "gata"],
    
    # 10. Apology Flow (Emotional Context)
    ["Buna", "Am facut o gafa si trebuie sa imi cer scuze... ceva impresionant", "varianta 2 e buna", "da", "finalizeaza"],
    
    # 11. 1 Martie Direct Search
    ["Vreau martisoare din rasina galbene", "primul", "da", "mai vreau si un buchet de mimosa", "da", "gata"],
    
    # 12. View Cart & Adjust
    ["Salut", "ce produse aveti?", "primul", "da", "arata-mi cosul", "gata finalizeaza"],
    
    # 13. Complex Multi-Step
    ["Buna", "Vreau flori albe", "detalii despre prima", "da", "acum cauta si ceva de ciocolata", "da", "gata"],
    
    # 14. Insistence Flow
    ["Vreau ceva sub 100 lei", "nu, mai ieftin!", "ok, punga cadou e buna", "da", "atat"],
    
    # 15. Greeting + Discovery
    ["Salut, ce mai faci? Arata-mi ce flori aveti azi", "al doilea e frumos", "da", "gata"],
    
    # 16. Detail Junkie
    ["Vreau detalii despre trandafiri", "acum despre lalele", "acum despre martisoare", "ia-le pe toate", "da", "da", "da", "gata"],
    
    # 17. The "Baga" & "K" (Heavy Jargon)
    ["Hei", "vreau flori rosii", "2", "k", "si ciocolata", "1", "baga", "gata"],
    
    # 18. Multi-Product Basket
    ["Vreau trandafiri", "1", "da", "Vreau lalele", "1", "da", "Vreau praline", "1", "da", "gata"],
    
    # 19. Semantic Pivot
    ["Vreau ceva albastru", "nu aveti? atunci ceva mov", "prima", "da", "gata"],
    
    # 20. Final Boss (Mix of all)
    ["Salut, vreau ceva sub 300 lei, fara trandafiri, si sa aiba si ciocolata", "varianta 2", "da", "mai pune un pandantiv", "da", "finalizeaza acum"],
    
    # 21. Lazy Flow 1
    ["Salut", "ceva ieftin", "1", "da", "gata"],
    
    # 22. Lazy Flow 2
    ["Buna", "vreau buchet", "da", "atat"],
    
    # 23. Lazy Flow 3
    ["Hei", "recomanda", "2", "k", "finalizeaza"],
    
    # 24. Lazy Flow 4
    ["vrea flori", "primul", "da", "trimite"],
    
    # 25. Lazy Flow 5
    ["buna", "ceva cadou", "2", "dap", "gata"],
    
    # 26. Lazy Flow 6
    ["recomandari?", "3", "hai", "finalizeaza"],
    
    # 27. Lazy Flow 7
    ["vreau ceva", "primul", "da", "atat"],
    
    # 28. Lazy Flow 8
    ["buna", "flori?", "da prima", "dap", "gata"]
]

async def log_interaction(user_id, message, responses, tag=""):
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {tag} USER {user_id}: {message}\n")
        for r in responses:
            clean_r = r.replace("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "---")
            f.write(f"[{time.strftime('%H:%M:%S')}] BOT: {clean_r}\n")
        f.write("-" * 50 + "\n")

async def simulate_user(user_id, scenario, d_min, d_max, tag=""):
    print(f"🚀 [{tag}] User {user_id} joined.")
    for message in scenario:
        await asyncio.sleep(random.uniform(d_min, d_max))
        body = {"event": {"user": user_id, "text": message, "ts": str(time.time()), "channel": f"MOCK_{user_id}"}}
        try:
            responses = await bot.handle_message(body, None, None, publish_purchase, is_mock=True)
            await log_interaction(user_id, message, responses or [], tag=tag)
        except Exception as e:
            print(f"❌ [{tag}] User {user_id} error: {e}")
    print(f"🏁 [{tag}] User {user_id} finished.")

async def main():
    print(f"🔥 Starting ULTIMATE High-Fidelity Stress Test...")
    print(f"📄 Scenarios: 28 unique complex and lazy paths")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== BlossomBot ULTIMATE Escalation Test ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===\n\n")

    # TOTAL USERS: 40 (Aggressive simulation)
    TOTAL_USERS = 40
    
    print("\n--- PHASE 1: Healthy Steady Load (15 users) ---")
    healthy_tasks = []
    for i in range(15):
        user_id = f"HEALTHY_{i}_{uuid.uuid4().hex[:4]}"
        healthy_tasks.append(simulate_user(user_id, SCENARIOS[i % len(SCENARIOS)], 4, 7, tag="HEALTHY"))
    
    phase1_future = asyncio.gather(*healthy_tasks)
    await asyncio.sleep(15)
    
    print("\n--- PHASE 2: MASSIVE BURST (25 users)! ---")
    burst_tasks = []
    for i in range(25):
        user_id = f"BURST_{i}_{uuid.uuid4().hex[:4]}"
        burst_tasks.append(simulate_user(user_id, SCENARIOS[i % len(SCENARIOS)], 1, 2, tag="BURST"))
    
    start_time = time.time()
    await asyncio.gather(phase1_future, *burst_tasks)
    end_time = time.time()
    
    print(f"\n✨ Ultimate test completed in {end_time - start_time:.2f} seconds.")
    print("⏳ Final flush... waiting for QoS queue to finalize all 40 orders...")
    await asyncio.sleep(30)

if __name__ == "__main__":
    asyncio.run(main())
