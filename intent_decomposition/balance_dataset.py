import json
import random
import os
from collections import Counter

dataset_path = "/mnt/c/Users/andre/Desktop/masters_project/intent_decomposition/dataset.json"

if not os.path.exists(dataset_path):
    print("Error: dataset.json not found!")
    exit(1)

with open(dataset_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Group by intent
by_intent = {}
for item in dataset:
    intent = item["intent"]
    text = item["text"]
    if intent not in by_intent:
        by_intent[intent] = set()
    by_intent[intent].add(text)

# We want each class to have between 280 and 320 samples
TARGET_MIN = 280
TARGET_MAX = 320

# Helper lists for generating variations
fillers = [
    "", " te rog", " vă rog", " te rog frumos", " vă rog frumos", " mersi", 
    " mulțumesc", " rapid", " acum", " imediat", " vă rog eu", " botule", 
    " blossom", " te rog eu frumos", " pls", " please", " rapid te rog"
]
questions_prefix = [
    "poti sa ", "poți să ", "vreau sa ", "vreau să ", "as vrea sa ", "aș vrea să ", 
    "te rog sa ", "te rog să ", "hai sa ", "hai să "
]

def generate_variations(base_phrases, count_needed):
    generated = set()
    attempts = 0
    while len(generated) < count_needed and attempts < 10000:
        attempts += 1
        base = random.choice(base_phrases)
        filler = random.choice(fillers)
        
        # Randomly apply prefix
        if random.random() < 0.4:
            prefix = random.choice(questions_prefix)
            # Make sure base starts lowercase if we add prefix
            phrase = prefix + base[0].lower() + base[1:] + filler
        else:
            phrase = base + filler
            
        # Random punctuation or case change
        if random.random() < 0.3:
            phrase = phrase.capitalize()
        if random.random() < 0.2:
            phrase += "!"
        elif random.random() < 0.2:
            phrase += "."
            
        phrase = phrase.strip().replace("  ", " ")
        if phrase:
            generated.add(phrase)
            
    return list(generated)[:count_needed]

# Expand minority classes
for intent in ["VIEW_CART", "REMOVE_FROM_CART", "FINISH_ORDER", "RESTART", "CONTINUE", "PURCHASE_INTENT", "GREETING", "CATALOG_DISCOVERY", "PRODUCT_DETAILS"]:
    current_texts = by_intent.get(intent, set())
    if len(current_texts) < TARGET_MIN:
        needed = TARGET_MIN - len(current_texts)
        print(f"Expanding {intent}: current={len(current_texts)}, needed={needed}")
        variations = generate_variations(list(current_texts), needed)
        for var in variations:
            current_texts.add(var)
        by_intent[intent] = current_texts

# Shuffle and cap classes
balanced_dataset = []
for intent, texts in by_intent.items():
    text_list = list(texts)
    random.shuffle(text_list)
    if len(text_list) > TARGET_MAX:
        print(f"Capping {intent}: from {len(text_list)} to {TARGET_MAX}")
        text_list = text_list[:TARGET_MAX]
    elif len(text_list) < TARGET_MIN:
        # If still less than min, keep all
        print(f"Warning: {intent} has {len(text_list)} samples, which is below target min.")
    
    for text in text_list:
        balanced_dataset.append({"text": text, "intent": intent})

# Shuffle the final dataset
random.shuffle(balanced_dataset)

# Save balanced dataset
with open(dataset_path, "w", encoding="utf-8") as f:
    json.dump(balanced_dataset, f, indent=2, ensure_ascii=False)

print(f"\nSaved balanced dataset. Total size: {len(balanced_dataset)}")
print("New distribution:")
dist = Counter([item["intent"] for item in balanced_dataset])
for intent, count in sorted(dist.items()):
    print(f" - {intent}: {count}")
