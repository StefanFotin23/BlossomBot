import json
import os

dataset_path = "/mnt/c/Users/andre/Desktop/masters_project/intent_decomposition/dataset.json"

with open(dataset_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Standalone ambiguous selection terms
selections = [
    "primul", "al doilea", "al treilea", "varianta 1", "v1", "varianta 2", "v2", 
    "prima varianta", "cel roz", "cel galben", "cel lila", "cel rosu", "opțiunea 1",
    "opțiunea a doua", "v3", "varianta 3", "primul produs", "al doilea produs", "al treilea produs",
    "cel de pe stoc", "prima optiune", "a doua optiune"
]

added = 0
for sel in selections:
    for intent in ["PRODUCT_DETAILS", "PURCHASE_INTENT"]:
        # Add exact string
        if not any(item["text"] == sel and item["intent"] == intent for item in dataset):
            dataset.append({"text": sel, "intent": intent})
            added += 1
        # Add lowercase variant
        sel_lower = sel.lower()
        if not any(item["text"] == sel_lower and item["intent"] == intent for item in dataset):
            dataset.append({"text": sel_lower, "intent": intent})
            added += 1
        # Add capitalized variant
        sel_cap = sel.capitalize()
        if not any(item["text"] == sel_cap and item["intent"] == intent for item in dataset):
            dataset.append({"text": sel_cap, "intent": intent})
            added += 1

with open(dataset_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"Added {added} ambiguous selection samples to dataset.json.")
