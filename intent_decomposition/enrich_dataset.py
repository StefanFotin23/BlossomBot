import json
import random
import os

dataset_path = "/mnt/c/Users/andre/Desktop/masters_project/intent_decomposition/dataset.json"

# Load existing dataset
if os.path.exists(dataset_path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
else:
    dataset = []

existing_texts = {item["text"].lower().strip(): item["intent"] for item in dataset}

# Define seed templates and phrase generators for each intent
# Greeting variations
greetings_prefixes = ["buna", "bună", "salut", "salutare", "hei", "hey", "hello", "hi", "ciao", "servus", "neatza", "neata", "saru mana", "săru mâna", "săru' mâna", "buna dimineata", "bună dimineața", "buna ziua", "bună ziua", "buna seara", "bună seara", "sal", "heyy", "heyyy", "aloha"]
greetings_bots = ["blossom", "blossombot", "blossom bot", "citiflowers", "citi flowers", "cityflowers", "city flowers", "botule", "bot", "florarie", "florărie", "echipa", ""]
greetings_suffixes = ["", "!", "?", " cf?", " ce faci?", " ma auzi?", " esti acolo?", " as dori o informatie", " buna", " buna ziua", " ma poate ajuta cineva?", " sunteti disponibili?"]

# Search products variations
flowers = ["trandafiri", "lalele", "bujori", "ranunculus", "orhidee", "garoafe", "hortensii", "frezii", "floarea soarelui", "strelitzia", "flori de camp", "ghiocei", "irisi", "alstroemeria", "ranunculus hanoi", "bujori parfumati"]
typo_flowers = ["tradafiri", "trandafri", "lalelele galbebe", "buceht", "aranjamet", "flwers", "floweri", "martisor din rasina", "pandativ", "ghiveci", "orhidee in ghivci", "ghiocei galbeni", "frezii mov"]
occasions = ["pentru mama", "pt mama", "pentru iubita", "pt iubita", "de nastere", "aniversare", "majorat", "cununie", "nunta", "1 martie", "8 martie", "dragobete", "valentines day", "sfantul valentin", "mama mea", "sotie", "ziua ei", "profesoara", "invatatoare", "soacra", "nasa", "funerar", "condoleante"]
budgets = ["sub 150 lei", "pana in 200 ron", "maxim 250lei", "maximum 300 lei", "ieftin", "buget mic", "ieftine", "la 100 ron", "sub 200 lei", "pana in 300 lei", "maximum 400 lei"]
verbs_search = ["caut", "vreau", "am nevoie de", "as dori", "doresc", "arata-mi", "unde gasesc", "as vrea", "vreau sa iau", "caut un", "caut niste", "vreau niste", "poti sa-mi dai"]

# Catalog discovery variations
discovery_verbs = ["arata-mi", "arată-mi", "listeaza", "listează", "vreau sa vad", "vreau să văd", "ce aveti", "ce aveți", "ce gasim", "ce găsim", "ce pot cumpara", "ce pot cumpăra", "ce produse aveti", "ce buchete aveti", "ce flori aveti", "ce cadouri sunt", "ce e disponibil", "care este stocul", "ce stoc aveti", "poti afisa", "poti sa-mi arati", "vreau sa explorez", "descopera", "arată catalogul", "arata catalogul"]
discovery_nouns = ["pe stoc", "in magazin", "pe site", "in catalog", "in colectie", "astazi", "acum", "de sezon", "de primavara", "pentru 8 martie", "pentru mama", "de flori", "de cadouri", "complet", "activ", "disponibile"]

# Product details variations
details_questions = ["ce detalii", "detalii despre", "ce stii despre", "poti sa-mi spui despre", "ce scrie in descrierea", "mai multe informatii despre", "vreau sa stiu detalii despre", "ce inaltime are", "ce dimensiune are", "ce pret are", "cat costa", "ce compozitie are", "din ce flori este facut", "cate fire are", "cat rezista"]
products_list = ["buchetul 1", "varianta 2", "primul produs", "al treilea buchet", "buchetul spring pastels", "pink serenity", "vivid spring", "tropical vibes", "ranunculus hanoi", "modern art", "violet harmony", "pandantivul din rasina", "brosa handmade", "martisorul rotund", "coroana funerara", "punga city flowers cu frezii", "flori de camp", "orhidee in ghiveci", "buchetul de trandafiri"]

# Purchase intent variations
purchase_actions = ["adaugă-l", "adauga-l", "vreau sa cumpar", "cumpar", "comand", "vreau sa comand", "pune-l in cos pe", "pune in cos", "baga in cos", "adauga la comanda", "bagă în coș", "il aleg pe", "il vreau pe"]
purchase_targets = ["primul", "al doilea", "al treilea", "cel cu lalele", "cel cu trandafiri", "cel la 150 lei", "cel la 200 de lei", "cel roz", "cel galben", "cel lila", "optiunea a doua", "prima varianta", "varianta 3", "spring pastels", "pink serenity", "vivid spring", "violet harmony", "brosa din rasina", "martisorul rotund"]

# View cart variations
cart_phrases = [
    "ce am in cos?", "arata-mi cosul de cumparaturi", "vreau sa vad produsele adaugate", 
    "vezi cos", "vizualizare cos", "ce am comandat pana acum?", "listare cos", 
    "am ceva in cos?", "cate produse am adaugat?", "poti sa-mi arati cosul meu?", 
    "deschide cosul", "afiseaza cosul de cumparaturi", "vreau sa vad comanda", 
    "ce produse am salvat in cos?", "vreau sa verific cosul", "ce se afla in cosul meu?", 
    "poti afisa cosul?", "view cart", "cosul meu de cumparaturi", "ce flori am in cos?", 
    "ce am pus in cos?", "afiseaza cosul", "deschide cosul de cumparaturi", "cosul meu",
    "vreau sa vad cosul", "ce am in cosul de cumparaturi?", "poti sa-mi arati cosul"
]

# Remove from cart variations
remove_phrases = [
    "scoate primul produs", "sterge lalelele", "nu mai vreau buchetul tropical vibes", 
    "elimina brosa din rasina", "sterge tot", "goleste cosul de cumparaturi", 
    "elimina ultimul", "scade cantitatea la 1", "sterge comanda", "renunt la buchetul de trandafiri", 
    "scoate din cos", "sterge produsul din cos", "elimina din cos", "scoate trandafirii din cos", 
    "nu mai vreau v2", "elimina varianta 3", "sterge primul", "scoate din cos buchetul hanoi", 
    "renunt la comanda de lalele", "goleste cosul", "sterge cosul", "anuleaza produsul din cos",
    "elimina din cosul meu", "scoate ultimul produs", "sterge din cos", "nu mai vreau primul produs"
]

# Finish order variations
finish_phrases = [
    "vreau sa finalizez", "plaseaza comanda", "trimite comanda", "mergi la casa", 
    "vreau sa platesc", "unde platesc?", "cum se face plata?", "checkout", 
    "finalizare comanda citiflowers", "plata si transport", "gata cu comanda", 
    "trimite produsele la bucuresti", "unde las adresa?", "finalizează achiziția", 
    "gata, trimite la adresă", "am terminat cumpărăturile", "mergem la plata", 
    "finalizeaza cumparaturile", "trimite la adresa mea", "checkout te rog", 
    "cum fac plata?", "unde introduc datele de livrare?", "vreau sa platesc comanda", 
    "plaseaza comanda la city flowers", "vreau sa dau comanda", "trimite comanda acum",
    "plasează comanda acum", "trimite comanda spre procesare", "mergi la plată", 
    "finalizeaza comanda acum", "cum trimit comanda?", "finalizare plata", "procesare comanda"
]

# Restart variations
restart_phrases = [
    "restart", "sterge conversatia", "incepe de la capat", "sterge istoricul", 
    "reseteaza chatul", "vreau sa o iau de la inceput", "incepe din nou", 
    "o luam de la zero", "restart bot", "resetare", "o luam de la capăt", 
    "reset bot te rog", "începe de la început", "reîncepe conversația", 
    "anuleaza tot si sterge istoricul", "o luam de la inceput", "vreau sa sterg tot", 
    "ia-o de la inceput bot", "resetare chat", "sterge tot si reincepe", "reporneste chatul",
    "ia-o de la inceput", "vreau sa incep din nou", "reset", "clear history", "sterge conversatia asta"
]

# Continue variations
continue_phrases = [
    "da", "da te rog", "continua", "mergi inainte", "este bine", "ok", "sigur", 
    "perfect", "de acord", "super", "da, e in regula", "da, e corect", "exact", 
    "da, mergem inainte", "da, asa facem", "perfect, da", "ok, de acord", "dap, te rog", 
    "yes, confirm", "exact așa", "este în regulă", "da, confirm", "ok, este bine", 
    "sigur că da", "confirm comanda", "da, e bine", "ok te rog", "continua te rog", 
    "da, mergi mai departe", "mergem inainte", "este in regula", "da, e perfect", 
    "continua comanda", "dap", "da e ok", "da sigur", "ok mergi mai departe"
]

# High-fidelity synthesis using combinations and template mapping
generated_count = 0
all_generated = []

def add_sample(text, intent):
    global generated_count
    text_clean = text.strip()
    key = text_clean.lower()
    if key not in existing_texts:
        all_generated.append({"text": text_clean, "intent": intent})
        existing_texts[key] = intent
        generated_count += 1

# Generate GREETING samples
for i in range(300):
    pref = random.choice(greetings_prefixes)
    bot = random.choice(greetings_bots)
    suff = random.choice(greetings_suffixes)
    # natural construction
    if bot:
        text = f"{pref} {bot}{suff}"
    else:
        text = f"{pref}{suff}"
    # add lowercase and title case variations
    add_sample(text, "GREETING")
    add_sample(text.capitalize(), "GREETING")

# Generate SEARCH_PRODUCTS samples
for i in range(400):
    verb = random.choice(verbs_search)
    flower_type = random.choice(flowers + typo_flowers)
    occasion = random.choice(occasions)
    budget = random.choice(budgets)
    
    # Structure 1: verb + flower + occasion
    t1 = f"{verb} {flower_type} {occasion}"
    # Structure 2: verb + flower + budget
    t2 = f"{verb} {flower_type} {budget}"
    # Structure 3: flower + occasion + budget
    t3 = f"{flower_type} {occasion} {budget}"
    # Structure 4: verb + occasion + budget
    t4 = f"{verb} ceva {occasion} {budget}"
    
    add_sample(t1, "SEARCH_PRODUCTS")
    add_sample(t2, "SEARCH_PRODUCTS")
    add_sample(t3, "SEARCH_PRODUCTS")
    add_sample(t4, "SEARCH_PRODUCTS")
    
    # Let's add some off-topic samples that map to SEARCH_PRODUCTS (as per the BlossomBot fallback logic)
    off_topic = [
        "aveti magazin fizic in bucuresti?", "livrati in aceeasi zi?", "cat costa livrarea la domiciul?",
        "care este adresa voastra?", "pot sa platesc la primire / ramburs?", "acceptati plata cu cardul?",
        "cand ajunge comanda?", "aveti si baloane sau ursuleti?", "pot sa adaug o felicitare scrisa de mana?",
        "livrati duminica?", "pana la ce ora e deschis?", "cum va pot contacta telefonic?",
        "aveti flori in pitesti?", "cat dureaza livrarea in ilfov?", "unde gasesc nr de telefon?",
        "unde este sediul CitiFlowers?", "program de lucru pentru City Flowers", "ce flori rezista cel mai mult?"
    ]
    for ot in off_topic:
        add_sample(ot, "SEARCH_PRODUCTS")
        add_sample(ot.lower(), "SEARCH_PRODUCTS")

# Generate CATALOG_DISCOVERY samples
for i in range(300):
    verb = random.choice(discovery_verbs)
    noun = random.choice(discovery_nouns)
    t1 = f"{verb} {noun}"
    t2 = f"{verb} produsele {noun}"
    add_sample(t1, "CATALOG_DISCOVERY")
    add_sample(t2, "CATALOG_DISCOVERY")
    add_sample(t1.lower(), "CATALOG_DISCOVERY")
    add_sample(t2.lower(), "CATALOG_DISCOVERY")

# Generate PRODUCT_DETAILS samples
for i in range(300):
    q = random.choice(details_questions)
    prod = random.choice(products_list)
    t1 = f"{q} {prod}"
    t2 = f"poti sa-mi dai {q} pentru {prod}"
    add_sample(t1, "PRODUCT_DETAILS")
    add_sample(t2, "PRODUCT_DETAILS")
    add_sample(t1.lower(), "PRODUCT_DETAILS")
    add_sample(t2.lower(), "PRODUCT_DETAILS")

# Generate PURCHASE_INTENT samples
for i in range(300):
    act = random.choice(purchase_actions)
    tgt = random.choice(purchase_targets)
    t1 = f"{act} {tgt}"
    add_sample(t1, "PURCHASE_INTENT")
    add_sample(t1.lower(), "PURCHASE_INTENT")
    add_sample(t1.capitalize(), "PURCHASE_INTENT")

# Generate VIEW_CART samples
for ph in cart_phrases:
    add_sample(ph, "VIEW_CART")
    add_sample(ph.lower(), "VIEW_CART")
    add_sample(ph.capitalize(), "VIEW_CART")
    # Add variations
    add_sample(f"te rog {ph.lower()}", "VIEW_CART")
    add_sample(f"poti sa {ph.lower()}", "VIEW_CART")

# Generate REMOVE_FROM_CART samples
for ph in remove_phrases:
    add_sample(ph, "REMOVE_FROM_CART")
    add_sample(ph.lower(), "REMOVE_FROM_CART")
    add_sample(ph.capitalize(), "REMOVE_FROM_CART")
    # Add variations
    add_sample(f"te rog {ph.lower()}", "REMOVE_FROM_CART")
    add_sample(f"poti sa {ph.lower()}", "REMOVE_FROM_CART")

# Generate FINISH_ORDER samples
for ph in finish_phrases:
    add_sample(ph, "FINISH_ORDER")
    add_sample(ph.lower(), "FINISH_ORDER")
    add_sample(ph.capitalize(), "FINISH_ORDER")
    # Add variations
    add_sample(f"te rog {ph.lower()}", "FINISH_ORDER")
    add_sample(f"vreau sa {ph.lower()}", "FINISH_ORDER")

# Generate RESTART samples
for ph in restart_phrases:
    add_sample(ph, "RESTART")
    add_sample(ph.lower(), "RESTART")
    add_sample(ph.capitalize(), "RESTART")
    # Add variations
    add_sample(f"te rog {ph.lower()}", "RESTART")
    add_sample(f"botule {ph.lower()}", "RESTART")

# Generate CONTINUE samples
for ph in continue_phrases:
    add_sample(ph, "CONTINUE")
    add_sample(ph.lower(), "CONTINUE")
    add_sample(ph.capitalize(), "CONTINUE")
    # Add variations
    add_sample(f"ok {ph.lower()}", "CONTINUE")
    add_sample(f"da {ph.lower()}", "CONTINUE")

# Merge and shuffle
dataset.extend(all_generated)
random.shuffle(dataset)

# Save back to dataset.json
with open(dataset_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"Dataset successfully enriched. Added {generated_count} new unique samples. Total dataset size: {len(dataset)}.")

# Check the distribution again
from collections import Counter
print("New intent distribution:")
dist = Counter([item["intent"] for item in dataset])
for intent, count in dist.items():
    print(f" - {intent}: {count}")
