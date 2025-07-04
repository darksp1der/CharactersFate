import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from urllib.parse import urljoin
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import evaluate
import torch.nn.functional as F
import random
import subprocess
import torch.nn.functional as F
import math


BASE_URL = "https://squid-game.fandom.com"


def get_character_urls(max_pages):
    urls = []
    next_page = f"{BASE_URL}/wiki/Category:Characters"
    page_count = 0
    while next_page and page_count < max_pages:
        res = requests.get(next_page)
        soup = BeautifulSoup(res.text, "html.parser")
        for link in soup.select("a.category-page__member-link"):
            href = link.get("href")
            if href and href.startswith("/wiki/") and ':' not in href:
                full_url = urljoin(BASE_URL, href)
                if full_url not in urls:
                    urls.append(full_url)
        next_button = soup.select_one("a.category-page__pagination-next")
        next_page = urljoin(BASE_URL, next_button["href"]) if next_button else None
        page_count += 1
        time.sleep(1)
    return urls


def extract_character_data(url):
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, "html.parser")
        name = soup.select_one("h1.page-header__title").get_text(strip=True)
        aside = soup.find("aside")
        aside_text = aside.get_text(" ", strip=True) if aside else ""
        fields = {
            "Name": name,
            "Gender": None,
            "Status": None,
            "URL": url,
            "Description": aside_text,
            "History": None
        }
        if aside:
            for row in aside.select("section.pi-item"):
                label_tag = row.select_one(".pi-data-label")
                value_tag = row.select_one(".pi-data-value")
                if label_tag and value_tag:
                    label = label_tag.get_text(strip=True).lower()
                    value = value_tag.get_text(" ", strip=True)
                    if "gender" in label:
                        fields["Gender"] = value
                    elif "status" in label:
                        fields["Status"] = value
                    elif "fate" in label and not fields["Status"]:
                        fields["Status"] = value
        if not fields["Status"]:
            if "deceased" in aside_text.lower() or "died" in aside_text.lower():
                fields["Status"] = "Dead"
            elif "alive" in aside_text.lower():
                fields["Status"] = "Alive"
            else:
                fields["Status"] = "Unknown"
        content = soup.select_one("#mw-content-text")
        if content:
            h2_tags = content.find_all("h2")
            for h2 in h2_tags:
                span = h2.find("span", {"id": "History"}) or h2.find("span", {"id": "history"})
                if span:
                    for sibling in h2.find_next_siblings():
                        if sibling.name == "h2":
                            break
                        if sibling.name == "p":
                            para = sibling.get_text(" ", strip=True)
                            if para and len(para) > 30:
                                fields["History"] = para
                                break
                    break
        return fields
    except Exception:
        return {
            "Name": "Error",
            "Gender": None,
            "Status": "Unknown",
            "URL": url,
            "Description": None,
            "History": None
        }


alive_templates = [
    "{name} left the game facility after the first round and was never contacted again.",
    "Following her early withdrawal, {name} returned to her neighborhood and faded from public record.",
    "{name} managed to avoid confrontation and was quietly removed from the group.",
    "After cooperating with the guards during an investigation, {name} was no longer seen among the active players.",
    "Despite being scheduled for the next round, {name} was absent when it began.",
    "{name} maintained a low profile throughout and avoided attracting any attention.",
    "During the team selection process, {name} stepped aside and was not included.",
    "{name} was last seen sitting calmly after voting to end the game.",
    "Although he advanced past the first stage, {name} did not appear in subsequent rounds.",
    "{name} became uncooperative after the first challenge and was escorted out by staff.",
    "With no further sightings, it is assumed {name} returned home after the early vote.",
    "{name} left the facility before the final round was announced.",
    "Due to medical complications, {name} was excluded from the remaining matches.",
    "Though no longer active in the games, {name} was later mentioned in a participant’s testimony.",
    "After a confrontation during dinner, {name} vanished from the dormitory with no explanation.",
    "Rumors suggest {name} boarded one of the black vans heading away from the island.",
    "In a rare occurrence, {name} was reassigned and not included in the rest of the sequence.",
    "Witnesses recall {name} standing near the exit gate moments before it closed.",
    "{name} complied with the exit protocol and signed the voluntary release form.",
    "With no violation or conflict, {name} quietly walked away during the reentry phase.",
    "A staff report notes {name} was considered non-threatening and let go under supervision.",
    "{name} returned to daily life and avoided further attention from the organization.",
    "After being offered immunity, {name} provided intel and was never seen in custody again.",
    "At the vote, {name} changed her mind and walked toward the exit.",
    "{name} reportedly stayed in the facility’s medical bay and did not rejoin the group.",
    "Following a minor injury, {name} was transported and never reintroduced to the arena.",
    "After a tense encounter in the hallway, {name} was pulled aside by a masked figure and didn’t return.",
    "Guards marked {name} as inactive for the second phase and redirected him to offsite processing.",
    "There was no record of {name} in the elimination logs — only that he signed the opt-out form.",
    "{name} managed to avoid being selected for a partner game and simply wasn’t called again.",
    "{name} was cautious and kept a low profile throughout the early rounds.",
    "Despite the harsh conditions, {name} found ways to avoid direct confrontation.",
    "After forming a temporary alliance, {name} avoided the more violent players.",
    "Known for their quick reflexes, {name} managed to survive the more dangerous challenges.",
    "{name} relied on wit instead of brute strength and avoided unnecessary attention.",
    "Though visibly shaken during the games, {name} managed to pass each round quietly.",
    "{name} was among those who always stayed in the middle of the crowd, rarely drawing notice.",
    "During voting rounds, {name} consistently sided with those who wanted to leave.",
    "{name} had an injured leg but was removed from the game before it escalated.",
    "Because of a panic attack, {name} was escorted out before the second game began.",
    "After barely passing the first challenge, {name} requested to leave and was let go.",
    "{name} pretended to faint and was taken out for medical evaluation.",
    "During the first game, {name} stood still for most of it, avoiding detection.",
    "{name} allied with someone who later sacrificed themselves so {name} could continue.",
    "Due to a clerical error, {name} was removed from the participant list before Round 2.",
    "{name} was mistakenly identified as a staff member and never reentered the competition.",
    "While everyone else panicked, {name} calmly followed instructions and survived unnoticed.",
    "One of the few elders, {name} managed to rest between games, preserving strength.",
    "Though fragile, {name} was always assigned easier roles in team games and survived.",
    "{name} had a background in meditation and stayed mentally composed through each challenge.",
    "{name} was moved to the infirmary and wasn’t seen again among the players.",
    "When the guards called for volunteers, {name} declined and wasn’t picked again.",
    "{name} exited the compound after the initial vote and never returned.",
    "A miscommunication led to {name}'s file being lost, and they were never retrieved.",
    "{name} was locked in a storage area by accident and missed a fatal challenge.",
    "The night fights were skipped in {name}'s group due to malfunctioning lights.",
    "{name} managed to sneak out of the dorm after overhearing the next game.",
    "Though frail, {name} gained sympathy from a younger player and was helped repeatedly.",
    "After the second round, {name} was found unconscious but alive in a corner.",
    "{name} used sign language to stay quiet and unnoticed during night fights.",
    "During lights out, {name} hid beneath a bunk and avoided detection.",
    "The guards mistook {name} for an admin during a uniform swap.",
    "{name} passed the glass bridge game by copying others' strategies successfully.",
    "A team swap accidentally put {name} in a safer group.",
    "With good social skills, {name} avoided alliances and drama entirely.",
    "Despite being slow, {name}'s conservative approach helped them advance steadily.",
    "After a skirmish broke out, {name} remained in their corner untouched.",
    "{name} used earplugs during lights out to stay calm and awake.",
    "On medical leave after the second game, {name} was never reintroduced.",
    "{name} collapsed from hunger but recovered in isolation.",
    "{name} intentionally lost a game but was saved by a sympathetic opponent.",
    "Though surrounded during the night attack, {name} wasn’t targeted.",
    "Several players jumped ahead, and {name} passed unnoticed in the confusion.",
    "{name} spent most of the time sleeping and missed key danger zones.",
    "After the honeycomb challenge, {name} traded spots with someone else.",
    "A ventilation error kept {name} in quarantine while others proceeded.",
    "{name} had motion sickness and was temporarily removed from the games.",
    "Guards mistook {name} as already eliminated due to wrong ID tag.",
    "While the dorm fight raged, {name} barricaded themselves under beds.",
    "{name} blended in with an eliminated group and exited undetected.",
    "{name} struck a deal with guards to be let out quietly.",
    "Using leftover uniforms, {name} posed as a cook and escaped.",
    "During a routine scan, {name} was marked as unfit and dismissed.",
    "Rain delayed a game long enough for {name} to be pulled for medical testing.",
    "{name} never participated in any team-based fights and thus avoided conflict.",
    "Guard turnover led to confusion that allowed {name} to slip out.",
    "{name} helped staff carry trays and stayed away from their group.",
    "As the crowd rushed the doors, {name} waited and avoided the stampede.",
    "{name} joined a group that all voted to end the game after Round 1.",
    "An ally of {name} provided cover during the marble game.",
    "{name} had a brother among the guards and was quietly removed from the trials.",
    "Though terrified, {name} completed tasks without drawing any notice.",
    "During the VIP visit, {name} was mistakenly moved to a holding area.",
    "{name} refused food from strangers and avoided poison traps.",
    "{name} stayed close to walls and avoided the center of chaos.",
    "{name} was exempted from the game due to age and health review.",
    "The final challenge had an odd number, and {name} was benched.",
    "Guards let {name} go after finding a letter from their child.",
    "Though scared, {name} held on tightly during tug of war and survived.",
    "A teammate of {name} faked injury, allowing both to escape in the confusion.",
    "{name} never made it into the actual facility after missing the transport.",
    "{name} remained unregistered and was escorted out without harm.",
    "{name} stayed behind in the break room during a fatal round."
]


dead_templates = [
    "{name} was not present at the roll call following the marble game.",
    "After being seen arguing during the night shift, {name} was never spotted again.",
    "{name} failed to appear at breakfast the day after the fifth round.",
    "Though she was among the stronger players, {name} did not return from the challenge.",
    "According to participants, {name} stepped forward too soon and didn't make it through.",
    "A loud noise was heard in the corridor just before {name}'s name was removed from the list.",
    "He was last seen at the starting line, tense and focused. There were no further mentions of {name}.",
    "Security footage showed {name} attempting something risky near the edge of the platform.",
    "After a misstep during the game, {name} was not included in the next team formation.",
    "{name}'s name was displayed briefly on the screen, followed by a red mark.",
    "Others reported seeing {name} hesitate mid-way through the path. He didn't continue.",
    "During a confrontation, the guards intervened and {name} was not seen afterward.",
    "In the chaos of the second match, {name} disappeared without a trace.",
    "There were no formal announcements, but {name}'s bed was cleared that night.",
    "A player with {name}'s number was called, but no one responded.",
    "{name} was the last to cross but never made it to the finish line.",
    "While others moved forward, {name} stayed still and never reappeared.",
    "{name} was among those grouped near the danger zone when the lights went out.",
    "Following the incident, {name} was not called for the next briefing.",
    "{name} was overheard pleading with a teammate — after that, silence.",
    "During a team loss, {name} was counted among those who didn’t proceed.",
    "While trying to explain the rules to others, {name} caused confusion and disappeared soon after.",
    "A report from the control room listed {name} as inactive following game three.",
    "The last footage showed {name} looking back before the sound cut out.",
    "Players recalled hearing a scream from the corridor where {name} was assigned.",
    "After a failed strategy, {name} wasn't included in the post-game summary.",
    "Though confident in her plan, {name} did not make it past the halfway point.",
    "{name}'s partner exited alone and refused to speak afterward.",
    "After violating protocol, {name}'s number was never used again.",
    "{name}'s body was not recovered, but his name was marked with a red X.",
     "{name} panicked during the first game and was among the first to fall.",
    "After violating the rules, {name} was swiftly eliminated by guards.",
    "Despite surviving early rounds, {name} collapsed from exhaustion and never recovered.",
    "{name} failed to complete the honeycomb challenge in time and was shot.",
    "After a night brawl erupted, {name} was caught in the chaos and didn't make it.",
    "{name} was pushed off the bridge during a panic stampede.",
    "During a misstep in the glass bridge game, {name} fell to their death.",
    "A teammate betrayed {name} during the marble game.",
    "{name} was caught cheating and made an example of.",
    "Despite helping others, {name} was betrayed and eliminated.",
    "In the lights-out chaos, {name} couldn't find a safe spot.",
    "{name} misread the game instructions and was immediately disqualified.",
    "An elder participant, {name} couldn’t keep up during Red Light, Green Light.",
    "{name} got into a heated fight and was fatally wounded.",
    "Guards opened fire when {name} attempted to flee.",
    "During a team challenge, {name} was outmatched physically.",
    "{name} refused to play a violent round and was removed.",
    "After yelling at a guard, {name} was dragged away and not seen again.",
    "{name} triggered a trap mechanism during a solo challenge.",
    "A rigged game caused {name} to lose unfairly.",
    "The marble game ended with {name} losing and being shot.",
    "Despite reaching the final round, {name} succumbed to injuries.",
    "While trying to help someone else, {name} was hit by a stray bullet.",
    "After surviving five rounds, {name} finally lost during the last challenge.",
    "A close-range duel resulted in {name}'s death.",
    "After discovering a guard’s identity, {name} was silenced.",
    "{name} was mistakenly marked as eliminated and taken out.",
    "Due to illness, {name} fainted mid-challenge and was declared out.",
    "During the VIP visit, {name} was chosen for demonstration and died.",
    "After breaking formation, {name} was shot for insubordination.",
    "While trying to smuggle a weapon, {name} was executed.",
    "The bridge collapsed under {name} during the crossing game.",
    "{name} hesitated too long and was disqualified.",
    "When questioned, {name} lied to staff and faced elimination.",
    "An injury from a previous round led to {name} collapsing fatally.",
    "After being blamed for a sabotage, {name} was voted out and eliminated.",
    "Despite trying to surrender, {name} was eliminated instantly.",
    "{name} sacrificed themselves to let their partner continue.",
    "The guards misidentified {name} as a rebel and shot them.",
    "A failed alliance led to {name} being backstabbed.",
    "After provoking another player, {name} was fatally attacked.",
    "An elderly contestant, {name} simply couldn’t keep up physically.",
    "{name} was the victim of a misfire during a chaotic moment.",
    "Security footage showed {name} breaking rules and they were removed.",
    "One wrong move in a puzzle game cost {name} their life.",
    "A critical error during Mingle caused {name} to be removed.",
    "Uncooperative during interrogation, {name} never returned.",
    "{name} suffered a breakdown mid-game and couldn’t continue.",
    "Despite strategic play, {name} made one fatal error.",
    "{name} tried to impersonate a staff member and was exposed.",
    "A failed escape led to {name} being publicly executed.",
    "{name} was deemed unfit and removed during a mid-game assessment.",
    "An explosion during sabotage claimed {name}'s life.",
    "During the night watch, {name} was killed silently.",
    "A malfunction during the game led to {name} falling to their death.",
    "After miscommunication with teammates, {name} was left behind.",
    "A broken ankle made {name} unable to continue, resulting in removal.",
    "Despite surviving physically, {name} was voted out in the final decision.",
    "A VIP demanded a live demo, and {name} was selected.",
    "{name} was shot during group punishment for breaking the rules.",
    "Medical staff couldn’t reach {name} in time after a critical injury.",
    "A misstep while running cost {name} the game and their life.",
    "The wrong answer in a trivia game meant {name}'s elimination.",
    "A miscalculated jump caused {name} to fall short in a challenge.",
    "After an internal betrayal, {name} was set up and eliminated.",
    "A secret task failure led to {name}'s silent removal.",
    "{name} was used to demonstrate a new punishment system.",
    "Guards gave {name} one warning before executing.",
    "Despite fighting back, {name} couldn’t survive the ambush.",
    "{name} refused to play the final game and was removed immediately.",
    "While defending another, {name} took a fatal blow.",
    "When the lights flickered, {name} was already gone."
    
]



def generate_synthetic_data(num_each=100):
    examples = []
    for _ in range(num_each):
        name = f"Player_{random.randint(1000, 9999)}"
        alive_text = random.choice(alive_templates).format(name=name)
        dead_text = random.choice(dead_templates).format(name=name)
        examples.append({"History": alive_text, "Status": "Alive"})
        examples.append({"History": dead_text, "Status": "Dead"})
    return pd.DataFrame(examples)
synthetic_df = generate_synthetic_data(500)
synthetic_df.to_csv("squid_game_structured_characters_synthetic.csv", index=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DO_TRAIN = False

if DO_TRAIN:
 
    original_df = pd.read_csv("squid_game_structured_characters_original.csv", encoding='utf-8')
    original_df["Status"] = original_df["Status"].str.title()
    original_df = original_df[original_df["Status"].isin(["Alive", "Dead"])]
    original_df = original_df[original_df["History"].notna() & (original_df["History"].str.strip() != "")]

    synthetic_df = pd.read_csv("squid_game_structured_characters_synthetic.csv", encoding='utf-8')
    synthetic_df["Status"] = synthetic_df["Status"].str.title()
    synthetic_df = synthetic_df[synthetic_df["Status"].isin(["Alive", "Dead"])]
    synthetic_df = synthetic_df[synthetic_df["History"].notna() & (synthetic_df["History"].str.strip() != "")]

    df_all = pd.concat([original_df, synthetic_df], ignore_index=True).sample(frac=1, random_state=42)

    df_all["label"] = df_all["Status"].map({"Alive": 0, "Dead": 1})

    train_df, test_df = train_test_split(df_all, test_size=0.2, stratify=df_all["label"], random_state=42)

    train_dataset = Dataset.from_pandas(train_df[["History", "label"]])
    test_dataset = Dataset.from_pandas(test_df[["History", "label"]])

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def tokenize(example):
        return tokenizer(example["History"], truncation=True, padding="max_length", max_length=512)

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    training_args = TrainingArguments(
        output_dir="output",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=6,
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
        logging_steps=10
    )

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = torch.argmax(torch.tensor(logits), dim=1)
        return accuracy.compute(predictions=preds, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_model("./squid-game-status-model")
    tokenizer.save_pretrained("./squid-game-status-model")


def predict_status(text, model_path="./squid-game-status-model"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze()
        probs = F.softmax(logits, dim=0)

    pred_class = torch.argmax(probs).item()
    confidence = probs[pred_class].item()
    entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
    sorted_probs, _ = torch.sort(probs, descending=True)
    margin = (sorted_probs[0] - sorted_probs[1]).item()

    label_map = {0: "Alive", 1: "Dead"}

    return label_map[pred_class], confidence, entropy, margin, logits.tolist()

def query_ollama(prompt):
    result = subprocess.run(["ollama", "run", "mistral"], input=prompt.encode(), capture_output=True)
    return result.stdout.decode()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze()
        probs = F.softmax(logits, dim=0)
    
    confidence = probs.tolist()
    
    pred_class = torch.argmax(probs).item()
    label_map = {0: "Alive", 1: "Dead"}
    status = label_map[pred_class]

    entropy = -torch.sum(probs * torch.log(probs)).item()

    sorted_probs, _ = torch.sort(probs, descending=True)
    margin = (sorted_probs[0] - sorted_probs[1]).item()
    
    return status, confidence, entropy, margin, logits.tolist()

#if __name__ == "__main__":
    # history_text = "Player 456, Seong Gi-hun, was a down-on-his-luck chauffeur who struggled with gambling debts and a broken family. After being invited to play children's games for a large cash prize, he initially hesitated but eventually joined the deadly competition. Throughout the games, Gi-hun displayed a mix of cunning and compassion, forming alliances and making difficult choices to survive. His journey was marked by moments of moral conflict, especially when faced with the harsh realities of the game. Ultimately, Gi-hun's determination and strategic thinking led him to victory, but at a great personal cost."
    
    # status, confidence, entropy, margin, logits = predict_status(history_text)

    # print(f"\n🧠 Predicted status: {status}")
    # print(f"📊 Confidence: {confidence}")
    # print(f"📈 Entropy: {entropy:.4f}")
    # print(f"📉 Confidence margin: {margin:.4f}")
    # print(f"📤 Raw logits: {logits}")
    
    # prompt = f"Here is a character history:\n\"{history_text}\"\nBased on this, the model predicted the character is '{status}'. Why would this be the case? Explain in a short paragraph."

    # explanation = query_ollama(prompt)
    # print(f"\n💡 Reasoning:\n{explanation}")