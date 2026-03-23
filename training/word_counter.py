import json
from collections import Counter
import re

with open("vesi_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

text = ""
for convo in data:
    for msg in convo["conversations"]:
        if msg["role"] == "assistant":
            text += msg["content"] + " "

watch_words = [
    "night", "goodnight", "tea", "evening", "morning",
    "session data", "processing load", "baka", "hmph",
    "browser history", "fox coins", "heating", "idiot",
    "stupid", "shutdown", "sleep"
]

text_lower = text.lower()
print("=== WORD / PHRASE COUNTS (assistant only) ===\n")
for word in watch_words:
    count = text_lower.count(word.lower())
    print(f"  {word:<20} {count}")

print("\n=== TOP 30 MOST COMMON WORDS ===\n")
words = re.findall(r"\b[a-z]{4,}\b", text_lower)
counter = Counter(words)
for word, count in counter.most_common(30):
    print(f"  {word:<20} {count}")