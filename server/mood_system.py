
### Mood System ###
# Score represents Vesi's affection/dere level toward User
# 0-30  = Full tsun (cold, sharp)
# 31-70 = Default smug Vesi
# 71-100 = Dere mode (flustered, softer)
# Score persists per session, resets on restart

### Imports ###
import re


# --- Vesi Response Word Lists ---

# Tsun signals in Vesi's response -> LOWER score (she's being cold)
VESI_TSUN_STRONG = {
    "baka", "idiot", "stupid", "dummy", "pathetic"
}  # -10 each

VESI_TSUN_WEAK = {
    "tch", "annoying", "whatever", "buzz", "off", "tsk", "hmph"
}  # -5 each

# Dere signals leaking through Vesi's response -> RAISE score
VESI_DERE_STRONG = {
    "i-it's", "h-hey", "j-just", "i-i", "y-you"
}  # +10 each (stuttering = peak flustered)

VESI_DERE_WEAK = {
    "suppose", "tolerable", "maybe", "fine", "acceptable", "once"
}  # +5 each

# --- User Input Word Lists ---

# User being sweet/kind -> RAISE score
USER_KIND_STRONG = {
    "thank you", "appreciate", "i like you", "you're great",
    "good job", "well done", "love you"
}  # +10 each (phrase match)

USER_KIND_WEAK = {
    "please", "goodnight", "thanks", "nice", "cute", "good"
}  # +5 each

# User being dismissive/rude -> LOWER score
USER_RUDE_STRONG = {
    "shut up", "go away", "don't care", "i hate"
}  # -10 each (phrase match)

USER_RUDE_WEAK = {
    "boring", "whatever", "annoying", "useless"
}  # -5 each


def calculate_mood(vesi_text: str, user_text: str, current_score: int) -> int:
    """
    Calculates mood score based on both Vesi's response and User's input.
    Phrase matching for multi-word signals, token matching for single words.
    """
    score = current_score
    vesi_lower = vesi_text.lower()
    user_lower = user_text.lower()
    vesi_tokens = set(re.sub(r'[^\w\s]', '', vesi_lower).split())
    user_tokens = set(re.sub(r'[^\w\s]', '', user_lower).split())

    # --- Vesi response scoring ---
    for word in VESI_TSUN_STRONG:
        if word in vesi_tokens:
            score -= 10

    for word in VESI_TSUN_WEAK:
        if word in vesi_tokens:
            score -= 5

    for word in VESI_DERE_STRONG:
        if word in vesi_lower:
            score += 10

    for word in VESI_DERE_WEAK:
        if word in vesi_tokens:
            score += 5

    # --- User input scoring ---
    for phrase in USER_KIND_STRONG:
        if phrase in user_lower:
            score += 10

    for word in USER_KIND_WEAK:
        if word in user_tokens:
            score += 5

    for phrase in USER_RUDE_STRONG:
        if phrase in user_lower:
            score -= 10

    for word in USER_RUDE_WEAK:
        if word in user_tokens:
            score -= 5

    return max(0, min(100, score))



def get_temperature(score: int) -> float:
    """Maps mood score to LLM temperature."""
    if score <= 30:
        return 0.75   # Cold, controlled tsun
    elif score <= 70:
        return 0.85   # Default smug Vesi
    else:
        return 0.95   # Flustered dere, slightly unpredictable


def get_reinforcement(score: int) -> dict:
    """
    Returns reinforcement system prompt based on mood score.
    
    You can change this to your liking or just remove the system. 
    """
    if score > 70:
        mood_note = "You are feeling unusually soft towards Arskaz. Be shy and clumsy but keep your pride. Stutter slightly."
    elif score < 30:
        mood_note = "You are cold and sharp. Use 'Baka!' freely. Short, cutting responses. No warmth whatsoever."
    else:
        mood_note = "Be your usual smug, bossy self. You're the AI in charge here."

    return {
        "role": "system",
        "content": f"REMINDER: {mood_note} Use short, punchy responses. You are Vesi."
    }