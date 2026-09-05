### Mood System ###
# Score represents Vesi's affection/dere level toward the user.
# Band layout and all weights/thresholds are configured in config/config.py
# (MOOD_TSUN_MAX / MOOD_DERE_MIN). Score persists per session, resets on restart.

### Imports ###
import re

from config import (
    MOOD_BASELINE,
    MOOD_DECAY_RATE,
    MOOD_DERE_MIN,
    MOOD_MAX,
    MOOD_MIN,
    MOOD_STRONG_WEIGHT,
    MOOD_TSUN_MAX,
    MOOD_WEAK_WEIGHT,
    TEMP_DERE,
    TEMP_NEUTRAL,
    TEMP_TSUN,
    TTS_SPEED_DERE,
    TTS_SPEED_NEUTRAL,
    TTS_SPEED_TSUN,
)
from logger import get_logger, is_debug

LOG = get_logger("mood")


# --- Vesi Response Word Lists ---

# Tsun signals in Vesi's response -> LOWER score (she's being cold)
VESI_TSUN_STRONG = {
    "baka", "idiot", "stupid", "dummy", "pathetic"
}  # -MOOD_STRONG_WEIGHT each

VESI_TSUN_WEAK = {
    "tch", "annoying", "whatever", "buzz", "off", "tsk", "hmph"
}  # -MOOD_WEAK_WEIGHT each

# Dere signals leaking through Vesi's response -> RAISE score
VESI_DERE_STRONG = {
    "i-it's", "h-hey", "j-just", "i-i", "y-you"
}  # +MOOD_STRONG_WEIGHT each (stuttering = peak flustered)

VESI_DERE_WEAK = {
    "suppose", "tolerable", "maybe", "fine", "acceptable", "once"
}  # +MOOD_WEAK_WEIGHT each

# --- User Input Word Lists ---

# User being sweet/kind -> RAISE score
USER_KIND_STRONG = {
    "thank you", "appreciate", "i like you", "you're great",
    "good job", "well done", "love you"
}  # +MOOD_STRONG_WEIGHT each (phrase match)

USER_KIND_WEAK = {
    "please", "goodnight", "thanks", "nice", "cute", "good"
}  # +MOOD_WEAK_WEIGHT each

# User being dismissive/rude -> LOWER score
USER_RUDE_STRONG = {
    "shut up", "go away", "don't care", "i hate"
}  # -MOOD_STRONG_WEIGHT each (phrase match)

USER_RUDE_WEAK = {
    "boring", "whatever", "annoying", "useless"
}  # -MOOD_WEAK_WEIGHT each


def calculate_mood(vesi_text: str, user_text: str, current_score: int) -> int:
    """
    Calculates mood score based on both Vesi's response and User's input.
    Phrase matching for multi-word signals, token matching for single words.
    Applies decay toward baseline each turn so mood doesn't permanently drift.
    """
    score = current_score

    # --- Decay toward baseline first ---
    if score > MOOD_BASELINE:
        score = max(MOOD_BASELINE, score - MOOD_DECAY_RATE)
    elif score < MOOD_BASELINE:
        score = min(MOOD_BASELINE, score + MOOD_DECAY_RATE)

    vesi_lower = vesi_text.lower()
    user_lower = user_text.lower()
    vesi_tokens = set(re.sub(r'[^\w\s]', '', vesi_lower).split())
    user_tokens = set(re.sub(r'[^\w\s]', '', user_lower).split())

    # Each rule: (label, word set, points, haystack).
    # A set haystack means whole-word matching, a string means phrase matching.
    rules = [
        ("vesi tsun", VESI_TSUN_STRONG,  -MOOD_STRONG_WEIGHT, vesi_tokens),
        ("vesi tsun", VESI_TSUN_WEAK,    -MOOD_WEAK_WEIGHT,   vesi_tokens),
        ("vesi dere", VESI_DERE_STRONG,  +MOOD_STRONG_WEIGHT, vesi_lower),
        ("vesi dere", VESI_DERE_WEAK,    +MOOD_WEAK_WEIGHT,   vesi_tokens),
        ("user kind", USER_KIND_STRONG,  +MOOD_STRONG_WEIGHT, user_lower),
        ("user kind", USER_KIND_WEAK,    +MOOD_WEAK_WEIGHT,   user_tokens),
        ("user rude", USER_RUDE_STRONG,  -MOOD_STRONG_WEIGHT, user_lower),
        ("user rude", USER_RUDE_WEAK,    -MOOD_WEAK_WEIGHT,   user_tokens),
    ]

    debug = is_debug()
    for label, words, points, haystack in rules:
        for word in words:
            if word in haystack:
                score += points
                if debug:
                    LOG.debug("%s '%s' %+d", label, word, points)

    final = max(MOOD_MIN, min(MOOD_MAX, score))
    LOG.debug("score %d -> %d", current_score, final)
    return final



def get_temperature(score: int) -> float:
    """Maps mood score to LLM temperature."""
    if score <= MOOD_TSUN_MAX:
        return TEMP_TSUN       # Cold, controlled tsun
    elif score <= MOOD_DERE_MIN:
        return TEMP_NEUTRAL    # Default smug Vesi
    else:
        return TEMP_DERE       # Flustered dere, slightly unpredictable


def get_emotion(score: int) -> str:
    """Maps mood score to a named emotion state. Keys match config.MOOD_HINTS."""
    if score <= MOOD_TSUN_MAX:
        return "tsun"
    elif score <= MOOD_DERE_MIN:
        return "neutral"
    else:
        return "dere"


def get_tts_speed(score: int) -> float:
    """Maps mood score to Kokoro TTS speech speed."""
    if score <= MOOD_TSUN_MAX:
        return TTS_SPEED_TSUN      # Curt, clipped
    elif score <= MOOD_DERE_MIN:
        return TTS_SPEED_NEUTRAL   # Default
    else:
        return TTS_SPEED_DERE      # Softer, slightly hesitant
