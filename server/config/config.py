### Central Configuration ###
# Every tunable backend value lives here.
#
# Machine-specific values (model path) read from the environment first, so you
# can keep them in a local .env instead of editing this file. See .env.example.

### Imports ###
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# Anchored to server/, so the backend runs from any working directory
BASE_DIR = Path(__file__).resolve().parent.parent


### Identity ###
# Change these two and the names propagate to the system prompt, stop tokens,
# role-leak filters, STT hints, mood hints and memory summaries.

USER_NAME = "Arskaz"    # What Vesi calls you. Fills {user} in vesi_config.yaml
AI_NAME = "Vesi"        # The AI's name. Fills {ai} in vesi_config.yaml


def render_names(text: str) -> str:
    """
    Substitutes {user} and {ai} placeholders with the configured names.
    Uses replace() rather than format() on purpose — user facts added via
    /remember are arbitrary text and a stray brace must not raise.
    """
    return text.replace("{user}", USER_NAME).replace("{ai}", AI_NAME)


### Paths ###

MODEL_PATH = Path(os.getenv(                            # GGUF LLM weights
    "VESI_MODEL_PATH",
    r"D:/models/lumi_vesi_v2.0_q6k.gguf",
))
MEMORY_PATH = Path(os.getenv(                           # Chat history JSON
    "VESI_MEMORY_PATH",
    str(BASE_DIR.parent / "logs" / "chat_log.json"),
))
CONFIG_PATH = BASE_DIR / "vesi_config.yaml"             # Personality YAML (gitignored)
STATIC_DIR = BASE_DIR / "static"                        # Generated TTS wavs land here
VOICES_DIR = BASE_DIR / "voices"                        # Kokoro model files
KOKORO_MODEL = VOICES_DIR / "kokoro-v0_19.onnx"         # Kokoro ONNX weights
KOKORO_VOICES = VOICES_DIR / "voices-v1.0.bin"          # Kokoro voice embeddings


### Server ###

HOST = "0.0.0.0"                                    # Interface uvicorn binds to
PORT = 8000                                         # Backend port
PUBLIC_HOST = "127.0.0.1"                           # Host the browser uses to reach us
PUBLIC_BASE_URL = f"http://{PUBLIC_HOST}:{PORT}"    # Prefix for returned audio URLs
STATIC_MOUNT = "/static"                            # URL path the static dir mounts at
CORS_ORIGINS = ["*"]                                # Allowed frontend origins


### Logging ###
# Logging is always on. INFO is deliberately minimal: startup, one line per
# request, warnings and errors. DEBUG adds stage timings and dumps the full
# prompt, response and mood detail.
# Flip to debug without editing this file:  set VESI_LOG_LEVEL=DEBUG

LOG_LEVEL = os.getenv("VESI_LOG_LEVEL", "INFO").strip().upper()   # INFO or DEBUG
if LOG_LEVEL not in ("DEBUG", "INFO", "WARNING", "ERROR"):
    LOG_LEVEL = "INFO"                                      # Ignore typos rather than crash
DEBUG_MODE = LOG_LEVEL == "DEBUG"                           # Convenience flag

# Truthy forms so a .env can say 1 / true / yes / on
LOG_TO_FILE = os.getenv("VESI_LOG_FILE", "0").strip().lower() in ("1", "true", "yes", "on")
LOG_DIR = BASE_DIR.parent / "logs"                          # Same folder as chat_log.json
LOG_FILE_PREFIX = "vesi"                                    # logs/vesi_<timestamp>.log

LOG_FORMAT = "%(asctime)s %(levelname)-5s [%(name)s] %(message)s"
LOG_TIME_FORMAT = "%H:%M:%S"
LOG_PREVIEW_CHARS = 80          # Chars of text shown at INFO. DEBUG logs it in full
LOG_QUIET_THIRD_PARTY = True    # Silence uvicorn access lines + llama-cpp unless DEBUG

# Third-party noise follows the debug switch
UVICORN_LOG_LEVEL = "debug" if DEBUG_MODE else "warning"
UVICORN_ACCESS_LOG = DEBUG_MODE                             # Per-request access lines


### LLM ###

CHAT_FORMAT = "chatml"      # Prompt template llama-cpp applies
N_CTX = 12288               # Context window in tokens
N_GPU_LAYERS = 99           # Layers offloaded to GPU. Lower this if VRAM is tight
LLM_VERBOSE = DEBUG_MODE    # llama-cpp perf spam — follows LOG_LEVEL

TOP_K = 40                  # Sample from the N most likely tokens
TOP_P = 0.85                # Nucleus sampling cutoff
MIN_P = 0.05                # Drop tokens below this share of the top token
REPEAT_PENALTY = 1.2        # Push the model toward varied wording
MAX_TOKENS = 100            # Response yap cap


### Speech to text (Faster Whisper) ###

WHISPER_MODEL_SIZE = "base"         # tiny / base / small / medium / large-v3
WHISPER_DEVICE = "cuda"             # "cuda" or "cpu"
WHISPER_COMPUTE_TYPE = "float16"    # float16 on GPU, int8 on CPU
WHISPER_BEAM_SIZE = 5               # Higher = more accurate, slower
WHISPER_LANGUAGE = "en"             # Forced transcription language
WHISPER_TASK = "transcribe"         # "transcribe" or "translate"

# Vocabulary hint so Whisper spells the names right instead of guessing
WHISPER_INITIAL_PROMPT = (
    f"{AI_NAME} is a girl's name. {USER_NAME} is the user. "
    f"{AI_NAME}, baka, hmph, smug, {USER_NAME}."
)


### Text to speech (Kokoro) ###

TTS_VOICE = "af_bella"          # Kokoro voice id
TTS_LANG = "en-us"              # Kokoro language code
AUDIO_CHANNELS = 1              # Mono
AUDIO_SAMPLE_WIDTH = 2          # Bytes per sample (2 = 16-bit)
AUDIO_SCALE = 32767             # Float -> int16 scaling factor
AUDIO_FILENAME_PREFIX = "vesi"  # Generated files are <prefix>_<timestamp>.wav
CLEAN_OLD_AUDIO = True          # Delete previous wavs on each /chat


### Memory ###

COMPRESSION_THRESHOLD = 60      # Raw turns before compression fires
KEEP_RECENT = 10                # Raw turns left untouched by compression
HOT_TURNS = 6                   # Recent turns sent verbatim to the LLM
MAX_COMPRESSED_BLOCKS = 4       # Newest memory blocks included in the prompt
TRUNCATE_FIRST = 120            # Chars kept from the opening line of a block
TRUNCATE_LAST = 100             # Chars kept from the closing exchange of a block
JSON_INDENT = 4                 # Indent for chat_log.json

# Separates recalled memories from the live conversation
SESSION_BREAK_TEXT = (
    "--- PAST MEMORIES END ---\n"
    "The above are memories from previous conversations, for background reference only.\n"
    "The CURRENT conversation starts now. Respond only to what follows."
)


### Mood ###
# Score runs 0-100. Low = cold tsun, high = flustered dere.
# The tsun/dere word lists themselves live in mood_system.py as character data.

MOOD_INITIAL = 45           # Score at startup — matches baseline so turn 1 doesn't jump
MOOD_BASELINE = 45          # Resting mood the score decays back toward
MOOD_DECAY_RATE = 3         # Points pulled toward baseline each turn
MOOD_STRONG_WEIGHT = 10     # Points per strong tsun/dere/kind/rude signal
MOOD_WEAK_WEIGHT = 5        # Points per weak signal
MOOD_MIN = 0                # Score floor
MOOD_MAX = 100              # Score ceiling

MOOD_TSUN_MAX = 30          # Score <= this is "tsun"
MOOD_DERE_MIN = 70          # Score > this is "dere", between the two is "neutral"

TEMP_TSUN = 0.45            # Cold and controlled
TEMP_NEUTRAL = 0.55         # Default smug
TEMP_DERE = 0.70            # Flustered, less predictable

TTS_SPEED_TSUN = 1.35       # Curt, clipped delivery
TTS_SPEED_NEUTRAL = 1.25    # Default pace
TTS_SPEED_DERE = 1.05       # Softer, slightly hesitant

# Injected into the system prompt so the model knows its current state.
# Keys must match the strings mood_system.get_emotion() returns.
MOOD_HINTS = {
    "tsun":    "[{ai} is currently in a cold, irritated mood.]",
    "neutral": "[{ai} is in her usual smug, composed mood.]",
    "dere":    "[{ai} is currently feeling flustered and softer than usual.]",
}


### Prompt labels ###

USER_FACTS_HEADER = "USER FACTS:"   # Precedes the fact list in the system prompt
CONTEXT_HEADER = "CONTEXT:"         # Precedes injected tool output
MEMORY_PREFIX = "MEMORY:"           # Precedes a compressed memory block


### Role-leak filtering ###
# Chat templates and role prefixes that mean the model stopped being the
# character and started writing the user's or system's turn.
# Matching is case-insensitive, so only one casing of each is needed.

_CHAT_TEMPLATE_TOKENS = [
    r"<\|im_end\|>",
    r"<\|im_start\|>",
    r"<\|eot_id\|>",
    r"<\|end_of_text\|>",
    r"<user",
    r"<system",
    r"<\|user",
    r"<\|system",
]

# Everything from the first match onward is cut from the response
ROLE_LEAK_PATTERNS = _CHAT_TEMPLATE_TOKENS + [
    rf"<\|{USER_NAME}",
    r"\nUser:",
    rf"\n{USER_NAME}:",
    r"\nSystem:",
]

# Passed to llama-cpp so generation halts before the leak happens.
# Only role-prefix forms of the user name — a bare name would truncate any
# reply that simply mentions you.
STOP_TOKENS = [
    "<|eot_id|>",
    "<|im_end|>",
    "<|end_of_text|>",
    "<|im_start|>",
    "<user>",
    "<|user",
    "<|User>",
    "\n<user>",
    f"<|{USER_NAME.lower()}>",
    "User:",
    "user:",
    "\nUser:",
    "\nuser:",
    f"{USER_NAME}:",
    f"{USER_NAME.lower()}:",
]
