### Memory Compression System ###
# Handles cold memory compression and prompt sandwich assembly.
# Compression fires after /chat saves

### Imports ###
import json
from pathlib import Path

### Config ###
COMPRESSION_THRESHOLD = 60   # Raw turns before compression fires
KEEP_RECENT = 10             # Raw turns to preserve after compression
MEMORY_PATH = Path("../logs/chat_log.json")

COMPRESSOR_PROMPT = (
    "You are a factual note-taker, NOT a character. Do NOT write as Vesi. "
    "Do NOT write in first person. Do NOT invent events that did not happen. "
    "Summarize ONLY what was explicitly said in the conversation below. "
    "Use third-person, past tense, factual tone. One paragraph, 2-3 sentences max. "
    "Format: 'Arskaz and Vesi discussed [topics]. [Key decisions or facts shared].' "
    "If unsure about something, omit it rather than guess."
)


### Helper functions ###

def _is_raw_turn(entry: dict) -> bool:
    """Returns True if entry is a raw uncompressed user or assistant turn."""
    return entry.get("role") in ("user", "assistant") and "type" not in entry


def _is_compressed_block(entry: dict) -> bool:
    """Returns True if entry is a compressed memory block."""
    return entry.get("type") == "compressed_block"


def _strip_type_field(entry: dict) -> dict:
    """Returns entry without the type field — safe to send to Llama."""
    return {k: v for k, v in entry.items() if k != "type"}


def _truncate(text: str, max_chars: int = 120) -> str:
    """Truncate at sentence boundary, fall back to word boundary. Adds … when cut."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    # Try sentence boundary (. ! ?) in the allowed region
    region = text[:max_chars]
    for punct in (". ", "! ", "? "):
        pos = region.rfind(punct)
        if pos > max_chars // 3:          # don't cut absurdly short
            return text[: pos + 1]
    # Fall back to word boundary
    last_space = region.rfind(" ")
    if last_space > max_chars // 3:
        return text[:last_space] + "…"
    return region + "…"


### Core functions ###

def should_compress(history: list) -> bool:
    """
    Returns True if raw uncompressed turns exceed the threshold.
    Only counts user/assistant turns, ignores system and compressed blocks.
    """
    raw_count = sum(1 for entry in history if _is_raw_turn(entry))
    return raw_count >= COMPRESSION_THRESHOLD


def get_compressible_turns(history: list) -> list:
    """
    Returns the oldest raw turns for compression, keeping
    the most recent KEEP_RECENT turns untouched.
    """
    raw_turns = [entry for entry in history if _is_raw_turn(entry)]
    compress_count = len(raw_turns) - KEEP_RECENT
    if compress_count <= 0:
        return []
    return raw_turns[:compress_count]


def compress(history: list, llm) -> list:
    """
    Compresses the oldest 50 raw turns into a single compressed block.
    Removes original turns from history, inserts the block, saves to disk.
    Returns the updated history.
    """
    turns_to_compress = get_compressible_turns(history)

    if not turns_to_compress:
        return history

    print(f"--- Compressing {len(turns_to_compress)} turns into a memory block ---")

    # Determine turn range from history indices
    indices = [i for i, e in enumerate(history) if _is_raw_turn(e)]
    turn_range = [indices[0], indices[len(turns_to_compress) - 1]]

    # Rule-based extraction: no LLM call, zero hallucination risk.
    # Extracts actual text from the conversation rather than generating new text.
    first_user = next((t for t in turns_to_compress if t["role"] == "user"), None)
    last_pair = [t for t in turns_to_compress[-4:] if t["role"] in ("user", "assistant")][-2:]

    parts = [f"[{len(turns_to_compress)} turns compressed]"]
    if first_user:
        parts.append(f'Started with: "{_truncate(first_user["content"], 120)}"')
    if len(last_pair) == 2:
        parts.append(
            f'Ended with user: "{_truncate(last_pair[0]["content"], 100)}" / '
            f'Vesi: "{_truncate(last_pair[1]["content"], 100)}"'
        )

    summary = " ".join(parts)

    # Build compressed block
    compressed_block = {
        "role": "system",
        "type": "compressed_block",
        "turn_range": turn_range,
        "content": f"MEMORY: {summary}"
    }

    # Remove the original turns from history
    compressed_set = set(id(t) for t in turns_to_compress)
    history = [e for e in history if id(e) not in compressed_set]

    # Insert compressed block after the main system prompt (index 0)
    # but before any existing compressed blocks and hot turns
    insert_at = 1
    history.insert(insert_at, compressed_block)

    save_memory(history)

    print(f"--- Compression complete. Summary: {summary[:80]}... ---")
    return history


def build_messages(history: list) -> list:
    """
    Assembles the prompt to send to Llama.
    Structure: [system prompt] + [compressed blocks] + [session break] + [hot turns]
    Strips the type field from compressed blocks before sending.
    Hot turns = last 6 raw user/assistant turns.
    """
    system_prompt = history[0]

    MAX_COMPRESSED_BLOCKS = 4
    compressed_blocks = [
        _strip_type_field(e) for e in history if _is_compressed_block(e)
    ]
    
    # Changed to only keep recent blocks to keep in context better
    compressed_blocks = compressed_blocks[-MAX_COMPRESSED_BLOCKS:]

    raw_turns = [e for e in history if _is_raw_turn(e)]
    hot_turns = raw_turns[-6:]

    session_break = {
        "role": "system",
        "content": (
            "--- PAST MEMORIES END ---\n"
            "The above are memories from previous conversations, for background reference only.\n"
            "The CURRENT conversation starts now. Respond only to what follows."
        )
    }

    return [system_prompt] + compressed_blocks + [session_break] + hot_turns


def save_memory(history_data: list):
    """Saves history to memory json."""
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=4)