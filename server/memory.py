### Memory Compression System ###
# Handles cold memory compression and prompt sandwich assembly.
# Compression fires after /chat saves, never blocks a response.

### Imports ###
import json
from pathlib import Path

### Config ###
COMPRESSION_THRESHOLD = 50   # Raw turns before compression fires
COMPRESSION_TEMP = 0.7       # Lower temp for consistent summaries
MEMORY_PATH = Path("../logs/chat_log.json")

COMPRESSOR_PROMPT = (
    "Summarize the following conversation between Vesi and Arskaz in a single short "
    "narrative paragraph in Vesi's voice. Focus on what happened and how Vesi felt, "
    "without admitting she cared. Do not invent events. No bullet points. No preamble. "
    "Write only the summary itself, nothing else."
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
    Returns the oldest 50 raw user/assistant turns for compression.
    Skips system messages and existing compressed blocks.
    """
    raw_turns = [entry for entry in history if _is_raw_turn(entry)]
    return raw_turns[:COMPRESSION_THRESHOLD]


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

    # Build conversation text for the compressor
    conversation_text = "\n".join(
        f"{t['role'].capitalize()}: {t['content']}" for t in turns_to_compress
    )

    # Determine turn range from history indices
    indices = [i for i, e in enumerate(history) if _is_raw_turn(e)]
    turn_range = [indices[0], indices[len(turns_to_compress) - 1]]

    # Call Llama for compression
    # Assistant prefill forces model to start summary directly, skipping preamble
    try:
        completion = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": COMPRESSOR_PROMPT},
                {"role": "user", "content": conversation_text},
                {"role": "assistant", "content": "Arskaz came to me"}
            ],
            temperature=COMPRESSION_TEMP,
            max_tokens=300,
            repeat_penalty=1.1,
        )
        summary = "Arskaz came to me " + completion["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"--- Compression failed: {e} ---")
        return history

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


def build_messages(history: list, reinforcement: dict) -> list:
    """
    Assembles the full prompt sandwich to send to Llama.
    Structure: [system prompt] + [compressed blocks] + [hot turns] + [reinforcement]
    Strips the type field from compressed blocks before sending.
    Hot turns = last 6 raw user/assistant turns.
    """
    system_prompt = history[0]

    compressed_blocks = [
        _strip_type_field(e) for e in history if _is_compressed_block(e)
    ]

    raw_turns = [e for e in history if _is_raw_turn(e)]
    hot_turns = raw_turns[-6:] 

    return [system_prompt] + compressed_blocks + hot_turns + [reinforcement]


def save_memory(history_data: list):
    """Saves history to memory json."""
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=4)