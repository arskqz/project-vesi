### Imports ###
import os
import json
import re
import wave
import time
import tempfile
import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
from pydantic import BaseModel
from llama_cpp import Llama
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro

from mood_system import calculate_mood, get_temperature, get_emotion, get_tts_speed
from memory import should_compress, compress, build_messages, save_memory
from tools import get_passive_context, run_active_tools

# Every tunable lives in config/config.py 
from config import (
    AI_NAME, USER_NAME, render_names,
    MODEL_PATH, MEMORY_PATH, CONFIG_PATH, STATIC_DIR, KOKORO_MODEL, KOKORO_VOICES,
    HOST, PORT, PUBLIC_BASE_URL, STATIC_MOUNT, CORS_ORIGINS,
    CHAT_FORMAT, N_CTX, N_GPU_LAYERS, LLM_VERBOSE,
    TOP_K, TOP_P, MIN_P, REPEAT_PENALTY, MAX_TOKENS,
    WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE,
    WHISPER_BEAM_SIZE, WHISPER_LANGUAGE, WHISPER_TASK, WHISPER_INITIAL_PROMPT,
    TTS_VOICE, TTS_LANG, AUDIO_CHANNELS, AUDIO_SAMPLE_WIDTH, AUDIO_SCALE,
    AUDIO_FILENAME_PREFIX, CLEAN_OLD_AUDIO,
    JSON_INDENT, MOOD_INITIAL, MOOD_HINTS,
    USER_FACTS_HEADER, CONTEXT_HEADER,
    ROLE_LEAK_PATTERNS, STOP_TOKENS,
    LOG_LEVEL, UVICORN_LOG_LEVEL, UVICORN_ACCESS_LOG,
)
from logger import setup_logging, get_logger, is_debug, preview

LOG = get_logger("main")     # startup and config
CHAT = get_logger("chat")    # per-request lines

### Setup ###
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

# Allow Frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(STATIC_MOUNT, StaticFiles(directory=str(STATIC_DIR)), name="static")

### Globals ###
llm = None
stt_model = None
vocal_cord = None
vesi_mood_score = MOOD_INITIAL
history = []

class ChatRequest(BaseModel):
    message: str

class RememberRequest(BaseModel):
    fact: str

### Config helpers ###
def load_config() -> dict:
    """Loads vesi_config.yaml. Crashes loudly if missing — it should always exist."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"vesi_config.yaml not found at {CONFIG_PATH}. Please create it.")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_config(yaml_cfg: dict):
    """Saves updated config back to vesi_config.yaml."""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(yaml_cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

def build_system_prompt(yaml_cfg: dict) -> str:
    """
    Builds the full system prompt string from config.
    {user} and {ai} placeholders are filled from config/config.py.
    """
    base_prompt = yaml_cfg["system_prompt"].strip()
    facts = yaml_cfg.get("user_facts", [])

    if facts:
        facts_block = "\n".join(f"- {fact}" for fact in facts)
        prompt = f"{base_prompt}\n\n{USER_FACTS_HEADER}\n{facts_block}"
    else:
        prompt = base_prompt

    return render_names(prompt)

### Helper functions ###

# Patterns that signal Vesi stopped talking and "became" the user or system.
# Built from USER_NAME in config so renaming stays a one-line change.
_ROLE_LEAK_RE = re.compile("|".join(ROLE_LEAK_PATTERNS), re.IGNORECASE)

def _ms(started: float) -> int:
    """Milliseconds elapsed since a time.perf_counter() mark."""
    return int((time.perf_counter() - started) * 1000)


def clean_response(text: str) -> str:
    """Strip everything from the first role-leak pattern onward."""
    match = _ROLE_LEAK_RE.search(text)
    if match:
        CHAT.debug("role leak cut at %r (-%d chars)",
                   match.group(0), len(text) - match.start())
        text = text[:match.start()]
    return text.strip()


def load_memory() -> list:
    """
    Loads history from file or creates it if missing.
    Always overwrites history[0] with the current YAML config.
    YAML is always the source of truth for the system prompt.
    """
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    yaml_cfg = load_config()
    system_prompt_content = build_system_prompt(yaml_cfg)
    system_message = {"role": "system", "content": system_prompt_content}

    if MEMORY_PATH.exists():
        try:
            with open(MEMORY_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            # Always overwrite index 0 with fresh config — YAML wins
            if loaded and loaded[0].get("role") == "system":
                loaded[0] = system_message
            else:
                loaded.insert(0, system_message)

            # Save back immediately so JSON stays in sync
            with open(MEMORY_PATH, "w", encoding="utf-8") as f:
                json.dump(loaded, f, indent=JSON_INDENT)

            LOG.info("memory loaded: %d entries", len(loaded))
            LOG.debug("memory file: %s", MEMORY_PATH)
            LOG.debug("system prompt rebuilt from vesi_config.yaml (%d chars)",
                      len(system_prompt_content))
            return loaded

        except Exception:
            # Log the traceback before falling back — this used to fail silently
            LOG.exception("could not load memory, starting fresh")

    # Fresh start
    initial_history = [system_message]
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(initial_history, f, indent=JSON_INDENT)
    LOG.info("created new memory file: %s", MEMORY_PATH)
    return initial_history


### Model initialization ###
def init_models():
    """Starts all the models"""
    global llm, stt_model, vocal_cord, history
    LOG.info("initializing %s", AI_NAME)
    # STT
    LOG.debug("whisper: %s on %s (%s)", WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE)
    stt_model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )
    LOG.info("whisper ready")
    # TTS
    # See --> README_Voices.md for info
    LOG.debug("kokoro: voice=%s lang=%s", TTS_VOICE, TTS_LANG)
    vocal_cord = Kokoro(str(KOKORO_MODEL), str(KOKORO_VOICES))
    LOG.info("kokoro ready")
    # LLM
    LOG.debug("llm: %s", MODEL_PATH)
    LOG.debug("llm: n_ctx=%d n_gpu_layers=%d chat_format=%s", N_CTX, N_GPU_LAYERS, CHAT_FORMAT)
    llm = Llama(
        model_path=str(MODEL_PATH),
        chat_format=CHAT_FORMAT,
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=LLM_VERBOSE,
    )
    LOG.info("llm ready")
    history = load_memory()
    LOG.info("%s is online at %s", AI_NAME, PUBLIC_BASE_URL)

### API Endpoint ###

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe audio to text using Faster Whisper"""

    # Save to tmp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        content = await audio.read()
        temp_audio.write(content)
        temp_path = temp_audio.name

    try:
        started = time.perf_counter()
        segments, info = stt_model.transcribe(
            temp_path,
            beam_size=WHISPER_BEAM_SIZE,
            language=WHISPER_LANGUAGE,
            task=WHISPER_TASK,
            initial_prompt=WHISPER_INITIAL_PROMPT,
        )
        text = " ".join([segment.text for segment in segments]).strip()
        elapsed = _ms(started)

        CHAT.info("transcribed %d chars in %dms", len(text), elapsed)
        CHAT.debug("transcript: %s", text)

        return {"text": text}

    finally:
        os.remove(temp_path)


@app.post("/remember")
async def remember(request: RememberRequest):
    """
    Adds a new user fact to vesi_config.yaml and immediately
    updates history[0] in the live session. No restart needed.
    """
    global history

    fact = request.fact.strip()
    if not fact:
        return {"status": "error", "message": "Empty fact ignored."}

    # Load, update, save config
    yaml_cfg = load_config()
    if "user_facts" not in yaml_cfg:
        yaml_cfg["user_facts"] = []

    yaml_cfg["user_facts"].append(fact)
    save_config(yaml_cfg)

    # Rebuild system prompt and update live history[0] immediately
    system_prompt_content = build_system_prompt(yaml_cfg)
    history[0] = {"role": "system", "content": system_prompt_content}
    save_memory(history)

    LOG.info("remembered fact (%d total)", len(yaml_cfg["user_facts"]))
    LOG.debug("fact: %s", fact)
    return {"status": "ok", "fact": fact, "total_facts": len(yaml_cfg["user_facts"])}


@app.post("/chat")
async def chat(request: ChatRequest):
    global vesi_mood_score, current_temp, history

    # Clean up old audio files
    if CLEAN_OLD_AUDIO:
        for f in os.listdir(STATIC_DIR):
            if f.endswith(".wav"):
                try:
                    os.remove(os.path.join(STATIC_DIR, f))
                except:
                    pass

    request_started = time.perf_counter()

    # Add user message to history
    user_input = request.message
    history.append({"role": "user", "content": user_input})
    CHAT.debug("user input: %s", user_input)

    mood_before = vesi_mood_score
    current_temp = get_temperature(vesi_mood_score)

    # build prompt
    messages_to_send = build_messages(history)

    # Passive tools always inject into system prompt
    passive_ctx = get_passive_context()
    emotion = get_emotion(vesi_mood_score)
    mood_hint = render_names(MOOD_HINTS[emotion])
    messages_to_send[0] = {
        "role": "system",
        "content": messages_to_send[0]["content"] + f"\n\n{CONTEXT_HEADER}\n{passive_ctx}\n\n{mood_hint}"
    }

    # Active tools only when triggered by user input
    active_ctx = run_active_tools(user_input)
    if active_ctx:
        messages_to_send.insert(-1, {"role": "system", "content": f"{CONTEXT_HEADER} {active_ctx}"})


    # The assembled prompt is the single most useful thing to see when the
    # model misbehaves. Guarded so the join is skipped entirely at INFO.
    if is_debug():
        CHAT.debug("temperature %.2f, %d messages:", current_temp, len(messages_to_send))
        for i, msg in enumerate(messages_to_send):
            CHAT.debug("  [%d] %s: %s", i, msg["role"], msg["content"])

    ### LLM
    # Sampling parameters and stop tokens all come from config/config.py
    llm_started = time.perf_counter()
    completion = llm.create_chat_completion(
        messages=messages_to_send,
        temperature=current_temp,
        top_k=TOP_K,
        top_p=TOP_P,
        min_p=MIN_P,
        repeat_penalty=REPEAT_PENALTY,
        max_tokens=MAX_TOKENS,
        stop=STOP_TOKENS,
    )

    llm_ms = _ms(llm_started)

    choice = completion["choices"][0]
    raw_response = choice["message"]["content"]
    full_response = clean_response(raw_response)

    CHAT.debug("llm %dms | finish_reason=%s | raw %d -> clean %d chars",
               llm_ms, choice.get("finish_reason"), len(raw_response), len(full_response))
    CHAT.debug("response: %s", full_response)

    vesi_mood_score = calculate_mood(full_response, user_input, vesi_mood_score)

    # TTS
    tts_started = time.perf_counter()
    samples, sample_rate = vocal_cord.create(
        full_response,
        voice=TTS_VOICE,
        speed=get_tts_speed(vesi_mood_score),
        lang=TTS_LANG
    )
    tts_ms = _ms(tts_started)
    CHAT.debug("tts %dms | %.1fs audio at %dHz",
               tts_ms, len(samples) / sample_rate, sample_rate)

    # Save generated audio to static/
    timestamp = int(time.time())
    audio_filename = f"{AUDIO_FILENAME_PREFIX}_{timestamp}.wav"
    audio_path = os.path.join(STATIC_DIR, audio_filename)

    with wave.open(audio_path, 'wb') as wf:
        wf.setnchannels(AUDIO_CHANNELS)
        wf.setsampwidth(AUDIO_SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes((samples * AUDIO_SCALE).astype(np.int16).tobytes())

    # Add Vesi response
    history.append({"role": "assistant", "content": full_response})
    save_memory(history)

    # Fire compression if raw turn count exceeds threshold
    # Runs after response is sent
    if should_compress(history):
        history = compress(history)

    # The one line INFO mode prints per request
    CHAT.info(
        "done in %dms | in %d -> out %d chars | mood %d->%d %s",
        _ms(request_started), len(user_input), len(full_response),
        mood_before, vesi_mood_score, emotion,
    )
    CHAT.debug("audio: %s", audio_filename)

    return {
        "text": full_response,
        "mood": vesi_mood_score,
        "emotion": emotion,
        "audio_url": f"{PUBLIC_BASE_URL}{STATIC_MOUNT}/{audio_filename}?t={os.urandom(4).hex()}"
    }

@app.on_event("shutdown")
def shutdown_models():
    global llm, stt_model
    LOG.info("shutting down")
    if llm is not None:
        llm.close()
    if stt_model is not None:
        del stt_model

def main():
    setup_logging()
    LOG.debug("log level %s", LOG_LEVEL)
    init_models()
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level=UVICORN_LOG_LEVEL,
        access_log=UVICORN_ACCESS_LOG,
    )

if __name__ == "__main__":
    main()
