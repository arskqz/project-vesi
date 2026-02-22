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
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
from pydantic import BaseModel
from llama_cpp import Llama
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from mood_system import calculate_mood, get_temperature, get_reinforcement


### Config and Paths ###
MODEL_PATH = r"D:\models\Llama-3-Lumimaid-8B-v0.1-OAS-Q6_K-imat.gguf"
MEMORY_PATH = Path("../logs/chat_log.json")
CONFIG_PATH = Path("vesi_config.yaml")
STATIC_DIR = "static"

if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

app = FastAPI()

# Allow Frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

### Globals ###
llm = None
stt_model = None
vocal_cord = None
vesi_mood_score = 50
current_temp = 0.85
history = []

class ChatRequest(BaseModel):
    message: str

class RememberRequest(BaseModel):
    fact: str

### Config helpers ###
def load_config() -> dict:
    """Loads vesi_config.yaml. Crashes loudly if missing — it should always exist."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"vesi_config.yaml not found at {CONFIG_PATH.resolve()}. Please create it.")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_config(config: dict):
    """Saves updated config back to vesi_config.yaml."""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

def build_system_prompt(config: dict) -> str:
    """Builds the full system prompt string from config."""
    base_prompt = config["system_prompt"].strip()
    facts = config.get("user_facts", [])

    if facts:
        facts_block = "\n".join(f"- {fact}" for fact in facts)
        return f"{base_prompt}\n\nUSER FACTS:\n{facts_block}"
    
    return base_prompt

### Helper functions ###
def load_memory() -> list:
    """
    Loads history from file or creates it if missing.
    Always overwrites history[0] with the current YAML config.
    YAML is always the source of truth for the system prompt.
    """
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    config = load_config()
    system_prompt_content = build_system_prompt(config)
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
                json.dump(loaded, f, indent=4)

            print(f"--- Memory loaded from {MEMORY_PATH.resolve()} ---")
            print(f"--- System prompt updated from vesi_config.yaml ---")
            return loaded

        except Exception as e:
            print(f"--- Error loading memory: {e}. Starting fresh ---")

    # Fresh start
    initial_history = [system_message]
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(initial_history, f, indent=4)
    print(f"--- Created new memory file at: {MEMORY_PATH.resolve()} ---")
    return initial_history
        

def save_memory(history_data):
    """Saves history to memory json"""
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=4)

# TODO: Func to input 'What do you want baka!' or similar
# to llm response while it gens next response for instant
# response.

### Model initialization ### 
def init_models():
    """Starts all the models"""
    global llm, stt_model, vocal_cord, history
    print("--- Initializing Vesi ---")
    # STT
    stt_model = WhisperModel("base", device="cuda", compute_type="float16")
    print("--- Faster Whisper Ready ---")
    # TTS
    # See --> README_Voices.md for info
    vocal_cord = Kokoro("voices/kokoro-v0_19.onnx", "voices/voices-v1.0.bin")
    print("--- Kokoro Ready ---")
    # LLM
    llm = Llama(model_path=MODEL_PATH, chat_format="chatml", n_ctx=12288, n_gpu_layers=-1, verbose=False)
    print("--- LLM Ready ---")
    history = load_memory()
    print("--- Vesi is Online ---")

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
        segments, info = stt_model.transcribe(
            temp_path, 
            beam_size=5, 
            language="en", 
            task="transcribe", 
            initial_prompt="Vesi is a girl's name. Arskaz is the user. Vesi, baka, hmph, smug, Arskaz."
        )
        text = " ".join([segment.text for segment in segments])
        
        return {"text": text.strip()}
    
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
    config = load_config()
    if "user_facts" not in config:
        config["user_facts"] = []
    
    config["user_facts"].append(fact)
    save_config(config)

    # Rebuild system prompt and update live history[0] immediately
    system_prompt_content = build_system_prompt(config)
    history[0] = {"role": "system", "content": system_prompt_content}
    save_memory(history)

    print(f"--- Remembered: '{fact}' ---")
    return {"status": "ok", "fact": fact, "total_facts": len(config["user_facts"])}


@app.post("/chat")
async def chat(request: ChatRequest):
    global vesi_mood_score, current_temp, history
    
    # Clean up old audio files
    for f in os.listdir(STATIC_DIR):
        if f.endswith(".wav"):
            try:
                os.remove(os.path.join(STATIC_DIR, f))
            except:
                pass
    
    # Add user message to history
    user_input = request.message
    history.append({"role": "user", "content": user_input})
    
    system_prompt = history[0] 
    recent_history = history[1:][-24:]  
    
    current_reinforcement = get_reinforcement(vesi_mood_score)
    
    # sandwhich prompt
    messages_to_send = [system_prompt] + recent_history + [current_reinforcement]
    
    ### LLM 
    completion = llm.create_chat_completion(
        messages=messages_to_send, 
        temperature=current_temp,
        min_p=0.05,
        repeat_penalty=1.2,
        max_tokens=300,
        stop=[
            "<|eot_id|>",          
            "<|im_end|>",          
            "<|end_of_text|>",      
            "<|im_start|>",         
            "User:", "arskaz:",       
            "user:", "Arskaz:"
        ]
    )

    full_response = completion["choices"][0]["message"]["content"]
    vesi_mood_score = calculate_mood(full_response, user_input, vesi_mood_score)
    
    # TTS
    samples, sample_rate = vocal_cord.create(
        full_response, 
        voice="af_bella", 
        speed=1.25, 
        lang="en-us"
    )
    
    # Save generated audio to static/
    timestamp = int(time.time())
    audio_filename = f"vesi_{timestamp}.wav"
    audio_path = os.path.join(STATIC_DIR, audio_filename)
    
    with wave.open(audio_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((samples * 32767).astype(np.int16).tobytes())
    
    # Add VEsi response
    history.append({"role": "assistant", "content": full_response})
    save_memory(history)

    print(current_temp)
    
    return {
        "text": full_response,
        "mood": vesi_mood_score,
        "audio_url": f"http://localhost:8000/static/{audio_filename}?t={os.urandom(4).hex()}"
    }

def main():
    init_models()
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()