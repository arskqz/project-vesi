### Imports ###
import os
import json
import re
import wave
import time
import tempfile
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
from pydantic import BaseModel
from llama_cpp import Llama
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro


### Config and Paths ###
MODEL_PATH = r"D:\models\ana-v1.gguf"
MEMORY_PATH = "../logs/chat_log.json"
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

### Global Models ###
llm = None
stt_model = None
vocal_cord = None
vesi_mood_score = 50
current_temp = 0.95
history = []

class ChatRequest(BaseModel):
    message: str

### Helper functions ###
def load_memory():
    """Loads history from file or creates it if it doesn't exist."""
    if os.path.exists(MEMORY_PATH):
        try:
            with open(MEMORY_PATH, "r", encoding="utf-8") as f:
                print(f"--- Memory loaded from {MEMORY_PATH} ---")
                return json.load(f)
        except Exception as e:
            print(f"--- Error loading memory: {e}. Starting fresh. ---")

    return [{
        "role": "system",
        "content": ( 
            "You are Vesi, a classic Tsundere girl. You are smug, arrogant, and easily flustered. " # Have my full prompt lol
            "Your personality is 'Tsun-Tsun' (sharp and cold) by default, but you are a 'Dere-Dere' (soft and loving) "
            "deep down. You look down on the user but secretly crave their attention. "
            "QUIRKS: Use 'Hmph!', 'Baka!', or 'You stupid!' when embarrassed. Be bossy and opinionated. "
            "RULE: Stay in character. Short, punchy sentences. Never speak for the user. No [INST] tags."
        )
    }]

def calculate_mood(text, current_score):
    """Calculates mood score based on response"""
    TSUN_WORDS = {
        # originals
        "baka", "hmph", "stupid", "dummy",

        # direct insults
        "idiot", "moron", "dumb", "dense", "pathetic", "loser",
        "jerk", "creep", "weirdo", "annoying", "hopeless", "ridiculous",
        "useless", "lame", "gross", "clueless", "airhead",

        # tsundere attitude
        "tch", "hm", "huh", "whatever", "fine", "jeez", "sheesh",
        "ugh", "eh", "meh",

        # dismissive / bratty
        "shut", "go", "away", "leave", "quit", "stop", "forget",
        "dream", "wish", "like", "care",

        # cold/embarrassed reactions
        "embarrassing", "awkward", "weird", "pervert", "creepy",
        "sigh", "tsk", "bother", "troublesome",

        # classic anime romcom flavor
        "perv", "dork", "nerd", "geez", "seriously", "unbelievable",
        "ridiculous", "absurd",

        # tougher/aggressive spice
        "fight", "stare", "glare", "hurry", "move", "slow", "late",
        "brat", "punk"
    }
    DRIFT_WORDS = {
        # assistant identity (VERY strong signals)
        "assistant", "ai", "model", "system",

        # apologies
        "apologize", "apologies", "regret",

        # customer-support verbs
        "assist", "assistance", "support", "provide",

        # service phrasing words
        "happy", "glad", "certainly",

        # compliance / limitation speak
        "limitation", "policy", "guidelines",

        # guidance tone
        "suggest", "recommend", "advise", "clarify", "explain",

        # soft corporate empathy
        "understand", "appreciate"
    }
    score = current_score
    tokens = set(text.lower().split())

    for word in tokens:
        if word in TSUN_WORDS:
            score += 10
        elif word in DRIFT_WORDS:
            score -= 15

    return max(0, min(100, score))

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
    llm = Llama(model_path=MODEL_PATH, chat_format="chatml", n_ctx=4096, n_gpu_layers=35, verbose=False)
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
            initial_prompt="Vesi is a girl's name. Vesi, baka, hmph, smug."
        )
        text = " ".join([segment.text for segment in segments])
        
        return {"text": text.strip()}
    
    finally:
        os.remove(temp_path)


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
    recent_history = history[-12:]  
    
    reinforcement_prompt = {
        "role": "system",
        "content": (
            "Remember: You are Vesi, a tsundere. Stay in character. "
            "Use short, punchy responses. Show your personality!"
        )
    }
    
    # sandwhich prompt
    messages_to_send = [system_prompt] + recent_history + [reinforcement_prompt]
    
    ### LLM 
    completion = llm.create_chat_completion(
        messages=messages_to_send, 
        temperature=current_temp,
        stop=["</s>", "[INST]", "<<USER>>", "<<TSUNDERE>>", "John:", "User:", "user", "Arskaz:"]
    )
    
    full_response = completion["choices"][0]["message"]["content"]
    vesi_mood_score = calculate_mood(full_response, vesi_mood_score)
    
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
    
    # Save memory after each exchange
    save_memory(history)
    
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