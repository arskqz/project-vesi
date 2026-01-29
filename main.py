### Imports ###
import os
import json
import textwrap
import pyaudio
import wave
import numpy as np
from llama_cpp import Llama
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
import sounddevice as sd

### Model and Memory path ###
MODEL_PATH = r"D:\models\ana-v1.gguf" # https://huggingface.co/TheBloke/Ana-v1-m7-GGUF Feel free to use different model
MEMORY_PATH = "logs/chat_log.json"

### Memory ###
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
            "You are Vesi, a classic Tsundere girl. You are smug, arrogant, and easily flustered." # ADD YOUR OWN PROMPt HERE
        )
    }]

## Model uses sandwich prompt meaning
# First initial prompt above
# then chat history last messages
# and lastly a reminder prompt
# Also a propt with audio
## These need to be adjusted with model parameters in order to change personality

### Helper functions ###
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

# If your model is not "tsundere these can be commented"
def calculate_mood(text, current_score):
    """Calculates mood score based on response"""
    score = current_score
    tokens = set(text.lower().split())

    for word in tokens:
        if word in TSUN_WORDS:
            score += 10
        elif word in DRIFT_WORDS:
            score -= 15

    return max(0, min(100, score))

## Uncomment if model tries to be too AI like
## NOTE: This can be a bit slow if checking long list

# def sanitize_drift(text):
#     """Checks if Vesi drifts into being an AI assistant"""
#     polite_triggers = ["I'm here to help", "As an AI", "How can I assist"]
#     for trigger in polite_triggers:
#         if trigger in text:
#             return "Hmph! Don't expect me to be your little servant, baka!"
#     return text

def save_memory(history_data):
    """Saves history to memory"""
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=4)


def display_mood_gauge(score):
    """Displays mood score"""
    bar_length = 20
    filled_length = int(bar_length * score / 100)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)

    print(f"\nVESI PERSONALITY GAUGE: [{bar}] {score}%")
    if score > 75:
        print("STATUS: Maximum tsundere (Safe)")
    elif score < 40:
        print("STATUS: WARNING - Assistant Drift Detected!")
    else:
        print("STATUS: Stable")


### STT ###
stt_model = WhisperModel("base", device="cuda", compute_type="float16")

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 500  # Adjust your mic sensitivity
SILENCE_LIMIT = 1.5      # Seconds of silence before stop record

def listen_continuously():
    '''Listen to speach continiously, stops after silence'''
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("\n[Vesi is listening... Speak now!]")

    frames = []
    silent_chunks = 0
    audio_started = False

    while True:
        data = stream.read(CHUNK)
        frames.append(data)

        amplitude = np.frombuffer(data, np.int16)
        volume = np.abs(amplitude).mean()

        if volume > SILENCE_THRESHOLD:
            if not audio_started:
                print("(Recording started...)")
                audio_started = True
            silent_chunks = 0
        else:
            if audio_started:
                silent_chunks += 1

        if audio_started and silent_chunks > (SILENCE_LIMIT * RATE / CHUNK):
            break

    print("(Processing speech...)")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save tmp wav file
    wf = wave.open("temp_input.wav", 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # Transcribe
    segments, _ = stt_model.transcribe(
        "temp_input.wav",
        beam_size=5,
        language="en",
        task="transcribe",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms = 500),
        initial_prompt = "Vesi is a girl's name. Vesi, tsundere, smug.", # Reminder for model that Vesi is her name
    )
    text = " ".join([s.text for s in segments])
    return text.strip()

### TTS Setup ###

# These voices need to be downloaded from kokoro repo and copied to voices/
vocal_cord = Kokoro("voices/kokoro-v0_19.onnx", "voices/voices-v1.0.bin")

def vesi_speak(text):
    """Vesi speaks using the Kokoro engine."""
    samples, sample_rate = vocal_cord.create(
        text,
        voice="af_bella",
        speed=1.1,
        lang="en-us"
    )

    sd.play(samples, sample_rate)
    sd.wait()

### Model set up ###
llm = Llama(
    model_path=MODEL_PATH,
    chat_format="chatml",
    n_ctx=4096,
    n_gpu_layers=35,
    verbose=False
)

# STOP generation if it generates "bad" word
STOP_LIST = ["</s>", "[INST]", "<<USER>>", "<<TSUNDERE>>", "John:", "User:"]
current_temp = 0.95
vesi_mood_score = 50

# Load memory
history = load_memory()

### Main loop ###
print("==============================")
print(" VESI INTERFACE SYSTEM ")
print("==============================")
print("1. Keyboard (Type)")
print("2. Voice (Speaking)")
choice = input("\nSelect mode (1 or 2): ").strip()

is_voice_mode = True if choice == "2" else False

if is_voice_mode:
    print("\n[System] Voice mode active. Listening for Vesi...")
else:
    print("\n[System] Keyboard mode active. Type your message below.")
print("==============================\n")

while True:
    if is_voice_mode:
        user_input = listen_continuously()
        # Filter background noice
        if not user_input or len(user_input) < 2:
            continue
        print(f"You (Voice): {user_input}")
    else:
        user_input = input("User > ")

    if user_input.lower() in ["exit", "quit"]:
        break

    history.append({"role": "user", "content": user_input})

    # Only remember last few chats to prevent model drift and improve performance
    if len(history) > 13:
        safe_history = history[0:1] + history[-12:]
    else:
        safe_history = history
    
    # reminder prompt
    vesi_anchor = {"role": "system", "content": "[VESI MODE: Stay smug, tsundere, and quirky.]"}
    message_to_vesi = safe_history + [vesi_anchor]

    completion = llm.create_chat_completion(
        messages=message_to_vesi,
        max_tokens=300,
        temperature=current_temp,
        top_p=0.92,
        frequency_penalty=0.65,
        presence_penalty=1.2,
        stop=STOP_LIST,
        stream=True
    )

    full_response = ""
    token_count = 0
    print("\n(Vesi is thinking...)",end="\r")

    for chunk in completion:
        if "content" in chunk["choices"][0]["delta"]:
            token = chunk["choices"][0]["delta"]["content"]
            # Prevent model leaking
            if "[" in token or "<" in token:
                break
            full_response += token
            token_count += 1

    ## Calls sanitize_drift()
    # full_response = sanitize_drift(full_response)

    vesi_mood_score = calculate_mood(full_response, vesi_mood_score)
    print(" " * 30, end="\r")

    print("Vesi >")
    print("-" * 30)
    wrapped_version = textwrap.fill(full_response.strip(), width=80)
    print(wrapped_version)

    vesi_speak(full_response)

    print("-" * 30)
    print(f"DEBUG INFO: Generated {token_count} tokens.")
    print("-" * 30)

    # Adjust Vesi temp based on score
    if vesi_mood_score < 40:
        current_temp = min(1.3, current_temp + 0.1)
    elif vesi_mood_score > 70:
        current_temp = 0.85

    display_mood_gauge(vesi_mood_score)
    history.append({"role": "assistant", "content": full_response.strip()})

    # Failsafe for memory saving
    save_memory(history)