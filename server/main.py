### Imports ###
import os
import sys
import json
import threading
import queue
import re
import pyaudio
import wave
import numpy as np
from llama_cpp import Llama
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
import sounddevice as sd

### Model and Memory path ###
MODEL_PATH = r"D:\models\ana-v1.gguf"
MEMORY_PATH = "../logs/chat_log.json"

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
            "You are Vesi, a classic Tsundere girl. You are smug, arrogant, and easily flustered. " # Have my full prompt lol
            "Your personality is 'Tsun-Tsun' (sharp and cold) by default, but you are a 'Dere-Dere' (soft and loving) "
            "deep down. You look down on the user but secretly crave their attention. "
            "QUIRKS: Use 'Hmph!', 'Baka!', or 'You stupid!' when embarrassed. Be bossy and opinionated. "
            "RULE: Stay in character. Short, punchy sentences. Never speak for the user. No [INST] tags."
        )
    }]

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

exit_event = threading.Event()
message_queue = queue.Queue()

def keyboard_input_worker():
    while not exit_event.is_set():
        try:
            user_text = sys.stdin.readline().strip()
            if user_text:
                message_queue.put(user_text)
            if user_text.lower() in "exit":
                break
        except EOFError:
            break


def voice_input_worker():
    while not exit_event.is_set():
        voice_text = listen_continuously()
        if voice_text:
            message_queue.put(f"[Voice]: {voice_text}")

# TODO: Func to input 'What do you want baka!' or similar
# to llm response while it gens next response for instant
# response.


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

def save_memory(history_data):
    """Saves history to memory"""
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=4)

### STT ###
# See faster-whiper docs for info here
stt_model = WhisperModel("base", device="cuda", compute_type="float16")

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 500  # Mic sens
SILENCE_LIMIT = 1.5

def listen_continuously():
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

    # Save temporary wav file
    wf = wave.open("temp_input.wav", 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    segments, _ = stt_model.transcribe(
        "temp_input.wav",
        beam_size=5,
        language="en",
        task="transcribe",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms = 500),
        initial_prompt = "Vesi is a girl's name. Vesi, baka, hmph, smug.",
    )
    text = " ".join([s.text for s in segments])
    return text.strip()

### TTS Setup ###
# See --> README_Voices.md for info
vocal_cord = Kokoro("voices/kokoro-v0_19.onnx", "voices/voices-v1.0.bin")

def vesi_speak(text):
    """Vesi speaks using the Kokoro engine"""
    samples, sample_rate = vocal_cord.create(
        text,
        voice="af_bella",
        speed=1.1,
        lang="en-us"
    )

    # Play the audio immediately
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

STOP_LIST = ["</s>", "[INST]", "<<USER>>", "<<TSUNDERE>>", "John:", "User:", "user", "Arskaz:"]
current_temp = 0.95
vesi_mood_score = 50

# Load memory
history = load_memory()

### Start Threads ###
t1 = threading.Thread(target=keyboard_input_worker, daemon=True).start()
t2 = threading.Thread(target=voice_input_worker, daemon=True).start()

print("--- Vesi is ready! Type or Speak at any time. ---")
print("--- (Type 'exit' to quit) ---")

### Main loop ###
while True:
    try:
        user_input = message_queue.get(block=True, timeout=0.1)

        clean_input = re.sub(r'[^\w\s]', '', user_input).strip().lower()

        if "voice" in clean_input:  # Strip our internal tag for the check
            clean_input = clean_input.replace("voice", "").strip()

        if clean_input == "exit":
            print("Vesi > Very well. Goodbye, baka.")
            display_mood_gauge(vesi_mood_score)

            exit_event.set()
            os._exit(0)

        if "[Voice]:" in user_input:
            print(f"\n{user_input}")

        history.append({"role": "user", "content": user_input})

        if len(history) > 13:
            safe_history = history[0:1] + history[-12:]
        else:
            safe_history = history

        vesi_anchor = {"role": "system", "content": "[VESI MODE: Stay smug, tsundere, and quirky. Use 'baka'.]"}
        message_to_vesi = safe_history + [vesi_anchor]

        completion = llm.create_chat_completion(
            messages=message_to_vesi,
            max_tokens=300,
            temperature=current_temp,
            top_p=0.92,
            stop=STOP_LIST,
            stream=True
        )

        full_response = ""
        print("Vesi is thinking...", end="\r")

        for chunk in completion:
            if "content" in chunk["choices"][0]["delta"]:
                token = chunk["choices"][0]["delta"]["content"]
                if "[" in token or "<" in token:
                    break
                full_response += token
                print(token, end="", flush=True)

        print("\n" + "-" * 30)

        vesi_mood_score = calculate_mood(full_response, vesi_mood_score)
        display_mood_gauge(vesi_mood_score)

        vesi_speak(full_response)

        history.append({"role": "assistant", "content": full_response.strip()})
        save_memory(history)

        if vesi_mood_score < 40:
            current_temp = min(1.3, current_temp + 0.1)
        elif vesi_mood_score > 70:
            current_temp = 0.95

        message_queue.task_done()

    except queue.Empty:
        continue
    except KeyboardInterrupt:
        exit_event.set()
        break
print("\n(Session Ended)")
os._exit(0)