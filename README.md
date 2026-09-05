![Vesi_temp](img/vesi_temp.png)

> [!WARNING]
> Ill be swapping models soon except chaos. Also tools are likely to get a purpose.

# :ocean: Project Vesi
Vesi is what happens when you give an anime personality a high-performance brain and vocal cords. It's a fully local, multimodal AI agent that can hear you, think for itself, remember conversations, and talk back **all without ever touching the cloud or relying on external APIs.**

Built with Three.js, VRM, and Python. Tested on Python 3.11.14 and Windows > 10.

## :rocket: Features

* :brain: **Local Brain**: Powered by a custom fine-tuned `Llama 3` model. No API keys or subscriptions.
* :ear: **Sharp Ears**: Uses `Faster-Whisper` to transcribe your voice. Fast and works well even for rally-english.
* :anger: **Mood System**: Bidirectional "dere meter" (0–100) that scans both Vesi's responses and user input for emotional keywords. Drives LLM temperature, TTS speech speed, and frontend mood bar in real time.
* :floppy_disk: **Memory Compression**: Long-term memory via LLM-powered summarization. Old conversation turns are compressed into narrative blocks in Vesi's voice, keeping recent context hot.
* :speech_balloon: **Clear Voice**: Uses `Kokoro-82M` for loud and clear human-like speech with lip sync.
* :microphone: **Push-to-Talk**: Hold the mic button to speak, release to auto-send. Seamless voice interaction.
* :art: **3D Avatar**: Interactive VRM character with natural idle animations, breathing, and blinking.
* :loop: **Hybrid Input**: Switch between speaking and keyboard on the fly.
* :wrench: **Tool System** *(Experimental)*: Context injection layer before LLM calls. Passive tools (datetime) always run; active tools trigger on user input keywords. Easily extensible.
* :memo: **Live Config**: Personality and user facts defined in `vesi_config.yaml`. Add facts at runtime via the `/remember` endpoint. No restart needed.

## :hammer: Tech Stack

### Backend
* Python 3.11
* FastAPI (REST API)
* Llama-cpp-python (The Brains)
* Faster-Whisper (The Hearing)
* Kokoro-ONNX (The Vocal Cords)

### Frontend
* Three.js (3D Rendering)
* @pixiv/three-vrm (VRM Character Support)
* Tailwind CSS (UI Styling)
* Web Audio API (Lip Sync & Audio Playback)

## :camera: Showcase

**DEMO VIDEO LIVE**

* https://www.youtube.com/watch?v=i9Aj_RLnwOU

## :brain: Custom Fine-Tuned Model

Vesi runs on a **custom fine-tuned Llama 3 8B model**, trained using [Unsloth](https://github.com/unslothai/unsloth) with LoRA on a hand-crafted conversational dataset (~400 examples). The model is quantized to Q6_K GGUF for efficient local inference.

The full dataset contains personal information and won't be shared, but a cleaned public version (~300 conversations) 
is available on HuggingFace: [Tsundere-AI Dataset](https://huggingface.co/datasets/arskaz/Tsundere-AI). 
It's a good starting point for training your own character AI with minimal tweaking.

However, the full training pipeline is documented in [`training/training.ipynb`](training/training.ipynb). Feel free to use it as a reference to fine-tune your own model with your own data.

If you are interested in training your own model check out:

- [Unsloth Documentation](https://docs.unsloth.ai)
- [Hugging Face Documentation](https://huggingface.co/docs)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Maxine Labonne LLM fine-tuning with unsloth](https://mlabonne.github.io/blog/posts/2024-07-29_Finetune_Llama31.html)


## :wrench: Setup

### 1. Clone the repo
```bash
git clone https://github.com/arskqz/project_vesi.git
cd project_vesi
```

### 2. Install Python dependencies

**NOTE:** You might want to use python 3.10 (Do your own research) 
```bash
cd server
pip install -r requirements.txt
```

### 3. Download models
- Place your GGUF model in `models/` (e.g., `ana-v1.gguf` or `lumimaid-v2.gguf`) It is also important to choose the right quantization for your GPU.
- Download Kokoro voices and place in `voices/`
- Place your VRM **1.0** model in `client/models` 

### 4. Add your own prompt
- Add your own prompt to `vesi_config.example.yaml`
- Change the file name to `vesi_config.yaml`
- Set `USER_NAME` and `AI_NAME` in `server/config/config.py`, then use `{user}` / `{ai}`
  in the prompt instead of writing names directly.
- Point `MODEL_PATH` at your `.gguf` — either in `config.py` or via `server/.env`
  (copy `server/.env.example`).
- Feel free to play around with the mood system and change it to your liking. 

### 5. Download three.js
- Download three.js, three VRM and Tailwind css using `npm`, `vite` or something else.

### 6. Run start script 
- Edit your conda path and environment name.
- run `start.bat`.

I'll add bash one soon... 

⚠️ Warning: This isn't a "one-click" install. You are going to encounter many errors. **Good luck!**

## :video_game: Usage
* **Type**: Enter text in the input box and click Send
* **Speak**: Press and hold the mic button, speak, then release to auto-send
* **Watch**: Vesi responds with voice, lip sync, and mood changes

## :gear: Configuration

**Everything tunable lives in `server/config/config.py`.** Each constant carries a
comment saying what it controls. The sections are:

* **Identity** — `USER_NAME` and `AI_NAME`. Change these two and the names propagate
  everywhere: system prompt, stop tokens, role-leak filters, STT vocabulary hints,
  mood hints and memory summaries.
* **Paths** — `MODEL_PATH` (your `.gguf`), memory log, Kokoro voice files. All anchored
  to `server/`, so the backend runs from any working directory.
* **Server** — `HOST`, `PORT`, `PUBLIC_BASE_URL`, CORS origins.
* **LLM** — `N_CTX`, `N_GPU_LAYERS` (tune for your VRAM), and the sampling params
  (`TOP_K`, `TOP_P`, `MIN_P`, `REPEAT_PENALTY`, `MAX_TOKENS`).
* **STT / TTS** — Whisper model size and device, Kokoro voice (`TTS_VOICE`) and language.
* **Memory** — `COMPRESSION_THRESHOLD`, `KEEP_RECENT`, `HOT_TURNS`, `MAX_COMPRESSED_BLOCKS`.
* **Mood** — baseline, decay rate, signal weights, the tsun/dere band boundaries, and the
  score-to-temperature / score-to-TTS-speed values.
* **Logging** — `LOG_LEVEL`, `LOG_TO_FILE` and the format. See below.

**Logging** — always on, minimal by default:

* `INFO` (default) — startup, one line per request, warnings and errors. Nothing else;
  uvicorn access lines and llama-cpp perf output are silenced.
* `DEBUG` — adds per-stage timings (LLM, TTS, total) and dumps the **fully assembled
  prompt** message by message, the full response, and every mood signal that matched.
  This is the fastest way to see what the model was actually given.

Flip modes without editing anything: `set VESI_LOG_LEVEL=DEBUG` (or put it in `server/.env`).
Set `VESI_LOG_FILE=1` to also write `logs/vesi_<timestamp>.log` — a fresh file per session,
so old ones stick around until you delete them.

```
16:41:46 INFO  [main] initializing Vesi
16:41:49 INFO  [main] Vesi is online at http://127.0.0.1:8000
16:42:07 INFO  [chat] done in 2392ms | in 25 -> out 101 chars | mood 45->55 neutral
```

**Machine-specific paths** — Copy `server/.env.example` to `server/.env` (gitignored) and
set `VESI_MODEL_PATH` there instead of editing `config.py`.

**Personality** — Edit `server/vesi_config.yaml` (gitignored, so your prompt stays private):
* `system_prompt` — Vesi's full personality prompt
* `user_facts` — List of facts about the user, injected into the system prompt. Can also be added at runtime via the `/remember` API endpoint.

Write `{user}` and `{ai}` in this file rather than literal names — they are filled in from
`USER_NAME` / `AI_NAME`. Quote any list entry that *starts* with `{`, or YAML will read it
as a mapping.

**Mood keywords** — Edit `server/mood_system.py`:
* Tsun/dere keyword lists for Vesi's responses and user input (the character vocabulary;
  the numbers behind them are in `config.py`)

**Tools** — Edit `server/tools.py`:
* Add passive tools (always injected) or active tools (keyword-triggered) to extend Vesi's context awareness

**Frontend** — If you change `PORT`, also update `API_BASE` at the top of `client/src/app.js`.


## 🗺️ TODO

* [ X ] Memory update: Fix the memory system to better keep in context

* [ ] More animations: Custom animations and multiple vrm model support.

* [ ] Voice Evolution: Custom voice with Kokoro or something different.



## :mag_right: Technical Challenges & Solutions

* **The VRAM Tightrope:** One of the biggest hurdles was managing the memory budget of a high-performance LLM alongside a GPU-intensive TTS. I optimized the system by utilizing **6-bit GGUF quantization** for the Llama model and dynamically offloading specific layers to system RAM, ensuring enough VRAM remained for real-time voice synthesis. With this optimization responses even with voice mode are almost instant.

* **Breaking the Dependency Loop:** I successfully navigated a "**dependency hell**" scenario where the original TTS library was unmaintained and conflicting with modern Python 3.11 environments. I solved this by surgically patching library imports and pivoting to a community-maintained ONNX-based architecture for better stability and performance.

* **Fine-Tuning a Personality:** Training a model to feel like a real character — not a generic chatbot — required careful dataset engineering. With only ~400 hand-crafted examples, the margin between a flat personality and heavy overfitting was razor-thin. I iterated on LoRA rank, epoch count, and dataset balance to land on a model that captures Vesi's tsundere voice without parroting the training data.

## 🤖 State of Development

This is an actively developed personal project. Expect rough edges and experimental features. It is built for learning and fun, not production use.

Parts of this project were developed with **agent-assisted programming** using [Claude Code](https://docs.anthropic.com/en/docs/claude-code), as an experiment in AI-augmented development workflows. From testing Anthropic does not have much training data for on AI waifus.

## :clap: Credits

Llama for model and training -> https://github.com/ggml-org/llama.cpp

Model used -> https://huggingface.co/Lewdiculous/Llama-3-Lumimaid-8B-v0.1-OAS-GGUF-IQ-Imatrix

STT Faster-Whisper -> https://github.com/SYSTRAN/faster-whisper

TTS Kokoro -> https://github.com/thewh1teagle/kokoro-onnx

Three JS VRM by pixiv -> https://github.com/pixiv/three-vrm

Inspiration for the project and vtube model -> https://www.youtube.com/@JustRayen


## :page_facing_up: License

MIT License - do whatever you want with it.

Go make your own AI Waifu !

Made with love and loads of coffee.
