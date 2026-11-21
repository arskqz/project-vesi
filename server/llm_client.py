### LLM client ###
# Runs the language model in llama.cpp's own llama-server and talks to it over
# its OpenAI-compatible HTTP API.
#
# Why not llama-cpp-python: it bundles its own, older copy of llama.cpp, so
# every new model architecture (e.g. qwen35) means waiting for a new wheel.
# llama-server is upstream llama.cpp, and --jinja applies the chat template
# stored in each GGUF, so any model in config.MODELS just works.
#
# If something already answers at LLM_BASE_URL (a llama-server or llama-swap
# you started yourself), it is used as is and nothing is spawned.

### Imports ###
import atexit
import subprocess
import threading
import time
from collections import deque

import httpx

from config import (
    ACTIVE_MODEL, MODEL_PATH, LLAMA_SERVER_EXE,
    LLM_HOST, LLM_PORT, LLM_BASE_URL,
    LLM_START_TIMEOUT, LLM_REQUEST_TIMEOUT, LLM_HEALTH_POLL,
    N_CTX, N_GPU_LAYERS, LLM_EXTRA_ARGS, LLM_VERBOSE, LLM_CHAT_TEMPLATE_KWARGS,
    TOP_K, TOP_P, MIN_P, REPEAT_PENALTY, MAX_TOKENS, STOP_TOKENS,
)
from logger import get_logger

LOG = get_logger("llm")

# Lines of llama-server output kept so a failed start can say why
_OUTPUT_TAIL_LINES = 25

_process = None             # The llama-server we spawned, or None in external mode
_output_tail = deque(maxlen=_OUTPUT_TAIL_LINES)
_client = None              # Shared httpx.AsyncClient, created on first chat()


### Server lifecycle ###

def _health() -> int | None:
    """Status code of GET /health, or None if nothing is listening."""
    try:
        return httpx.get(f"{LLM_BASE_URL}/health", timeout=2).status_code
    except httpx.HTTPError:
        return None


def _pump_output(stream) -> None:
    """Forwards llama-server output to the log at DEBUG and keeps a short tail."""
    for raw in stream:
        line = raw.rstrip()
        if not line:
            continue
        _output_tail.append(line)
        if LLM_VERBOSE:
            LOG.debug("server: %s", line)


def _wait_until_ready() -> None:
    """Polls /health until 200. 503 means the model is still loading."""
    deadline = time.monotonic() + LLM_START_TIMEOUT
    while time.monotonic() < deadline:
        if _process is not None and _process.poll() is not None:
            tail = "\n".join(_output_tail)
            raise RuntimeError(
                f"llama-server exited with code {_process.returncode} while loading "
                f"{MODEL_PATH}.\nLast output:\n{tail}"
            )
        if _health() == 200:
            return
        time.sleep(LLM_HEALTH_POLL)
    raise TimeoutError(f"llama-server not ready after {LLM_START_TIMEOUT}s at {LLM_BASE_URL}")


def start_server() -> None:
    """
    Makes sure a model server is ready at LLM_BASE_URL, spawning
    llama-server for the active model if nothing is listening yet.
    """
    global _process

    if _health() is not None:
        LOG.info("using existing llm server at %s", LLM_BASE_URL)
        _wait_until_ready()
        return

    if not LLAMA_SERVER_EXE.exists():
        raise FileNotFoundError(
            f"llama-server not found at {LLAMA_SERVER_EXE}. "
            "Set VESI_LLAMA_SERVER in server/.env, or start a server yourself at VESI_LLM_URL."
        )
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    cmd = [
        str(LLAMA_SERVER_EXE),
        "-m", str(MODEL_PATH),
        "--jinja",                      # chat template from the GGUF
        "-c", str(N_CTX),
        "-ngl", str(N_GPU_LAYERS),
        "--host", LLM_HOST,
        "--port", str(LLM_PORT),
        *LLM_EXTRA_ARGS,
    ]
    LOG.info("starting llama-server: model=%s (%s)", ACTIVE_MODEL, MODEL_PATH.name)
    LOG.debug("llama-server cmd: %s", " ".join(cmd))

    _process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    # Kill the child even if the backend dies without a clean shutdown
    atexit.register(stop_server)
    threading.Thread(target=_pump_output, args=(_process.stdout,), daemon=True).start()

    started = time.monotonic()
    _wait_until_ready()
    LOG.info("llm ready in %.1fs", time.monotonic() - started)


def stop_server() -> None:
    """Terminates the llama-server we spawned. No-op in external mode."""
    global _process
    if _process is None:
        return
    if _process.poll() is None:
        LOG.info("stopping llama-server")
        _process.terminate()
        try:
            _process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _process.kill()
    _process = None


async def close_client() -> None:
    """Closes the shared HTTP client."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


### Inference ###

def _normalize_messages(messages: list) -> list:
    """
    Reshapes the prompt so strict chat templates accept it.

    Many GGUF templates (Qwen, Gemma, Mistral...) allow one system message
    and only at index 0, but memory.py and the active tools add several.
    Leading system messages are merged into one; a system message later in
    the list is folded into the next user turn so it keeps its position.
    """
    merged = []
    pending = []    # mid-conversation system text waiting for the next user turn

    for msg in messages:
        role, content = msg["role"], msg["content"]
        if role == "system":
            if not merged:
                merged.append({"role": "system", "content": content})
            elif len(merged) == 1 and merged[0]["role"] == "system":
                merged[0] = {"role": "system", "content": f"{merged[0]['content']}\n\n{content}"}
            else:
                pending.append(content)
            continue
        if role == "user" and pending:
            content = "\n\n".join(pending + [content])
            pending = []
        merged.append({"role": role, "content": content})

    if pending:     # trailing system text with no user turn after it
        merged.append({"role": "user", "content": "\n\n".join(pending)})
    return merged


async def chat(messages: list, temperature: float) -> tuple[str, str | None]:
    """
    Runs one chat completion. Returns (text, finish_reason).
    Sampling values beyond the OpenAI set (top_k, min_p, repeat_penalty) are
    llama-server extensions it reads from the same request body.
    """
    global _client
    if _client is None:
        _client = httpx.AsyncClient(base_url=LLM_BASE_URL, timeout=LLM_REQUEST_TIMEOUT)

    payload = {
        "model": ACTIVE_MODEL,          # ignored by llama-server, used by llama-swap
        "messages": _normalize_messages(messages),
        "temperature": temperature,
        "top_k": TOP_K,
        "top_p": TOP_P,
        "min_p": MIN_P,
        "repeat_penalty": REPEAT_PENALTY,
        "max_tokens": MAX_TOKENS,
        "stop": STOP_TOKENS,
        "chat_template_kwargs": LLM_CHAT_TEMPLATE_KWARGS,
    }
    response = await _client.post("/v1/chat/completions", json=payload)
    if response.is_error:
        # Surface llama-server's own message (e.g. a template exception)
        raise RuntimeError(f"llama-server {response.status_code}: {response.text[:500]}")

    choice = response.json()["choices"][0]
    text = choice["message"].get("content") or ""
    return text, choice.get("finish_reason")
