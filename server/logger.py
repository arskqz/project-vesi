### Logging ###
# Thin wrapper over stdlib logging. Configured entirely from config/config.py.
#
# INFO  — minimal: startup, one line per request, warnings and errors
# DEBUG — adds stage timings and the full prompt / response / mood detail

### Imports ###
import logging
import sys
from datetime import datetime

from config import (
    DEBUG_MODE,
    LOG_DIR,
    LOG_FILE_PREFIX,
    LOG_FORMAT,
    LOG_LEVEL,
    LOG_PREVIEW_CHARS,
    LOG_QUIET_THIRD_PARTY,
    LOG_TIME_FORMAT,
    LOG_TO_FILE,
)

# Set once setup_logging() has run, so a second call is a no-op
_configured = False

# Path of this session's log file, or None when file logging is off
log_file_path = None


def setup_logging() -> None:
    """
    Configures the root logger. Call once, before anything else logs.
    Writes to the console always, and to logs/<prefix>_<timestamp>.log when
    LOG_TO_FILE is set — a fresh file per session.
    """
    global _configured, log_file_path

    if _configured:
        return

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_TIME_FORMAT)

    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)
    root.handlers.clear()   # drop anything a library installed at import time

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if LOG_TO_FILE:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = LOG_DIR / f"{LOG_FILE_PREFIX}_{stamp}.log"
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    if LOG_QUIET_THIRD_PARTY:
        # Always noisy and never useful here — HTTP connection pools, model
        # downloads, file locks. Pinned even at DEBUG so debug output stays
        # about Vesi rather than about urllib3.
        for name in ("urllib3", "httpx", "httpcore", "filelock",
                     "huggingface_hub", "asyncio", "matplotlib", "numba"):
            logging.getLogger(name).setLevel(logging.WARNING)

        if not DEBUG_MODE:
            for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
                logging.getLogger(name).setLevel(logging.WARNING)
            # phonemizer warns "words count mismatch" on almost every Kokoro call
            logging.getLogger("phonemizer").setLevel(logging.ERROR)

    _configured = True

    if log_file_path:
        get_logger("main").info("logging to %s", log_file_path)


def get_logger(name: str) -> logging.Logger:
    """Returns the named logger. The name becomes the [tag] in each line."""
    return logging.getLogger(name)


def is_debug() -> bool:
    """
    True when running at DEBUG. Guard expensive debug formatting with this so
    the work is skipped entirely at INFO.
    """
    return DEBUG_MODE


def preview(text: str) -> str:
    """
    Compact form for INFO: length plus the opening characters.
    DEBUG logs the full text instead.
    """
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= LOG_PREVIEW_CHARS:
        return f'{len(text)} chars: "{text}"'
    return f'{len(text)} chars: "{text[:LOG_PREVIEW_CHARS]}..."'
