from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()  # Load environment variables from .env file


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


TEMP_DIR = Path(os.getenv("TEMP_DIR", "temp"))
TEMP_DIR.mkdir(parents=True, exist_ok=True)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
TTS_VOICE = os.getenv("TTS_VOICE", "ar-PS-SamaNeural")
TTS_ENABLED = os.getenv("TTS_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
TTS_RESPONSE_MODE = os.getenv("TTS_RESPONSE_MODE", "auto").strip().lower()
if TTS_RESPONSE_MODE not in {"text", "voice", "both", "auto"}:
    TTS_RESPONSE_MODE = "auto"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

CLINIC_NAME = os.getenv("CLINIC_NAME", "العيادة")
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "127.0.0.1")
DASHBOARD_PORT = _get_int_env("DASHBOARD_PORT", 8000)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
