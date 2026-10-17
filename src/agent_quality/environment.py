from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv


def load_environment(explicit: Path | None = None) -> Path | None:
    project = Path(__file__).resolve().parents[2]
    for candidate in [explicit, project / ".env", project.parent / "LangfuseTest" / ".env"]:
        if candidate and candidate.exists():
            load_dotenv(candidate, override=False)
            return candidate
    return None


def require_openrouter_key() -> str:
    value = os.getenv("OPENROUTER_API_KEY")
    if not value:
        raise RuntimeError("OPENROUTER_API_KEY is missing; pass --env-file or create .env")
    return value
