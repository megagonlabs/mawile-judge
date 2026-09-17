from __future__ import annotations

from typing import Any

try:
    from dotenv import find_dotenv, load_dotenv
except ImportError:  # pragma: no cover - dependency is declared, fallback is defensive.

    def find_dotenv(*args: Any, **kwargs: Any) -> str:
        return ""

    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False


def load_environment() -> None:
    """Load a local .env file without overriding already-exported variables."""

    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)
        return
    load_dotenv(override=False)
