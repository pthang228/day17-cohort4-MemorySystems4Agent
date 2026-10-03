from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import DEFAULT_MODELS, OLLAMA_BASE_URL, OPENROUTER_BASE_URL, ProviderConfig, normalize_provider

DEFAULT_COMPACT_THRESHOLD_TOKENS = 800
DEFAULT_COMPACT_KEEP_MESSAGES = 4

# provider -> (api key env vars in priority order, base url env var, default base url)
PROVIDER_ENV = {
    "openai": (("OPENAI_API_KEY",), "OPENAI_BASE_URL", None),
    "custom": (("CUSTOM_API_KEY", "OPENAI_API_KEY"), "CUSTOM_BASE_URL", None),
    "gemini": (("GEMINI_API_KEY", "GOOGLE_API_KEY"), None, None),
    "anthropic": (("ANTHROPIC_API_KEY",), "ANTHROPIC_BASE_URL", None),
    "ollama": ((), "OLLAMA_BASE_URL", OLLAMA_BASE_URL),
    "openrouter": (("OPENROUTER_API_KEY",), "OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
}


@dataclass
class LabConfig:
    """Shared configuration for the lab: paths, compact-memory knobs and model settings."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def _load_dotenv(root: Path) -> None:
    env_file = root / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_file, override=False)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def provider_config_from_env(prefix: str = "LLM", fallback: ProviderConfig | None = None) -> ProviderConfig:
    """Build a ProviderConfig from `<prefix>_PROVIDER`, `<prefix>_MODEL`, ... env vars."""

    raw_provider = os.getenv(f"{prefix}_PROVIDER", "").strip()
    if not raw_provider and fallback is not None:
        provider = fallback.provider
    else:
        provider = normalize_provider(raw_provider or "openai")

    same_as_fallback = fallback is not None and fallback.provider == provider
    key_vars, base_url_var, default_base_url = PROVIDER_ENV[provider]
    api_key = (
        os.getenv(f"{prefix}_API_KEY")
        or next((os.getenv(v) for v in key_vars if os.getenv(v)), None)
        or (fallback.api_key if same_as_fallback else None)
    )
    base_url = (
        os.getenv(f"{prefix}_BASE_URL")
        or (os.getenv(base_url_var) if base_url_var else None)
        or (fallback.base_url if same_as_fallback else None)
        or default_base_url
    )

    default_model = fallback.model_name if same_as_fallback else DEFAULT_MODELS[provider]
    return ProviderConfig(
        provider=provider,
        model_name=os.getenv(f"{prefix}_MODEL", "").strip() or default_model,
        temperature=_env_float(f"{prefix}_TEMPERATURE", 0.0),
        api_key=api_key,
        base_url=base_url,
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load `.env` (if present) and environment variables into a LabConfig.

    Env knobs:
    - LLM_PROVIDER / LLM_MODEL / LLM_TEMPERATURE / LLM_API_KEY / LLM_BASE_URL
    - JUDGE_PROVIDER / JUDGE_MODEL / ... (defaults to the main model)
    - OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY,
      CUSTOM_BASE_URL / CUSTOM_API_KEY, OLLAMA_BASE_URL
    - COMPACT_THRESHOLD_TOKENS, COMPACT_KEEP_MESSAGES, LAB_STATE_DIR
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    _load_dotenv(root)

    state_dir = Path(os.getenv("LAB_STATE_DIR", "") or root / "state")
    if not state_dir.is_absolute():
        state_dir = root / state_dir
    state_dir.mkdir(parents=True, exist_ok=True)

    model = provider_config_from_env("LLM")
    judge_model = provider_config_from_env("JUDGE", fallback=model)

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=max(50, _env_int("COMPACT_THRESHOLD_TOKENS", DEFAULT_COMPACT_THRESHOLD_TOKENS)),
        compact_keep_messages=max(1, _env_int("COMPACT_KEEP_MESSAGES", DEFAULT_COMPACT_KEEP_MESSAGES)),
        model=model,
        judge_model=judge_model,
    )
