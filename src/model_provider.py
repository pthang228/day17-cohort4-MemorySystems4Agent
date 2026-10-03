from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_PROVIDERS = ("openai", "custom", "gemini", "anthropic", "ollama", "openrouter")

# Common typos / alternative names students (and .env files) tend to use.
PROVIDER_ALIASES = {
    "openai": "openai",
    "oai": "openai",
    "gpt": "openai",
    "custom": "custom",
    "openai_compatible": "custom",
    "openai-compatible": "custom",
    "compatible": "custom",
    "gemini": "gemini",
    "google": "gemini",
    "google_genai": "gemini",
    "google-genai": "gemini",
    "anthropic": "anthropic",
    "anthorpic": "anthropic",
    "antropic": "anthropic",
    "anthropics": "anthropic",
    "claude": "anthropic",
    "ollama": "ollama",
    "local": "ollama",
    "openrouter": "openrouter",
    "open_router": "openrouter",
    "open-router": "openrouter",
}

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "custom": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-sonnet-5-5",
    "ollama": "qwen2.5:7b",
    "openrouter": "openai/gpt-4o-mini",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OLLAMA_BASE_URL = "http://localhost:11434"


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Supported providers: openai, custom (OpenAI-compatible base URL), gemini,
    anthropic, ollama, openrouter.
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None

    def is_ready(self) -> bool:
        """True when the provider has enough settings to make a live call."""

        if self.provider == "ollama":
            return bool(self.model_name)
        if self.provider == "custom":
            return bool(self.base_url and self.model_name)
        return bool(self.api_key and self.model_name)


def normalize_provider(value: str) -> str:
    """Map aliases like `anthorpic` -> `anthropic`; raise on unknown providers."""

    key = (value or "").strip().lower().replace(" ", "_")
    if key in PROVIDER_ALIASES:
        return PROVIDER_ALIASES[key]
    raise ValueError(f"Unsupported provider {value!r}. Supported: {', '.join(SUPPORTED_PROVIDERS)}")


def build_chat_model(config: ProviderConfig):
    """Instantiate the real LangChain chat model for the selected provider.

    Imports are lazy so the offline benchmark/tests never need provider SDKs.
    """

    provider = normalize_provider(config.provider)

    if provider in ("openai", "custom"):
        from langchain_openai import ChatOpenAI

        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        elif provider == "custom":
            raise ValueError("Provider 'custom' requires CUSTOM_BASE_URL.")
        if provider == "custom" and not config.api_key:
            # Many self-hosted OpenAI-compatible servers ignore the key but the SDK requires one.
            kwargs["api_key"] = "not-needed"
        return ChatOpenAI(**kwargs)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs = {"model": config.model_name, "temperature": config.temperature, "api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatAnthropic(**kwargs)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=config.base_url or OLLAMA_BASE_URL,
        )

    if provider == "openrouter":
        try:
            from langchain_openrouter import ChatOpenRouter

            return ChatOpenRouter(model=config.model_name, temperature=config.temperature, api_key=config.api_key)
        except ImportError:
            # OpenRouter is OpenAI-compatible, so ChatOpenAI is a safe fallback.
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
                base_url=config.base_url or OPENROUTER_BASE_URL,
            )

    raise ValueError(f"Unsupported provider {config.provider!r}")
