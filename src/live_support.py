"""Small helpers shared by the live (LangChain / LangGraph) code paths."""

from __future__ import annotations

from typing import Any


def langchain_available() -> bool:
    try:
        import langchain.agents  # noqa: F401
        import langgraph.checkpoint.memory  # noqa: F401
    except ImportError:
        return False
    return True


def message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # Anthropic / Gemini content blocks
        parts = [block.get("text", "") if isinstance(block, dict) else str(block) for block in content]
        return "".join(parts)
    return str(content)


def turn_usage(messages: list[Any], start_index: int) -> tuple[int, int]:
    """Sum (input_tokens, output_tokens) reported by the provider for AI messages of this turn."""

    prompt_tokens = output_tokens = 0
    for message in messages[start_index:]:
        usage = getattr(message, "usage_metadata", None) or {}
        prompt_tokens += int(usage.get("input_tokens", 0) or 0)
        output_tokens += int(usage.get("output_tokens", 0) or 0)
    return prompt_tokens, output_tokens


def build_summarization_middleware(model: Any, threshold_tokens: int, keep_messages: int):
    from langchain.agents.middleware import SummarizationMiddleware

    try:  # langchain >= 1.0 signature
        return SummarizationMiddleware(model=model, trigger=("tokens", threshold_tokens), keep=("messages", keep_messages))
    except TypeError:  # older pre-release signature
        return SummarizationMiddleware(
            model=model, max_tokens_before_summary=threshold_tokens, messages_to_keep=keep_messages
        )
