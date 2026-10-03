from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from live_support import langchain_available, message_text, turn_usage
from memory_store import estimate_tokens, extract_profile_updates, merge_fact_value, nfc
from model_provider import build_chat_model
from responder import SYSTEM_PROMPT, compose_ack, compose_recall_answer, is_recall_request, requested_fields


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0
    # What the model can "see" in this thread's transcript. Lives and dies with the thread.
    facts: dict[str, str] = field(default_factory=dict)


class BaselineAgent:
    """Agent A: within-session memory only.

    - Keeps the full transcript of each thread and re-sends all of it every turn.
    - No `User.md`, no compaction: a new thread id starts from zero.
    """

    name = "Baseline"

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.live_error: str | None = None

        self.langchain_agent = None
        if not force_offline:
            try:
                self.langchain_agent = self._maybe_build_langchain_agent()
            except Exception as exc:  # missing SDK, bad key format, ... -> stay offline
                self.live_error = repr(exc)

    @property
    def mode(self) -> str:
        return "live" if self.langchain_agent is not None else "offline"

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response and token accounting for one user turn.

        `user_id` is accepted for API symmetry but deliberately ignored: the baseline
        has no per-user memory.
        """

        message = nfc(message)
        if self.langchain_agent is not None:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def memory_file_size(self, user_id: str) -> int:
        # Baseline has no persistent memory file.
        return 0

    def _session(self, thread_id: str) -> SessionState:
        return self.sessions.setdefault(thread_id, SessionState())

    def build_prompt_context(self, thread_id: str) -> str:
        """Everything the baseline sends to the model: system prompt + the whole thread."""

        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in self._session(thread_id).messages)
        return f"{SYSTEM_PROMPT}\n\n## Conversation\n{transcript}"

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})
        prompt_tokens = estimate_tokens(self.build_prompt_context(thread_id))
        session.prompt_tokens_processed += prompt_tokens

        # The offline "model" reads facts out of the transcript it was given, i.e. this thread only.
        updates = extract_profile_updates(message)
        for key, value in updates.items():
            session.facts[key] = merge_fact_value(key, session.facts.get(key), value)

        if is_recall_request(message, updates):
            response = compose_recall_answer(
                session.facts, requested_fields(message), "chưa có thông tin này trong phiên hiện tại"
            )
        else:
            response = compose_ack(message)

        session.messages.append({"role": "assistant", "content": response})
        agent_tokens = estimate_tokens(message) + estimate_tokens(response)
        session.token_usage += agent_tokens
        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
            "compactions": 0,
            "mode": "offline",
        }

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            {"configurable": {"thread_id": thread_id}},
        )
        messages = result["messages"]
        start = max(i for i, m in enumerate(messages) if getattr(m, "type", "") == "human")
        response = message_text(messages[-1])
        provider_prompt_tokens, _ = turn_usage(messages, start)
        prompt_tokens = provider_prompt_tokens or estimate_tokens(self.build_prompt_context(thread_id))
        session.prompt_tokens_processed += prompt_tokens

        session.messages.append({"role": "assistant", "content": response})
        agent_tokens = estimate_tokens(message) + estimate_tokens(response)
        session.token_usage += agent_tokens
        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
            "compactions": 0,
            "mode": "live",
        }

    def _maybe_build_langchain_agent(self):
        """Build `create_agent` + `InMemorySaver` (thread-scoped memory only), if possible."""

        if not self.config.model.is_ready() or not langchain_available():
            return None
        from langchain.agents import create_agent
        from langgraph.checkpoint.memory import InMemorySaver

        return create_agent(
            build_chat_model(self.config.model),
            tools=[],
            system_prompt=SYSTEM_PROMPT,
            checkpointer=InMemorySaver(),
        )
