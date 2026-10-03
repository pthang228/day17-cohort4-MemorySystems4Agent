from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from live_support import build_summarization_middleware, langchain_available, message_text, turn_usage
from memory_store import (
    FACT_LABELS,
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
    nfc,
)
from model_provider import build_chat_model
from responder import (
    ADVANCED_MEMORY_INSTRUCTIONS,
    SYSTEM_PROMPT,
    compose_ack,
    compose_recall_answer,
    is_recall_request,
    requested_fields,
)


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: short-term memory + persistent `User.md` + compact memory.

    Per turn: extract stable facts -> upsert User.md -> append to compact memory
    (auto-compacts past the threshold) -> prompt = User.md + summary + recent messages.
    """

    name = "Advanced"

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.live_error: str | None = None
        self._active_user_id: str | None = None

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
        message = nfc(message)
        if self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _remember(self, user_id: str, message: str) -> tuple[dict[str, str], dict[str, str]]:
        """Guardrailed write path: only confident, non-question facts reach User.md."""

        updates = extract_profile_updates(message)
        changed = self.profile_store.upsert_facts(user_id, updates)
        return updates, changed

    def _record(self, thread_id: str, message: str, response: str, prompt_tokens: int) -> int:
        agent_tokens = estimate_tokens(message) + estimate_tokens(response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_tokens
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        return agent_tokens

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        updates, changed = self._remember(user_id, message)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        response = self._offline_response(user_id, thread_id, message, updates=updates, changed=changed)
        self.compact_memory.append(thread_id, "assistant", response)
        agent_tokens = self._record(thread_id, message, response, prompt_tokens)
        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
            "profile_updates": changed,
            "compactions": self.compaction_count(thread_id),
            "mode": "offline",
        }

    def build_prompt_context(self, user_id: str, thread_id: str) -> str:
        """The context carried into one turn: User.md + compact summary + recent messages."""

        ctx = self.compact_memory.context(thread_id)
        parts = [SYSTEM_PROMPT, ADVANCED_MEMORY_INSTRUCTIONS, "## User.md", self.profile_store.read_text(user_id)]
        if ctx["summary"]:
            parts += ["## Tóm tắt hội thoại cũ (compact)", str(ctx["summary"])]
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in ctx["messages"])  # type: ignore[union-attr]
        parts += ["## Recent messages", transcript]
        return "\n\n".join(parts)

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        return estimate_tokens(self.build_prompt_context(user_id, thread_id))

    def _offline_response(
        self,
        user_id: str,
        thread_id: str,
        message: str,
        updates: dict[str, str] | None = None,
        changed: dict[str, str] | None = None,
    ) -> str:
        """Deterministic answer that reads persisted memory (User.md), not the thread."""

        if updates is None:
            updates = extract_profile_updates(message)
        if is_recall_request(message, updates):
            facts = self.profile_store.facts(user_id)
            return compose_recall_answer(facts, requested_fields(message), "chưa có trong User.md")
        return compose_ack(message, changed)

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        _, changed = self._remember(user_id, message)
        self._active_user_id = user_id
        self.compact_memory.append(thread_id, "user", message)  # mirror for metrics
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            {"configurable": {"thread_id": thread_id}},
            context=AgentContext(user_id=user_id, memory_path=str(self.profile_store.path_for(user_id))),
        )
        messages = result["messages"]
        start = max(i for i, m in enumerate(messages) if getattr(m, "type", "") == "human")
        response = message_text(messages[-1])
        provider_prompt_tokens, _ = turn_usage(messages, start)
        prompt_tokens = provider_prompt_tokens or self._estimate_prompt_context_tokens(user_id, thread_id)
        self.compact_memory.append(thread_id, "assistant", response)
        agent_tokens = self._record(thread_id, message, response, prompt_tokens)
        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
            "profile_updates": changed,
            "compactions": self.compaction_count(thread_id),
            "mode": "live",
        }

    def _maybe_build_langchain_agent(self):
        """Live agent: provider model + InMemorySaver + User.md tools + dynamic prompt + summarization."""

        if not self.config.model.is_ready() or not langchain_available():
            return None
        from langchain.agents import create_agent
        from langchain.agents.middleware import ModelRequest, dynamic_prompt
        from langchain.tools import tool
        from langgraph.checkpoint.memory import InMemorySaver

        model = build_chat_model(self.config.model)
        store = self.profile_store
        agent = self

        def current_user(runtime_context: Any = None) -> str:
            return getattr(runtime_context, "user_id", None) or agent._active_user_id or "anonymous"

        @tool
        def read_user_memory() -> str:
            """Đọc toàn bộ User.md (persistent memory) của người dùng hiện tại."""

            return store.read_text(current_user())

        @tool
        def save_user_fact(key: str, value: str) -> str:
            """Ghi hoặc ghi đè một fact ổn định vào User.md.

            key phải là một trong: name, location, profession, response_style, interests,
            favorite_drink, favorite_food, pet. Không lưu câu hỏi, câu đùa hay thông tin tạm thời.
            """

            if key not in FACT_LABELS:
                return f"Bỏ qua: key '{key}' không thuộc schema User.md."
            changed = store.upsert_facts(current_user(), {key: value})
            return f"Đã cập nhật {key}." if changed else "Không có thay đổi."

        @tool
        def edit_user_memory(search_text: str, replacement: str) -> str:
            """Sửa User.md bằng cách thay thế một đoạn văn bản (một lần)."""

            ok = store.edit_text(current_user(), search_text, replacement)
            return "Đã sửa User.md." if ok else "Không tìm thấy đoạn cần sửa."

        @dynamic_prompt
        def inject_user_memory(request: ModelRequest) -> str:
            runtime = getattr(request, "runtime", None)
            user_id = current_user(getattr(runtime, "context", None))
            return f"{SYSTEM_PROMPT}\n\n{ADVANCED_MEMORY_INSTRUCTIONS}\n\n## User.md\n{store.read_text(user_id)}"

        return create_agent(
            model,
            tools=[read_user_memory, save_user_fact, edit_user_memory],
            middleware=[
                inject_user_memory,
                build_summarization_middleware(
                    model, self.config.compact_threshold_tokens, self.config.compact_keep_messages
                ),
            ],
            context_schema=AgentContext,
            checkpointer=InMemorySaver(),
        )
