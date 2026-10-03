"""Deterministic offline "LLM" shared by both agents.

Both agents answer with the same rules; the only difference is *which memory*
they are allowed to read (baseline: facts seen in the current thread, advanced:
User.md). That keeps the benchmark a fair comparison of memory architectures.
"""

from __future__ import annotations

import re

from memory_store import FACT_LABELS, is_question, nfc

SYSTEM_PROMPT = (
    "Bạn là trợ lý AI tiếng Việt. Trả lời đúng trọng tâm, ưu tiên thông tin mới nhất "
    "khi người dùng đính chính, không bịa thông tin không có trong ngữ cảnh."
)

ADVANCED_MEMORY_INSTRUCTIONS = (
    "Bạn có persistent memory `User.md`. Chỉ ghi fact ổn định (tên, nơi ở, nghề nghiệp, style trả lời, "
    "sở thích); khi người dùng đính chính thì ghi đè fact cũ; không ghi câu hỏi, câu đùa hay thông tin tạm thời. "
    "Áp dụng style trả lời trong User.md."
)

# requested field -> trigger keywords inside a recall question
FIELD_TRIGGERS: dict[str, tuple[str, ...]] = {
    "name": ("tên",),
    "location": ("ở đâu", "nơi ở", "đang ở", "còn ở", "sống ở"),
    "profession": ("nghề", "làm gì", "công việc"),
    "response_style": ("style", "kiểu trả lời", "cách trả lời"),
    "interests": ("quan tâm", "sở thích"),
    "favorite_drink": ("đồ uống", "uống gì"),
    "favorite_food": ("món ăn", "ăn gì"),
    "pet": ("nuôi", "thú cưng"),
}
PROFILE_SUMMARY_FIELDS = ("name", "profession", "location", "interests")
SUMMARY_TRIGGERS = ("là ai", "tóm tắt", "mô tả", "giới thiệu", "có biết")
RECALL_VERBS = ("nhắc lại", "nhớ lại", "tóm tắt", "mô tả", "cho mình biết")


def requested_fields(message: str) -> list[str]:
    text = nfc(message).lower()
    fields = [key for key, triggers in FIELD_TRIGGERS.items() if any(t in text for t in triggers)]
    if any(t in text for t in SUMMARY_TRIGGERS):
        fields += [f for f in PROFILE_SUMMARY_FIELDS if f not in fields]
    return fields


def is_recall_request(message: str, updates: dict[str, str] | None = None) -> bool:
    """A question about the user, or an imperative "nhắc lại ..." that does not also state new facts."""

    if updates:
        return False
    text = nfc(message).lower()
    if not requested_fields(message):
        return False
    return is_question(message) or any(verb in text for verb in RECALL_VERBS)


def compose_recall_answer(facts: dict[str, str], fields: list[str], missing_note: str) -> str:
    lines = []
    for key in fields:
        value = facts.get(key)
        lines.append(f"- {FACT_LABELS[key]}: {value if value else missing_note}")
    return "\n".join(lines)


def _gist(message: str, max_words: int = 12) -> str:
    words = re.sub(r"\s+", " ", nfc(message)).strip().split(" ")
    gist = " ".join(words[:max_words])
    return gist + ("..." if len(words) > max_words else "")


def compose_ack(message: str, changed: dict[str, str] | None = None) -> str:
    lines = [f"- Đã ghi nhận: {_gist(message)}"]
    if changed:
        updates = "; ".join(f"{FACT_LABELS.get(k, k)} = {v}" for k, v in changed.items())
        lines.append(f"- Đã cập nhật User.md: {updates}")
    return "\n".join(lines)
