from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

# --- Guardrails / bonus knobs -------------------------------------------------
# Confidence threshold: a fact candidate is persisted only when its confidence
# reaches this value. Negated, joked, past-tense or hypothetical mentions are
# pushed below it, so "product manager" (a joke) never lands in User.md.
PROFILE_CONFIDENCE_THRESHOLD = 0.6
# Memory decay: multi-value fields and logs keep only the most recent entries,
# so User.md cannot grow without bound.
MAX_INTERESTS = 6
MAX_CHANGE_LOG = 6
MAX_SUMMARY_LINES = 8

# Structured entity schema for User.md (order = render order).
FACT_LABELS: dict[str, str] = {
    "name": "Tên",
    "location": "Nơi ở hiện tại",
    "profession": "Nghề nghiệp hiện tại",
    "response_style": "Style trả lời",
    "interests": "Mối quan tâm",
    "favorite_drink": "Đồ uống yêu thích",
    "favorite_food": "Món ăn yêu thích",
    "pet": "Thú cưng",
}
SINGLE_VALUE_KEYS = ("name", "location", "profession", "favorite_drink", "favorite_food", "pet")


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def estimate_tokens(text: str) -> int:
    """Heuristic token estimate (~4 characters per token), stable for offline benchmarks."""

    stripped = (text or "").strip()
    if not stripped:
        return 0
    return max(1, math.ceil(len(stripped) / 4))


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", nfc(text).strip())
    return [p.strip() for p in parts if p.strip()]


# --- User.md ------------------------------------------------------------------


def _slugify(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", (value or "").replace("đ", "d").replace("Đ", "D"))
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    slug = re.sub(r"[^0-9A-Za-z_-]+", "_", ascii_only).strip("_")
    return slug or "anonymous"


def render_profile(user_id: str, facts: dict[str, str], change_log: list[str] | None = None) -> str:
    lines = [
        f"# User.md - {user_id}",
        "",
        "> Persistent memory của Advanced Agent: chỉ lưu fact ổn định, fact mới nhất ghi đè fact cũ.",
        "",
        "## Stable facts",
    ]
    ordered = [k for k in FACT_LABELS if k in facts] + [k for k in facts if k not in FACT_LABELS]
    if ordered:
        lines.extend(f"- {key}: {facts[key]}" for key in ordered)
    else:
        lines.append("- (chưa có fact nào)")
    if change_log:
        lines += ["", "## Change log"]
        lines.extend(f"- {entry}" for entry in change_log)
    return "\n".join(lines) + "\n"


def parse_profile(text: str) -> tuple[dict[str, str], list[str]]:
    facts: dict[str, str] = {}
    change_log: list[str] = []
    section = ""
    for line in nfc(text).splitlines():
        if line.startswith("## "):
            section = line[3:].strip().lower()
            continue
        if not line.startswith("- "):
            continue
        if section == "stable facts":
            match = re.match(r"^- ([a-z_]+): (.+)$", line)
            if match:
                facts[match.group(1)] = match.group(2).strip()
        elif section == "change log":
            change_log.append(line[2:].strip())
    return facts, change_log


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md` (one markdown file per user id)."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        return Path(self.root_dir) / _slugify(user_id) / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return render_profile(user_id, {})

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(nfc(content))
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        path = self.path_for(user_id)
        if not path.exists() or not search_text:
            return False
        content = path.read_text(encoding="utf-8")
        search_text = nfc(search_text)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, nfc(replacement), 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    # Structured helpers on top of the raw markdown API.

    def facts(self, user_id: str) -> dict[str, str]:
        return parse_profile(self.read_text(user_id))[0]

    def change_log(self, user_id: str) -> list[str]:
        return parse_profile(self.read_text(user_id))[1]

    def upsert_fact(self, user_id: str, key: str, value: str) -> bool:
        return bool(self.upsert_facts(user_id, {key: value}))

    def upsert_facts(self, user_id: str, updates: dict[str, str]) -> dict[str, str]:
        """Merge updates into User.md. Returns {key: new_value} for facts that changed.

        Conflict handling: single-value facts are overwritten by the newer value and
        the superseded value moves to a bounded change log instead of staying "current".
        """

        if not updates:
            return {}
        facts, change_log = parse_profile(self.read_text(user_id))
        changed: dict[str, str] = {}
        for key, value in updates.items():
            old = facts.get(key)
            merged = merge_fact_value(key, old, value)
            if not merged or merged == old:
                continue
            if old and key in SINGLE_VALUE_KEYS:
                change_log.append(f"{key}: {old} -> {merged}")
            facts[key] = merged
            changed[key] = merged
        if changed or not self.path_for(user_id).exists():
            self.write_text(user_id, render_profile(user_id, facts, change_log[-MAX_CHANGE_LOG:]))
        return changed


# --- Fact extraction ------------------------------------------------------------


@dataclass
class FactCandidate:
    key: str
    value: str
    confidence: float
    reason: str = ""


QUESTION_RE = re.compile(r"\?|\b(?:là gì|là ai|ở đâu|thế nào|ra sao|bao nhiêu|có phải)\b", re.IGNORECASE)
NEGATION_RE = re.compile(r"không còn|không phải|chứ không|đừng|chẳng còn|thôi không|không ở", re.IGNORECASE)
PAST_RE = re.compile(r"lúc đầu|trước đó|trước đây|hồi trước|ngày xưa|\btừng\b", re.IGNORECASE)
JOKE_RE = re.compile(r"\bđùa\b|\bgiỡn\b|giả sử|giả như", re.IGNORECASE)
HYPOTHETICAL_RE = re.compile(r"\bnếu\b|\blỡ\b", re.IGNORECASE)
HYPOTHETICAL_START_RE = re.compile(r"\s*(?:nếu|giả sử|lỡ)\b", re.IGNORECASE)
TEMPORARY_RE = re.compile(r"tạm thời|chỉ trong cuộc này", re.IGNORECASE)
LOCATION_SUBJECT_RE = re.compile(r"\b(?:mình|tôi|hiện|sống|nơi ở)\b", re.IGNORECASE)
PROFESSION_CONTEXT_RE = re.compile(r"\b(?:làm|nghề|chuyển sang|là)\b", re.IGNORECASE)
STYLE_TRIGGER_RE = re.compile(r"trả lời|giải thích|trình bày|\bstyle\b", re.IGNORECASE)

NAME_RE = re.compile(
    r"(?:^|[,:;]\s*|\b(?:mình|tôi|em|anh|chị)\s+)tên(?:\s+(?:mình|tôi|của mình))?(?:\s+là)?\s+",
    re.IGNORECASE,
)
LOCATION_AT_RE = re.compile(r"\bở\s+", re.IGNORECASE)
LOCATION_EXPLICIT_RE = re.compile(r"nơi ở(?:\s+hiện tại)?\s+(?:là|:)\s+", re.IGNORECASE)
LOCATION_MOVE_RE = re.compile(r"\b(?:sang|chuyển (?:về|đến|tới|vào))\s+", re.IGNORECASE)
ROLE_RE = re.compile(
    r"(?<![\w-])([A-Za-z][A-Za-z0-9+./-]*)\s+(engineer|developer|manager|scientist|analyst|designer|researcher|architect)\b"
)
INTEREST_RE = re.compile(
    r"\bmình\s+(?:(?:vẫn|còn|rất|cũng|đang)\s+)*(?:thích|quan tâm(?:\s+nhiều)?\s+(?:đến|tới)|học(?:\s+thêm)?\s+về)\s+(.+)$",
    re.IGNORECASE,
)
DRINK_FAVORITE_RE = re.compile(r"đồ uống yêu thích(?:\s+của\s+mình)?(?:\s+vẫn)?\s+(?:là|:)\s+(.+)", re.IGNORECASE)
DRINK_HABIT_RE = re.compile(r"\bmình\s+(?:(?:vẫn|thường|hay|chỉ)\s+)*uống\s+(.+)", re.IGNORECASE)
FOOD_RE = re.compile(
    r"món(?:\s+ăn)?\s+(?:yêu thích|ruột|khoái khẩu)(?:\s+của\s+mình)?\s+(?:là|:)\s+(.+)", re.IGNORECASE
)
PET_RE = re.compile(
    r"(?:nuôi\s+(?:một\s+)?(?:(?:bé|con|chú|em)\s+)?|\b(?:con|bé|chú)\s+)"
    r"(corgi|chó|mèo|cún|poodle|husky|shiba|alaska|vẹt|hamster|thỏ)\b(?:\s+(?:tên\s+)?(\S+))?",
    re.IGNORECASE,
)
PHRASE_STOP_RE = re.compile(
    r"[.,;:!?]|\s(?:như|nhưng|và|mỗi|để|vì|rồi|khi|lúc|mà|thì)\s", re.IGNORECASE
)
INTEREST_STOP_RE = re.compile(r"[:;]|\s(?:vì|để|hơn là|nhưng|khi|mà)\s", re.IGNORECASE)

# (slot, pattern, canonical value). The first match per slot wins inside one text.
STYLE_RULES: list[tuple[str, re.Pattern[str], str]] = [
    ("length", re.compile(r"chi tiết hơn|dài hơn|đầy đủ hơn|trả lời (?:thật )?chi tiết", re.I), "chi tiết"),
    ("length", re.compile(r"ngắn gọn|\bngắn\b|\bgọn\b|lan man", re.I), "ngắn gọn"),
    ("format", re.compile(r"(\d+)\s*bullet", re.I), "{0} bullet"),
    ("format", re.compile(r"bullet", re.I), "bullet"),
    ("examples", re.compile(r"ví dụ thực chiến", re.I), "có ví dụ thực chiến"),
    ("examples", re.compile(r"ví dụ thực tế", re.I), "có ví dụ thực tế"),
    ("quant", re.compile(r"định lượng|số liệu", re.I), "có số liệu minh họa"),
    ("structure", re.compile(r"có cấu trúc", re.I), "có cấu trúc"),
    ("clarity", re.compile(r"rõ ý", re.I), "rõ ý"),
    ("emphasis", re.compile(r"trade-off", re.I), "nhấn trade-off"),
]
STYLE_SLOT_ORDER = ["length", "format", "examples", "quant", "structure", "clarity", "emphasis"]


def is_question(text: str) -> bool:
    return bool(QUESTION_RE.search(nfc(text)))


def _cap_phrase(text: str, max_words: int = 4) -> str:
    """Leading run of capitalised words, e.g. 'Đà Nẵng và ...' -> 'Đà Nẵng'."""

    words: list[str] = []
    for raw in text.split():
        word = raw.strip(".,;:!?()\"'“”")
        if not word or not word[0].isupper():
            break
        words.append(word)
        if raw[-1] in ".,;:!?)" or len(words) >= max_words:
            break
    return " ".join(words)


def _short_phrase(text: str, max_words: int = 5) -> str:
    match = PHRASE_STOP_RE.search(" " + text + " ")
    cut = text[: max(0, match.start() - 1)] if match else text
    return " ".join(cut.split()[:max_words]).strip()


def _clause_prefix(sentence: str, start: int) -> str:
    prefix = sentence[:start]
    cut = max(prefix.rfind(","), prefix.rfind(";"), prefix.rfind(":"), prefix.lower().rfind(" nhưng "))
    return prefix[cut + 1 :] if cut >= 0 else prefix


def _factual_confidence(base: float, sentence: str, prefix: str) -> tuple[float, str]:
    """Lower confidence for negated / past / joked / hypothetical mentions."""

    if JOKE_RE.search(sentence):
        return 0.0, "joke"
    if NEGATION_RE.search(prefix):
        return 0.0, "negated"
    if PAST_RE.search(prefix):
        return 0.2, "past"
    if HYPOTHETICAL_RE.search(prefix) or HYPOTHETICAL_START_RE.match(sentence):
        return base - 0.4, "hypothetical"
    return base, "stated"


def extract_style_features(text: str) -> dict[str, str]:
    features: dict[str, str] = {}
    for slot, pattern, canonical in STYLE_RULES:
        if slot in features:
            continue
        match = pattern.search(text)
        if match:
            features[slot] = canonical.format(*match.groups()) if match.groups() else canonical
    return features


def render_style(features: dict[str, str]) -> str:
    return ", ".join(features[slot] for slot in STYLE_SLOT_ORDER if slot in features)


def merge_fact_value(key: str, old: str | None, new: str) -> str:
    new = nfc(new).strip()
    if not old:
        return new
    if key == "response_style":
        merged = extract_style_features(old)
        incoming = extract_style_features(new)
        if incoming.get("format") == "bullet" and merged.get("format", "").endswith("bullet"):
            incoming.pop("format")  # do not lose "3 bullet" because of a vaguer "bullet"
        merged.update(incoming)
        return render_style(merged)
    if key == "interests":
        items = [i.strip() for i in old.split(",") if i.strip()]
        for item in (i.strip() for i in new.split(",") if i.strip()):
            items = [i for i in items if i.casefold() != item.casefold()] + [item]
        return ", ".join(items[-MAX_INTERESTS:])  # recency-based decay
    if old.casefold().startswith(new.casefold()):
        return old  # "corgi" must not erase the richer "corgi tên Bơ"
    return new


def extract_profile_candidates(message: str) -> list[FactCandidate]:
    """Extract every fact candidate with a confidence score (including rejected ones)."""

    candidates: list[FactCandidate] = []
    for sentence in split_sentences(message):
        if is_question(sentence) or TEMPORARY_RE.search(sentence):
            continue

        for match in NAME_RE.finditer(sentence):
            name = _cap_phrase(sentence[match.end() :])
            if name:
                conf, why = _factual_confidence(0.95, sentence, "")
                candidates.append(FactCandidate("name", name, conf, why))

        for pattern, base in ((LOCATION_AT_RE, 0.85), (LOCATION_EXPLICIT_RE, 0.95), (LOCATION_MOVE_RE, 0.9)):
            for match in pattern.finditer(sentence):
                place = _cap_phrase(sentence[match.end() :], max_words=3)
                if not place:
                    continue
                prefix = _clause_prefix(sentence, match.start())
                if pattern is LOCATION_AT_RE and not LOCATION_SUBJECT_RE.search(prefix):
                    continue
                if pattern is LOCATION_MOVE_RE and not (
                    re.search(r"nơi ở", sentence, re.I) or match.group(0).lower().startswith("chuyển")
                ):
                    continue
                if ROLE_RE.match(sentence[match.end() :]):
                    continue  # "chuyển sang MLOps engineer" is a job change, not a move
                conf, why = _factual_confidence(base, sentence, prefix)
                candidates.append(FactCandidate("location", place, conf, why))

        if PROFESSION_CONTEXT_RE.search(sentence):
            for match in ROLE_RE.finditer(sentence):
                role = f"{match.group(1)} {match.group(2)}"
                conf, why = _factual_confidence(0.8, sentence, _clause_prefix(sentence, match.start()))
                candidates.append(FactCandidate("profession", role, conf, why))

        if STYLE_TRIGGER_RE.search(sentence):
            style = render_style(extract_style_features(sentence))
            if style:
                candidates.append(FactCandidate("response_style", style, 0.8, "instruction"))

        match = INTEREST_RE.search(sentence)
        if match:
            raw = match.group(1)
            stop = INTEREST_STOP_RE.search(" " + raw + " ")
            raw = raw[: max(0, stop.start() - 1)] if stop else raw
            for item in re.split(r",\s*|\s+và\s+", raw):
                item = re.sub(r"^và\s+", "", item.strip(" .!")).strip()
                words = item.split()
                if words and len(words) <= 4 and any(w[0].isascii() and w[0].isupper() for w in words):
                    candidates.append(FactCandidate("interests", item, 0.7, "interest"))

        for pattern, base in ((DRINK_FAVORITE_RE, 0.95), (DRINK_HABIT_RE, 0.7)):
            match = pattern.search(sentence)
            if match and (drink := _short_phrase(match.group(1))):
                conf, why = _factual_confidence(base, sentence, _clause_prefix(sentence, match.start()))
                candidates.append(FactCandidate("favorite_drink", drink, conf, why))

        match = FOOD_RE.search(sentence)
        if match and (food := _short_phrase(match.group(1))):
            candidates.append(FactCandidate("favorite_food", food, 0.95, "stated"))

        match = PET_RE.search(sentence)
        if match:
            animal = match.group(1).lower()
            pet_name = (match.group(2) or "").strip(".,;:!?")
            value = f"{animal} tên {pet_name}" if pet_name and pet_name[0].isupper() else animal
            candidates.append(FactCandidate("pet", value, 0.85, "stated"))

    return candidates


def extract_profile_updates(message: str, threshold: float = PROFILE_CONFIDENCE_THRESHOLD) -> dict[str, str]:
    """Convert raw user text into stable profile facts that pass the confidence threshold.

    Question-only sentences and temporary instructions are skipped; for single-value
    facts the last confident mention in the message wins (handles in-message corrections).
    """

    updates: dict[str, str] = {}
    for candidate in extract_profile_candidates(message):
        if candidate.confidence < threshold:
            continue
        if candidate.key in ("interests", "response_style") and candidate.key in updates:
            updates[candidate.key] = merge_fact_value(candidate.key, updates[candidate.key], candidate.value)
        else:
            updates[candidate.key] = candidate.value
    return updates


# --- Compact memory -------------------------------------------------------------


def _key_terms(text: str, limit: int = 5) -> list[str]:
    terms: list[str] = []
    for sentence in split_sentences(text):
        for word in sentence.split()[1:]:
            token = word.strip(".,;:!?()\"'“”")
            if token and (token[0].isupper() or any(ch.isdigit() for ch in token)) and token not in terms:
                terms.append(token)
    return terms[:limit]


def _gist(text: str, max_chars: int = 70) -> str:
    sentences = split_sentences(text)
    first = sentences[0] if sentences else ""
    if len(first) > max_chars:
        first = first[:max_chars].rsplit(" ", 1)[0] + "..."
    terms = _key_terms(text)
    return f"{first} [{', '.join(terms)}]" if terms else first


def summarize_lines(messages: list[dict[str, str]]) -> list[str]:
    """One bullet per user message: first sentence + key entities/numbers.

    Assistant turns are folded away: in this lab they mostly acknowledge or re-state
    memory, which is already persisted in User.md.
    """

    return [f"- {m['role']}: {_gist(m['content'])}" for m in messages if m.get("role") == "user" and m.get("content")]


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Heuristic summary of older messages (replaceable by an LLM summary in live mode)."""

    lines = summarize_lines(messages)
    dropped = len(lines) - max_items
    lines = lines[-max_items:] if max_items > 0 else []
    if dropped > 0:
        lines.insert(0, f"- ({dropped} ý cũ hơn đã được lược bỏ)")
    return "\n".join(lines)


@dataclass
class CompactMemoryManager:
    """Short-term memory per thread with automatic compaction.

    - Keep the `keep_messages` most recent messages verbatim.
    - When message tokens exceed `threshold_tokens`, fold older messages into a
      bounded summary (at most `max_summary_lines` lines).
    - Track compactions for benchmarking.
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)
    max_summary_lines: int = MAX_SUMMARY_LINES

    def _thread(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(
            thread_id,
            {"messages": [], "summary": "", "summary_lines": [], "compactions": 0, "compacted_messages": 0},
        )

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self._thread(thread_id)
        thread["messages"].append({"role": role, "content": nfc(content)})  # type: ignore[union-attr]
        if self.should_compact(thread_id):
            self.compact(thread_id)

    def message_tokens(self, thread_id: str) -> int:
        messages = self._thread(thread_id)["messages"]
        return sum(estimate_tokens(m["content"]) for m in messages)  # type: ignore[union-attr]

    def should_compact(self, thread_id: str) -> bool:
        messages = self._thread(thread_id)["messages"]
        return len(messages) > self.keep_messages and self.message_tokens(thread_id) > self.threshold_tokens  # type: ignore[arg-type]

    def compact(self, thread_id: str) -> bool:
        thread = self._thread(thread_id)
        messages: list[dict[str, str]] = thread["messages"]  # type: ignore[assignment]
        if len(messages) <= self.keep_messages:
            return False
        older, recent = messages[: -self.keep_messages], messages[-self.keep_messages :]
        lines = list(thread["summary_lines"]) + summarize_lines(older)  # type: ignore[arg-type]
        thread["summary_lines"] = lines[-self.max_summary_lines :]
        thread["summary"] = "\n".join(thread["summary_lines"])  # type: ignore[arg-type]
        thread["messages"] = recent
        thread["compactions"] = int(thread["compactions"]) + 1  # type: ignore[arg-type]
        thread["compacted_messages"] = int(thread["compacted_messages"]) + len(older)  # type: ignore[arg-type]
        return True

    def context(self, thread_id: str) -> dict[str, object]:
        thread = self._thread(thread_id)
        return {
            "messages": list(thread["messages"]),  # type: ignore[arg-type]
            "summary": thread["summary"],
            "compactions": thread["compactions"],
            "compacted_messages": thread["compacted_messages"],
        }

    def compaction_count(self, thread_id: str) -> int:
        return int(self.state.get(thread_id, {}).get("compactions", 0))  # type: ignore[arg-type]
