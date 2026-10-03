from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import heuristic_quality, load_conversations, recall_points, run_agent_benchmark
from config import load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_candidates,
    extract_profile_updates,
)
from model_provider import normalize_provider

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def make_config(tmp_path: Path, threshold: int = 200, keep: int = 4):
    """Isolated config: state lives in tmp_path and compaction triggers quickly."""

    config = load_config(Path(__file__).resolve().parent.parent)
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return replace(config, state_dir=state_dir, compact_threshold_tokens=threshold, compact_keep_messages=keep)


# --- User.md ------------------------------------------------------------------------


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")

    assert store.file_size("dungct") == 0
    assert "Stable facts" in store.read_text("dungct")  # default template before first write

    path = store.write_text("dungct", "# User.md\n\n## Stable facts\n- name: DũngCT\n- location: Đà Nẵng\n")
    assert path.exists() and path.name == "User.md"
    assert store.facts("dungct") == {"name": "DũngCT", "location": "Đà Nẵng"}

    assert store.edit_text("dungct", "location: Đà Nẵng", "location: Huế") is True
    assert store.edit_text("dungct", "text that is not there", "x") is False
    assert store.facts("dungct")["location"] == "Huế"
    assert store.file_size("dungct") == len(store.read_text("dungct").encode("utf-8"))


def test_user_id_is_sanitised_into_a_safe_path(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    path = store.path_for("../Dũng CT")
    assert path.parent.parent == tmp_path / "profiles"
    assert ".." not in path.parts


def test_advanced_agent_writes_user_markdown(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("dungct", "t1", "Chào bạn, mình tên là DũngCT.")
    agent.reply("dungct", "t1", "Mình ở Đà Nẵng và đang làm backend engineer cho startup AI.")

    text = agent.profile_store.read_text("dungct")
    assert "- name: DũngCT" in text
    assert "- location: Đà Nẵng" in text
    assert "- profession: backend engineer" in text
    assert agent.memory_file_size("dungct") > 0


# --- Compact memory -------------------------------------------------------------------


def test_compact_trigger(tmp_path: Path) -> None:
    manager = CompactMemoryManager(threshold_tokens=60, keep_messages=2)
    for i in range(6):
        manager.append("t", "user", f"Tin số {i}: NASA Artemis III và X-59 " + "chi tiết dài " * 10)
        manager.append("t", "assistant", "Đã ghi nhận.")

    ctx = manager.context("t")
    assert manager.compaction_count("t") >= 1
    assert len(ctx["messages"]) <= 3  # 2 kept + at most one appended after the last compaction
    assert "NASA" in ctx["summary"]  # older content survives as a summary, not verbatim
    assert manager.compaction_count("another-thread") == 0


def test_compact_does_not_trigger_on_short_thread(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path, threshold=800), force_offline=True)
    for turn in load_conversations(DATA_DIR / "conversations.json")[0]["turns"]:
        agent.reply("dungct", "short", turn)
    assert agent.compaction_count("short") == 0


def test_summary_is_bounded(tmp_path: Path) -> None:
    manager = CompactMemoryManager(threshold_tokens=40, keep_messages=2, max_summary_lines=5)
    for i in range(50):
        manager.append("t", "user", f"Lượt {i}: " + "nội dung khá dài " * 8)
    summary = manager.context("t")["summary"]
    assert len(summary.splitlines()) <= 5
    assert "Lượt 49" not in summary  # the most recent turns stay verbatim, not summarised


# --- Cross-session recall -----------------------------------------------------------------


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    turns = ["Chào bạn, mình tên là DũngCT.", "Đồ uống yêu thích là cà phê sữa đá."]
    for agent in (baseline, advanced):
        for turn in turns:
            agent.reply("dungct", "session-1", turn)

    question = "Mình tên gì và đồ uống yêu thích là gì?"
    advanced_answer = advanced.reply("dungct", "session-2", question)["response"]
    baseline_answer = baseline.reply("dungct", "session-2", question)["response"]

    assert "DũngCT" in advanced_answer and "cà phê sữa đá" in advanced_answer
    assert "DũngCT" not in baseline_answer and "cà phê sữa đá" not in baseline_answer


def test_baseline_remembers_within_the_same_thread(tmp_path: Path) -> None:
    baseline = BaselineAgent(make_config(tmp_path), force_offline=True)
    baseline.reply("dungct", "same", "Chào bạn, mình tên là DũngCT.")
    answer = baseline.reply("dungct", "same", "Bạn có thể nhắc lại tên mình không?")["response"]
    assert "DũngCT" in answer


def test_memory_persists_across_agent_restarts(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    AdvancedAgent(config, force_offline=True).reply("dungct", "t1", "Món ăn yêu thích là mì Quảng.")
    fresh_process = AdvancedAgent(config, force_offline=True)
    answer = fresh_process.reply("dungct", "t2", "Món ăn yêu thích của mình là gì?")["response"]
    assert "mì Quảng" in answer


def test_profiles_are_isolated_per_user(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("alice", "a", "Chào bạn, mình tên là Alice.")
    answer = agent.reply("bob", "b", "Mình tên gì?")["response"]
    assert "Alice" not in answer


# --- Conflict handling & guardrails (bonus) -----------------------------------------------


def test_correction_overwrites_old_fact(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("dungct", "t1", "Mình ở Đà Nẵng và đang làm backend engineer cho startup AI.")
    agent.reply("dungct", "t2", "À, mình đính chính một chút: giờ mình đang ở Huế chứ không còn ở Đà Nẵng mỗi ngày nữa.")
    agent.reply("dungct", "t3", "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.")

    facts = agent.profile_store.facts("dungct")
    assert facts["location"] == "Huế"
    assert facts["profession"] == "MLOps engineer"
    # the old values are kept only as history, never as the current fact
    assert "location: Đà Nẵng -> Huế" in agent.profile_store.change_log("dungct")

    answer = agent.reply("dungct", "t4", "Hiện tại mình làm nghề gì và mình còn ở Huế không?")["response"]
    assert "MLOps engineer" in answer and "backend" not in answer


def test_noise_jokes_and_trips_are_not_persisted() -> None:
    message = (
        "Có lúc mình đùa với đồng nghiệp rằng hay là chuyển sang product manager cho đỡ phải ngồi canh pipeline, "
        "nhưng đó chỉ là câu đùa. Nghề nghiệp hiện tại vẫn là MLOps engineer. Tương tự, Hà Nội chỉ là nơi mình "
        "vừa bay ra họp hai ngày với đối tác chứ không phải nơi ở hiện tại."
    )
    updates = extract_profile_updates(message)
    assert updates.get("profession") == "MLOps engineer"
    assert "location" not in updates

    rejected = [c for c in extract_profile_candidates(message) if c.value == "product manager"]
    assert rejected and rejected[0].confidence < 0.6 and rejected[0].reason == "joke"


@pytest.mark.parametrize(
    "message",
    [
        "Bạn có thể nhắc lại tên mình không?",
        "Bạn thử nhớ lại xem đồ uống yêu thích của mình là gì.",
        "Nếu ai đó nhắc Huế, Hà Nội hay product manager, đâu mới là nghề nghiệp và nơi ở hiện tại của mình?",
    ],
)
def test_questions_do_not_write_facts(message: str) -> None:
    assert extract_profile_updates(message) == {}


def test_negated_and_hypothetical_mentions_are_rejected() -> None:
    assert extract_profile_updates("Nếu nhắc lại nghề nghiệp, đừng nói backend engineer nữa nhé.") == {}
    updates = extract_profile_updates(
        "Lúc đầu mình nói hiện ở Huế, nhưng thực ra từ tuần này mình đang làm việc ở Đà Nẵng vài tháng."
    )
    assert updates["location"] == "Đà Nẵng"


def test_job_change_is_not_mistaken_for_a_move() -> None:
    updates = extract_profile_updates("Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.")
    assert updates == {"profession": "MLOps engineer"}


def test_style_preferences_merge_instead_of_overwrite(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    for message in [
        "Mình muốn bạn trả lời ngắn gọn thành 3 bullet.",
        "Khi giải thích kỹ thuật, hãy trả lời thành bullet và có ví dụ thực tế.",
    ]:
        store.upsert_facts("u", extract_profile_updates(message))
    style = store.facts("u")["response_style"]
    assert "ngắn gọn" in style and "3 bullet" in style and "ví dụ thực tế" in style


def test_interests_decay_keeps_most_recent(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    for i in range(10):
        store.upsert_fact("u", "interests", f"Topic{i}")
    interests = store.facts("u")["interests"].split(", ")
    assert len(interests) == 6
    assert interests[-1] == "Topic9" and "Topic0" not in interests


# --- Prompt load ---------------------------------------------------------------------------


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    conversation = load_conversations(DATA_DIR / "advanced_long_context.json")[0]
    config = make_config(tmp_path, threshold=800)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    for turn in conversation["turns"]:
        baseline.reply(conversation["user_id"], "long", turn)
        advanced.reply(conversation["user_id"], "long", turn)

    assert advanced.compaction_count("long") >= 2
    assert advanced.prompt_token_usage("long") < 0.75 * baseline.prompt_token_usage("long")
    # the last-turn context stays bounded instead of growing with the whole thread
    last_advanced = advanced._estimate_prompt_context_tokens(conversation["user_id"], "long")
    last_baseline = estimate_tokens(baseline.build_prompt_context("long"))
    assert last_advanced < last_baseline / 2


def test_advanced_costs_more_on_short_threads(tmp_path: Path) -> None:
    """The honest trade-off: without compaction, User.md is pure overhead per turn."""

    config = make_config(tmp_path, threshold=800)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    for turn in load_conversations(DATA_DIR / "conversations.json")[0]["turns"]:
        baseline.reply("dungct", "short", turn)
        advanced.reply("dungct", "short", turn)
    assert advanced.prompt_token_usage("short") > baseline.prompt_token_usage("short")


# --- Benchmark ------------------------------------------------------------------------------


def test_recall_points_and_quality() -> None:
    assert recall_points("Tên: DũngCT, đồ uống: cà phê sữa đá", ["DũngCT", "cà phê sữa đá"]) == 1.0
    assert recall_points("Tên: DũngCT", ["DũngCT", "cà phê sữa đá"]) == 0.5
    assert recall_points("Không biết", ["DũngCT"]) == 0.0
    assert heuristic_quality("- Tên: DũngCT", ["DũngCT"]) > heuristic_quality("chưa có thông tin", ["DũngCT"])


def test_standard_benchmark_separates_the_agents(tmp_path: Path) -> None:
    config = make_config(tmp_path, threshold=800)
    conversations = load_conversations(DATA_DIR / "conversations.json")
    baseline = run_agent_benchmark("Baseline", BaselineAgent(config, force_offline=True), conversations, config)
    advanced = run_agent_benchmark("Advanced", AdvancedAgent(config, force_offline=True), conversations, config)

    assert baseline.recall_score == 0.0 and baseline.memory_growth_bytes == 0
    assert advanced.recall_score == 1.0
    assert advanced.memory_growth_bytes > 0


def test_provider_aliases() -> None:
    assert normalize_provider("anthorpic") == "anthropic"
    assert normalize_provider("Google") == "gemini"
    assert normalize_provider("openai-compatible") == "custom"
    for provider in ("openai", "custom", "gemini", "anthropic", "ollama", "openrouter"):
        assert normalize_provider(provider) == provider
    with pytest.raises(ValueError):
        normalize_provider("not-a-provider")
