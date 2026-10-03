from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import unicodedata
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from memory_store import estimate_tokens

COLUMNS = [
    "Agent",
    "Agent tokens only",
    "Prompt tokens processed",
    "Cross-session recall",
    "Response quality",
    "Memory growth (bytes)",
    "Compactions",
]

Judge = Callable[[str, str, list[str]], "float | None"]


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int
    details: list[dict[str, Any]] = field(default_factory=list)


def load_conversations(path: Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", text or "").casefold()


def _coverage(answer: str, expected: list[str]) -> float:
    if not expected:
        return 1.0
    return sum(1 for item in expected if _norm(item) in _norm(answer)) / len(expected)


def recall_points(answer: str, expected: list[str]) -> float:
    """1 if every expected fact appears, 0.5 if only some do, 0 if none."""

    coverage = _coverage(answer, expected)
    if coverage >= 1.0:
        return 1.0
    return 0.5 if coverage > 0 else 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight 0..1 quality score for offline mode.

    - correctness dominates: the form only counts in proportion to the facts covered
      (a nicely formatted non-answer is still a non-answer);
    - concise: <= 80 tokens gets full credit, then decays;
    - structured: bullet / numbered lines get full credit;
    - an honest "I don't know" scores a little above a silent miss.
    """

    coverage = _coverage(answer, expected)
    tokens = estimate_tokens(answer)
    concise = 1.0 if tokens <= 80 else max(0.0, 1.0 - (tokens - 80) / 160)
    structured = 1.0 if re.search(r"^\s*(?:[-*•]|\d+[.)])\s+", answer, re.MULTILINE) else 0.5
    score = coverage * (0.6 + 0.2 * concise + 0.2 * structured)
    if coverage == 0 and re.search(r"chưa có|không biết|không nhớ", _norm(answer)):
        score = 0.1
    return round(score, 4)


def make_llm_judge(config: LabConfig) -> Judge | None:
    """LLM-as-judge using `config.judge_model` (live mode only). Returns None if unavailable."""

    if not config.judge_model.is_ready():
        return None
    try:
        from model_provider import build_chat_model

        judge_model = build_chat_model(config.judge_model)
    except Exception:
        return None

    def judge(question: str, answer: str, expected: list[str]) -> float | None:
        prompt = (
            "Bạn là giám khảo chấm câu trả lời của một AI agent có bộ nhớ.\n"
            f"Câu hỏi của người dùng: {question}\n"
            f"Các fact đúng cần có: {', '.join(expected)}\n"
            f"Câu trả lời của agent:\n{answer}\n\n"
            "Chấm từ 0 đến 10 theo: đúng fact mới nhất (quan trọng nhất), không bịa, ngắn gọn, có cấu trúc. "
            "Chỉ trả về một con số."
        )
        try:
            content = judge_model.invoke(prompt).content
            text = content if isinstance(content, str) else str(content)
            match = re.search(r"\d+(?:\.\d+)?", text)
            return min(10.0, float(match.group())) / 10 if match else None
        except Exception:
            return None

    return judge


def run_agent_benchmark(
    agent_name: str,
    agent,
    conversations: list[dict[str, Any]],
    config,
    judge: Judge | None = None,
) -> BenchmarkRow:
    """Feed every conversation to the agent, then ask its recall questions in fresh threads."""

    user_ids = sorted({conv["user_id"] for conv in conversations})
    start_sizes = {user_id: agent.memory_file_size(user_id) for user_id in user_ids}
    threads: list[str] = []
    details: list[dict[str, Any]] = []

    for conv in conversations:
        main_thread = f"{conv['id']}::main"
        threads.append(main_thread)
        for turn in conv["turns"]:
            agent.reply(conv["user_id"], main_thread, turn)

        for index, item in enumerate(conv.get("recall_questions", [])):
            recall_thread = f"{conv['id']}::recall-{index}"  # new thread = new session
            threads.append(recall_thread)
            answer = agent.reply(conv["user_id"], recall_thread, item["question"])["response"]
            expected = item["expected_contains"]
            quality = judge(item["question"], answer, expected) if judge else None
            details.append(
                {
                    "conversation": conv["id"],
                    "question": item["question"],
                    "expected": expected,
                    "answer": answer,
                    "recall": recall_points(answer, expected),
                    "quality": quality if quality is not None else heuristic_quality(answer, expected),
                }
            )

    count = max(1, len(details))
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=sum(agent.token_usage(t) for t in threads),
        prompt_tokens_processed=sum(agent.prompt_token_usage(t) for t in threads),
        recall_score=round(sum(d["recall"] for d in details) / count, 4),
        response_quality=round(sum(d["quality"] for d in details) / count, 4),
        memory_growth_bytes=sum(agent.memory_file_size(u) - start_sizes[u] for u in user_ids),
        compactions=sum(agent.compaction_count(t) for t in threads),
        details=details,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    table = [
        [
            row.agent_name,
            f"{row.agent_tokens_only:,}",
            f"{row.prompt_tokens_processed:,}",
            f"{row.recall_score:.2f}",
            f"{row.response_quality:.2f}",
            f"{row.memory_growth_bytes:,}",
            str(row.compactions),
        ]
        for row in rows
    ]
    try:
        from tabulate import tabulate

        return tabulate(table, headers=COLUMNS, tablefmt="github", disable_numparse=True)
    except ImportError:
        lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
        lines += ["| " + " | ".join(r) + " |" for r in table]
        return "\n".join(lines)


def _pct(new: int, old: int) -> str:
    if old == 0:
        return "n/a"
    return f"{(new - old) / old * 100:+.1f}%"


def format_comparison(baseline: BenchmarkRow, advanced: BenchmarkRow) -> str:
    return (
        f"Advanced vs Baseline: agent tokens {_pct(advanced.agent_tokens_only, baseline.agent_tokens_only)}, "
        f"prompt tokens {_pct(advanced.prompt_tokens_processed, baseline.prompt_tokens_processed)}, "
        f"recall {advanced.recall_score - baseline.recall_score:+.2f}, "
        f"quality {advanced.response_quality - baseline.response_quality:+.2f}"
    )


def format_details(row: BenchmarkRow) -> str:
    lines = [f"\n[{row.agent_name}] recall questions"]
    for d in row.details:
        answer = d["answer"].replace("\n", " | ")
        lines.append(f"  {d['conversation']} recall={d['recall']:.1f} quality={d['quality']:.2f}")
        lines.append(f"    Q: {d['question']}")
        lines.append(f"    A: {answer}")
    return "\n".join(lines)


def run_suite(
    title: str,
    dataset: Path,
    config: LabConfig,
    live: bool = False,
    judge: Judge | None = None,
) -> list[BenchmarkRow]:
    """Run baseline and advanced on the same dataset with an isolated, fresh state dir."""

    suite_dir = config.state_dir / "benchmark" / re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    if suite_dir.exists():
        shutil.rmtree(suite_dir)  # reproducible memory-growth numbers on every run
    suite_dir.mkdir(parents=True)
    suite_config = replace(config, state_dir=suite_dir)

    conversations = load_conversations(dataset)
    agents = [
        BaselineAgent(suite_config, force_offline=not live),
        AdvancedAgent(suite_config, force_offline=not live),
    ]
    return [run_agent_benchmark(agent.name, agent, conversations, suite_config, judge) for agent in agents]


def threshold_sweep(config: LabConfig, dataset: Path, thresholds: list[int]) -> str:
    """Offline sweep of the compact threshold on one dataset (Advanced only, Baseline as reference)."""

    conversations = load_conversations(dataset)
    sweep_dir = config.state_dir / "benchmark" / "threshold-sweep"
    baseline = run_agent_benchmark("Baseline", BaselineAgent(config, force_offline=True), conversations, config)
    table = [["Baseline (no compact)", f"{baseline.prompt_tokens_processed:,}", "-", "0", f"{baseline.recall_score:.2f}"]]
    for threshold in thresholds:
        run_dir = sweep_dir / str(threshold)
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_config = replace(config, state_dir=run_dir, compact_threshold_tokens=threshold)
        row = run_agent_benchmark(
            "Advanced", AdvancedAgent(run_config, force_offline=True), conversations, run_config
        )
        table.append(
            [
                f"Advanced @ {threshold} tokens",
                f"{row.prompt_tokens_processed:,}",
                _pct(row.prompt_tokens_processed, baseline.prompt_tokens_processed),
                str(row.compactions),
                f"{row.recall_score:.2f}",
            ]
        )
    headers = ["Config", "Prompt tokens processed", "vs Baseline", "Compactions", "Cross-session recall"]
    try:
        from tabulate import tabulate

        return tabulate(table, headers=headers, tablefmt="github", disable_numparse=True)
    except ImportError:
        return "\n".join(" | ".join(r) for r in [headers, *table])


def main(argv: list[str] | None = None) -> None:
    """Run the Standard benchmark and the Long-Context Stress benchmark."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Vietnamese output on Windows consoles

    parser = argparse.ArgumentParser(description="Benchmark Baseline vs Advanced memory agents.")
    parser.add_argument("--live", action="store_true", help="use the configured LLM provider instead of offline mode")
    parser.add_argument("--details", action="store_true", help="print every recall question and answer")
    parser.add_argument("--only", choices=["standard", "stress"], help="run just one benchmark suite")
    parser.add_argument("--sweep", action="store_true", help="also sweep the compact threshold on the stress set")
    parser.add_argument("--json", type=Path, help="also write raw results to this JSON file")
    args = parser.parse_args(argv)

    config = load_config(Path(__file__).resolve().parent.parent)
    judge = make_llm_judge(config) if args.live else None
    suites = [
        ("Standard Benchmark", config.data_dir / "conversations.json"),
        ("Long-Context Stress Benchmark", config.data_dir / "advanced_long_context.json"),
    ]

    mode = f"live ({config.model.provider}:{config.model.model_name})" if args.live else "offline (deterministic)"
    print(f"Mode: {mode} | compact threshold = {config.compact_threshold_tokens} tokens, "
          f"keep = {config.compact_keep_messages} messages")

    results: dict[str, list[dict[str, Any]]] = {}
    selected = {"standard": suites[:1], "stress": suites[1:]}.get(args.only or "", suites)
    for title, dataset in selected:
        rows = run_suite(title, dataset, config, live=args.live, judge=judge)
        print(f"\n## {title} ({dataset.name})\n")
        print(format_rows(rows))
        print("\n" + format_comparison(rows[0], rows[1]))
        if args.details:
            for row in rows:
                print(format_details(row))
        results[title] = [asdict(row) for row in rows]

    if args.sweep:
        print("\n## Compact threshold sweep (advanced_long_context.json, offline)\n")
        print(threshold_sweep(config, suites[1][1], [300, 600, 800, 1200, 2000, 4000]))

    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nRaw results written to {args.json}")


if __name__ == "__main__":
    main()
