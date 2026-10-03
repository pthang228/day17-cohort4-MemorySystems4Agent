# src/ – bài làm hoàn chỉnh

| File | Nội dung |
|---|---|
| `model_provider.py` | `ProviderConfig`, `normalize_provider()` (alias như `anthorpic`, `google`), `build_chat_model()` cho openai / custom / gemini / anthropic / ollama / openrouter |
| `config.py` | `LabConfig`, `load_config()` đọc `.env` + biến môi trường (`LLM_*`, `JUDGE_*`, `COMPACT_*`, `LAB_STATE_DIR`) |
| `memory_store.py` | `estimate_tokens()`, `UserProfileStore` (read/write/edit + `facts()` / `upsert_facts()`), `extract_profile_candidates()` có confidence, `extract_profile_updates()`, `summarize_messages()`, `CompactMemoryManager` |
| `responder.py` | "LLM offline" dùng chung cho cả hai agent, gồm system prompt, phát hiện câu hỏi recall và câu trả lời bullet |
| `live_support.py` | Helper cho chế độ live: đọc text và usage từ message LangChain, tạo `SummarizationMiddleware` |
| `agent_baseline.py` | Agent A: chỉ có transcript trong thread, không có `User.md` |
| `agent_advanced.py` | Agent B: short-term + `User.md` + compact; live = `create_agent` + `InMemorySaver` + tool User.md + `dynamic_prompt` + summarization |
| `benchmark.py` | Standard + Long-Context Stress benchmark (6 cột), `--details`, `--sweep`, `--live`, `--json` |
| `test_agents.py` | 24 test: User.md, compact trigger/bounded summary, cross-session recall, correction/noise/question guardrails, prompt load |

Chạy từ root repo:

```bash
python src/benchmark.py            # offline, deterministic
python src/benchmark.py --sweep    # thêm bảng sweep ngưỡng compact
python src/benchmark.py --live     # dùng provider trong .env (+ LLM judge)
pytest src/test_agents.py -v
```

Phân tích kết quả nằm trong [`../REPORT.md`](../REPORT.md).
