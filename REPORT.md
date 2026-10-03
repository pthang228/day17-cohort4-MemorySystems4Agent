# Báo cáo Day 17: Memory Systems for AI Agent

Báo cáo gồm kết quả benchmark, phân tích trade-off và các phần bonus. Toàn bộ số liệu bên dưới lấy từ chế độ **offline (deterministic)**, nên chạy lại sẽ ra đúng các con số này:

```bash
python src/benchmark.py --sweep --details
pytest src/test_agents.py -v
```

Cấu hình mặc định: `COMPACT_THRESHOLD_TOKENS=800`, `COMPACT_KEEP_MESSAGES=4`, token ước lượng bằng `ceil(len(text)/4)`.

## 1. Kiến trúc: tách bạch 3 lớp memory

| Lớp | Baseline | Advanced | Nằm ở đâu |
|---|---|---|---|
| Short-term (trong thread) | Có: giữ **toàn bộ** transcript và gửi lại mỗi lượt | Có: chỉ giữ `keep_messages` message gần nhất ở dạng nguyên văn | `SessionState` / `CompactMemoryManager` |
| Persistent (`User.md`) | Không | Có: fact ổn định của từng user, sống qua thread và qua cả lần khởi động lại process | `UserProfileStore` → `state/profiles/<user>/User.md` |
| Compact (tóm tắt) | Không | Có: khi thread vượt ngưỡng thì gộp message cũ vào summary (tối đa 8 dòng) | `CompactMemoryManager.compact()` |

Luồng một lượt của Advanced:

```
message → extract_profile_candidates() → lọc theo confidence ≥ 0.6 → upsert_facts() vào User.md
        → CompactMemoryManager.append() (tự compact nếu vượt ngưỡng)
        → prompt = system + User.md + summary + recent messages → trả lời → cộng token
```

Để so sánh công bằng, cả hai agent dùng chung một "LLM offline" (`src/responder.py`). Khác biệt duy nhất giữa chúng là **được phép đọc memory nào**: Baseline chỉ thấy fact xuất hiện trong thread hiện tại, Advanced đọc `User.md`. Ở chế độ live (`--live`), Baseline là `create_agent + InMemorySaver`. Advanced là `create_agent + InMemorySaver`, cộng thêm `dynamic_prompt` để chèn `User.md`, 3 tool (`read_user_memory`, `save_user_fact`, `edit_user_memory`) và `SummarizationMiddleware`. Live chạy được với 6 provider: openai, custom, gemini, anthropic, ollama, openrouter.

`User.md` sau Standard Benchmark:

```markdown
## Stable facts
- name: DũngCT
- location: Huế
- profession: MLOps engineer
- response_style: ngắn gọn, bullet, có ví dụ thực chiến, có số liệu minh họa, có cấu trúc, rõ ý, nhấn trade-off
- interests: RAG, AI agent, Python, AI ứng dụng
- favorite_drink: cà phê sữa đá
- favorite_food: mì Quảng
- pet: corgi tên Bơ

## Change log
- location: Đà Nẵng -> Huế
- profession: backend engineer -> MLOps engineer
```

## 2. Kết quả benchmark

### Standard Benchmark (`data/conversations.json`, 10 hội thoại, 14 câu recall)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|------:|-------:|-----:|-----:|----:|--:|
| Baseline | 4,030 | 23,860 | 0.00 | 0.10 | 0   | 0 |
| Advanced | 4,094 | 45,435 | 1.00 | 1.00 | 586 | 0 |

So với Baseline, Advanced tốn thêm **+1.6% agent tokens** và **+90.4% prompt tokens**, đổi lại recall tăng **+1.00**.

### Long-Context Stress Benchmark (`data/advanced_long_context.json`, 16 lượt dài, 3 câu recall)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|------:|-------:|-----:|-----:|----:|--:|
| Baseline | 2,827 | 24,304 | 0.00 | 0.10 | 0   | 0 |
| Advanced | 2,945 | 14,330 | 1.00 | 1.00 | 454 | 4 |

Advanced dùng **ít hơn 41% prompt tokens** so với Baseline mà recall vẫn là 1.00.

### Thử các ngưỡng compact khác nhau (stress set, `--sweep`)

| Config | Prompt tokens processed | So với Baseline | Compactions | Recall |
|---|---:|---:|---:|---:|
| Baseline (không compact) | 24,304 | - | 0 | 0.00 |
| Advanced @ 300 | 11,439 | -52.9% | 28 | 1.00 |
| Advanced @ 600 | 12,727 | -47.6% | 7 | 1.00 |
| **Advanced @ 800 (mặc định)** | **14,330** | **-41.0%** | **4** | **1.00** |
| Advanced @ 1200 | 17,143 | -29.5% | 2 | 1.00 |
| Advanced @ 2000 | 20,525 | -15.5% | 1 | 1.00 |
| Advanced @ 4000 | 28,721 | +18.2% | 0 | 1.00 |

## 3. Phân tích

### Vì sao Advanced recall tốt hơn Baseline?
Câu hỏi recall được hỏi ở **thread mới**. Baseline chỉ có within-session memory, nên sang thread mới nó không còn gì và trả lời "chưa có thông tin này trong phiên hiện tại" (recall = 0). Đây là hành vi đúng, không phải lỗi. Advanced đọc `User.md`, nơi các fact ổn định đã được trích ra từ các phiên trước. Fact đúng ngay cả với các case khó:

- **Correction**: Đà Nẵng → Huế (standard) và Huế → Đà Nẵng (stress); backend → MLOps engineer.
- **Nhiễu**: "product manager" chỉ là câu đùa và "Hà Nội" chỉ là nơi đi họp, nên cả hai đều không được ghi.

### Vì sao Advanced tốn hơn ở hội thoại ngắn?
Ở Standard, mỗi thread chỉ khoảng 10 lượt ngắn và chưa bao giờ chạm ngưỡng 800 tokens, nên compact không kích hoạt (0 compactions). Trong khi đó, **mỗi lượt** Advanced phải kéo theo `User.md` (~150 tokens) và phần hướng dẫn memory. Đây là chi phí cố định cộng vào mọi lượt, không bù lại được bằng gì, nên prompt tokens tăng +90%. Agent tokens chỉ tăng nhẹ (+1.6%) vì lời xác nhận của Advanced dài hơn một chút ("Đã cập nhật User.md: ..."). Sweep @ 4000 cho thấy điều tương tự ở hội thoại dài: nếu ngưỡng quá cao thì compact không xảy ra, và Advanced **đắt hơn** Baseline 18%. Persistent memory luôn có giá. Compact mới là thứ trả lại số token đó.

### Vì sao compact giúp Advanced thắng ở hội thoại dài, và vì sao nó tối ưu chủ yếu *prompt tokens processed*?
- Baseline gửi lại toàn bộ thread ở mỗi lượt, nên tổng prompt tăng theo **bậc hai** với số lượt (Σ k·m ≈ n²·m/2). Advanced giữ ngữ cảnh mỗi lượt gần như **cố định**, khoảng ngưỡng + User.md + summary có giới hạn. Vì vậy tổng của nó chỉ tăng **tuyến tính**. Hội thoại càng dài thì khoảng cách càng lớn (test `test_compact_reduces_prompt_load_on_long_thread` kiểm tra ngữ cảnh ở lượt cuối của Advanced nhỏ hơn một nửa của Baseline).
- Compact **không** giảm *agent tokens only*. Số token người dùng nói và agent trả lời là như nhau ở cả hai agent (Advanced còn cao hơn 4%). Thứ compact cắt bớt là phần **ngữ cảnh lặp lại** mỗi lượt. Đây cũng là phần chiếm phần lớn chi phí API thực tế.
- Trade-off của ngưỡng: ngưỡng thấp (300) tiết kiệm nhiều nhất nhưng compact gần như mọi lượt (28 lần). Summary là heuristic nên bị mất dữ kiện (lossy), và các chi tiết trong thread như số liệu X-59 hay WMO bị nén sớm. Ở chế độ live, mỗi lần compact còn là một lần gọi LLM để tóm tắt. Ngưỡng cao thì giữ ngữ cảnh tốt hơn nhưng tiết kiệm ít. Mức 800 là điểm cân bằng: chỉ compact 4 lần và vẫn giảm 41%.
- Recall vẫn bằng 1.00 ở mọi ngưỡng vì **fact quan trọng không nằm trong summary mà nằm trong `User.md`**. Summary chỉ phục vụ follow-up trong thread. Tách hai vai trò này ra là điểm thiết kế quan trọng nhất: compact được phép làm mất dữ kiện vì những gì cần nhớ lâu đã được lưu ở chỗ khác.

### Memory file tăng trưởng thế nào, và có rủi ro gì?
- `User.md` tăng **586 bytes** sau 10 phiên và **454 bytes** sau stress test. Kích thước bị chặn vì nó lưu **trạng thái hiện tại** chứ không lưu log hội thoại: fact đơn trị bị ghi đè, `interests` giữ tối đa 6 mục, `Change log` giữ tối đa 6 dòng.
- Kích thước đó vẫn là chi phí thật: mỗi byte của `User.md` được gửi ở **mọi lượt của mọi thread**. Đó chính là nguồn của mức +90% ở Standard.
- Rủi ro **lưu sai fact**: một fact sai sẽ bị "nhớ mãi" và được tiêm vào mọi prompt sau đó, tệ hơn nhiều so với Baseline quên. Vì vậy cần các guardrail ở mục 4.
- Rủi ro **style phình to**: `response_style` tích lũy 7 đặc điểm qua 10 phiên. Mỗi đặc điểm đều đúng, nhưng cả chuỗi bắt đầu dài và khó áp dụng hết cùng lúc.
- Rủi ro **privacy**: `User.md` là PII ở dạng plain text, nên cần quyền xóa/sửa cho người dùng và không được commit lên repo (`state/` đã nằm trong `.gitignore`).

## 4. Bonus

| Bonus | Cài đặt | Giải quyết vấn đề gì | Rủi ro / giới hạn |
|---|---|---|---|
| **Confidence threshold** | `extract_profile_candidates()` gán confidence cho từng candidate; chỉ những candidate ≥ `PROFILE_CONFIDENCE_THRESHOLD = 0.6` mới được ghi. Bị đưa về 0: câu đùa (`đùa`, `giả sử`) và phủ định (`không còn`, `chứ không`, `đừng`). Bị hạ điểm: thì quá khứ (`lúc đầu`, `trước đó`) và câu giả định (`Nếu ...`). | "product manager" (câu đùa), "backend engineer" trong "đừng nói backend engineer" và "Huế" trong "Lúc đầu mình nói hiện ở Huế" không lọt vào `User.md`. | Dựa trên rule tiếng Việt nên có thể bỏ sót cách nói lạ (recall giảm) hoặc nhận nhầm. Ngưỡng cao hơn thì an toàn hơn nhưng nhớ ít hơn. |
| **Không lưu câu hỏi / thông tin tạm thời** | Câu có `?`, `là gì`, `ở đâu`... và câu có `tạm thời` bị bỏ qua khi trích fact. | "Nếu ai đó nhắc Huế, Hà Nội hay product manager...?" không làm hỏng profile. "Tạm thời cho cuộc benchmark này..." không thành preference lâu dài. | Câu hỏi tu từ có chứa fact sẽ bị bỏ qua. |
| **Conflict handling** | Fact đơn trị bị ghi đè, giá trị cũ chuyển xuống `Change log` (`location: Đà Nẵng -> Huế`). Khi chạy qua cùng một message, candidate hợp lệ cuối cùng thắng. `response_style` được merge theo slot (độ dài, format, ví dụ...) thay vì ghi đè cả chuỗi, nên "bullet" không xóa mất "3 bullet". "corgi" cũng không xóa "corgi tên Bơ". | Agent luôn trả lời fact mới nhất mà vẫn giữ lại lịch sử để giải thích. Test `test_correction_overwrites_old_fact` kiểm tra câu trả lời không còn chữ "backend". | Nếu chính bản đính chính là sai thì nó vẫn ghi đè fact đúng. Change log có giới hạn nên lịch sử rất cũ sẽ mất. |
| **Entity extraction có cấu trúc** | Schema cố định gồm 8 field (`FACT_LABELS`): name, location, profession, response_style, interests, favorite_drink, favorite_food, pet. File được render và parse lại dưới dạng `- key: value`. | Câu trả lời recall chỉ cần tra field, nên không cần cả LLM đọc lại toàn văn. `User.md` gọn và diff được. | Thông tin nằm ngoài schema (ví dụ lịch chạy bộ 6h sáng) sẽ không được nhớ. Ở chế độ live, tool `save_user_fact` từ chối key lạ. |
| **Memory decay** | `interests` sắp xếp theo lần nhắc gần nhất và giữ tối đa 6 mục. `Change log` giữ 6 dòng. Summary của compact giữ 8 dòng mới nhất. | Chặn `User.md` và summary phình vô hạn, nên chi phí mỗi lượt có trần. | Đây là decay theo độ mới, không theo độ quan trọng: một sở thích cốt lõi lâu không được nhắc lại vẫn có thể bị đẩy ra. |

## 5. Kết quả live với LLM thật (DeepSeek `deepseek-chat`)

Chạy bằng `python src/benchmark.py --live --details`. Provider `custom` trỏ tới endpoint OpenAI-compatible của DeepSeek. Response quality ở đây do LLM-as-judge chấm (`judge_model` cũng là DeepSeek). Prompt tokens lấy từ `usage_metadata` thật của provider.

| Benchmark | Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---:|---:|---:|---:|---:|---:|
| Standard | Baseline | 21,507 | 155,527 | 0.21 | 0.00 | 0 | 0 |
| Standard | Advanced | 18,653 | 318,961 | 1.00 | 0.98 | 1,448 | 33 |
| Stress | Baseline | 4,589 | 57,297 | 0.00 | 0.07 | 0 | 0 |
| Stress | Advanced | 6,288 | 135,852 | 1.00 | 1.00 | 1,130 | 23 |

**Recall vẫn đúng như kỳ vọng.** Advanced đạt 1.00 ở cả hai bộ, và trả lời đúng cả các câu bẫy (Đà Nẵng chứ không phải Huế; MLOps engineer chứ không phải product manager). Baseline có recall 0.21 ở Standard nhưng đây là **nhiễu của metric so khớp chuỗi**: LLM lặp lại từ trong câu hỏi ("Huế"), gợi ý lựa chọn ("ngắn gọn, chi tiết...?") hoặc bịa ("DũngCT là streamer game..."). Judge chấm các câu này 0 điểm, cho thấy metric substring cần được bổ sung bằng judge.

**Nhưng ở live, compact không tiết kiệm token: Advanced tốn +137% prompt tokens ở stress.** Mình chạy chẩn đoán 6 lượt stress đầu tiên và thấy:

| Lượt | Số lần gọi model | Input tokens mỗi lần gọi | Độ dài summary (ký tự) |
|---:|---:|---|---:|
| 0 | 2 | 974, 1,536 | 1,306 |
| 1 | 2 | 2,040, 2,141 | 1,664 |
| 3 | 2 | 3,078, 3,475 | 3,908 |
| 5 | 2 | 4,632, 4,533 | 5,953 |

1. **Mỗi lượt có 2 lần gọi model.** Model gọi tool `save_user_fact` rồi mới trả lời, nên toàn bộ ngữ cảnh bị gửi hai lần. Việc này dư thừa, vì lớp trích fact rule-based đã cập nhật `User.md` trước khi gọi model.
2. **Summary của `SummarizationMiddleware` không bị chặn.** Prompt mặc định yêu cầu "extract the highest quality context" theo 4 mục, nên mỗi lần compact LLM lại tóm tắt cả summary cũ lẫn message mới. Summary phình từ 1,3k lên 6k ký tự chỉ sau 6 lượt. Message gần đây được cắt bớt, nhưng ngữ cảnh vẫn tăng tuyến tính.
3. Schema của 3 tool cộng với `User.md` là chi phí cố định cho mỗi lần gọi.

**Bài học.** Compact chỉ tiết kiệm khi **summary có trần kích thước**. Bản offline có trần (tối đa 8 dòng) nên giảm 41%. Bản live dùng prompt tóm tắt mặc định thì không có trần, nên không tiết kiệm. Hướng sửa:

- Truyền `summary_prompt` giới hạn độ dài (ví dụ ≤ 120 từ, chỉ giữ ý chính, vì fact đã nằm trong `User.md`).
- Bỏ tool ghi fact khỏi lượt thường, hoặc dặn model chỉ gọi tool khi gặp fact chưa có trong `User.md`.

Đây cũng là minh họa trực tiếp cho câu "hệ thống mạnh hơn nhưng phức tạp hơn và cần guardrail tốt hơn".

## 6. Hạn chế và hướng mở rộng
- Trích fact bằng regex chỉ chạy tốt trên dataset này. Ở production nên dùng một LLM extractor trả về JSON kèm confidence, nhưng giữ nguyên lớp guardrail (threshold, chống câu hỏi, change log).
- Token đang được ước lượng bằng `len/4`. Ở chế độ live, prompt tokens lấy từ `usage_metadata` của provider nên chính xác hơn.
- Response quality ở chế độ offline là heuristic, tính theo độ phủ fact × (ngắn gọn, có cấu trúc). Với `--live`, benchmark dùng `judge_model` làm LLM-as-judge.
