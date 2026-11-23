# Mở rộng agent ngân hàng

## Trả lời câu hỏi của agent

Trong web chat, chọn một đáp án (hoặc nhiều đáp án nếu thẻ bật `multi_select`) ngay trên thẻ khi agent cần làm rõ hoặc xin chốt
phương án. Có thể dùng **Khác** để nhập câu trả lời khi làm rõ, hoặc chọn
**Hủy yêu cầu** hay **Không tiếp tục** để dừng. Tải lại cùng tab sẽ lấy lại thẻ
đang chờ. Checkpoint SQLite giữ các bước đã hoàn thành khi server khởi động lại.

Với API local, chat đồng bộ trả `status: awaiting_input` và SSE kết thúc bằng
sự kiện `awaiting_input`. Gửi `session_id`, `interaction_id` (từ `interaction.id`)
và `choice_id` (hoặc mảng `choice_ids` khi `interaction.multi_select` là `true`)
từ `interaction.options` đến `POST /api/agent/resume`.
Thêm `text` nếu chọn `other`. Gửi `session_id` đến `POST /api/agent/status`
để lấy lại thẻ hoặc kết quả tiếp tục gần nhất. ID phiên là dữ liệu truy cập của
phiên trong bản local và cần được giữ riêng.

## Đăng ký tool và skill

Skill dùng [Agent Skills Open Standard](https://agentskills.io/specification).
Mỗi thư mục bên trong `agent_banking/skills/` có một tệp `SKILL.md`.
Tên thư mục phải trùng với `name` trong YAML frontmatter, ví dụ
`service-troubleshooting`.

Đầu mỗi lượt, agent đọc danh mục tên, mô tả và nhóm. Khi chọn một skill,
`inspect_skill` nạp nội dung Markdown và danh sách tệp trong `references/`.
Agent có thể yêu cầu một tệp Markdown bằng đường dẫn tương đối từ danh sách
đó. Hướng dẫn đã nạp được giữ trong ngữ cảnh của lượt, kể cả khi lượt đang tạm
dừng. Bản mới của `SKILL.md` và cấu hình binding có hiệu lực từ lượt mới.

## Biểu đồ và sơ đồ trong web chat

Chạy các lệnh sau từ thư mục gốc dự án với Node.js 22.12 trở lên trước khi chạy FastAPI
hoặc đóng gói bản phát hành. Triển khai toàn bộ `static/visualizations/` cùng thư mục `static/`.

```powershell
npm.cmd --prefix frontend/visualizations ci
npm.cmd --prefix frontend/visualizations run build
```

Ví dụ, hỏi “Vẽ biểu đồ so sánh doanh số giả lập tháng 1 là 12, tháng 2 là 18 và tháng 3 là 15”
rồi tiếp tục “Đổi sang biểu đồ đường, giữ nguyên số liệu”. Agent có thể tham chiếu hai hình gần nhất
trong cùng phiên. Mỗi lần chỉnh sửa tạo một phiên bản mới trong lịch sử chat.

Agent có thể chủ động gọi `present_visualization` khi hình ảnh hoặc tương tác giúp
khách hiểu thông tin và so sánh lựa chọn. API đồng bộ, sự kiện SSE `done` và lịch sử
trả cấu hình đã kiểm duyệt trong `visualization`. Web chat dựng phần trực quan ngay
cùng câu trả lời và trích dẫn.

Với `schema_version: 1`, `engine` và `kind` xác định cách render, còn `id` và `revision`
đánh dấu phiên bản trong phiên chat. ECharts nhận trực tiếp `x_name`, `x`, `series` và
`config`. Mermaid nhận `source` và `theme`. Cây `layout` trong tệp quyết định cách dựng
giao diện. Agent tự đặt tiêu đề và nhóm, lồng hàng/cột/lưới và bố trí ô nhập, kết quả,
công thức, biểu đồ hoặc bảng dữ liệu. Nếu dùng CSP khi triển khai, cho phép tải
module cùng origin và `blob:` trong `img-src`.

### Bảng tính tương tác

Thử hỏi “Tạo bảng tính khoản vay trả góp đều 650 triệu VND, lãi suất giả định
9,5%/năm, kỳ hạn 16 năm. Cho phép chỉnh cả ba biến và hiện công thức, tổng gốc lãi”.
Agent có thể chọn các trường, khoảng kéo, bước kéo, đơn vị và phép tính theo yêu cầu.
Ô nhập số cho phép nhập chính xác trong khoảng đã chọn và nhận cả dấu phẩy lẫn dấu chấm
thập phân. Các kết quả phụ thuộc cập nhật
ngay trên trình duyệt. Các số chỉnh trong bảng chỉ tồn tại trong trang đang mở.

Cấu hình dùng `engine: simulator` và `kind: calculator` với `inputs`,
`calculations` và `assumptions`. Backend kiểm tra các tham chiếu và chạy thử công thức tại
giá trị mặc định. Trình duyệt dùng các phép tính được tích hợp sẵn. Agent có thể sửa cấu
hình từ lời nhắn tiếp theo và lưu cấu hình mới vào lịch sử phiên.

## Thêm tool

Định nghĩa công khai dùng cấu trúc [MCP Tool](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)
và JSON Schema Draft 2020-12. A-Bank lưu định nghĩa bằng YAML và gọi tool qua
registry nội bộ. `get_tool_contracts()` xuất định nghĩa công khai.
`get_tool_specs()` chuyển cùng schema đầu vào sang dạng function calling.

Đặt hàm trong `agent_banking/tools/extensions/document_copy.py`:

~~~python
def estimate_copy_cost(pages: int) -> dict:
    return {"estimated_cost_vnd": pages * 1000, "simulation": True}
~~~

Tạo `agent_banking/tools/estimate_copy_cost/tool.yaml` với tên trùng thư mục:

~~~yaml
name: estimate_copy_cost
title: Ước tính phí sao lục
description: Tính chi phí sao lục giả lập theo số trang.
inputSchema:
  type: object
  additionalProperties: false
  properties:
    pages: {type: integer, minimum: 1, maximum: 100}
  required: [pages]
outputSchema:
  type: object
  properties:
    estimated_cost_vnd: {type: integer}
    simulation: {type: boolean}
  required: [estimated_cost_vnd, simulation]
annotations:
  readOnlyHint: true
~~~

Trong cùng thư mục, thêm `runtime.yaml`:

~~~yaml
category: CALCULATION
handler:
  module: agent_banking.tools.extensions.document_copy
  function: estimate_copy_cost
policy:
  access: read_only
  workers: [financial_analysis]
~~~

Registry kiểm tra hợp đồng đầu vào trước và sau guardrail, gọi handler trong
namespace được cấp rồi kiểm tra đầu ra trước khi trả kết quả. `annotations`
mô tả hành vi của tool. Handler, quyền và danh sách subagent nằm trong
`runtime.yaml`. Bộ xử lý batch vẫn nhận dữ liệu JSON và các tham số cũ đã hỗ
trợ. Khi sửa mã Python của module đã import, khởi động lại tiến trình để dùng
mã mới.

## Thêm skill

Tạo `agent_banking/skills/estimate-copy-cost/SKILL.md`:

~~~markdown
---
name: estimate-copy-cost
description: Ước tính chi phí sao lục giả lập khi khách đã cho biết số trang.
metadata:
  a-bank-category: "Nghiệp vụ"
  a-bank-tool: "estimate_copy_cost"
---

# Ước tính chi phí sao lục

Hỏi số trang nếu khách chưa nêu. Dùng tham số `pages` để tính. Ghi rõ đây là phí mẫu.
~~~

Với `metadata.a-bank-tool`, skill dùng schema và quyền của tool đọc hoặc tính
toán đã đăng ký. Nếu cần ánh xạ tham số, cấu hình binding trong
`assets/a-bank-runtime.yaml`. Các bộ thực thi chuyên biệt đi kèm repo cũng dùng
tệp cấu hình này. Một skill chỉ có hướng dẫn thì chỉ cần `name`, `description`
và nội dung Markdown. Agent có thể áp dụng hướng dẫn bằng công cụ đã được cấp.

Các ID cũ dùng dấu gạch dưới vẫn được chấp nhận khi gọi. Trường `allowed-tools`
không cấp thêm quyền trong A-Bank. Runtime đọc các tệp tham khảo Markdown
trong gói đã kích hoạt và giữ quyền thực thi ở registry công cụ.


## Chuẩn hóa câu hỏi ngân hàng

Tool `analyze_banking_query` và skill `banking-query-understanding` dùng cùng bộ xử lý xác định. Agent chọn gọi khi cần làm rõ câu hỏi. Tra cứu RAG tự dùng bộ xử lý này để chuẩn bị từ khóa và bộ lọc tìm kiếm. Chạy batch retrieval với tệp queries.json, queries.txt hoặc parsed_queries.json qua tham số --input.

## Kiểm tra độ phủ và đánh giá A-Bank

Từ thư mục gốc, sau khi xuất publication bằng `agent_banking.chunking.run_chunking`,
dùng `python -m agent_banking.evaluator.harness_index --publication <dir> --index <dir>`
để tạo chỉ mục riêng. Lệnh chỉ chấp nhận publication đã kiểm tra đủ tài liệu và
khớp mã băm với bộ PDF/DOCX hiện tại. `harness_snapshot.json` ghi scope và mã
băm để dùng cùng một snapshot khi so sánh. Có thể chọn collection đã xác minh
cho chat local qua `ABANK_QDRANT_STORAGE_PATH` và `ABANK_KNOWLEDGE_SCOPE`.

Chạy `python -m agent_banking.evaluator.harness_eval --index <dir> --output <dir>
--version <label> --scenarios agent_banking/evaluator/harness_scenarios.json`
để đánh giá 30 ca A–F và 785 câu golden với cloud. Dùng thư mục kết quả riêng
cho mỗi phiên bản. File `results.jsonl` lưu từng ca để chạy tiếp khi gián đoạn.
Dùng `--retry-errors` để chạy lại ca hạ tầng. Bộ golden, cache lịch sử và
checkpoint chat thông thường vẫn tách khỏi lần chạy này.

## Tự động chọn model cloud

Đặt `AUTO_MODEL_ROUTING=true` trong môi trường chạy, sau đó khởi động lại server hoặc runner. Giá trị mặc định là `false`. Cấu hình một model cho mỗi vai trò đã đủ để chạy. Thêm ứng viên `deep` để router có thể chuyển model khi tác vụ cần thêm năng lực.

### Khai báo vai trò và profile

| Vai trò | Prefix môi trường | Giao thức |
| --- | --- | --- |
| Điều phối, worker và tóm tắt chat | `AGENT` | `chat` |
| Review câu trả lời | `AGENT_REVIEW` | `chat` |
| Enrichment tài liệu | `CHUNKING_TEXT` | `chat` |
| OCR ảnh và trang PDF | `CHUNKING_VISION` | `vision` |
| Đọc lại ảnh và vùng crop | `CHUNKING_VISION_REVIEW` | `vision` |
| Quyết định cấu trúc tài liệu | `CHUNKING_DECISION` | `systemone` |

Mỗi prefix nhận hai profile, `STANDARD` và `DEEP`. Ví dụ `AGENT_STANDARD_MODEL`, `AGENT_DEEP_MODEL` và `AGENT_REVIEW_STANDARD_MODEL`. Cấu hình standard kế thừa endpoint cũ của vai trò. Deep chỉ được thêm khi có biến `*_DEEP_MODEL`. Deep kế thừa URL và credential từ standard, cùng các giới hạn của endpoint cũ, có thể ghi đè riêng. Vai trò review chat cần đủ model, URL và API key, sau đó tự fallback sang ứng viên agent khi cần.

Các hậu tố cấu hình sau được đặt sau tên profile.

| Hậu tố | Ý nghĩa |
| --- | --- |
| `MODEL`, `BASE_URL`, `API_KEY` | Model ID và thông tin kết nối qua môi trường |
| `PROTOCOL` | Giao thức phù hợp vai trò trong bảng trên |
| `CONTEXT_WINDOW` | Số token context tối đa theo cấu hình triển khai model. Giá trị `0` dùng ngân sách cũ |
| `MAX_OUTPUT_TOKENS` | Trần output của model. Giá trị `0` dùng giới hạn của tác vụ |
| `TOKEN_PARAMETER` | `max_tokens` hoặc `max_completion_tokens` |
| `SEND_TEMPERATURE`, `TEMPERATURE` | Có gửi temperature và giá trị trong khoảng 0–2 |
| `REASONING_EFFORT` | Chỉ khai báo mức provider hỗ trợ, gồm `none`, `minimal`, `low`, `medium`, `high`, `xhigh` |
| `THINKING` | `true` hoặc `false`, dùng cho provider hỗ trợ `chat_template_kwargs.enable_thinking` |
| `JSON_MODE` | `on`, `off` hoặc `auto`. `auto` cho phép thêm JSON object khi sửa contract chat |
| `IMAGE_DETAIL` | `auto`, `low`, `high` khi provider Vision hỗ trợ |
| `IMAGE_TOKEN_RESERVE` | Ước lượng dự phòng token cho mỗi ảnh. Mặc định 2048, điều chỉnh theo provider và độ phân giải |
| `TIMEOUT`, `ATTEMPTS` | Timeout mỗi lần gọi và retry, vẫn chịu ngân sách chung của tác vụ |

Ví dụ cấu hình giới hạn cho model deep đã được admin đăng ký, dùng provider hỗ trợ reasoning và completion tokens.

```dotenv
AUTO_MODEL_ROUTING=true
AGENT_DEEP_MODEL=<model-id-trong-danh-muc-cloud>
AGENT_DEEP_CONTEXT_WINDOW=<gioi-han-context-do-provider-cong-bo>
AGENT_DEEP_MAX_OUTPUT_TOKENS=8192
AGENT_DEEP_TOKEN_PARAMETER=max_completion_tokens
AGENT_DEEP_SEND_TEMPERATURE=false
AGENT_DEEP_REASONING_EFFORT=high
```

Điền các placeholder trước khi chạy. Model cloud quyết định context window khả dụng. Router dùng một token dự phòng cho mỗi byte UTF-8 của prompt để kiểm tra an toàn khi chưa có tokenizer khớp model, đồng thời dành chỗ cho output và overhead. Cách tính này có thể chuyển sang model context lớn sớm hơn cần thiết. `AGENT_CONTEXT_BYTES`, nếu được đặt, tiếp tục giới hạn kích thước payload chat.

Prompt bắt buộc gồm yêu cầu, phản hồi làm rõ, phạm vi đã xác nhận và schema. Review giữ toàn bộ evidence và support đã đưa vào review. Lịch sử và quan sát được đóng gói theo các record đầy đủ. Khi các ứng viên đều thiếu context, trace ghi `context_budget_exceeded` và agent dùng đường phản hồi lỗi hiện có.

### Khi nào router chuyển model

- Coordinator ưu tiên deep khi có từ hai nguồn khác nhau, kết quả worker đã xác minh hoặc feedback yêu cầu sửa. Review ưu tiên deep khi kiểm chứng kết quả tool có số, có nhiều nguồn hoặc sửa record review.
- Review chat dùng danh mục `AGENT_REVIEW` khi đủ cấu hình, sau đó fallback sang danh mục `AGENT`.
- Decision tài liệu ưu tiên deep cho nối bảng PDF và phân cấp tiêu đề. Các endpoint `/systemone` tiếp tục dùng hợp đồng `state/questions/answers`.
- Vision gọi review khi layout thiếu cấu trúc hoặc còn bất định. Các crop ưu tiên ứng viên review nếu có. Bộ đếm gọi cloud được chia sẻ cho trang và các crop, tối đa ba lần theo cấu hình pipeline hiện có.
- Mỗi structured request chat có tối đa ba lần gọi cloud khi bật router, gồm retry, đổi model và sửa JSON. Quyết định cuối lượt sau review vẫn đi qua validator và guardrail.

### Trace và UAT

Toàn bộ tracing của dự án sử dụng **LangSmith** (`agent_banking/tracing/tracer.py`). Khi `ENABLE_TRACING=true` và `LANGSMITH_API_KEY` được cấu hình, mọi lượt chat, SSE, resume, LangGraph node, gọi model LLM (`model.attempt`), tool/RAG (`query_parser`, `top_k_hybrid_retrieval`, `top_k_cross_encoder_reranker`) và xử lý tài liệu (`document`) đều được đồng bộ lên LangSmith project (`LANGSMITH_PROJECT`). Khi chạy offline không có `LANGSMITH_API_KEY`, tracer tự động dùng `OfflineLangSmithClient` an toàn không gọi mạng; khi chạy đánh giá harness hoặc truyền `LOCAL_TRACE_EXPORT=true`, bản tóm tắt LangSmith được xuất tại `TRACE_DIR/langsmith_runs.jsonl`. Mỗi `model.attempt` ghi model thực thi, profile, role, tham số thực gửi, latency, kết quả và usage (`usage_metadata` / `usage`). Các lỗi context được ghi trước khi gửi request. Trace giữ metadata chẩn đoán, mã phiên đã băm và các ID liên kết.

API chat, SSE `done` và resume trả `trace_id` và `turn_id` của LangSmith khi bật trace. Một lần resume có trace mới và tiếp tục turn đang chờ. API xử lý ảnh chat giữ hợp đồng text hiện tại, Vision nằm trong pipeline tài liệu RAG.

Chạy test offline từ thư mục gốc repo.

```powershell
.\.venv\Scripts\python.exe -m pytest agent_banking/agent/tests agent_banking/chunking/tests -q
```

Để so sánh cloud, tạo chỉ mục riêng từ publication đủ tài liệu và `index_ready=true` bằng `agent_banking.evaluator.harness_index`. Dùng cùng chỉ mục và judge cho hai lần chạy.

```powershell
$env:AUTO_MODEL_ROUTING="false"
.\.venv\Scripts\python.exe -m agent_banking.evaluator.harness_eval --index <index> --output <baseline> --version routing-off --scenarios agent_banking/evaluator/harness_scenarios.json
$env:AUTO_MODEL_ROUTING="true"
.\.venv\Scripts\python.exe -m agent_banking.evaluator.harness_eval --index <index> --output <candidate> --version routing-on --scenarios agent_banking/evaluator/harness_scenarios.json
.\.venv\Scripts\python.exe -m agent_banking.evaluator.harness_eval --compare <baseline> <candidate>
```

`identity.json` ghi fingerprint policy và ứng viên của runtime, cùng fingerprint judge cố định. Mỗi case trong `results.jsonl` giữ trace ID. `metrics.json` thống kê p50/p95 theo request agent, số lần gọi và usage khi có. Lệnh compare trả thay đổi verdict kèm trace ID và áp dụng tiêu chí A–F/785 hiện có. Model judge dùng cấu hình legacy của agent cho cả hai lần chạy. Giữ các biến `MODEL`, `BASE_URL`, `API_KEY` của judge cố định khi thay danh mục routing.

Tắt router bằng `AUTO_MODEL_ROUTING=false` và khởi động lại tiến trình để quay về danh sách model cũ. Embedding giữ cùng model và vector space 1024 chiều theo cấu hình hiện tại.

Hướng dẫn xây lại kho tài liệu, đo truy xuất và kiểm thử chất lượng agent theo bộ 785 câu nằm trong [TUNING.md](TUNING.md).

### Tra cứu văn bản pháp luật trên web

`search_legal_web` tra các trang `/van-ban/` trên `thuvienphapluat.vn` và tạo trích dẫn có liên kết tới văn bản. Dùng từ khóa pháp luật công khai, ví dụ `Thông tư 17/2024/TT-NHNN mở tài khoản thanh toán`. Lượt chat dùng đoạn trích tìm kiếm làm căn cứ. Khi đoạn trích thiếu nội dung điều khoản hoặc thông tin hiệu lực, agent hướng người dùng tới văn bản để đối chiếu.

Điền khóa vào `.env` ở thư mục gốc repo rồi khởi động lại app. Tool xuất hiện khi có ít nhất một khóa. Thứ tự mặc định là Brave rồi Tavily. Đặt `LEGAL_WEB_SEARCH_PRIMARY=tavily` để đảo thứ tự. Agent tra API dự phòng khi API chính báo lỗi, hết hạn mức, thiếu văn bản phù hợp hoặc đoạn trích chưa nhắc tới số điều được hỏi.

```dotenv
TAVILY_API_KEY=
BRAVE_SEARCH_API_KEY=
LEGAL_WEB_SEARCH_PRIMARY=brave
```

Mỗi API có tối đa 1.000 lượt thử gọi từ dự án trong một tháng theo giờ Việt Nam. Bộ đếm lưu cục bộ trong SQLite và dùng chung giữa các tiến trình chạy cùng checkout. Tavily dùng `basic` để mỗi lượt tìm kiếm dùng một credit. Có thể giảm trần cục bộ bằng `LEGAL_WEB_SEARCH_TAVILY_MONTHLY_LIMIT` hoặc `LEGAL_WEB_SEARCH_BRAVE_MONTHLY_LIMIT` khi cùng khóa được dùng ở nơi khác.

Từ thư mục gốc repo, chạy phép so sánh với 7 truy vấn pháp lý công khai giống nhau cho cả hai API.

```powershell
.\.venv\Scripts\python.exe -m agent_banking.evaluator.legal_web_search_eval
```

Báo cáo xếp theo số truy vấn tìm đúng văn bản trong 4 kết quả đầu, số cụm từ cần có trong đoạn trích, thứ hạng trung bình nghịch đảo và độ trễ trung vị. Các lượt đánh giá dùng cùng hạn mức với agent.
