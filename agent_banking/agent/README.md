# Mở rộng agent ngân hàng

## Trả lời câu hỏi của agent

Trong web chat, chọn một đáp án ngay trên thẻ khi agent cần làm rõ hoặc xin chốt
phương án. Có thể dùng **Khác** để nhập câu trả lời khi làm rõ, hoặc chọn
**Hủy yêu cầu** hay **Không tiếp tục** để dừng. Tải lại cùng tab sẽ lấy lại thẻ
đang chờ. Checkpoint SQLite giữ các bước đã hoàn thành khi server khởi động lại.

Với API local, chat đồng bộ trả `status: awaiting_input` và SSE kết thúc bằng
sự kiện `awaiting_input`. Gửi `session_id`, `interaction_id` (từ `interaction.id`)
và `choice_id` (từ `interaction.options`) đến `POST /api/agent/resume`.
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
