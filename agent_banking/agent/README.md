# Mở rộng agent ngân hàng

Agent đọc lại danh mục khi bắt đầu lượt chat. Agent chính và các subagent dùng cùng một danh mục trong lượt đó. Thêm hoặc đổi tệp JSON sẽ có hiệu lực ở lượt tiếp theo.

## Thêm tool

Tạo thư mục `agent_banking/tools/extensions/` nếu cần. Trong tệp `document_copy.py`, khai báo hàm Python trả về một dictionary.

~~~python
def estimate_copy_cost(pages: int) -> dict:
    return {"estimated_cost_vnd": pages * 1000, "simulation": True}
~~~

Thêm tệp `agent_banking/tools/estimate_copy_cost.json`. Dùng tên duy nhất, khai báo schema cho tham số và chọn subagent được dùng tool nếu cần. Danh sách ví dụ cấp quyền cho `financial_analysis`.

~~~json
{
  "name": "estimate_copy_cost",
  "description": "Tính chi phí sao lục giả lập theo số trang.",
  "parameters": {
    "type": "object",
    "properties": {"pages": {"type": "integer", "minimum": 1, "maximum": 100}},
    "required": ["pages"]
  },
  "handler": {
    "module": "agent_banking.tools.extensions.document_copy",
    "function": "estimate_copy_cost"
  },
  "policy": {"access": "read_only", "workers": ["financial_analysis"]}
}
~~~

Tool mở rộng chạy bằng mã Python do người phát triển đặt trong project. Runtime kiểm tra namespace, schema, quyền và tham số trước khi gọi. Khi sửa mã Python của module đã được import, khởi động lại tiến trình để dùng mã mới.

## Thêm skill dùng tool sẵn có

Thêm tệp `agent_banking/skills/estimate_copy_cost_skill.json`. `arg_mapping` ánh xạ từ tên tham số của tool sang tên tham số của skill.

~~~json
{
  "skill_id": "estimate_copy_cost_skill",
  "description": "Ước tính chi phí sao lục giả lập cho số trang khách cung cấp.",
  "parameters_schema": {
    "type": "object",
    "properties": {"page_count": {"type": "integer", "minimum": 1, "maximum": 100}},
    "required": ["page_count"]
  },
  "execution": {
    "type": "registered_tool",
    "tool": "estimate_copy_cost",
    "arg_mapping": {"pages": "page_count"}
  }
}
~~~

Skill chỉ xuất hiện khi tool đích đã đăng ký và được phép đọc hoặc tính toán. Subagent nhận skill này theo quyền của tool đích.


## Chuẩn hóa câu hỏi ngân hàng

Tool analyze_banking_query và skill banking_query_understanding dùng cùng bộ xử lý xác định. Agent chọn gọi khi cần làm rõ câu hỏi. Tra cứu RAG tự dùng bộ xử lý này để chuẩn bị từ khóa và bộ lọc tìm kiếm. Chạy batch retrieval với tệp queries.json, queries.txt hoặc parsed_queries.json qua tham số --input.
