---
name: banking-query-understanding
description: Phân tích câu hỏi tiếng Việt về A-Bank có viết tắt, thiếu dấu, lỗi gõ
  hoặc nhiều nghĩa. Dùng trước khi tra cứu nếu đối tượng hoặc tham số chưa rõ.
compatibility: Các thao tác nghiệp vụ sử dụng các công cụ được cấp trong A-Bank.
metadata:
  version: 1.0.0
  a-bank-category: KNOWLEDGE
  a-bank-display-name: Làm rõ câu hỏi ngân hàng
  a-bank-tool: analyze_banking_query
---

# Làm rõ câu hỏi ngân hàng

## Cách làm

1. Gọi `analyze_banking_query` với câu gốc. Xem dạng chuẩn hóa, từ khóa và các nghĩa ứng viên.
2. Dùng lượt trước để xác định đối tượng đang nói tới. Giữ nguyên số tiền, kỳ hạn, mã dịch vụ và tên sản phẩm khách nêu.
3. Nếu hai nghĩa dẫn tới câu trả lời khác nhau, hỏi lại bằng thẻ lựa chọn. Nếu đã đủ rõ, tạo truy vấn tự chứa đối tượng và điều kiện.

Ví dụ, chỉ mở rộng “STK” thành một nghĩa khi câu hoặc ngữ cảnh hỗ trợ. Kết quả phân tích là gợi ý cho truy xuất, không phải căn cứ về nghiệp vụ.
