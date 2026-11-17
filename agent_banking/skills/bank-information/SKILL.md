---
name: bank-information
description: Tra cứu giờ làm việc, kênh hỗ trợ, địa chỉ và thông tin chung của A-Bank.
  Dùng khi khách muốn biết liên hệ ở đâu hoặc khi nào.
compatibility: Các thao tác nghiệp vụ sử dụng các công cụ được cấp trong A-Bank.
metadata:
  version: 1.0.0
  a-bank-category: Thông tin chung
  a-bank-display-name: bank-information
  a-bank-tool: search_banking_knowledge
---

# bank-information

## Cách làm

1. Xác định kênh hoặc địa điểm khách hỏi. Nếu câu trả lời phụ thuộc vào khu vực hoặc loại yêu cầu, hỏi lại đúng thông tin đó.
2. Tìm nguồn bằng `search_banking_knowledge`, kết hợp tên kênh với loại hỗ trợ. Phân biệt trợ giúp trong A-Bank Plus, chi nhánh và tổng đài.
3. Trả kênh, cách tiếp cận và giờ phục vụ ghi trong cùng nguồn. Gắn nguồn ngay sau thông tin có thể sử dụng.

Nếu nguồn không ghi giờ hoặc số liên hệ, nói rõ phần chưa xác minh. Không dùng thông tin của ngân hàng khác hoặc số từ công cụ giả lập.
