---
name: business-procedure
description: Tra quy trình, điều kiện, hồ sơ và thủ tục A-Bank. Dùng cho yêu cầu mở,
  thay đổi, đóng hoặc xử lý hồ sơ nghiệp vụ.
compatibility: Các thao tác nghiệp vụ sử dụng các công cụ được cấp trong A-Bank.
metadata:
  version: 1.0.0
  a-bank-category: Nghiệp vụ
  a-bank-display-name: business-procedure
  a-bank-tool: search_banking_knowledge
---

# business-procedure

## Cách làm

1. Xác định nghiệp vụ, đối tượng khách hàng và giai đoạn trong quy trình. Tách các điều kiện khác nhau nếu câu hỏi có nhiều trường hợp.
2. Tra bằng `search_banking_knowledge`. Đưa tên thủ tục và điều kiện vào `query`. Dùng `exact` khi khách cung cấp mã quy trình hoặc tên cần khớp.
3. Tóm tắt điều kiện, giấy tờ và các bước theo thứ tự. Gắn nguồn cho từng nhóm yêu cầu.

Khi tài liệu chỉ bao phủ một trường hợp, giới hạn đáp án trong trường hợp đó hoặc hỏi lại. Không biến tình huống minh họa thành thủ tục bắt buộc.
