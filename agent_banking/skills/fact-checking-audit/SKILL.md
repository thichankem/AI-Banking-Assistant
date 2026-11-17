---
name: fact-checking-audit
description: Kiểm tra dự thảo hoặc mức độ đáp ứng của bằng chứng A-Bank. Dùng khi
  câu trả lời có nhiều điều kiện, giá trị bảng hoặc nguồn mâu thuẫn.
compatibility: Các thao tác nghiệp vụ sử dụng các công cụ được cấp trong A-Bank.
metadata:
  version: 2.0.0
  a-bank-category: AUDIT
  a-bank-display-name: Kiểm soát Ảo giác & Thẩm định Tài liệu
---

# Kiểm soát Ảo giác & Thẩm định Tài liệu

## Cách làm

- Dùng `grade_documents` với `query` để kiểm tra tài liệu đã bao phủ câu hỏi.
- Dùng `audit_answer` với `draft_answer` để đối chiếu dự thảo. Runtime lấy tài liệu từ các nguồn đã truy xuất trong lượt.
- Kiểm tra đối tượng, điều kiện, số liệu, đơn vị và nhãn nguồn. Nếu còn thiếu, nêu chính xác dữ kiện cần tra thêm.
- Khi nguồn cùng điều kiện mâu thuẫn, giữ cả hai nguồn và nêu phần chưa thể kết luận.

Kết quả đối soát giúp sửa dự thảo. Bước review cuối vẫn kiểm tra câu trả lời trước khi gửi.
