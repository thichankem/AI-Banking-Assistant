---
name: product-advice
description: So sánh sản phẩm A-Bank như thẻ, khoản vay, tiết kiệm, hạn mức, biểu
  phí và ưu đãi. Dùng khi khách muốn chọn phương án theo nhu cầu cụ thể.
compatibility: Các thao tác nghiệp vụ sử dụng các công cụ được cấp trong A-Bank.
metadata:
  version: 1.0.0
  a-bank-category: Tư vấn
  a-bank-display-name: product-advice
  a-bank-tool: search_banking_knowledge
---

# product-advice

## Cách làm

1. Xác định tên sản phẩm và tiêu chí. Hỏi lại nếu thiếu đối tượng hoặc kỳ hạn làm thay đổi so sánh.
2. Tra bằng `search_banking_knowledge`, dùng `exact` khi cần khớp tên. Đối chiếu tên dòng, kỳ hạn, đơn vị và điều kiện cùng nhau.
3. Khi so sánh nhiều phương án, dùng bảng với các cột cùng tiêu chí. Gắn nguồn cho các mức phí, hạn mức hoặc điều kiện.
4. Nếu khách muốn xem ví dụ tính tiền, dùng skill `financial-calculator` và nói rõ giả định.

Nếu nguồn có nội dung khác nhau cho cùng đối tượng và điều kiện, nêu cả hai thay vì tự chọn giá trị.
