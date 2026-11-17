---
name: financial-calculator
description: Tính khoản trả nợ theo dư nợ giảm dần và mô phỏng tiền lãi tiết kiệm.
  Dùng khi khách yêu cầu phép tính hoặc ví dụ lãi suất, tỷ giá hoặc kỳ hạn.
compatibility: Các thao tác nghiệp vụ sử dụng các công cụ được cấp trong A-Bank.
metadata:
  version: 2.0.0
  a-bank-category: CALCULATION
  a-bank-display-name: Tính toán Tài chính & Lịch Trả Nợ Vay
---

# Tính toán Tài chính & Lịch Trả Nợ Vay

## Cách làm

1. Xác nhận các tham số ảnh hưởng đến phép tính. Với trả nợ, cần tiền gốc, lãi suất năm và kỳ hạn tháng. Với tiết kiệm, cần tiền gốc và kỳ hạn.
2. Chọn `operation` phù hợp. Xem [bảng thao tác](references/operations.md) nếu cần đối chiếu tham số.
3. Giải thích kết quả bằng VND và nêu giả định. Với khoản vay, phân biệt khoản trả tháng đầu, tổng lãi và tổng phải trả.
4. Đánh dấu lãi suất, tỷ giá và lãi tiết kiệm mẫu là mô phỏng. Nếu khách hỏi biểu niêm yết, tra tài liệu A-Bank đúng sản phẩm và điều kiện.

Ví dụ, chuyển “12 tháng” thành `term_months: 12`. Giữ nguyên lãi suất và số tiền khách đưa.
