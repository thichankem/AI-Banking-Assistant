---
name: rag-knowledge-retrieval
description: Truy xuất bằng chứng từ kho A-Bank đã lập chỉ mục. Dùng để kiểm tra quy
  định, biểu phí, hạn mức, thủ tục, hướng dẫn ứng dụng hoặc bảng dữ liệu.
compatibility: Các thao tác nghiệp vụ sử dụng các công cụ được cấp trong A-Bank.
metadata:
  version: 2.0.0
  a-bank-category: RAG
  a-bank-display-name: Truy xuất Tri thức Quy chế Ngân hàng (Agentic RAG)
---

# Truy xuất Tri thức Quy chế Ngân hàng (Agentic RAG)

## Cách làm

1. Tạo `query` tự chứa đối tượng, câu hỏi và điều kiện đã biết. Dùng `exact` để ràng buộc tên hoặc mã khi cần.
2. Đọc kết quả theo đúng tên sản phẩm và mục. Nếu kết quả rỗng hoặc lệch, đổi cách diễn đạt, thêm mã rõ hơn hoặc tách câu hỏi theo đối tượng.
3. Khi sử dụng bảng, đối chiếu cột, đơn vị, dòng sản phẩm và ghi chú. Gắn nhãn nguồn của lượt hiện tại ngay sau nhận định.
4. Khi có nguồn đối nghịch, kiểm tra điều kiện của từng nguồn trước khi kết luận. Xem [cách rà bằng chứng](references/evidence-check.md) nếu cần.

Chỉ sử dụng chunks đã truy xuất. Tài liệu A-Bank thiếu ngày vẫn dùng được theo quy ước của kho nội bộ. Không suy ra hiệu lực từ mã phiên bản nội dung.
