# ĐỀ XUẤT ĐỀ TÀI GR2

## XÂY DỰNG TRỢ LÝ AI SỬ DỤNG KIẾN TRÚC AI AGENT HỖ TRỢ KHÁCH HÀNG TRONG LĨNH VỰC NGÂN HÀNG (HOÀN TOÀN CHƯA ĐƯỢC KIỂM CHỨNG)

*AI Banking Assistant*

---

## 1. Chi tiết bối cảnh đề tài

Trong hoạt động ngân hàng, khách hàng thường phải tra cứu một lượng lớn thông tin liên quan đến sản phẩm, quy trình, phí, hạn mức, điều kiện sử dụng dịch vụ và cách xử lý sự cố. Các thông tin này thường nằm rải rác trong ứng dụng ngân hàng, website, biểu phí, tài liệu hướng dẫn, điều khoản sản phẩm và các quy định. Việc tìm đúng thông tin và hiểu cách áp dụng vào từng tình huống cụ thể có thể gây khó khăn cho khách hàng.

**Các khó khăn chính của khách hàng:**

| Vấn đề | Mô tả |
|---|---|
| Thông tin phân tán | Khách hàng phải tìm kiếm ở nhiều nơi như ứng dụng, website, FAQ, biểu phí, điều khoản và tài liệu hướng dẫn. |
| Khó tìm đúng thông tin | Một câu hỏi có thể liên quan đến nhiều tài liệu hoặc nhiều điều khoản; tìm kiếm bằng từ khóa đơn giản dễ trả về kết quả không phù hợp. |
| Quy trình nhiều bước | Các nghiệp vụ như mở thẻ, đổi thông tin, chuyển tiền, đăng ký dịch vụ hoặc xử lý giao dịch cần thực hiện theo nhiều bước. |
| Khó mô tả và chẩn đoán lỗi | Khách hàng thường chỉ biết biểu hiện như "chuyển khoản bị treo", "thẻ bị từ chối", "không nhận được OTP" nhưng không biết nguyên nhân và cách xử lý. |
| Khó lựa chọn sản phẩm | Khách hàng phải tự so sánh thẻ, tiền gửi, khoản vay, bảo hiểm hoặc các sản phẩm đầu tư dựa trên nhiều tiêu chí. |
| Không hiểu thuật ngữ và quy định | Các khái niệm về lãi suất, hạn mức, phí, điều kiện, kỳ hạn, bảo hiểm hoặc đầu tư có thể khó hiểu đối với khách hàng phổ thông. |
| Câu hỏi cần nhiều nguồn thông tin | Một câu hỏi có thể cần kết hợp thông tin sản phẩm, quy trình, chính sách và dữ liệu giao dịch để đưa ra câu trả lời phù hợp. |
| Khó kiểm chứng câu trả lời | Khách hàng cần biết câu trả lời dựa trên tài liệu hoặc quy định nào, đặc biệt đối với thông tin về phí, lãi suất và điều kiện sản phẩm. |

Từ các vấn đề trên, đề tài hướng tới xây dựng một trợ lý AI hội thoại có khả năng hiểu yêu cầu tự nhiên của khách hàng, xác định loại nhu cầu, truy xuất đúng nguồn thông tin, sử dụng các công cụ cần thiết, xử lý nhiều bước và đưa ra câu trả lời có cấu trúc, có nguồn tham chiếu khi phù hợp.

---

## 2. Chi tiết nghiệp vụ và chức năng

Hệ thống tập trung vào bốn nhóm nhu cầu chính của khách hàng ngân hàng. Người dùng không bắt buộc phải chọn nhóm trước; AI Agent có thể tự xác định ý định từ câu hỏi và điều phối nghiệp vụ tương ứng.

### 2.1. Nhóm 1 — Nghiệp vụ

**Mục tiêu:** trả lời câu hỏi dạng *"Tôi phải làm thế nào?"* và hướng dẫn khách hàng theo từng bước bao gồm cả điều kiện cần chuẩn bị.

- Mở tài khoản hoặc đăng ký dịch vụ ngân hàng số.
- Mở, khóa hoặc đăng ký lại thẻ.
- Đăng ký khoản vay, gửi tiết kiệm hoặc các sản phẩm/dịch vụ khác.

### 2.2. Nhóm 2 — Vận hành và xử lý sự cố

**Mục tiêu:** hỗ trợ khách hàng xác định tình trạng, thu thập thông tin cần thiết và hướng dẫn cách xử lý.

- Chuyển khoản thất bại, đang xử lý hoặc đã trừ tiền nhưng người nhận chưa nhận được.
- Giao dịch thẻ bị từ chối hoặc thanh toán trực tuyến không thành công.
- ATM giữ thẻ, thẻ bị khóa hoặc không sử dụng được.
- Không nhận được OTP, không đăng nhập được ứng dụng hoặc ứng dụng báo lỗi.

### 2.3. Nhóm 3 — Tư vấn sản phẩm và tài chính

**Mục tiêu:** cung cấp thông tin, so sánh và hỗ trợ khách hàng lựa chọn sản phẩm dựa trên nhu cầu.

- **Thẻ:** so sánh phí, ưu đãi, hoàn tiền, điểm thưởng, nhu cầu du lịch hoặc mua sắm.
- **Tiền gửi và lãi suất:** so sánh kỳ hạn, lãi suất và ước tính tiền lãi.
- **Khoản vay:** cung cấp điều kiện tham khảo, ước tính khoản trả và so sánh phương án.
- **Bảo hiểm:** giải thích quyền lợi, phạm vi, phí và so sánh các gói.
- **Cổ phiếu/đầu tư:** cung cấp thông tin và phân tích tham khảo, nêu rõ rủi ro và không thay thế quyết định đầu tư của khách hàng.

### 2.4. Nhóm 4 — Cung cấp thông tin chung

- Thông tin sản phẩm hoặc dịch vụ.
- Biểu phí, hạn mức, điều kiện và thời gian xử lý.
- Giải thích thuật ngữ ngân hàng và tài chính.
- Thông tin chi nhánh, kênh hỗ trợ và giờ hoạt động.

### 2.5. Các chức năng chính

| Chức năng | Mô tả |
|---|---|
| Chatbot hội thoại | Cho phép khách hàng đặt câu hỏi tự nhiên và tiếp tục hội thoại nhiều lượt. |
| Intent Routing | Xác định nhu cầu thuộc nghiệp vụ, vận hành, tư vấn hay thông tin và điều phối xử lý. |
| Knowledge Retrieval | Tìm kiếm thông tin từ kho tài liệu, FAQ, quy trình, chính sách và sản phẩm. |
| Tool Calling | Cho phép Agent gọi các công cụ như tra cứu trạng thái giao dịch, tra cứu thẻ, tính lãi, tính khoản vay và so sánh sản phẩm. |
| Business Rules | Áp dụng các quy tắc nghiệp vụ để tránh trả lời sai hoặc bỏ qua điều kiện quan trọng. |
| Multi-step Reasoning | Xử lý các câu hỏi cần nhiều bước hoặc nhiều nguồn dữ liệu. |
| Citation/Evidence | Hiển thị nguồn tài liệu hoặc căn cứ cho câu trả lời khi phù hợp. |
| Escalation | Chuyển sang nhân viên/hỗ trợ khi yêu cầu vượt phạm vi hoặc cần xử lý nghiệp vụ thực tế. |
| Conversation Memory | Duy trì ngữ cảnh trong phiên hội thoại để khách không phải lặp lại thông tin. |
| Admin/Knowledge Management | Quản lý tài liệu, sản phẩm, quy trình, FAQ và phiên bản thông tin. |

---

## 3. Kiến trúc dự kiến phát triển

Hệ thống dự kiến sử dụng kiến trúc ứng dụng web kết hợp LLM, AI Agent, RAG và các công cụ nghiệp vụ. Trong phạm vi đồ án, các chức năng liên quan đến dữ liệu thực tế có thể được mô phỏng bằng Mock Banking System để bảo đảm an toàn và độc lập với hệ thống ngân hàng thật.

### 3.1. Kiến trúc dự kiến cho AI Agent

- **AI Agent / Orchestrator:** trung tâm điều phối, nhận diện ý định, lựa chọn tool và kiểm tra kết quả.
- **Skill:** bộ các hướng dẫn cho AI để sử dụng tool và xử lý tình huống theo đúng nghiệp vụ.
- **Knowledge Base / RAG:** lưu và truy xuất FAQ, quy trình, chính sách, biểu phí, thông tin sản phẩm và tài liệu nghiệp vụ.
- **Tool Layer:** các công cụ tra cứu giao dịch/thẻ, tính lãi, tính khoản vay, so sánh sản phẩm và truy vấn dữ liệu.
- **Logging/Evaluation:** lưu lịch sử debug Agent, tool calls, lỗi và kết quả để kiểm thử chất lượng hệ thống.

### 3.2. Kiến trúc logic

```text
User → Web UI → Backend API → AI Agent / Orchestrator
AI Agent → Intent Router → Skill tương ứng
Skill → RAG / Knowledge Base và/hoặc Tool Layer
Tool Layer → Call API / Script python
Kết quả → Verification → LLM Response → Citation / Escalation → User
```

### 3.6. Công nghệ dự kiến

| Thành phần | Công nghệ dự kiến |
|---|---|
| Frontend | React + TypeScript + Tailwind CSS |
| Backend | Python + FastAPI |
| AI | LLM có hỗ trợ tool calling; embedding và reranking |
| RAG | Hybrid retrieval |
| Database | PostgreSQL |
| Document Processing | PyMuPDF/Docling và các thư viện OCR |
| Mock Banking | REST API + dữ liệu ngân hàng mô phỏng |

### 3.7. Phạm vi đồ án

Đồ án tập trung vào xây dựng một nguyên mẫu ứng dụng Banking AI Assistant có khả năng xử lý bốn nhóm nhu cầu nêu trên. Đối với tư vấn tài chính, hệ thống cung cấp thông tin, so sánh và tính toán tham khảo, không tự động thực hiện giao dịch và không thay thế quyết định chuyên môn của ngân hàng hoặc quyết định đầu tư của khách hàng.

---

*— Hết —*