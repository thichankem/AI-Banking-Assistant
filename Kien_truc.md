# BÁO CÁO KỸ THUẬT: KIẾN TRÚC HỆ THỐNG AI BANKING AGENT

## PHẦN 1: TỔNG QUAN KIẾN TRÚC HỆ THỐNG VÀ SƠ ĐỒ DÒNG PHÂN CẤP

### 1.1. Kiến trúc khái quát (Conceptual Architecture)

```text
[ Người Dùng (Web / Mobile Banking App) ]
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│ 1. TẦNG TIẾP NHẬN & BẢO MẬT (Ingress & Security)        │
│    • Tiếp nhận yêu cầu & Lọc an toàn đầu vào            │
│    • Xác thực danh tính & Trích xuất ngữ cảnh           │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│ 2. TẦNG ĐIỀU PHỐI TRUNG TÂM (Central Orchestrator)      │
│    • Phân tích ý định & Định tuyến luồng xử lý          │
│    • Phản biện & Duyệt an toàn     │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│ 3. TẦNG TÁC NHÂN NGHIỆP VỤ (Business Subgraphs)         │
│    • Hướng dẫn Quy trình & Thủ tục                      │
│    • Vận hành & Xử lý Sự cố Giao dịch                   │
│    • Tư vấn Sản phẩm & Lập Kế hoạch Tài chính          │
│    • Tra cứu FAQ, Biểu phí & Điều khoản                 │
└───────┬──────────────────┬──────────────────┬───────────┘
        │                  │                  │
        ▼                  ▼                  ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  4. BỘ NHỚ   │   │ 5. TRA CỨU   │   │ 6. KẾT NỐI   │
│  KHÁCH HÀNG  │   │   TRI THỨC   │   │   CÔNG CỤ    │
│  (Memory)    │   │ (Knowledge)  │   │   (Tools)    │
│ • Ngắn hạn   │   │ • Hybrid RAG │   │ • MCP Core   │
│ • Dài hạn    │   │ • GraphRAG   │   │ • Finance    │
└──────────────┘   └──────────────┘   └──────────────┘
```

---

### 1.2. Sơ đồ dòng phân cấp chi tiết

*   **1. Tầng Giao diện Người dùng (Frontend Layer)**
    *   Ứng dụng Web Chat / Mobile Banking App (React.js, TypeScript, WebSockets)
*   **2. Tầng Cổng giao tiếp & An toàn (API Gateway & Security Shield)**
    *   FastAPI Ingress Gateway: Tiếp nhận kết nối và quản lý Session
    *   Input Sanitization Shield: Bộ lọc làm sạch dữ liệu, ngăn chặn Prompt Injection
    *   JWT Auth & Context Extraction: Xác thực Bearer Token và trích xuất `user_id`
*   **3. Tầng Điều phối Đồ thị Trạng thái (LangGraph Stateful Orchestrator)**
    *   **3.1 Intent Router (Bộ định tuyến ban đầu)**
        *   Phân tích ý định câu hỏi (Intent Classification)
        *   Quyết định chuyển luồng: Trả lời trực tiếp, tra cứu RAG nhanh, hay chuyển vào Subgraph chuyên sâu
    *   **3.2 Business Subgraphs (Các Đồ thị Nghiệp vụ Chuyên biệt)**
        *   *Subgraph 1: Hướng dẫn Quy trình & Thủ tục* (Mở tài khoản, Đăng ký thẻ, Vay vốn)
        *   *Subgraph 2: Vận hành & Sự cố* (Lỗi chuyển tiền, Khóa thẻ khẩn cấp, Lỗi OTP)
        *   *Subgraph 3: Tư vấn Sản phẩm & Tài chính* (So sánh thẻ tín dụng, Tính lãi suất vay/tiết kiệm)
        *   *Subgraph 4: Tra cứu FAQ & Chính sách* (Biểu phí dịch vụ, Giờ làm việc, Điều khoản chung)
    *   **3.3 Evaluator & HITL Gate (Kiểm duyệt Phản biện & Duyệt An toàn)**
        *   Critic Node: Đánh giá độ trung thực (Faithfulness Check), chống ảo giác thông tin
*   **4. Tầng Hạ tầng Dịch vụ & Dữ liệu Ngoại vi (External Services & Data Layer)**
    *   **4.1 Phân hệ Agentic RAG (Dịch vụ Tra cứu Tri thức)**
        *   Late Chunking & Semantic Chunking (Chạy ngầm Offline)
        *   Search Engine: Hybrid Search
    *   **4.2 Tầng Công cụ Chuẩn hóa MCP (MCP Tool Execution Layer)**
        *   MCP Server Core Banking: Tra cứu lịch sử giao dịch, Khóa/Mở thẻ
        *   MCP Server Financial Engine: Máy tính lịch trả nợ vay, Lãi suất tiết kiệm
        *   MCP Server Policy API: Resources cung cấp tài liệu chính sách
    *   **4.3 Phân hệ Bộ nhớ Đa người dùng (Multi-User Memory Subsystem)**
        *   Redis Session Cache (Lưu ngữ cảnh hội thoại ngắn hạn, TTL 24h)
        *   PostgreSQL `pgvector` với RLS
        *   Async Consolidation Worker: Chạy ngầm trích xuất Fact, tổng hợp tri thức và cập nhật hàm suy giảm Ebbinghaus Decay

---


## PHẦN 2: HẠ TẦNG BỘ NHỚ

---

| Tầng Bộ Nhớ | Công Nghệ Lưu Trữ | Thời Gian Tồn Tại (TTL) | Mục Đích Sử Dụng 
| :--- | :--- | :--- | :--- | :--- |
| **1. In-Context RAM** | LangGraph State (RAM) | Trong phiên gọi API | Chứa prompt active & scratchpad
| **2. Short-Term Buffer**| Redis Cluster | 24 giờ | Lưu vết 10 lượt thoại gần nhất 
| **3. Episodic Memory** | PostgreSQL (`pgvector`) | Vĩnh viễn (Lọc Decay) | Lưu vết sự cố, lỗi giao dịch từng gặp 
| **4. Semantic Memory** | PostgreSQL (`pgvector` + RLS) | Vĩnh viễn (Có Cập nhật) | Lưu thuộc tính định danh, sở thích tài chính 
| **5. Archival Storage** | MinIO Object Storage | Lưu trữ lâu dài (Cold) | Snapshot toàn bộ lịch sử phục vụ Audit


---

## PHẦN 3: KHUNG BENCHMARK VÀ BỘ TIÊU CHÍ CHẤM ĐIỂM ĐỘNG THỰC TẾ

### 3.1. Tại sao loại bỏ phương pháp đánh giá Q&A tĩnh truyền thống?
Bộ câu hỏi - câu trả lời tĩnh không thể đo đạc được tác động thực tế của Agent đối với CSDL (Side-effects) và dễ bị quá khớp (Overfitting).



### 3.2. Phương pháp 1: Kiểm thử Biến đổi Trạng thái Cơ sở Dữ liệu (State-Verification Testing)
Truy vấn trực tiếp CSDL **Mock Banking Database Sandbox** sau tương tác:
$$\text{State\_Match} = \mathbb{I}\left( \text{DB}_{\text{actual\_after}} \equiv \text{DB}_{\text{expected\_after}} \right)$$

---

### 3.3. Phương pháp 2: Mô phỏng Hội thoại Động Đa lượt (TAU-Bench Style Agent Simulation)
Sử dụng **User Simulator Agent** đóng vai khách hàng với các Persona phức tạp (thiếu thông tin, thiếu kiên nhẫn, tấn công Red-Teaming) để đo chỉ số **Goal Completion Rate (GCR)**.

---

### 3.4. Phương pháp 3: Chấm điểm Cấu trúc & Thuật toán Độc lập (Deterministic Algorithmic Scoring)
1. **Trajectory Match Rate**: So sánh chuỗi Tool Calls thực tế với chuỗi tối ưu bằng **Tree Edit Distance**.
2. **Exact Deterministic Math Verification**: Kiểm tra kết quả tính toán tài chính bằng AST Math Engine độc lập (Sai số = 0).
3. **JSON Schema Precision Rate**: Đo tỷ lệ tham số MCP Tools tuân thủ Pydantic Schema.

---

### 4.5. Phương pháp 4: LLM-as-a-Judge kết hợp Ước lượng Độ tin cậy (Uncertainty Estimation)
Chạy $N=5$ lần đánh giá độc lập để tính độ lệch chuẩn $\sigma$. Nếu $\sigma > 0.5$, hệ thống tự động đẩy kịch bản về cho Human Reviewer thẩm định thủ công.


