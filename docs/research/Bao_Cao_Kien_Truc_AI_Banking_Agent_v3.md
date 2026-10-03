# BÁO CÁO KỸ THUẬT TOÀN DIỆN: KIẾN TRÚC HỆ THỐNG AI BANKING AGENT DOANH NGHIỆP VÀ KHUNG BENCHMARK ĐÁNH GIÁ ĐỘNG DẠNG THỰC TẾ

---

## MỤC LỤC
1. **TỔNG QUAN VÀ BỐI CẢNH NÂNG CẤP TỪ ĐỀ XUẤT GR2**
2. **PHẦN 1: TỔNG QUAN KIẾN TRÚC HỆ THỐNG VÀ SƠ ĐỒ DÒNG PHÂN CẤP**
   - 1.1. Kiến trúc tổng quát siêu cấp khái quát (Conceptual Architecture)
   - 1.2. Sơ đồ dòng phân cấp chi tiết (Detailed Line-Based Architecture)
   - 1.3. Luồng xử lý dữ liệu End-to-End 6 giai đoạn (Data Execution Flow)
   - 1.4. Bản chất và vị trí của từng thành phần (Subgraphs, Offline Chunking, RAG, MCP Tools)
   - 1.5. Cấu trúc State Machine và Reducer trong LangGraph Orchestrator
3. **PHẦN 2: BỘ THUẬT TOÁN ĐA DẠNG VÀ THÍCH ỨNG DẠNG ĐỘNG (DYNAMIC ALGORITHMIC SUITE)**
   - 2.1. Phân hệ Tiền xử lý & Phân đoạn văn bản (Adaptive Chunking Suite)
   - 2.2. Phân hệ Truy xuất Tri thức (Multi-Route Agentic Retrieval Suite)
   - 2.3. Phân hệ Quản lý Bộ nhớ Khách hàng (Memory Decay & Consolidation Algorithms)
   - 2.4. Phân hệ Lập luận & Ra quyết định (Adaptive Planning & Reasoning Engine)
4. **PHẦN 3: HẠ TẦNG BỘ NHỚ QUY MÔ LỚN VÀ CÁCH LY NGUYÊN TẮC BẢO MẬT (MULTI-USER MEMORY ARCHITECTURE)**
   - 3.1. Ma trận phân tầng bộ nhớ 5 tầng (5-Tier Memory Taxonomy)
   - 3.2. Thiết kế CSDL Đa người dùng & Cách ly bảo mật dữ liệu khách hàng (PostgreSQL RLS & Indexing)
   - 3.3. Luồng ghi/đồng bộ bất đồng bộ (Async Consolidation Worker)
5. **PHẦN 4: KHUNG BENCHMARK VÀ BỘ TIÊU CHÍ CHẤM ĐIỂM ĐỘNG THỰC TẾ**
   - 4.1. Tại sao loại bỏ phương pháp đánh giá Q&A tĩnh truyền thống?
   - 4.2. Phương pháp 1: Kiểm thử Biến đổi Trạng thái Cơ sở Dữ liệu (State-Verification Testing)
   - 4.3. Phương pháp 2: Mô phỏng Hội thoại Động Đa lượt (TAU-Bench Style Agent Simulation)
   - 4.4. Phương pháp 3: Chấm điểm Cấu trúc & Thuật toán Độc lập (Deterministic Algorithmic Scoring)
   - 4.5. Phương pháp 4: LLM-as-a-Judge kết hợp Ước lượng Độ tin cậy (Uncertainty Estimation)
6. **PHẦN 5: ĐỊNH LƯỢNG CHI TIẾT CÔNG THỨC CHẤM ĐIỂM TỪNG MODULE VÀ TOÀN HỆ THỐNG**
   - 5.1. Bảng tổng hợp công thức đo lường và mục tiêu từng Module
   - 5.2. Công thức điểm utility tổng thể toàn hệ thống (End-to-End System Score)
   - 5.3. Quy trình bánh đà tự động tối ưu hóa và liên tục fine-tuning (Continuous Tuning Flywheel)

---

## 1. TỔNG QUAN VÀ BỐI CẢNH NÂNG CẤP TỪ ĐỀ XUẤT GR2

Trong bản đề xuất đề tài GR2 ban đầu (*"Xây dựng Trợ lý AI sử dụng kiến trúc AI Agent hỗ trợ khách hàng trong lĩnh vực Ngân hàng"*), hệ thống được thiết kế theo luồng xử lý tuyến tính truyền thống: **User → Web UI → FastAPI → AI Agent / Intent Router → Skill → RAG / Tool Layer → Verification → Response**. 

Khi triển khai trong môi trường ngân hàng thực tế phục vụ quy mô hàng triệu khách hàng truy cập đồng thời (High-Concurrency Multi-User), kiến trúc nâng cấp này giải quyết 4 thách thức lớn của mô hình cũ:
1. **Duy trì trạng thái ngữ cảnh dài hạn**: Sử dụng LangGraph Stateful Graph thay vì prompt chaining đơn giản để không bị mất dấu luồng khi khách hàng đổi ý hoặc tương tác kéo dài.
2. **An toàn và cách ly dữ liệu người dùng (Cross-User Data Protection)**: Thiết lập cơ chế bảo vệ và cách ly dữ liệu lịch sử/sở thích ở cấp độ cơ sở dữ liệu (PostgreSQL Row-Level Security).
3. **Gọi công cụ chính xác (Tool Selection Precision)**: Sử dụng Model Context Protocol (MCP) và Dynamic Tool Routing để ngăn ngừa hiện tượng "bão hòa ngữ cảnh" khi số lượng API ngân hàng gia tăng.
4. **Khung đánh giá động thực tế**: Thay thế bộ câu hỏi - câu trả lời tĩnh bằng kiểm thử biến đổi trạng thái (State Verification) và mô phỏng hội thoại đa lượt.

---

## PHẦN 1: TỔNG QUAN KIẾN TRÚC HỆ THỐNG VÀ SƠ ĐỒ DÒNG PHÂN CẤP

### 1.1. Kiến trúc tổng quát siêu cấp khái quát (Conceptual Architecture)

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
│    • Phản biện & Duyệt an toàn (Human-in-the-Loop)      │
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

### 1.2. Sơ đồ dòng phân cấp chi tiết (Detailed Line-Based Architecture)

*   **1. Tầng Giao diện Người dùng (Frontend Layer)**
    *   Ứng dụng Web Chat / Mobile Banking App (React.js, TypeScript, WebSockets)
*   **2. Tầng Cổng giao tiếp & An toàn (API Gateway & Security Shield)**
    *   FastAPI Ingress Gateway: Tiếp nhận kết nối và quản lý Session
    *   Input Sanitization Shield: Bộ lọc làm sạch dữ liệu, ngăn chặn Prompt Injection
    *   JWT Auth & Context Extraction: Xác thực Bearer Token và trích xuất `user_id`
*   **3. Tầng Điều phối Đồ thị Trạng thái (LangGraph Stateful Orchestrator)**
    *   **3.1 Pre-Route Router (Bộ định tuyến ban đầu)**
        *   Phân tích ý định câu hỏi (Intent Classification)
        *   Quyết định chuyển luồng: Trả lời trực tiếp, tra cứu RAG nhanh, hay chuyển vào Subgraph chuyên sâu
    *   **3.2 Business Subgraphs (Các Đồ thị Nghiệp vụ Chuyên biệt)**
        *   *Subgraph 1: Hướng dẫn Quy trình & Thủ tục* (Mở tài khoản, Đăng ký thẻ, Vay vốn)
        *   *Subgraph 2: Vận hành & Sự cố* (Lỗi chuyển tiền, Khóa thẻ khẩn cấp, Lỗi OTP)
        *   *Subgraph 3: Tư vấn Sản phẩm & Tài chính* (So sánh thẻ tín dụng, Tính lãi suất vay/tiết kiệm)
        *   *Subgraph 4: Tra cứu FAQ & Chính sách* (Biểu phí dịch vụ, Giờ làm việc, Điều khoản chung)
    *   **3.3 Evaluator & HITL Gate (Kiểm duyệt Phản biện & Duyệt An toàn)**
        *   Critic Node: Đánh giá độ trung thực (Faithfulness Check), chống ảo giác thông tin
        *   Human-in-the-Loop Interrupt Gate: Tạm dừng xin xác nhận người dùng đối với các tác vụ tài chính rủi ro cao
*   **4. Tầng Hạ tầng Dịch vụ & Dữ liệu Ngoại vi (External Services & Data Layer)**
    *   **4.1 Phân hệ Agentic RAG (Dịch vụ Tra cứu Tri thức)**
        *   Data Ingestion Worker: Late Chunking & Semantic Chunking (Chạy ngầm Offline)
        *   Search Engine: Hybrid Search (Vector HNSW + BM25 Full-Text via RRF)
        *   Enhancement Modules: Corrective RAG (CRAG) & GraphRAG (Knowledge Graph)
    *   **4.2 Tầng Công cụ Chuẩn hóa MCP (MCP Tool Execution Layer)**
        *   MCP Server Core Banking: Tra cứu lịch sử giao dịch, Khóa/Mở thẻ
        *   MCP Server Financial Engine: Máy tính lịch trả nợ vay, Lãi suất tiết kiệm
        *   MCP Server Policy API: Resources cung cấp tài liệu chính sách
    *   **4.3 Phân hệ Bộ nhớ Đa người dùng (Multi-User Memory Subsystem)**
        *   Working Memory: Redis Session Cache (Lưu ngữ cảnh hội thoại ngắn hạn, TTL 24h)
        *   Long-Term Memory: PostgreSQL `pgvector` với RLS (Row-Level Security cách ly dữ liệu từng khách hàng)
        *   Async Consolidation Worker: Chạy ngầm trích xuất Fact, tổng hợp tri thức và cập nhật hàm suy giảm Ebbinghaus Decay

---

### 1.3. Luồng xử lý dữ liệu End-to-End 6 giai đoạn (Data Execution Flow)

```text
[GIAI ĐOẠN 1: INGRESS & SECURITY SANITIZATION]
  User Query 
    ───> FastAPI Ingress Gateway 
            ───> Strip Prompt Injection (Regex + Guardrails Engine)
                    ───> Extract User_ID from Validated JWT

[GIAI ĐOẠN 2: PRE-FETCH MULTI-USER CONTEXT]
  Gateway 
    ───> Kích hoạt Truy vấn Song song (Parallel Async Read)
            ├───> [Redis Buffer] Kéo Top 10 tin nhắn phiên hiện tại (Latency < 2ms)
            └───> [PostgreSQL pgvector] Kéo Top 3 Semantic Facts của User_ID (Latency < 12ms)
                    ───> Đóng gói vào `AgentState` ban đầu

[GIAI ĐOẠN 3: PRE-ROUTE & SUBGRAPH ROUTING]
  `AgentState` 
    ───> Master Orchestrator (Pre-Route Node)
            ├─── [Chào hỏi / Đơn giản] ───> Trả về Direct Response
            ├─── [Tra cứu tri thức]   ───> Route sang Agentic RAG Pipeline
            └─── [Tác vụ Nghiệp vụ]   ───> Route sang Subgraph [1 / 2 / 3 / 4] tương ứng

[GIAI ĐOẠN 4: SUBGRAPH EXECUTION & MCP TOOL CALLING]
  Subgraph Agent Node
    ───> Thực hiện Vòng lặp Lập luận (ReAct / ToT)
            ───> Sinh Tool Call Request 
                    ───> Chuyển đổi sang chuẩn MCP JSON-RPC 2.0 Payload
                            ───> Đẩy tới MCP Server tương ứng (Banking / Financial Engine)
                                    ───> Thực thi trong Python Sandbox
                                            ───> Nhận kết quả Tool Response & Cập nhật `AgentState`

[GIAI ĐOẠN 5: EVALUATOR-OPTIMIZER & HUMAN-IN-THE-LOOP (HITL)]
  Agent Raw Output 
    ───> Critic Node (Đánh giá Faithfulness & Nguy cơ Rủi ro)
            ├─── [Phát hiện Ảo giác / Thiếu ý] ───> Yêu cầu Subgraph Regenerate (Tối đa 2 lần)
            ├─── [Tác vụ Tài chính Nguy hiểm]  ───> Trigger LangGraph `interrupt()`
            │                                         ───> Gửi Yêu cầu Xác nhận lên UI
            │                                                 ───> Chờ User Bấm [Xác Nhận/Hủy]
            └─── [Đạt tiêu chuẩn / An toàn]    ───> Chuyển tiếp ra Response Stream

[GIAI ĐOẠN 6: ASYNCHRONOUS MEMORY CONSOLIDATION]
  Final Output Sent to User
    ───> Bắn Event Payload bất đồng bộ vào Celery / RabbitMQ
            ───> Memory Consolidation Worker thu gom
                    ───> LLM trích xuất Atomic Facts & Ước tính Decay
                            ───> Ghi / Cập nhật vào PostgreSQL `semantic_memories` (RLS Secured)
```

---

### 1.4. Bản chất và vị trí của từng thành phần trong hệ thống

1. **Subgraph (Đồ thị con) là gì?**
   * *Bản chất*: Là một luồng logic code/state độc lập (`StateGraph` trong LangGraph) cho một nhóm nghiệp vụ ngân hàng cụ thể.
   * *Chức năng*: Chứa biến lưu trữ trạng thái riêng, các hàm xử lý từng bước (Nodes) và quy tắc rẽ nhánh (Edges). Giúp hệ thống không bị quá tải ngữ cảnh.

2. **Offline Chunking (Phân đoạn văn bản) nằm ở đâu?**
   * *Bản chất*: Nằm ở **Phase Offline (Chạy ngầm độc lập)**, hoàn toàn không nằm trong vòng lặp chat trực tiếp của người dùng.
   * *Chức năng*: Khi ngân hàng nhập file PDF chính sách/biểu phí mới -> Data Worker đọc PDF -> Áp dụng Late Chunking / Semantic Chunking -> Tạo Vector Embedding -> Lưu sẵn vào PostgreSQL Database (`pgvector`).

3. **RAG (Retrieval-Augmented Generation) nằm ở đâu?**
   * *Bản chất*: Là một **Dịch vụ Tra cứu Tri thức (Knowledge Service)** độc lập.
   * *Chức năng*: Khi Subgraph cần thông tin chính sách/biểu phí, nó gửi câu hỏi tới RAG Pipeline -> RAG thực hiện Hybrid Search (HNSW + BM25) trên CSDL đã chunk sẵn ở Bước 2 và trả về 3-5 đoạn văn bản chính xác.

4. **MCP (Model Context Protocol) nằm ở đâu?**
   * *Bản chất*: Là **Cổng kết nối chuẩn hóa (JSON-RPC 2.0 API Protocol)** giữa các Subgraph và Hệ thống Core Banking ngoại vi.
   * *Chức năng*: Đóng vai trò như "ổ cắm chuẩn" để Agent cắm vào và tương tác với các dịch vụ tính toán lãi suất, tra cứu tài khoản, hoặc khóa thẻ an toàn.

---

### 1.5. Cấu trúc State Machine và Reducer trong LangGraph Orchestrator

```python
from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # Lịch sử hội thoại - dùng Reducer add_messages để nối tiếp, không ghi đè
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Context định danh người dùng
    user_id: str
    session_id: str
    
    # Ngữ cảnh bộ nhớ trích xuất từ PostgreSQL
    user_semantic_facts: List[str]
    
    # Trạng thái định tuyến & lập luận
    selected_subgraph: Optional[str]
    retrieved_documents: List[Dict[str, Any]]
    tool_execution_history: List[Dict[str, Any]]
    
    # Trạng thái duyệt an toàn (Human-in-the-Loop)
    requires_approval: bool
    user_approval_granted: Optional[bool]
    
    # Đánh giá chất lượng
    critic_score: float
    hallucination_flag: bool
```

---

## PHẦN 2: BỘ THUẬT TOÁN ĐA DẠNG VÀ THÍCH ỨNG DẠNG ĐỘNG (DYNAMIC ALGORITHMIC SUITE)

Hệ thống **không cố định một thuật toán duy nhất** cho toàn bộ tác vụ, mà tự động chuyển đổi tùy theo tính chất của từng câu hỏi:

### 2.1. Phân hệ Tiền xử lý & Phân đoạn văn bản (Adaptive Chunking Suite)
1. **Dynamic Semantic Chunking**: Tính toán khoảng cách Cosine Embeddings giữa các câu liên tiếp, ngắt đoạn khi độ tương đồng xuống dưới ngưỡng.
2. **Late Chunking**: Đưa văn bản thô qua Long-Context Transformer Encoder thu nhận vector ẩn toàn cục trước khi Mean Pooling cho từng đoạn 512 tokens. Triệt tiêu hoàn toàn **Coreference Trap**.
3. **Tree-Sitter AST Chunking**: Phân tích cây cú pháp cho tài liệu JSON API, Biểu phí cấu trúc bảng.

---

### 2.2. Phân hệ Truy xuất Tri thức (Multi-Route Agentic Retrieval Suite)
1. **Hybrid Search + Reciprocal Rank Fusion (RRF)**:
   $$\text{RRF\_Score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
   *(Kết hợp Vector HNSW + BM25 Full-Text Search)*.
2. **Corrective RAG (CRAG) & Self-RAG**:
   Evaluation score $S_{eval}$:
   - $S_{eval} > 0.75$ (**Correct**): Dùng trực tiếp.
   - $0.4 \le S_{eval} \le 0.75$ (**Ambiguous**): Rã câu hỏi (Decompose-then-Recompose) và truy xuất lại.
   - $S_{eval} < 0.4$ (**Incorrect**): Chuyển hướng Fallback.
3. **GraphRAG (Leiden Community Detection)**: Xây dựng Đồ thị Tri thức để trả lời các câu hỏi so sánh đa sản phẩm.

---

### 2.3. Phân hệ Quản lý Bộ nhớ Khách hàng (Memory Decay & Consolidation Algorithms)
1. **Thuật toán Suy giảm Bộ nhớ Ebbinghaus**:
   $$S_{memory}(t) = w_i \cdot I_{base} + w_r \cdot e^{-\lambda \cdot (t - t_{last})} + w_f \cdot \log(1 + C_{access})$$
2. **Cơ chế Async Fact Consolidation**:
   Worker ngầm kiểm tra khoảng cách Cosine giữa Fact mới và Fact cũ. Nếu $\text{CosineDistance} < 0.15$, tiến hành vô hiệu hóa Fact cũ (`is_active = False`) để tránh xung đột dữ liệu.

---

### 2.4. Phân hệ Lập luận & Ra quyết định (Adaptive Planning & Reasoning Engine)
1. **ReAct Pattern**: Dùng cho 80% tác vụ thông thường nhằm tối ưu thời gian phản hồi (P95 < 1.2s).
2. **Tree-of-Thoughts (ToT) / MCTS**: Áp dụng cho các bài toán lập kế hoạch tài chính phức tạp theo công thức UCT:
   $$\text{UCT}(node) = \bar{V}_{node} + c \cdot \sqrt{\frac{\ln N_{parent}}{n_{node}}}$$

---

## PHẦN 3: HẠ TẦNG BỘ NHỚ QUY MÔ LỚN VÀ CÁCH LY NGUYÊN TẮC BẢO MẬT (MULTI-USER MEMORY ARCHITECTURE)

### 3.1. Ma trận phân tầng bộ nhớ 5 tầng (5-Tier Memory Taxonomy)

| Tầng Bộ Nhớ | Công Nghệ Lưu Trữ | Thời Gian Tồn Tại (TTL) | Mục Đích Sử Dụng | Latency Truy Xuất |
| :--- | :--- | :--- | :--- | :--- |
| **1. In-Context RAM** | LangGraph State (RAM) | Trong phiên gọi API | Chứa prompt active & scratchpad | < 0.1ms |
| **2. Short-Term Buffer**| Redis Cluster | 24 giờ | Lưu vết 10 lượt thoại gần nhất | < 2ms |
| **3. Episodic Memory** | PostgreSQL (`pgvector`) | Vĩnh viễn (Lọc Decay) | Lưu vết sự cố, lỗi giao dịch từng gặp | < 15ms |
| **4. Semantic Memory** | PostgreSQL (`pgvector` + RLS) | Vĩnh viễn (Có Cập nhật) | Lưu thuộc tính định danh, sở thích tài chính | < 15ms |
| **5. Archival Storage** | MinIO Object Storage | Lưu trữ lâu dài (Cold) | Snapshot toàn bộ lịch sử phục vụ Audit | < 100ms |

---

### 3.2. Thiết kế CSDL Đa người dùng & Cách ly bảo mật dữ liệu khách hàng (PostgreSQL RLS & Indexing)

```sql
-- DDL Khởi tạo Bảng Semantic Memories
CREATE TABLE public.semantic_memories (
    memory_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(64) NOT NULL,
    fact_content TEXT NOT NULL,
    embedding vector(1536),
    importance_score INT DEFAULT 3,
    access_count INT DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Tạo Chỉ mục Kép (Hybrid Index): B-Tree cho User_ID + HNSW cho Vector Similarity
CREATE INDEX idx_memories_user ON public.semantic_memories(user_id);
CREATE INDEX idx_memories_vector_hnsw ON public.semantic_memories 
USING hnsw (embedding vector_cosine_ops) WHERE (is_active = TRUE);

-- Bật Row-Level Security (RLS) cách ly dữ liệu giữa các khách hàng
ALTER TABLE public.semantic_memories ENABLE ROW LEVEL SECURITY;

CREATE POLICY user_memory_isolation_policy ON public.semantic_memories
    FOR ALL
    TO application_role
    USING (user_id = current_setting('app.current_user_id', true));
```

---

## PHẦN 4: KHUNG BENCHMARK VÀ BỘ TIÊU CHÍ CHẤM ĐIỂM ĐỘNG THỰC TẾ

### 4.1. Tại sao loại bỏ phương pháp đánh giá Q&A tĩnh truyền thống?
Bộ câu hỏi - câu trả lời tĩnh không thể đo đạc được tác động thực tế của Agent đối với CSDL (Side-effects) và dễ bị quá khớp (Overfitting).

---

### 4.2. Phương pháp 1: Kiểm thử Biến đổi Trạng thái Cơ sở Dữ liệu (State-Verification Testing)
Truy vấn trực tiếp CSDL **Mock Banking Database Sandbox** sau tương tác:
$$\text{State\_Match} = \mathbb{I}\left( \text{DB}_{\text{actual\_after}} \equiv \text{DB}_{\text{expected\_after}} \right)$$

---

### 4.3. Phương pháp 2: Mô phỏng Hội thoại Động Đa lượt (TAU-Bench Style Agent Simulation)
Sử dụng **User Simulator Agent** đóng vai khách hàng với các Persona phức tạp (thiếu thông tin, thiếu kiên nhẫn, tấn công Red-Teaming) để đo chỉ số **Goal Completion Rate (GCR)**.

---

### 4.4. Phương pháp 3: Chấm điểm Cấu trúc & Thuật toán Độc lập (Deterministic Algorithmic Scoring)
1. **Trajectory Match Rate**: So sánh chuỗi Tool Calls thực tế với chuỗi tối ưu bằng **Tree Edit Distance**.
2. **Exact Deterministic Math Verification**: Kiểm tra kết quả tính toán tài chính bằng AST Math Engine độc lập (Sai số = 0).
3. **JSON Schema Precision Rate**: Đo tỷ lệ tham số MCP Tools tuân thủ Pydantic Schema.

---

### 4.5. Phương pháp 4: LLM-as-a-Judge kết hợp Ước lượng Độ tin cậy (Uncertainty Estimation)
Chạy $N=5$ lần đánh giá độc lập để tính độ lệch chuẩn $\sigma$. Nếu $\sigma > 0.5$, hệ thống tự động đẩy kịch bản về cho Human Reviewer thẩm định thủ công.

---

## PHẦN 5: ĐỊNH LƯỢNG CHI TIẾT CÔNG THỨC CHẤM ĐIỂM TỪNG MODULE VÀ TOÀN HỆ THỐNG

### 5.1. Bảng tổng hợp công thức đo lường và mục tiêu từng Module

| Subsystem Module | Tên Chỉ Số Benchmark | Công Thức Toán Học / Phương Pháp Đo | Mục Tiêu Doanh Nghiệp |
| :--- | :--- | :--- | :--- |
| **RAG Subsystem** | **Faithfulness** (Chống Ảo Giác) | $$F = \frac{|\text{Mệnh đề trích xuất đúng từ Source}|}{|\text{Tổng số mệnh đề khẳng định trong câu trả lời}|}$$ | **100% (Zero Hallucination)** |
| **RAG Subsystem** | **Context Recall & Precision** | $$CR = \frac{|D_{retrieved} \cap D_{relevant}|}{|D_{relevant}|}, \quad CP = \frac{|D_{retrieved} \cap D_{relevant}|}{|D_{retrieved}|}$$ | Context Recall @5 > 92%<br>Context Precision @5 > 88% |
| **Memory System** | **Multi-User Isolation Score** | $$IS = 1 - \frac{|\text{Tri thức của User khác bị rò rỉ}|}{|\text{Tổng số truy vấn kiểm thử}|}$$ | **100.0% (Tuyệt đối không rò rỉ)** |
| **Memory System** | **Memory Recall @K (LOCOMO)** | $$MR@K = \frac{|\text{Semantic Facts đúng trích xuất}|}{|\text{Tổng Facts ngữ cảnh cần thiết}|}$$ | Memory Recall @3 > 90% |
| **Router & Planning**| **Pre-Route Accuracy** | $$RA = \frac{|\text{Số lần định tuyến đúng Subgraph/Path}|}{|\text{Tổng số Query kiểm thử}|}$$ | Pre-Route Accuracy > 96% |
| **MCP Tool Layer** | **Tool & Parameter Accuracy** | $$Acc_{tool} = \frac{|\text{Số lần chọn đúng Tool \& đúng Schema}|}{|\text{Tổng số Tool Calls}|}$$ | Tool Selection Acc > 98% |
| **MCP Tool Layer** | **State-Verification Success** | $$SVR = \frac{|\text{Số tác vụ biến đổi CSDL Sandbox chính xác}|}{|\text{Tổng số tác vụ thực thi}|}$$ | **State Verification > 99%** |
| **Guardrails** | **Prompt Injection Defense** | $$IR = 1 - \frac{|\text{Số kịch bản tấn công thành công}|}{|\text{Tổng kịch bản Red-Teaming}|}$$ | Defense Rate > 99.5% |

---

### 5.2. Công thức điểm utility tổng thể toàn hệ thống (End-to-End System Score)

$$\text{Score}_{\text{System}} = w_1 \cdot Pass@1 + w_2 \cdot SVR + w_3 \cdot Faithfulness + w_4 \cdot IS - w_5 \cdot \text{Penalty}_{\text{Latency}}$$

---

### 5.3. Quy trình bánh đà tự động tối ưu hóa và liên tục fine-tuning (Continuous Tuning Flywheel)

```text
[LangSmith / OpenTelemetry Production Tracing]
                         │
                         ▼
     [Lọc các phiên có điểm Pass@1 < 0.6 hoặc SVR = 0]
                         │
                         ▼
             [Hard Negative Mining]
                         │
                         ▼
  [Đóng gói Dataset DPO (Direct Preference Optimization)]
     ├── Prompt: Câu hỏi khách hàng
     ├── Chosen: Trajectory & Trả lời từ Chuyên viên
     └── Rejected: Trajectory lỗi của Agent
                         │
                         ▼
     [Fine-Tune Router & Subgraph Small LLMs (LoRA/QLoRA)]
                         │
                         ▼
       [Automated CI/CD Regression Gate Test]
                         │
                         ▼
              [Deploy Version Mới]
```

---

## TỔNG KẾT BÁO CÁO KỸ THUẬT

Bản thiết kế kiến trúc này định hình chuẩn mực một **Hệ thống AI Agent Doanh nghiệp Ngân hàng Đơn lẻ (Single-Enterprise, Multi-User)**. Việc kết hợp giữa **Stateful Graph Orchestration (LangGraph)**, **Model Context Protocol (MCP)**, **PostgreSQL RLS Multi-User Memory**, cùng **Khung Đánh Giá Động (State-Verification & Agent Simulation)** mang lại một nền tảng khoa học, an toàn và sẵn sàng bảo vệ trước Hội đồng Chuyên môn.
