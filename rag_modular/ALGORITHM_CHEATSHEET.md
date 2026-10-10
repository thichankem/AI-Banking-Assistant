# 📚 BẢNG TRA CỨU CÁC THUẬT TOÁN MODULAR RAG (BANKING ASSISTANT)
> **Tài liệu tham chiếu nhanh cho đồ án tốt nghiệp AI Banking Assistant**  
> Định dạng: **Siêu ngắn gọn - Súc tích - Tập trung vào thực chiến & Nghiệp vụ Ngân hàng**

---

## 📑 MỤC LỤC
1. [Module 1: Chunking (Phân đoạn văn bản)](#1-module-chunking-phân-đoạn-văn-bản)
2. [Module 2: Embedding (Mã hóa ngữ nghĩa)](#2-module-embedding-mã-hóa-ngữ-nghĩa)
3. [Module 3: Keyword Search (Tìm kiếm từ khóa - Sparse)](#3-module-keyword-search-tìm-kiếm-từ-khóa)
4. [Module 4: Vector Database (Cơ sở dữ liệu Vector)](#4-module-vector-database-lưu-trữ-vector)
5. [Module 5: Query Parsing (Xử lý & Chuyển đổi truy vấn)](#5-module-query-parsing-xử-lý-truy-vấn)
6. [Module 6: Retrieval & Fusion (Truy xuất & Hợp nhất)](#6-module-retrieval--fusion-truy-xuất--hợp-nhất)
7. [Module 7: Reranking (Tái xếp hạng)](#7-module-reranking-tái-xếp-hạng)
8. [Module 8: Context Synthesis (Tổng hợp ngữ cảnh)](#8-module-context-synthesis-tổng-hợp-ngữ-cảnh)
9. [Module 9: LLM Generation & Guardrails (Sinh câu trả lời & An toàn)](#9-module-llm-generation--guardrails-sinh-phản-hồi)
10. [Lộ trình gợi ý tự code từ đầu (Recommended Roadmap)](#10-lộ-trình-gợi-ý-tự-code-từ-đầu)

---

## 1. Module: Chunking (Phân đoạn văn bản)

| Thuật toán | Ý nghĩa cốt lõi | Điểm cộng (+) | Điểm trừ (-) |
| :--- | :--- | :--- | :--- |
| **Fixed-size Window** | Cắt văn bản theo độ dài cố định (ký tự/token) kèm bước trượt (overlap). | Cực kỳ dễ code, tốc độ siêu nhanh ($O(N)$), kiểm soát chính xác token budget. | Cắt cụt câu giữa chừng, phá vỡ ngữ nghĩa ngữ cảnh. |
| **Sentence / Regex Split** | Tách theo ranh giới câu (`.`, `?`, `!`, regex xuống dòng). | Giữ trọn vẹn ngữ nghĩa từng câu đơn, dễ đọc hiểu. | Kích thước chunk không đồng đều (câu ngắn quá hoặc câu dài quá). |
| **Paragraph Split** | Tách theo khối đoạn văn bản (`\n\n`). | Giữ trọn vẹn một ý/chủ đề nhỏ do tác giả định dạng. | Đoạn văn dài dễ tràn token; văn bản thiếu ngắt đoạn sẽ bị dính chùm. |
| **FAQ / Q&A Pair** | Bóc tách chính xác cặp Hỏi - Đáp (Question & Answer) từ tài liệu hỗ trợ. | **Hoàn hảo cho Banking FAQ**; query khớp trực tiếp câu hỏi mẫu. | Chỉ áp dụng được cho dữ liệu dạng Q&A, không dùng được cho hợp đồng/luật. |
| **Hierarchy & Header** | Tách theo cây phân cấp tiêu đề Markdown (`# H1`, `## H2`, `### H3`), giữ breadcrumb cha. | Chunk luôn mang metadata nguồn gốc điều khoản (VD: `Thẻ tín dụng > Biểu phí`). | Cần tài liệu có cấu trúc Heading chuẩn chỉnh (Markdown/HTML). |
| **Semantic Sliding Window** | Tính cosine similarity giữa các câu liên tiếp; ngắt chunk khi độ tương đồng tụt dốc. | Chunk đồng nhất hoàn toàn về mặt chủ đề ngữ nghĩa. | Chậm vì phải chạy model embedding cho từng câu đơn lẻ. |
| **Late Chunking** | Đưa toàn bộ tài liệu qua Long-Context Transformer trước, sau đó mới pooling theo ranh giới token. | Mỗi chunk giữ được ngữ cảnh toàn cục của cả văn bản lớn. | Phức tạp, tốn tài nguyên GPU, phụ thuộc model hỗ trợ late pooling. |
| **Layout-Aware / Table-Aware** | Tách riêng cấu trúc bảng biểu (HTML/Markdown table), bảo toàn dòng tiêu đề cột. | Không làm nát dữ liệu dạng bảng số liệu (lãi suất, hạn mức, phí phạt). | Yêu cầu parser chuyên biệt (OCR hoặc parser tài liệu phức tạp). |

---

## 2. Module: Embedding (Mã hóa ngữ nghĩa)

| Thuật toán / Mô hình | Ý nghĩa cốt lõi | Điểm cộng (+) | Điểm trừ (-) |
| :--- | :--- | :--- | :--- |
| **Vietnamese-SBERT / BGE-M3** | Model Transformer mã hóa câu chuyên ngữ tiếng Việt / Đa ngữ 1024d. | **Tốt nhất cho tiếng Việt ngân hàng**, hiểu từ đồng nghĩa tự nhiên. | Nặng, cần GPU hoặc máy RAM tốt để suy luận mượt mà. |
| **Multilingual-E5-Large** | Mô hình embedding của Microsoft, tối ưu hóa riêng cho tác vụ Retrieval với tiền tố `query:` / `passage:`. | Chuẩn benchmark quốc tế cao, hỗ trợ đa ngữ xuất sắc. | Bắt buộc phải gắn đúng tiền tố (prefix) thì mới đạt độ chính xác tối đa. |
| **Cloud Embedding (OpenAI / Gemini)** | Gọi API mã hóa vector từ xa (OpenAI `text-embedding-3`, Gemini `text-embedding-004`). | Chất lượng cực cao, không tốn RAM máy cục bộ, chiều vector linh hoạt. | Mất phí API, phụ thuộc Internet, rủi ro tuân thủ bảo mật dữ liệu ngân hàng. |
| **Deterministic Dense** | Chiếu vector ngẫu nhiên cố định (hash MD5/SHA256 kết hợp seed cố định ra mảng float). | $0$ tài nguyên, không cần tải model, kết quả đồng nhất tuyệt đối để test luồng. | Hoàn toàn không hiểu ngữ nghĩa ngôn ngữ tự nhiên. |
| **Hashing TF-IDF Vector** | Mã hóa sparse/dense dựa trên băm từ khóa và tần suất từ trong từ điển. | Siêu nhẹ, chạy thuần CPU bằng NumPy, nắm bắt tốt từ khóa hiếm. | Không nắm được từ đồng nghĩa hay ngữ cảnh sâu của câu. |

---

## 3. Module: Keyword Search (Tìm kiếm từ khóa - Sparse)

| Thuật toán | Ý nghĩa cốt lõi | Điểm cộng (+) | Điểm trừ (-) |
| :--- | :--- | :--- | :--- |
| **BM25 Okapi** | Thuật toán xếp hạng xác suất dựa trên tần suất từ (TF), nghịch đảo văn bản (IDF) và độ dài tài liệu. | **Chuẩn công nghiệp**, bắt chính xác tên riêng, mã biểu phí, số hotline. | Không hiểu từ đồng nghĩa (VD: gõ "mất thẻ" không tìm ra "khóa thẻ"). |
| **BM25+ / BM25L** | Biến thể của BM25 bổ sung ngưỡng chặn dưới cho tần suất từ nhằm tránh phạt quá nặng tài liệu dài. | Khắc phục điểm yếu tài liệu ngân hàng dài (như hợp đồng mở thẻ, biểu phí tổng). | Cần tinh chỉnh thêm siêu tham số $\delta$ ($delta$). |
| **TF-IDF Vector Space** | Tính toán ma trận trọng số TF-IDF và đo khoảng cách Cosine giữa truy vấn và tài liệu. | Cực kỳ kinh điển, dễ triển khai thuần Python/scikit-learn. | Bị ảnh hưởng nặng bởi độ dài văn bản nếu không chuẩn hóa tốt. |
| **Exact Substring Match** | Kiểm tra chuỗi con chính xác (`query in doc`). | Điểm tự tin $100\%$ khi tìm số hotline, mã biểu phí (`VIB01`, `19001122`). | Quá cứng nhắc, sai một dấu cách hoặc gõ nhầm là trượt. |
| **Fuzzy Levenshtein / N-gram** | So khớp chuỗi dựa trên khoảng cách chỉnh sửa ký tự (Edit Distance). | Bắt được lỗi chính tả, gõ teencode hoặc gõ tiếng Việt thiếu dấu. | Chi phí tính toán $O(M \times N)$ lớn khi tập dữ liệu văn bản phình to. |

---

## 4. Module: Vector Database (Lưu trữ Vector)

| Cơ sở dữ liệu / Thuật toán | Ý nghĩa cốt lõi | Điểm cộng (+) | Điểm trừ (-) |
| :--- | :--- | :--- | :--- |
| **NumPy Cosine Matrix (Brute-force Flat)** | Tính toán tích vô hướng ma trận thuần túy: $\frac{A \cdot B}{\|A\| \|B\|}$. | **Tuyệt đối chính xác 100% (Exact KNN)**, $0$ phụ thuộc bên ngoài, hoàn hảo để học code. | Độ phức tạp $O(N)$, chậm khi dữ liệu vượt quá $100.000$ chunks. |
| **ChromaDB** | Vector DB nhúng in-process, lưu trữ SQLite + chỉ mục HNSW qua hnswlib. | Rất phổ biến trong cộng đồng RAG, dễ dùng, có sẵn bộ lọc metadata. | Quản lý tiến trình Python đôi khi gặp xung đột file lock trên Windows. |
| **Qdrant** | Vector Search Engine viết bằng Rust, tối ưu mạnh mẽ cho lọc Payload. | **Tốc độ cực nhanh**, lọc metadata phức tạp (ngày tháng, loại tài khoản) siêu việt. | Cần chạy Docker container hoặc kết nối Qdrant Cloud. |
| **LanceDB** | Vector DB serverless dựa trên định dạng Apache Arrow và lưu trữ đĩa. | Đọc ghi đĩa cực nhanh, không tốn RAM, chi phí vận hành thấp. | Hệ sinh thái tính năng lọc nâng cao còn tương đối mới. |
| **FAISS (HNSW / IVF-PQ)** | Thư viện chuyên biệt từ Meta chuyên cho tìm kiếm láng giềng gần nhất (ANN). | Xử lý hàng triệu vector mượt mà trên cả CPU và GPU. | Khó cài đặt trên một số môi trường Windows, không hỗ trợ sẵn CRUD document. |

---

## 5. Module: Query Parsing (Xử lý truy vấn)

| Thuật toán / Kỹ thuật | Ý nghĩa cốt lõi | Điểm cộng (+) | Điểm trừ (-) |
| :--- | :--- | :--- | :--- |
| **Pass-Through** | Giữ nguyên văn bản câu hỏi mộc của người dùng đưa thẳng vào retriever. | Trễ $0$ ms, không phụ thuộc LLM, giữ đúng nguyên tác. | Thất bại nếu câu hỏi mơ hồ, chứa từ lóng hoặc từ viết tắt. |
| **Banking Synonym Normalizer** | Từ điển ánh xạ từ viết tắt và thuật ngữ (VD: `VCB` $\rightarrow$ `Vietcombank`, `đáo hạn`, `CASA`, `CVV`). | **Cực kỳ quan trọng cho nghiệp vụ ngân hàng**, chuẩn hóa tức thì ($<1$ ms). | Phải tự xây dựng và bảo trì danh mục từ điển thuật ngữ. |
| **Keyword / NER Extractor** | Tách từ khóa chính và thực thể (số tiền, loại thẻ, dịch vụ) loại bỏ từ dừng (stop-words). | Giúp thuật toán BM25 tập trung vào từ khóa mang tải thông tin cao. | Dễ cắt mất ý nghĩa nếu câu hỏi có ngữ cảnh đảo ngược (VD: "không muốn mở thẻ"). |
| **Multi-Query Expansion** | Dùng LLM viết lại câu hỏi thành 3-5 phiên bản góc nhìn khác nhau. | Bao quát toàn diện các cách diễn đạt khác nhau của khách hàng. | Tăng độ trễ do gọi LLM ($+500$ ms) và nhân số lượt truy xuất lên gấp 3-5 lần. |
| **HyDE (Hypothetical Doc Embed)** | Dùng LLM sinh câu trả lời giả định trước, rồi lấy embedding của câu giả định đi tìm kiếm. | Chuyển đổi không gian Vector từ dạng `Câu hỏi` sang dạng `Tài liệu`. | Dễ bị "ảo giác dẫn đường" (hallucination drift) nếu LLM ban đầu sinh sai nghiệp vụ. |
| **Step-Back Prompting** | Đặt câu hỏi lùi về khái niệm gốc tổng quát hơn (VD: từ "lãi thẻ Visa Cash Back" lùi về "chính sách lãi thẻ tín dụng"). | Giúp lấy được tài liệu nguyên lý nền tảng thay vì sa đà vào chi tiết hẹp. | Đòi hỏi thêm 1 bước gọi LLM phân tích ngữ nghĩa trừu tượng. |

---

## 6. Module: Retrieval & Fusion (Truy xuất & Hợp nhất)

| Thuật toán | Ý nghĩa cốt lõi | Điểm cộng (+) | Điểm trừ (-) |
| :--- | :--- | :--- | :--- |
| **Dense Only** | Chỉ truy xuất dựa trên độ tương đồng Vector (Cosine Similarity). | Hiểu tốt ý định ngữ nghĩa, câu hỏi diễn đạt quanh co vẫn tìm được. | Thường trượt khi khách hàng hỏi đích danh số hotline, mã biểu phí, tên gói vay. |
| **Sparse Only** | Chỉ truy xuất dựa trên so khớp từ khóa (BM25 / TF-IDF). | Chính xác tuyệt đối với thuật ngữ hiếm và mã ký hiệu. | Thất bại hoàn toàn với từ đồng nghĩa hoặc câu hỏi trừu tượng. |
| **Weighted Linear Hybrid** | Cộng tuyến tính điểm chuẩn hóa: $Score = \alpha \cdot S_{dense} + (1-\alpha) \cdot S_{sparse}$. | Cân bằng hoàn hảo giữa ngữ nghĩa và từ khóa; dễ hiểu, dễ tinh chỉnh $\alpha$. | Điểm số Dense và Sparse có phân phối khác nhau, cần bước chuẩn hóa Min-Max cẩn thận. |
| **Reciprocal Rank Fusion (RRF)** | Hợp nhất theo thứ hạng nghịch đảo: $RRF(d) = \sum \frac{1}{k + rank_i(d)}$ (thường chọn $k=60$). | **Chuẩn thực chiến tốt nhất**; không quan tâm thang điểm gốc, chỉ cần thứ hạng. | Không phản ánh được khoảng cách chênh lệch điểm số thực tế giữa các vị trí. |
| **Threshold-Filtered Hybrid** | Chỉ giữ lại các tài liệu có điểm số vượt qua một ngưỡng tự tin nhất định ($score \ge \tau$). | Ngăn chặn chunk rác/nhiễu lọt vào context prompt của LLM. | Nếu đặt ngưỡng quá cao sẽ làm rỗng kết quả tìm kiếm (zero recall). |

---

## 7. Module: Reranking (Tái xếp hạng)

| Thuật toán / Kỹ thuật | Ý nghĩa cốt lõi | Điểm cộng (+) | Điểm trừ (-) |
| :--- | :--- | :--- | :--- |
| **Identity (No-op)** | Giữ nguyên thứ tự kết quả trả về từ tầng Retrieval. | Trễ $0$ ms, tiết kiệm toàn bộ tài nguyên tính toán. | Phụ thuộc hoàn toàn vào chất lượng của bộ Retriever. |
| **Lexical Overlap / Jaccard** | Đếm tỷ lệ phần trăm từ khóa trong câu hỏi xuất hiện trực tiếp trong văn bản chunk. | Cực nhanh, giúp kéo các đoạn chứa trọn vẹn từ khóa quan trọng lên đầu. | Chỉ nhìn vào từ vựng bề mặt, không đo lường được tính liên quan ngữ nghĩa sâu. |
| **FAQ Golden Match** | Boost điểm ưu tiên cao nhất ($+10.0$) nếu chunk là cặp Q&A khớp intent câu hỏi. | **Cực chuẩn cho Banking bot**: giải quyết tức thì các câu hỏi thường gặp. | Chỉ có tác dụng với kho tài liệu đã gán nhãn metadata FAQ. |
| **Score-Blended Weighted** | Kết hợp điểm vector + điểm từ vựng + metadata boost (ngày hiệu lực, độ tin cậy nguồn). | Phản ánh đa chiều giá trị của tài liệu ngân hàng. | Cần điều chỉnh nhiều trọng số phụ thuộc vào cảm tính thử nghiệm. |
| **Cross-Encoder Neural** | Cho cặp `(Query, Chunk)` đi qua chung một mô hình Transformer (`bge-reranker-v2-m3`). | **Độ chính xác cao nhất**, hiểu sâu sắc mối quan hệ ngữ nghĩa trực tiếp. | Độ trễ cao ($50 - 200$ ms), ngốn tài nguyên GPU nếu rerank nhiều chunk ($K > 20$). |
| **LLM Listwise Reranker** | Gửi danh sách Top-K văn bản cho LLM và yêu cầu xếp hạng lại từ phù hợp nhất đến ít nhất. | Tận dụng được khả năng suy luận logic siêu việt của LLM lớn. | Chi phí API đắt đỏ, độ trễ rất lớn ($1 - 3$ giây). |

---

## 8. Module: Context Synthesis (Tổng hợp ngữ cảnh)

| Kỹ thuật | Ý nghĩa cốt lõi | Điểm cộng (+) | Điểm trừ (-) |
| :--- | :--- | :--- | :--- |
| **Raw Concatenation** | Ghép nối thô danh sách các văn bản bằng dấu ngắt dòng `\n\n`. | Đơn giản nhất, không qua xử lý phức tạp. | Dễ gây rối mắt cho LLM, không biết thông tin nào đến từ tài liệu nào. |
| **Structured Markdown & Citation** | Đánh số thứ tự và gắn nhãn nguồn rõ ràng: `[Nguồn 1: Biểu phí thẻ]`, `[Nguồn 2: Điều khoản...]`. | Giúp LLM dễ dàng trích dẫn nguồn số mấy (`[1]`, `[2]`), tăng tính minh bạch. | Tốn thêm một lượng nhỏ token cho phần tiêu đề nhãn. |
| **Lost-in-the-Middle Reorder** | Đưa các chunk quan trọng nhất lên vị trí đầu tiên và cuối cùng của ngữ cảnh. | Khắc phục hiện tượng LLM hay "quên" thông tin nằm ở đoạn giữa prompt. | Cần thêm logic sắp xếp vị trí mảng. |
| **Token Budget Truncation** | Cắt bớt văn bản nghiêm ngặt khi chạm ngưỡng tối đa (VD: $2000$ tokens). | **Đảm bảo không bao giờ bị lỗi tràn context window** hoặc vượt ngân sách chi phí. | Có thể cắt cụt thông tin ở chunk nằm cuối danh sách. |
| **Context Compression** | Dùng LLM nhỏ tóm tắt, chỉ giữ lại các câu trực tiếp trả lời câu hỏi trước khi ghép prompt. | Tiết kiệm token tối đa, loại bỏ hoàn toàn thông tin thừa. | Tốn thêm thời gian tiền xử lý và có rủi ro tóm tắt sót ý quan trọng. |

---

## 9. Module: LLM Generation & Guardrails (Sinh phản hồi)

| Cơ chế / Mô hình | Ý nghĩa cốt lõi | Điểm cộng (+) | Điểm trừ (-) |
| :--- | :--- | :--- | :--- |
| **Rule-based / Mock LLM** | Trả lời theo mẫu định sẵn hoặc trích xuất trực tiếp câu trả lời từ chunk phù hợp nhất. | Chạy tức thì, $0$ chi phí, kết quả xác định $100\%$ phục vụ viết Unit Test. | Không thể đàm thoại tự nhiên linh hoạt với khách hàng. |
| **Local LLM (Qwen-2.5 / PhoGPT)** | Triển khai mô hình nguồn mở trên máy chủ nội bộ qua Ollama hoặc vLLM. | **Bảo mật tuyệt đối dữ liệu nội bộ ngân hàng**, kiểm soát hoàn toàn hệ thống. | Yêu cầu máy chủ có GPU cấu hình mạnh (VRAM $\ge 16$GB). |
| **Cloud Frontier LLM** | Kết nối API các mô hình cao cấp (DeepSeek-V3, Gemini 2.5 Flash, GPT-4o). | Khả năng suy luận đỉnh cao, giọng văn mượt mà, trả lời câu hỏi khó xuất sắc. | Rủi ro rò rỉ dữ liệu qua bên thứ ba; phụ thuộc mạng và chi phí token. |
| **Banking Guardrails Prompt** | Prompt hệ thống nghiêm ngặt: cấm bịa đặt, cấm tư vấn tài chính trái luật, bắt trích dẫn nguồn. | **Bảo vệ an toàn nghiệp vụ**, chống hallucination, đảm bảo tuân thủ pháp lý. | Đôi khi khiến câu trả lời bị khô cứng hoặc từ chối trả lời quá an toàn. |

---

## 10. Lộ trình gợi ý tự code từ đầu (Recommended Roadmap)

Nếu bạn mới bắt đầu tự lập trình lại hệ thống từ đầu, hãy làm tuần tự theo **5 giai đoạn thực chiến** sau:

```
[Giai đoạn 1: Chuẩn bị dữ liệu]
  └── Code Chunking: Fixed Window & FAQ Pair (cơ bản)
        └── Nâng cấp: Paragraph Split & Hierarchy Markdown (nâng cao)

[Giai đoạn 2: Tìm kiếm từ vựng]
  └── Code Keyword Search: BM25 Okapi hoặc TF-IDF thuần NumPy

[Giai đoạn 3: Tìm kiếm ngữ nghĩa]
  └── Code Vector DB: Brute-force Cosine Similarity với NumPy
        └── Tích hợp Embedding: Dùng mô hình SentenceTransformers hoặc API

[Giai đoạn 4: Hợp nhất lai & Tái sắp xếp]
  └── Code Retrieval: Reciprocal Rank Fusion (RRF) kết hợp BM25 + Vector
        └── Code Reranking: Lexical Overlap hoặc FAQ Golden Match

[Giai đoạn 5: Tổng hợp & Hoàn thiện]
  └── Code Context Synthesis: Ghép prompt kèm thẻ trích dẫn [Nguồn X]
        └── Gọi LLM & Viết System Prompt an toàn nghiệp vụ ngân hàng
```
