# QUY TẮC PHÁT TRIỂN & NGHIÊN CỨU DỰ ÁN AI BANKING ASSISTANT

Tài liệu này định hình hành vi cốt lõi của Trợ lý AI khi làm việc trong dự án **AI Banking Assistant**.

---

## 1. NGUYÊN TẮC BẮT BUỘC: ƯU TIÊN SỬ DỤNG NOTEBOOKLM (NOTEBOOKLM-FIRST)

Khi người dùng trao đổi, thảo luận, thiết kế kiến trúc, lập trình hoặc viết báo cáo học thuật:
* **Luôn luôn truy vấn NotebookLM trước**: Gọi công cụ `notebook_query` từ server `gemini-notebook-mcp` để đối soát tri thức với tài liệu nguồn đã được chuẩn hóa.
* **Định tuyến sổ tay chuẩn xác**:
  * Đề tài GR2, Kiến trúc 4 tầng, Luồng dữ liệu, Checklist: `1938151d-f91d-47df-b395-4b27aac0a211` (`AI Agent banking`)
  * Lý thuyết AI Agent, CoALA, LangGraph, MCP: `f07ce8fc-f536-4624-af36-3f614e6637ca` (`AI Agent`)
  * Kỹ thuật RAG nâng cao, Hybrid Search, CRAG, Reranking: `880c1277-4ac1-4b2b-9565-54913bda81df` (`RAG`)
  * Phân đoạn văn bản, Late Chunking, Coreference Trap: `d3f1452a-ba79-4ca9-875e-a5d7bb20ca74` (`Strategic Document Chunking`)
  * Nghiệp vụ lừa đảo, an toàn ngân hàng, gian lận: `f3f23d9f-31db-4025-873e-fd47fec79a87` (`Khảo sát về những gian lận trong ngân hàng`)
* **Dẫn chứng minh bạch**: Trích dẫn cụ thể số hiệu citation từ nguồn NotebookLM trong mọi câu trả lời phân tích chuyên sâu.

---

## 2. QUY CHUẨN XÓA LỊCH SỬ & LÀM SẠCH NHẬT KÝ TRÒ CHUYỆN (SESSION HYGIENE)

Để ngăn ngừa hiện tượng trôi dạt ngữ cảnh (context drift), giảm thiểu độ trễ truy vấn và tránh ảo giác do lịch sử tích tụ quá dài:
* **Thường xuyên làm mới phiên hội thoại**: Khi chuyển đổi sang một khía cạnh/chủ đề nghiên cứu mới hoặc sau chuỗi **3–5 câu hỏi sâu**, **BẮT BUỘC** truyền tham số `"new_conversation": true` vào công cụ `notebook_query`.
* **Cơ chế hoạt động**: `"new_conversation": true` sẽ khởi động một phiên làm việc độc lập, xóa bỏ hoàn toàn nhật ký các lượt hỏi đáp cũ trong phiên, giữ cho bộ nhớ ngữ cảnh của NotebookLM luôn tinh gọn, sắc bén và chính xác nhất.
* **Lưu vết trước khi xóa**: Nếu các lượt hội thoại trước đó rút ra kết luận mang tính bước ngoặt, hãy dùng công cụ `note` (`action="create"`) để lưu thành Note vĩnh viễn trên sổ tay trước khi khởi động phiên hội thoại mới.

---

## 3. CƠ CHẾ TỰ ĐỘNG DỌN DẸP BÁO CÁO & TẠO TÁC LỖI (AUTOMATED ERROR CLEANUP)

Để giữ cho kho lưu trữ dự án và Studio trên NotebookLM luôn sạch sẽ, không bị tích tụ bởi các báo cáo hỏng, tệp rỗng hoặc tạo tác sinh lỗi:
* **Tự động xóa tạo tác lỗi trên Cloud**: Khi kiểm tra hoặc tạo báo cáo/tạo tác qua Studio (`studio_status`), nếu phát hiện bất kỳ artifact nào có trạng thái `failed`, `error` hoặc có `error_reason != null`, **BẮT BUỘC** gọi ngay `studio_delete` (hoặc `nlm studio delete <notebook_id> <artifact_id> --confirm`) để xóa vĩnh viễn khỏi sổ tay NotebookLM.
* **Tự động dọn dẹp tệp hỏng cục bộ (Local Storage Cleanup)**: Khi tải báo cáo/tạo tác (`download_artifact`, `download_all_artifacts`) về thư mục `docs/research/` hoặc thư mục tải về:
  * Tự động kiểm tra tính toàn vẹn của tệp ngay sau khi tải.
  * Nếu phát hiện tệp rỗng (0 bytes), tệp tải dở (`.tmp`, `.crdownload`, `.part`), hoặc tệp phản hồi payload lỗi (`{"status": "error"}`), **BẮT BUỘC** xóa bỏ ngay lập tức.
  * Định kỳ thực thi công cụ hoặc script bảo trì `scripts/cleanup_failed_artifacts.py` để rà soát và đảm bảo tính nguyên vẹn của thư mục tài liệu.

---

## 4. QUY ĐỊNH BẮT BUỘC: LUÔN SỬ DỤNG CHẾ ĐỘ DEEP RESEARCH (DEEP RESEARCH DIRECTIVE)

Khi thực hiện tìm kiếm, mở rộng tri thức và khám phá các nguồn tài liệu học thuật/kỹ thuật mới cho đề tài AI Banking Assistant (sử dụng công cụ `research_start` từ server `gemini-notebook-mcp` hoặc CLI `nlm research start`):
* **Luôn luôn chọn chế độ `deep`**: **BẮT BUỘC** truyền tham số `"mode": "deep"` (tuyệt đối KHÔNG dùng `"mode": "fast"`).
* **Mục đích khoa học**: Chế độ `deep` kích hoạt khả năng duyệt web đa tầng, khảo sát sâu không gian học thuật toàn cầu, thu thập đến ~40 nguồn tài liệu chất lượng cao (bài báo hội nghị hàng đầu NeurIPS/ICLR/ACL, preprint arXiv, báo cáo tài chính quốc tế) thay vì chỉ quét sơ bộ ~10 nguồn của chế độ fast.
* **Quy trình chuẩn**:
  1. Khởi chạy: Gọi `research_start(query="...", mode="deep", source="web", notebook_id="...")`.
  2. Giám sát: Kiểm tra trạng thái bằng `research_status`.
  3. Nhập nguồn: Phê duyệt và nạp các tài liệu đối soát vào sổ tay bằng `research_import`.


