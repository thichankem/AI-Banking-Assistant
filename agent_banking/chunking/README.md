# Chunking PDF, DOCX và Excel

## Cấu trúc mã nguồn

| Vị trí trong `universal/` | Trách nhiệm |
| --- | --- |
| `pipeline.py`, `fingerprint.py` | Điều phối xử lý tài liệu và tính fingerprint cho cache |
| `extraction/docx/`, `extraction/pdf/`, `extraction/xlsx/` | Đọc nội dung, bố cục, bảng và tọa độ nguồn theo định dạng |
| `enrichment/`, `enrichment/decisions/` | Bổ sung bằng chứng, kiểm tra cấu trúc và quyết định về heading, bảng, ảnh |
| `planning/` | Lập chunk, dựng bảng, ghép ngữ cảnh và kiểm tra độ phủ |
| `text/` | Nhận diện ngôn ngữ, tìm biên Unicode và tính ngân sách token |
| `models/` | Gọi model văn bản, vision và lưu cache phản hồi |

Điểm vào Python là `agent_banking.chunking.universal.pipeline.UniversalDocumentChunker`. Lệnh CLI chạy qua `python -m agent_banking.chunking.run_chunking`.

Phần OCR và các bước cần model dùng API cloud đã cấu hình. Python đọc cấu trúc DOCX, render PDF, tổ chức kết quả và xuất chunk cùng thông tin nguồn.

## Excel

Đặt file `.xlsx` vào thư mục input. CLI tự nhận định dạng và đọc dữ liệu từ các sheet hiển thị. Kết quả trích dẫn dùng tên sheet và vùng ô A1. Ô chứa công thức và kết quả đã lưu của ô đó được bỏ qua. Manifest ghi số ô đã bỏ qua và các đối tượng đọc thiếu theo vị trí.

Mọi ảnh nhúng đọc được trên sheet được xử lý qua model OCR Cloud đã cấu hình. Pipeline dùng lại kết quả OCR từ cache, gọi Vision Review khi cần và thử lại OCR chưa hoàn tất trong lượt nạp sau. Text và Decision được dùng khi bố cục bảng hoặc cặp hỏi đáp chưa rõ.

Khi quản lý nhiều bản của cùng biểu phí, giữ các file phiên bản dưới đường dẫn riêng rồi khai báo mã tài liệu và ngày hiệu lực trong một file JSON. Khóa JSON là đường dẫn nguồn tương đối như trong manifest.

```json
{
  "bieu-phi-v2.xlsx": {
    "document_key": "BP-01",
    "business_version": "2",
    "effective_from": "2025-07-01"
  }
}
```

```powershell
python -m agent_banking.chunking.run_chunking --input "<thu_muc_input>" --output "<thu_muc_output>" --excel-metadata "<revisions.json>"
```

Khi file JSON được bỏ qua, hệ thống đọc các nhãn như `Mã tài liệu`, `Phiên bản` và `Ngày hiệu lực` trong workbook. Ngày thiếu hoặc có nhiều giá trị khác nhau được giữ ở trạng thái chưa xác định. Các phiên bản nên giữ cùng tên Excel Table hoặc cùng vị trí bắt đầu bảng trên sheet để tra cứu theo cùng một ID bảng.

Trong `search_banking_knowledge`, dùng tìm kiếm thông thường để lấy `business_metadata.excel_table_id`. Tra theo hàng bằng tham số `excel`, ví dụ `{"table_id":"<id>","as_of":"2025-07-01","filters":[{"column":"Phí VND","op":"gte","value":100000,"unit":"VND"}]}`. Điều kiện được kiểm tra trên cùng một hàng của bản có hiệu lực. Giá trị phần trăm dùng điểm phần trăm, ví dụ `2.5` cho `2,5%`.

## PDF scan, ảnh và chú thích

- Pipeline render từng trang PDF và gửi ảnh trang cho model OCR cloud. DOCX được đọc từ cấu trúc OpenXML.
- Mỗi chunk xuất tên file trong `business_metadata.citation.file_name`. Chunk PDF có thêm `page_numbers`, liệt kê các trang nguồn theo số trang bắt đầu từ 1. Trích dẫn DOCX chỉ gồm tên file. Tọa độ hình ảnh phục vụ bố cục nội bộ khi có và thiếu tọa độ không kích hoạt retry.
- DOCX giữ thân bài, bảng, hộp văn bản, caption và chú thích có tham chiếu. Bật `--ocr-docx-images` để thêm chữ nhìn thấy trong ảnh raster nhúng. Pipeline gửi từng ảnh qua cloud và lưu XPath cùng hash ảnh vào chứng cứ nguồn.
- OCR PDF giữ chữ nhìn thấy trong nhãn, trục, chú giải, bảng, chú thích cuối trang và chỉ dẫn trong ảnh màn hình.
- Mỗi yêu cầu OCR gửi một trang, một vùng hoặc một ảnh nhúng, với trần mặc định 6.144 token đầu ra mỗi lần gọi. OCR giữ kết quả đầu tiên đọc được. Khi kết quả chưa dùng được, pipeline thử model cloud dự phòng và có thể gửi các vùng ảnh nhỏ hơn nếu phản hồi chạm giới hạn kích thước hoặc sai định dạng.
- Khi phản hồi thiếu trường bắt buộc, pipeline bổ sung yêu cầu sửa JSON rồi gọi lại. Mỗi trang PDF, ảnh DOCX hoặc cửa sổ phân tích có tối đa 3 lượt gọi cloud tính cả lượt đầu. Với OCR, lượt dự phòng và đọc các vùng nhỏ dùng chung hạn mức này. Model chính tự sửa phản hồi khi đã đọc được nội dung. Model dự phòng được dùng nếu model chính chưa trả nội dung dùng được. Python giữ nội dung cloud đọc được. Artifact của tài liệu ghi cảnh báo, các trang OCR một phần và các trang lỗi. Tài liệu bị ghi lỗi khi quá trình OCR gặp lỗi và toàn bộ file thiếu đầu ra có thể dùng. Khi đã render trang lỗi, hash ảnh trong báo cáo liên kết với ảnh trang trong `assets/`.


## Cấu hình và đổi model cloud

Dùng tiền tố `CHUNKING_TEXT`, `CHUNKING_VISION`, `CHUNKING_VISION_REVIEW` hoặc `CHUNKING_DECISION` theo vai trò. Ở mỗi vai trò, đặt hậu tố `_MODEL`, `_BASE_URL` và `_API_KEY` trong môi trường hoặc `.env` của dự án. Dùng URL gốc tương thích OpenAI, ví dụ URL kết thúc bằng `/v1`. SystemOne là vai trò tùy chọn và dùng endpoint `/systemone`.

| Hậu tố theo vai trò | Cách dùng |
| --- | --- |
| `_TEMPERATURE` | Số từ 0 đến 2. Đặt `_SEND_TEMPERATURE=false` khi endpoint yêu cầu bỏ trường này. |
| `_MAX_OUTPUT_TOKENS`, `_TOKEN_PARAMETER` | Giới hạn đầu ra và tên trường `max_tokens` hoặc `max_completion_tokens`. Trần nghiệp vụ của từng tác vụ vẫn áp dụng. |
| `_JSON_MODE` | `on` gửi JSON mode, `off` bỏ trường đó, `auto` dùng tùy chọn của tác vụ. Phản hồi luôn được kiểm tra theo schema của tác vụ. |
| `_DISABLE_THINKING`, `_IMAGE_DETAIL` | Tùy chọn cho endpoint hỗ trợ tham số thinking và mức chi tiết ảnh. |
| `_TIMEOUT`, `_ATTEMPTS`, `_ENABLED` | Điều chỉnh thời gian chờ, số lượt thử hoặc tắt ứng viên tùy chọn. |

Các biến cũ như `MODEL`, `VISION_MODEL`, `DECISION_MODEL`, `BASE_URL` và `API_KEY` vẫn được đọc khi vai trò chưa có biến riêng. Tùy chỉnh prompt OCR cloud qua `CLOUD_OCR_PAGE_PROMPT` hoặc `config/prompts/pdf_extractor/cloud_page.md`. Cache tự đổi fingerprint khi model, URL hoặc cấu hình ảnh hưởng kết quả thay đổi.

```powershell
python -m agent_banking.chunking.run_chunking --check-models
python -m agent_banking.chunking.golden_suite --output "tmp/chunking-model-a" --cache "tmp/chunking-responses.db" --regression-input "agent_banking/chunking/input"
```

Sau khi đổi model qua môi trường, chạy lại hai lệnh với thư mục xuất riêng. Lệnh kiểm tra gửi dữ liệu tổng hợp nhỏ đến từng ứng viên đã bật. Benchmark tự tạo 24 tài liệu thuộc sáu nhóm. `--regression-input` thêm hai PDF HDSD. File `golden_report.json` liệt kê mục trọng yếu còn thiếu theo nguồn. Ngưỡng bàn giao là đủ mọi mục trọng yếu, các file đều thành công và 0 trang OCR lỗi.

## Ranh giới chunk và ngôn ngữ

- Mỗi file được xử lý riêng, theo thứ tự. Các tác vụ trong một file có thể chạy theo lô với tối đa hai worker.
- Mỗi batch SystemOne xử lý tối đa 6 câu hỏi. Rà cấu trúc gửi tối đa 6 khối chính trong một cửa sổ bằng chứng. Chấm ranh giới gửi 12 đoạn cần chấm và tối đa 3 đoạn ngữ cảnh ở mỗi phía.
- Heading có tối đa 7 cấp. Heading gốc rõ ràng được giữ, cấp sâu hơn được quy về cấp 7.
- Danh sách Word được đọc như văn bản thường. Nhãn số hoặc dấu đầu dòng gõ trực tiếp được giữ nguyên, còn nhãn Word tự sinh từ cấu hình danh sách được bỏ qua. LLM chấm ranh giới chunk dựa trên nội dung.
- Danh sách PDF được đọc như văn bản thường. Giữ nguyên nhãn số, chữ cái và dấu đầu dòng đã hiển thị trên trang, kể cả khi thứ tự bị ngắt quãng. Với PDF scan, OCR chép các nhãn nhìn thấy và để nguyên mục thiếu nhãn.
- Chunk nhắm 512 token, ưu tiên tối thiểu 128 token khi nội dung và cấu trúc cho phép, với trần thông thường 1024. Phần ngắn hợp lệ có thể dưới 128. Công thức, tiêu đề bảng hay hàng bảng không thể cắt an toàn được giữ nguyên.
- Pipeline dùng model embedding được cấu hình qua biến môi trường (`EMBEDDING_MODEL`). Mặc định mỗi request embedding nhận tối đa 512 token trên một đoạn. Chunk dài hơn được biểu diễn bằng vector tổng hợp có trọng số từ các đoạn gửi đi, còn nội dung gốc giữ nguyên trong chunk. Giới hạn dịch vụ có thể đặt bằng `MAX_PASSAGE_TOKENS` khi endpoint đã được xác nhận.
- Mặc định tìm kiếm giữ tất cả ngôn ngữ. Khi chạy `--search-language vi` hoặc chọn `search_language: vi` trong cấu hình ứng dụng, nội dung nhận diện được là ngoại ngữ giữ nguyên trong `business_metadata.elements` với kiểu `foreign_language` và vị trí nguồn. Bộ nhận diện kiểm tra cả câu khác ngôn ngữ trên cùng dòng và ưu tiên dấu hiệu tiếng Việt rõ ràng khi thẻ ngôn ngữ Word không khớp. Văn bản Latin ngắn hoặc mơ hồ được giữ trong ngữ cảnh để tránh loại nhầm tiếng Việt không dấu. Phần chỉ có ngoại ngữ được giữ trong đầu ra với vai trò `attachment_only` khi không có nội dung tìm kiếm tương ứng, kể cả ở một phần của bảng nhiều chunk.
- Biểu thức Word hỗ trợ được lưu dưới dạng Markdown có TeX, chỉ số trên/dưới có thể dùng `<sup>` và `<sub>`. Công thức Word chưa chuyển được sang TeX vẫn còn dưới dạng OMML để khôi phục. Bảng lưu lưới và tọa độ ô gộp ngoài biểu diễn văn bản.

## Tên và kết quả theo file

Định danh nguồn dùng đường dẫn tương đối có đuôi file, tính từ thư mục input chung khi nguồn nằm bên trong. Trước khi xử lý, hệ thống kiểm tra trùng tên trong các thư mục con, không phân biệt hoa/thường. Các bản cùng định dạng như `Hướng dẫn.pdf` và `Hướng dẫn (1).pdf` cũng được coi là trùng nếu cùng có mặt. Báo lỗi liệt kê từng đường dẫn theo số thứ tự. Đổi tên nguồn có ý nghĩa rồi chạy lại.

Xem `chunking_manifest.json` để tra file kết quả riêng trong `chunks_per_file/` và bằng chứng trong `artifacts/`. Với file xử lý lỗi, pipeline hoàn thành lượt đầu của các file khác rồi thử lại các lỗi cấp tài liệu còn có thể hồi phục tối đa ba lượt, chờ lần lượt 30, 60 và 120 giây. File vẫn lỗi được ghi vào báo cáo. File thành công được xuất bản và CLI trả mã lỗi nếu còn file thất bại sau cùng. Khi chạy toàn bộ pipeline, các file lỗi vẫn giữ bản đã index trước đó, nếu có.

Các trường `ocr_pages_processed`, `ocr_partial_pages` và `ocr_failed_pages` giúp đối chiếu số trang PDF đã xử lý, thiếu một phần và lỗi. Tài liệu có đầu ra OCR một phần vẫn có thể được xuất bản kèm thông tin chất lượng. Artifact ghi `issues` cho trang thiếu trường đánh giá, trang model tự báo chưa chắc chắn và trang thiếu nội dung OCR dùng được. `--force` chạy lại tài liệu, dùng lại các trang OCR đã có kết quả đầy đủ và gửi lại các trang thiếu hoặc lỗi. Bản xuất PDF từ policy cũ cần chạy lại chunking trước khi đưa vào chỉ mục.

Khi lập chỉ mục theo manifest, phạm vi đồng bộ được xác định theo thư mục xuất bản. Tài liệu đã bỏ khỏi snapshot sẽ được dọn khỏi chỉ mục của phạm vi đó. Điểm cũ chưa có phạm vi được dọn khi khớp đường dẫn nguồn hoặc có duy nhất một đối chiếu theo hậu tố đường dẫn trong lượt thành công. Điểm cũ có nguồn chưa đối chiếu được hoặc tài liệu xử lý lỗi vẫn được giữ.

`--clean` dựng lại kết quả và chỉ dọn các file xuất bản cũ sau khi toàn bộ lượt chạy thành công. Thư mục `runtime`, dữ liệu benchmark và tệp người dùng được giữ lại.

## Đồng bộ chỉ mục

Khi chạy từng bước, tạo embedding từ manifest chunking. Sau đó chạy lệnh sau từ thư mục gốc của dự án.

```text
python -m agent_banking.vector_db.run_vector_db --input "<embedded_chunks.json>" --snapshot --publication "<thư_mục_xuất_bản_chunking>"
```

## Chạy kiểm thử

Từ thư mục gốc của dự án, chạy `uv run --group dev pytest -q agent_banking/chunking/tests`. Các ca OCR và lập chỉ mục trong suite dùng phản hồi mô phỏng và kho dữ liệu tạm.
