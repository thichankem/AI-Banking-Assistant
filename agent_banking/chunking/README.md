# Chunking PDF và DOCX

## Cấu trúc mã nguồn

| Vị trí trong `universal/` | Trách nhiệm |
| --- | --- |
| `pipeline.py`, `fingerprint.py` | Điều phối xử lý tài liệu và tính fingerprint cho cache |
| `extraction/docx/`, `extraction/pdf/` | Đọc nội dung, bố cục, bảng và tọa độ nguồn theo định dạng |
| `enrichment/`, `enrichment/decisions/` | Bổ sung bằng chứng, kiểm tra cấu trúc và quyết định về heading, bảng, ảnh |
| `planning/` | Lập chunk, dựng bảng, ghép ngữ cảnh và kiểm tra độ phủ |
| `text/` | Nhận diện ngôn ngữ, tìm biên Unicode và tính ngân sách token |
| `models/` | Gọi model văn bản, vision và lưu cache phản hồi |

Điểm vào Python cho PDF và DOCX là `agent_banking.chunking.universal.pipeline.UniversalDocumentChunker`. Lệnh CLI chạy qua `python -m agent_banking.chunking.run_chunking`.

Phần OCR và các bước cần model dùng API cloud đã cấu hình. Python đọc cấu trúc DOCX, render PDF, tổ chức kết quả và xuất chunk cùng thông tin nguồn.

## PDF scan, ảnh và chú thích

- Pipeline render từng trang PDF và gửi ảnh trang cho model OCR cloud. DOCX được đọc từ cấu trúc OpenXML.
- **Ảnh minh họa, sơ đồ và biểu đồ đứng riêng không được gửi riêng cho vision hoặc OCR.** DOCX giữ chữ được tác giả viết trong thân bài, ô bảng, hộp văn bản và caption của thân bài. Nhận chữ trên ảnh cả trang PDF scan phục vụ việc đọc nội dung văn bản của trang.
- Mỗi yêu cầu OCR gửi một trang hoặc một vùng ảnh, với tối đa 6.144 token đầu ra mỗi lần gọi. OCR giữ kết quả đầu tiên đọc được. Khi kết quả chưa dùng được, pipeline thử model cloud dự phòng và có thể gửi các vùng ảnh nhỏ hơn nếu phản hồi chạm giới hạn kích thước hoặc sai định dạng.
- Python giữ nội dung cloud đọc được. Artifact của tài liệu ghi cảnh báo, các trang OCR một phần và các trang lỗi. Tài liệu bị ghi lỗi khi quá trình OCR gặp lỗi và toàn bộ file thiếu đầu ra có thể dùng. Khi đã render trang lỗi, hash ảnh trong báo cáo liên kết với ảnh trang trong `assets/`.
- Cấu hình model cloud chính và dự phòng qua `CHUNKING_VISION_MODEL` và `CHUNKING_VISION_REVIEW_MODEL`.

## Ranh giới chunk và ngôn ngữ

- Mỗi file được xử lý riêng, theo thứ tự. Các tác vụ trong một file có thể chạy theo lô với tối đa hai worker.
- Mỗi batch SystemOne xử lý tối đa 6 câu hỏi. Rà cấu trúc gửi tối đa 6 khối chính trong một cửa sổ bằng chứng. Chấm ranh giới gửi 12 đoạn cần chấm và tối đa 3 đoạn ngữ cảnh ở mỗi phía.
- Heading có tối đa 7 cấp. Heading gốc rõ ràng được giữ, cấp sâu hơn được quy về cấp 7.
- Danh sách Word được đọc như văn bản thường. Nhãn số hoặc dấu đầu dòng gõ trực tiếp được giữ nguyên, còn nhãn Word tự sinh từ cấu hình danh sách được bỏ qua. LLM chấm ranh giới chunk dựa trên nội dung.
- Danh sách PDF được đọc như văn bản thường. Giữ nguyên nhãn số, chữ cái và dấu đầu dòng đã hiển thị trên trang, kể cả khi thứ tự bị ngắt quãng. Với PDF scan, OCR chép các nhãn nhìn thấy và để nguyên mục thiếu nhãn.
- Chunk tiếng Việt nhắm 512 token, ưu tiên tối thiểu 128 token khi nội dung và cấu trúc cho phép, với trần thông thường 1024. Phần ngắn hợp lệ có thể dưới 128. Công thức, tiêu đề bảng hay hàng bảng không thể cắt an toàn được giữ nguyên.
- Pipeline vẫn dùng embedding multilingual-e5-large. Mặc định mỗi request embedding nhận tối đa 512 token trên một đoạn. Chunk dài hơn được biểu diễn bằng vector tổng hợp có trọng số từ các đoạn gửi đi, còn nội dung gốc giữ nguyên trong chunk. Giới hạn dịch vụ có thể đặt bằng `MAX_PASSAGE_TOKENS` khi endpoint đã được xác nhận.
- Nội dung nhận diện được là ngoại ngữ giữ nguyên trong `business_metadata.elements` với kiểu `foreign_language` và vị trí nguồn. Bộ nhận diện kiểm tra cả câu khác ngôn ngữ trên cùng dòng và ưu tiên dấu hiệu tiếng Việt rõ ràng khi thẻ ngôn ngữ Word không khớp. Văn bản Latin ngắn hoặc mơ hồ được giữ trong ngữ cảnh để tránh loại nhầm tiếng Việt không dấu. Phần chỉ có ngoại ngữ được giữ trong đầu ra với vai trò `attachment_only` khi không có nội dung tìm kiếm tương ứng, kể cả ở một phần của bảng nhiều chunk.
- Biểu thức Word hỗ trợ được lưu dưới dạng Markdown có TeX, chỉ số trên/dưới có thể dùng `<sup>` và `<sub>`. Công thức Word chưa chuyển được sang TeX vẫn còn dưới dạng OMML để khôi phục. Bảng lưu lưới và tọa độ ô gộp ngoài biểu diễn văn bản.

## Tên và kết quả theo file

Định danh nguồn dùng đường dẫn tương đối có đuôi file, tính từ thư mục input chung khi nguồn nằm bên trong. Trước khi xử lý, hệ thống kiểm tra trùng tên trong các thư mục con, không phân biệt hoa/thường. Các bản cùng định dạng như `Hướng dẫn.pdf` và `Hướng dẫn (1).pdf` cũng được coi là trùng nếu cùng có mặt. Báo lỗi liệt kê từng đường dẫn theo số thứ tự. Đổi tên nguồn có ý nghĩa rồi chạy lại.

Xem `chunking_manifest.json` để tra file kết quả riêng trong `chunks_per_file/` và bằng chứng trong `artifacts/`. Với file xử lý lỗi, pipeline hoàn thành lượt đầu của các file khác rồi thử lại nhóm lỗi tối đa ba lượt, chờ lần lượt 30, 60 và 120 giây. File vẫn lỗi được ghi vào báo cáo. File thành công được xuất bản và CLI trả mã lỗi nếu còn file thất bại sau cùng. Khi chạy toàn bộ pipeline, các file lỗi vẫn giữ bản đã index trước đó, nếu có.

Các trường `ocr_pages_processed`, `ocr_partial_pages` và `ocr_failed_pages` giúp đối chiếu số trang PDF đã xử lý, thiếu một phần và lỗi. Tài liệu có đầu ra OCR một phần vẫn có thể được xuất bản kèm thông tin chất lượng. `--force` chạy lại tài liệu, dùng lại các trang OCR đã có kết quả đầy đủ và gửi lại các trang thiếu hoặc lỗi. Bản xuất PDF từ policy cũ cần chạy lại chunking trước khi đưa vào chỉ mục.

Khi lập chỉ mục theo manifest, phạm vi đồng bộ được xác định theo thư mục xuất bản. Tài liệu đã bỏ khỏi snapshot sẽ được dọn khỏi chỉ mục của phạm vi đó. Điểm cũ chưa có phạm vi được dọn khi khớp đường dẫn nguồn hoặc có duy nhất một đối chiếu theo hậu tố đường dẫn trong lượt thành công. Điểm cũ có nguồn chưa đối chiếu được hoặc tài liệu xử lý lỗi vẫn được giữ.

`--clean` dựng lại kết quả và chỉ dọn các file xuất bản cũ sau khi toàn bộ lượt chạy thành công. Thư mục `runtime`, dữ liệu benchmark và tệp người dùng được giữ lại.

## Đồng bộ chỉ mục

Khi chạy từng bước, tạo embedding từ manifest chunking. Sau đó chạy lệnh sau từ thư mục gốc của dự án.

```text
python -m agent_banking.vector_db.run_vector_db --input "<embedded_chunks.json>" --snapshot --publication "<thư_mục_xuất_bản_chunking>"
```

## Chạy kiểm thử

Từ thư mục gốc của dự án, chạy `uv run --group dev pytest -q agent_banking/chunking/tests`. Các ca OCR và lập chỉ mục trong suite dùng phản hồi mô phỏng và kho dữ liệu tạm.
