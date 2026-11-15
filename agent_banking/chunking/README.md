# Chunking PDF và DOCX

## PDF scan, ảnh và chú thích

- Trình xử lý kiểm tra từng trang PDF trước khi trích xuất. Khi phát hiện trang scan, kể cả ảnh scan nhỏ trên trang thiếu chữ thân bài, nó render **từng trang của cả PDF** và gửi ảnh trang lần lượt để nhận chữ. PDF có lớp chữ đầy đủ dùng chữ và cấu trúc có sẵn trong file.
- **Ảnh minh họa, sơ đồ và biểu đồ đứng riêng không được gửi riêng cho vision hoặc OCR.** DOCX giữ chữ được tác giả viết trong thân bài, ô bảng, hộp văn bản và caption của thân bài. Nhận chữ trên ảnh cả trang PDF scan phục vụ việc đọc nội dung văn bản của trang.
- DOCX bỏ header, footer, footnote và endnote. PDF bỏ phần được nhận diện là header, footer hoặc footnote. Với PDF có lớp chữ, dòng lặp gần lề chỉ được loại khi vị trí và kiểu chữ cũng cho thấy đó là nội dung lề. Bố cục nhiều cột được đọc theo cột trong từng vùng trang.
- Phần chữ trích từ PDF được đối chiếu với lớp chữ có thể dùng để kiểm chứng. Ca không đủ bằng chứng hoặc không nhất quán được ghi lỗi theo file.

## Ranh giới chunk và ngôn ngữ

- Mỗi file được xử lý riêng, theo thứ tự. Các tác vụ trong một file có thể chạy theo lô với tối đa hai worker.
- Heading có tối đa 7 cấp. Heading gốc rõ ràng được giữ, cấp sâu hơn được quy về cấp 7.
- Danh sách Word được đọc như văn bản thường. Nhãn số hoặc dấu đầu dòng gõ trực tiếp được giữ nguyên, còn nhãn Word tự sinh từ cấu hình danh sách được bỏ qua. LLM chấm ranh giới chunk dựa trên nội dung.
- Danh sách PDF được đọc như văn bản thường. Giữ nguyên nhãn số, chữ cái và dấu đầu dòng đã hiển thị trên trang, kể cả khi thứ tự bị ngắt quãng. Với PDF scan, OCR chép các nhãn nhìn thấy và để nguyên mục thiếu nhãn.
- Chunk tiếng Việt nhắm 512 token, ưu tiên tối thiểu 128 token khi nội dung và cấu trúc cho phép, với trần thông thường 1024. Phần ngắn hợp lệ có thể dưới 128. Công thức, tiêu đề bảng hay hàng bảng không thể cắt an toàn được giữ nguyên.
- Pipeline vẫn dùng embedding multilingual-e5-large. Mặc định mỗi request embedding nhận tối đa 512 token trên một đoạn. Chunk dài hơn được biểu diễn bằng vector tổng hợp có trọng số từ các đoạn gửi đi, còn nội dung gốc giữ nguyên trong chunk. Giới hạn dịch vụ có thể đặt bằng `MAX_PASSAGE_TOKENS` khi endpoint đã được xác nhận.
- Nội dung nhận diện được là ngoại ngữ giữ nguyên trong `business_metadata.elements` với kiểu `foreign_language` và vị trí nguồn. Bộ nhận diện kiểm tra cả câu khác ngôn ngữ trên cùng dòng và ưu tiên dấu hiệu tiếng Việt rõ ràng khi thẻ ngôn ngữ Word không khớp. Văn bản Latin ngắn hoặc mơ hồ được giữ trong ngữ cảnh để tránh loại nhầm tiếng Việt không dấu. Phần chỉ có ngoại ngữ được giữ trong đầu ra với vai trò `attachment_only` khi không có nội dung tìm kiếm tương ứng, kể cả ở một phần của bảng nhiều chunk.
- Biểu thức Word hỗ trợ được lưu dưới dạng Markdown có TeX, chỉ số trên/dưới có thể dùng `<sup>` và `<sub>`. Công thức Word chưa chuyển được sang TeX vẫn còn dưới dạng OMML để khôi phục. Bảng lưu lưới và tọa độ ô gộp ngoài biểu diễn văn bản.

## Tên và kết quả theo file

Định danh dùng tên file có đuôi. Trước khi xử lý, hệ thống kiểm tra trùng tên trong các thư mục con, không phân biệt hoa/thường. Các bản cùng định dạng như `Hướng dẫn.pdf` và `Hướng dẫn (1).pdf` cũng được coi là trùng nếu cùng có mặt. Báo lỗi liệt kê từng đường dẫn theo số thứ tự. Đổi tên nguồn có ý nghĩa rồi chạy lại.

Xem `chunking_manifest.json` để tra file kết quả riêng trong `chunks_per_file/` và bằng chứng trong `artifacts/`. Với file xử lý lỗi, pipeline hoàn thành lượt đầu của các file khác rồi thử lại nhóm lỗi tối đa ba lượt, chờ lần lượt 30, 60 và 120 giây. File vẫn lỗi được ghi vào báo cáo. File thành công được xuất bản và CLI trả mã lỗi nếu còn file thất bại sau cùng. Khi chạy toàn bộ pipeline, các file lỗi vẫn giữ bản đã index trước đó, nếu có.

Khi lập chỉ mục theo manifest, phạm vi đồng bộ được xác định theo thư mục xuất bản. Tài liệu đã bỏ khỏi snapshot sẽ được dọn khỏi chỉ mục của phạm vi đó. Bản cũ lập chỉ mục trước khi có thông tin phạm vi sẽ được nhận diện khi nguồn xuất hiện lại trong một lượt thành công.

`--clean` dựng lại kết quả và chỉ dọn các file xuất bản cũ sau khi toàn bộ lượt chạy thành công. Thư mục `runtime`, các file sửa tay và dữ liệu benchmark được giữ lại.

## Sửa nội dung thủ công

Dùng một file JSON cho mỗi tài liệu cần sửa, ví dụ `output/manual_edits/Hướng dẫn.pdf.json`. Có thể chỉnh thư mục chứa JSON bằng `--manual-edits`. Lấy `document_version` từ metadata của chunk và `chunk_id` gốc từ file mà manifest ghi ở `base_output_file`. Cấu hình ví dụ:

```json
{
  "document_version": "hash SHA-256 của tài liệu hiện tại",
  "replacements": {
    "chunk_id_gốc": "Nội dung đã kiểm tra và sửa. Công thức có thể viết là $x^2$."
  }
}
```

Chạy lại chunking để xuất bản bản đã sửa. Ngữ cảnh trích dẫn từ đoạn được sửa được loại khỏi các chunk phụ thuộc. Nếu bản sửa bao gồm một heading, nhãn cũ được bỏ khỏi breadcrumb vì bản thay toàn chunk không chỉ rõ ranh giới heading mới. Lần chạy toàn bộ pipeline tiếp theo sẽ cập nhật embedding và chỉ mục. Bản sửa phải trỏ đúng phiên bản tài liệu và một chunk văn bản hiện có. Để chỉnh ô bảng hoặc bố cục, sửa trong file nguồn rồi chạy lại. Khi file nguồn đổi, kiểm tra lại bản sửa và cập nhật những ID liên quan.

Khi chạy từng bước, tạo embedding từ manifest chunking. Sau đó chạy lệnh sau từ thư mục gốc của dự án.

```text
python -m agent_banking.vector_db.run_vector_db --input "<embedded_chunks.json>" --snapshot --publication "<thư_mục_xuất_bản_chunking>"
```
