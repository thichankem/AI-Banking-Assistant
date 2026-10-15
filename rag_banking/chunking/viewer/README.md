# 🔍 TRÌNH SOI BOUNDING BOX & PHÂN ĐOẠN CHUNKING (DLA INSPECTOR)

Frontend trực quan hóa dành riêng cho phân hệ **Phase 1: Ingestion & Chunking** của hệ thống **AI Banking Assistant** (Vietcombank).

---

## 🎯 2 MỤC TIÊU CỐT LÕI

1. **Soi Bounding Box trên Tệp Gốc (PDF / DOCX)**:
   * Hiển thị nguyên vẹn tệp gốc thực tế (logo Vietcombank, biểu bảng, con dấu, căn lề).
   * Phủ khung Bounding Box tọa độ (`x_min, y_min, x_max, y_max`) chuẩn xác từng milimet đè lên văn bản gốc để phát hiện lỗi lệch dòng hoặc cắt xén.
2. **Kiểm tra Phân đoạn (Chunking)**:
   * Xem cách hệ thống gom nhóm elements thành từng Chunk theo cấu trúc pháp lý (Chương > Điều > Khoản > Bảng biểu).
   * Bấm vào bất kỳ box nào ở nửa trái sẽ **ngay lập tức định vị và highlight Chunk tương ứng ở nửa phải**!
   * Bấm vào Chunk bên phải sẽ **nhảy đến trang và viền sáng Bounding Box ở nửa trái**!

---

## 🚀 CÁCH KHỞI CHẠY (TRONG THƯ MỤC `viewer/`)

Toàn bộ mã nguồn, dữ liệu và launcher của giao diện đã được đóng gói tập trung hoàn toàn trong thư mục [`viewer/`](file:///d:/Desktop/GitHub/AI%20Banking%20Asistant/rag_banking/chunking/viewer).

### Cách 1: Chạy file Batch hoặc PowerShell (Khuyên dùng)
Tại thư mục `rag_banking/chunking/viewer`:
* Nhấp đúp chuột vào `run_viewer.bat`
* Hoặc chạy trong PowerShell:
```powershell
.\run_viewer.ps1
```
*(Hệ thống tự động phát hiện `uv`, môi trường ảo `.venv` hoặc Python 3.14/3.11 và mở trình duyệt tại `http://127.0.0.1:8501/index.html`)*

### Cách 2: Chạy trực tiếp qua `uv run`
```bash
cd viewer
uv run python serve.py
```

### Cách 3: Mở trực tiếp file HTML (Chạy Offline 100% không cần Python)
Chỉ cần **nhấp đúp chuột vào tệp `index.html`** trong thư mục này để mở ngay trên trình duyệt Google Chrome hoặc Microsoft Edge. Toàn bộ thư viện PDF.js, Mammoth.js và dữ liệu mẫu đã được nhúng sẵn chạy ngầm.

---

## 🖥️ THIẾT KẾ GIAO DIỆN CHIA 2 NỬA (DUAL-PANE SPLIT VIEW)

```
┌────────────────────────────────────────┬────────────────────────────────────────┐
│  📑 NỬA TRÁI: FILE GỐC (PDF / DOCX)    │  🧩 NỬA PHẢI: KẾT QUẢ CHUNKING        │
├────────────────────────────────────────┼────────────────────────────────────────┤
│ • Hiển thị nguyên bản tệp PDF / DOCX   │ • Danh sách Chunk kèm Breadcrumbs      │
│ • Phủ Bounding Box trực tiếp lên trang │ • Tiêu đề mục (Section title)          │
│ • Màu sắc phân cấp (Title, H1, H2,     │ • Số từ, số ký tự, loại chunk          │
│   Body, Table, Header/Footer)          │ • Nội dung văn bản chi tiết            │
│ • Thanh trượt điều chỉnh Độ mờ Box     │ • Tìm kiếm & Lọc theo loại chunk       │
│ • Bật/tắt lớp nền File Gốc             │ • Lọc theo danh sách 108 tài liệu      │
│                                        │                                        │
│  👉 BẤM VÀO BOX TRÁI ➔                 │  👉 Focus & Highlight Chunk bên phải!  │
│  👈 Highlight box trái                 │  👈 BẤM VÀO CHUNK PHẢI ➔               │
└────────────────────────────────────────┴────────────────────────────────────────┘
```

---

## 🎨 BẢNG MÀU PHÂN BẬC BOUNDING BOX (DLA HIERARCHY)

* 🟣 **Title (Tím)**: Tiêu đề lớn của tài liệu (Biểu phí, Điều khoản dịch vụ).
* 🔵 **H1 / Chương / Điều (Xanh dương)**: Phân cấp bậc 1.
* 🔷 **H2 / Mục / Khoản (Xanh lơ)**: Phân cấp bậc 2.
* 🟢 **Body (Xanh lá)**: Đoạn văn bản nội dung thông thường.
* 🟡 **Bảng biểu / Table (Vàng cam)**: Khối ma trận phí, lãi suất, bảng đối soát.
* 🔴 **Header / Footer (Đỏ)**: Tiêu đề đầu/cuối trang cần loại bỏ.

---

## 📂 NÚT CHỌN FILE GỐC & CÁC CÁCH NẠP DỮ LIỆU

1. **Nút "📂 Chọn file gốc (PDF / DOCX)"**: Bấm để chọn trực tiếp bất kỳ tệp PDF hoặc Word từ máy tính để hiển thị tài liệu gốc và tự động dựng bounding boxes.
2. **Kéo & Thả (Drag & Drop)**: Thả trực tiếp file `.pdf`, `.docx` hoặc `.json` vào giữa cửa sổ trình duyệt.
3. **Menu chọn sẵn (Dropdown)**: Xem ngay các mẫu Vietcombank thực tế hoặc xem toàn bộ 108 tài liệu (1,384 chunks).
