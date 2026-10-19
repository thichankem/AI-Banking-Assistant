import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import subprocess
from PIL import Image, ImageOps

def render_mermaid():
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif;
            background: #ffffff;
            margin: 0;
            padding: 30px;
            width: fit-content;
            height: fit-content;
        }
        #container {
            background: #ffffff;
            padding: 24px;
        }
        .mermaid {
            font-size: 15px;
        }
        .cluster rect {
            rx: 8px;
            ry: 8px;
            fill: #F8FAFC !important;
            stroke: #CBD5E1 !important;
            stroke-width: 1.5px !important;
        }
        .cluster-label {
            font-weight: 700 !important;
            font-size: 15px !important;
            fill: #1E293B !important;
        }
    </style>
</head>
<body>
<div id="container">
<pre class="mermaid">
flowchart TD
    %% TẦNG 1: TIẾP NHẬN
    subgraph S1["1. Tiếp nhận & Phân loại Tài liệu"]
        Input["Tài liệu Ngân hàng"] --> Router{"Phân loại Định dạng"}
    end

    %% TẦNG 2: PHÂN TÍCH BỐ CỤC
    subgraph S2["2. Phân tích Bố cục & Trích xuất Cấu trúc"]
        Router -->|"Tệp Word"| WordParser["Nhánh Tài liệu Word<br/>Trích xuất cấu trúc phân cấp và cây tiêu đề"]
        Router -->|"Tệp PDF"| PDFParser["Nhánh Tài liệu PDF<br/>Lọc lề trang và kiểm soát bộ đệm nối trang"]
        
        WordParser --> WordSplit{"Phân loại Khối"}
        PDFParser --> PDFSplit{"Phân loại Dữ liệu"}
    end

    %% TẦNG 3: XỬ LÝ CHUYÊN SÂU BẢNG BIỂU & SCAN
    subgraph S3["3. Xử lý Chuyên sâu Bảng biểu & Tài liệu Scan"]
        WordSplit -->|"Bảng phức tạp"| VisionTable["Mô hình Vision AI<br/>Đọc hiểu bảng có ô gộp và đa tầng tiêu đề"]
        WordSplit -->|"Bảng tiêu chuẩn"| SimpleTable["Trích xuất Ma trận Bảng"]
        
        PDFSplit -->|"Bảng biểu"| AutoFootnote["Mở rộng Vùng Bảng Động<br/>Tự động bao trọn vùng chú thích chân bảng"]
        AutoFootnote --> VisionTable
        AutoFootnote --> SimpleTable

        PDFSplit -->|"Trang ảnh quét"| OCREngine["Bộ xử lý OCR Tesseract & OpenCV<br/>Lọc dấu mộc đỏ bằng dải màu HSV và cân thẳng góc nghiêng"]
        
        VisionTable --> TableMerge["Nhúng Chú thích Nội dòng<br/>Hàn gắn trực tiếp ghi chú vào đúng ô dữ liệu mang ký hiệu neo"]
        SimpleTable --> TableMerge
    end

    %% TẦNG 4: CHUẨN HÓA & PHÂN TÁCH NGỮ NGHĨA
    subgraph S4["4. Chuẩn hóa & Phân tách Ngữ nghĩa"]
        WordSplit -->|"Đoạn văn"| TextStream["Dòng Văn bản Thô"]
        PDFSplit -->|"Văn bản số"| TextStream
        OCREngine --> TextStream
        
        TextStream --> Dehyphen["Hàn gắn Từ ngắt dòng & Khử nhiễu định dạng"]
        Dehyphen --> WTPSplit["Phân tách Câu Ngữ nghĩa<br/>Mô hình wtpsplit nhận diện ranh giới câu tự nhiên theo ngữ cảnh"]
        WTPSplit --> BudgetSplit["Cân đối Độ dài Phân đoạn<br/>Tối ưu 150 đến 400 âm tiết theo dấu câu và triệt tiêu chồng lấn"]
    end

    %% TẦNG 5: ĐÓNG GÓI CHUẨN HÓA
    subgraph S5["5. Đóng gói Dữ liệu Đầu ra Chuẩn hóa"]
        TableMerge --> AddContext["Gắn Tiền tố Ngữ cảnh Phân cấp Tiêu đề"]
        BudgetSplit --> AddContext
        
        AddContext --> OutputContract[/"Bộ Dữ liệu Đoạn Chuẩn hóa 7 Trường<br/>Sẵn sàng trực tiếp cho tầng nhúng Vector và Tìm kiếm kết hợp"/]
    end

    %% STYLING
    classDef default fill:#FFFFFF,stroke:#64748B,stroke-width:1.5px,color:#1E293B;
    classDef primary fill:#EFF6FF,stroke:#3B82F6,stroke-width:2px,color:#1D4ED8;
    classDef decision fill:#FEF3C7,stroke:#D97706,stroke-width:2px,color:#92400E;
    classDef process fill:#F0FDF4,stroke:#10B981,stroke-width:1.5px,color:#065F46;
    classDef ocr fill:#FDF2F8,stroke:#DB2777,stroke-width:1.5px,color:#9D174D;
    classDef finalNode fill:#EEF2FF,stroke:#6366F1,stroke-width:2px,color:#3730A3;

    class Input primary;
    class Router,WordSplit,PDFSplit decision;
    class VisionTable,SimpleTable,TableMerge,WTPSplit,BudgetSplit,Dehyphen,TextStream,WordParser,PDFParser,AutoFootnote process;
    class OCREngine ocr;
    class AddContext,OutputContract finalNode;
</pre>
</div>
<script>
    mermaid.initialize({
        startOnLoad: true,
        theme: 'neutral',
        flowchart: {
            useMaxWidth: false,
            htmlLabels: true,
            curve: 'basis'
        }
    });
</script>
</body>
</html>"""

    base_dir = r"d:\Desktop\GitHub\AI Banking Asistant\docs\flow diagram"
    os.makedirs(base_dir, exist_ok=True)
    temp_html = os.path.join(base_dir, "chunking_architecture_simplified.html")
    temp_png = os.path.join(base_dir, "chunking_architecture_simplified.png")

    with open(temp_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    cmd = [
        chrome_path,
        "--headless",
        "--disable-gpu",
        "--virtual-time-budget=6000",
        "--window-size=2400,2800",
        f"--screenshot={temp_png}",
        temp_html
    ]
    print("Running Chrome to capture screenshot...")
    subprocess.run(cmd, check=True)

    if os.path.exists(temp_png):
        img = Image.open(temp_png).convert("RGB")
        print("Initial image size:", img.size)
        gray = img.convert("L")
        inv = ImageOps.invert(gray)
        mask = inv.point(lambda p: 255 if p > 8 else 0)
        bbox = mask.getbbox()
        if bbox:
            pad = 25
            bbox = (
                max(0, bbox[0] - pad),
                max(0, bbox[1] - pad),
                min(img.size[0], bbox[2] + pad),
                min(img.size[1], bbox[3] + pad)
            )
            cropped = img.crop(bbox)
            cropped.save(temp_png)
            print("Cropped image size:", cropped.size)
        return temp_png
    return None

if __name__ == "__main__":
    render_mermaid()
