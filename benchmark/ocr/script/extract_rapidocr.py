"""
Script chạy trích xuất RapidOCR trên 20 ảnh mẫu (10 PDF + 10 DOCX) và lưu kết quả vào benchmark/ocr/rapidocr_output/
"""

import sys
import time
import json
from pathlib import Path
from PIL import Image

# Đảm bảo UTF-8 cho console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BENCHMARK_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_banking.chunking.ocr_extractor import OCRExtractor

IMG_DIR = BENCHMARK_DIR / "rendered_images"
OUT_OCR_DIR = BENCHMARK_DIR / "rapidocr_output"
OUT_OCR_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    # 10 PDF
    "pdf_01_bctc_2024_hop_nhat",
    "pdf_02_bctc_2024_rieng_le",
    "pdf_03_trai_phieu_cbtt",
    "pdf_04_nq_hdqt_vcbs",
    "pdf_05_bieu_phi_the_tdqt",
    "pdf_06_bieu_phi_vay_khcn",
    "pdf_07_bieu_phi_forex_digi",
    "pdf_08_bieu_phi_so_dep",
    "pdf_09_bieu_phi_debit_haha",
    "pdf_10_dkdk_tktt_truc_tuyen",
    # 10 DOCX
    "docx_01_faq_the_ngan_hang",
    "docx_02_le_loyalty",
    "docx_03_dkdk_ekyc",
    "docx_04_faq_digibank_otp",
    "docx_05_dkdk_tai_khoan_khcn",
    "docx_06_faq_kieu_hoi",
    "docx_07_faq_loyalty_doi_qua",
    "docx_08_faq_cho_vay_khcn",
    "docx_09_urbox_evergreen",
    "docx_10_vibe_platinum"
]


def run_rapidocr_extractions():
    print("=" * 70)
    print("  CHẠY TRÍCH XUẤT RAPIDOCR TRÊN 20 ẢNH MẪU (BENCHMARK)")
    print("=" * 70)

    ocr = OCRExtractor()
    if not ocr.is_available():
        print("[Lỗi] RapidOCR không khả dụng!")
        return

    stats = {}

    for idx, sample_name in enumerate(SAMPLES, start=1):
        img_path = IMG_DIR / f"{sample_name}.png"
        out_md_path = OUT_OCR_DIR / f"{sample_name}_ocr.md"

        if not img_path.exists():
            print(f"  - [{idx:02d}/20] Bỏ qua vì không thấy file: {img_path.name}")
            continue

        print(f"  -> [{idx:02d}/20] Đang OCR: {sample_name}...")
        t0 = time.time()
        try:
            pil_img = Image.open(str(img_path))
            text = ocr.extract_text_from_image(pil_img)
            dur = time.time() - t0

            # Ghi ra file markdown
            md_content = f"""# RAPIDOCR TRÍCH XUẤT: `{sample_name}`
- **Thời gian xử lý**: {dur:.3f} giây
- **Số ký tự trích xuất**: {len(text)}
- **Số từ trích xuất**: {len(text.split())}

---

```text
{text}
```
"""
            with open(out_md_path, "w", encoding="utf-8") as f:
                f.write(md_content)

            stats[sample_name] = {
                "chars": len(text),
                "words": len(text.split()),
                "duration_seconds": round(dur, 3),
                "text": text
            }
            print(f"     Xong: {len(text)} ký tự, {dur:.2f}s")
        except Exception as e:
            print(f"     Lỗi: {e}")

    # Ghi metadata
    with open(OUT_OCR_DIR / "_rapidocr_meta.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n[Hoàn tất] Toàn bộ 20 kết quả RapidOCR đã lưu tại: {OUT_OCR_DIR}\n")


if __name__ == "__main__":
    run_rapidocr_extractions()
