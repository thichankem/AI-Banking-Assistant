#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
VIETOCR PIPELINE: DBNET DETECTION + VIETOCR VGG-TRANSFORMER (100% LOCAL)
File: benchmark/ocr/script/run_vietocr.py
=============================================================================
Pipeline kết hợp hiện đại:
1. Text Detection: Dùng DBNet (ONNX) từ RapidOCR để định vị bounding boxes các dòng chữ.
2. Sắp xếp thứ tự đọc (Reading Order): Gom cụm dòng và sắp xếp Top-to-Bottom, Left-to-Right.
3. Text Recognition: Dùng VietOCR (VGG-Transformer) nhận dạng từng dòng tiếng Việt.
4. Xuất kết quả vào: benchmark/ocr/vietocr_output/
=============================================================================
"""

import sys
import os
import time
import json
from pathlib import Path
from PIL import Image
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rapidocr_onnxruntime import RapidOCR
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
IMAGES_DIR = BENCHMARK_DIR / "rendered_images"
OUTPUT_DIR = BENCHMARK_DIR / "vietocr_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sort_boxes_reading_order(boxes: list, y_thresh: float = 12.0) -> list:
    """
    Sắp xếp bounding boxes theo thứ tự đọc tự nhiên (từ trên xuống dưới, từ trái sang phải).
    """
    if not boxes:
        return []

    # Mỗi item: (box, y_min, x_min)
    items = []
    for item in boxes:
        box = item[0]
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        items.append({
            "box": box,
            "xmin": min(xs),
            "xmax": max(xs),
            "ymin": min(ys),
            "ymax": max(ys),
            "orig": item
        })

    # Sắp xếp thô theo ymin
    items.sort(key=lambda it: it["ymin"])

    # Gom theo dòng
    lines = []
    current_line = []
    curr_y = None

    for it in items:
        if curr_y is None:
            curr_y = it["ymin"]
            current_line.append(it)
        else:
            if abs(it["ymin"] - curr_y) <= y_thresh:
                current_line.append(it)
            else:
                # Sắp xếp các cụm trong cùng dòng theo xmin
                current_line.sort(key=lambda x: x["xmin"])
                lines.extend(current_line)
                current_line = [it]
                curr_y = it["ymin"]

    if current_line:
        current_line.sort(key=lambda x: x["xmin"])
        lines.extend(current_line)

    return [it["orig"] for it in lines]


def process_image_with_vietocr(
    img_path: Path,
    rapid_detector: RapidOCR,
    vietocr_predictor: Predictor,
) -> tuple[str, float]:
    """
    Nhận diện toàn trang bằng pipeline DBNet + VietOCR.
    """
    t0 = time.time()
    pil_img = Image.open(img_path).convert("RGB")
    w_img, h_img = pil_img.size

    # 1. Phát hiện vùng chữ qua DBNet
    results, _ = rapid_detector(str(img_path))
    if not results:
        elapsed = time.time() - t0
        return "", elapsed

    # 2. Sắp xếp thứ tự đọc
    sorted_results = sort_boxes_reading_order(results)

    # 3. Cắt từng dòng và nhận diện qua VietOCR
    lines_text = []
    for item in sorted_results:
        box = item[0]
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]

        xmin = max(0, int(min(xs)) - 2)
        ymin = max(0, int(min(ys)) - 2)
        xmax = min(w_img, int(max(xs)) + 2)
        ymax = min(h_img, int(max(ys)) + 2)

        if xmax <= xmin or ymax <= ymin:
            continue

        crop = pil_img.crop((xmin, ymin, xmax, ymax))
        try:
            line_pred = vietocr_predictor.predict(crop)
            if line_pred and line_pred.strip():
                lines_text.append(line_pred.strip())
        except Exception:
            pass

    full_text = "\n".join(lines_text)
    elapsed = time.time() - t0
    return full_text, elapsed


def main():
    print("=" * 80)
    print("   KHỞI CHẠY PIPELINE OCR LOCAL: DBNET DETECTION + VIETOCR VGG-TRANSFORMER")
    print(f"   - Thư mục ảnh đầu vào: {IMAGES_DIR}")
    print(f"   - Thư mục kết quả:     {OUTPUT_DIR}")
    print("=" * 80)

    # 1. Khởi tạo DBNet
    print("[*] Đang nạp mô hình Text Detection (RapidOCR ONNX)...", flush=True)
    rapid = RapidOCR()

    # 2. Khởi tạo VietOCR
    print("[*] Đang nạp mô hình VietOCR VGG-Transformer (CPU)...", flush=True)
    config = Cfg.load_config_from_name("vgg_transformer")
    config["device"] = "cpu"
    config["predictor"]["beamsearch"] = False  # Greedy search để suy luận siêu nhanh trên CPU
    vietocr = Predictor(config)
    print("[+] Cả 2 mô hình đã sẵn sàng!", flush=True)

    image_files = sorted(list(IMAGES_DIR.glob("*.png")))
    print(f"[*] Tìm thấy {len(image_files)} tệp ảnh cần xử lý.\n", flush=True)

    meta_timings = {}

    for idx, img_path in enumerate(image_files, 1):
        stem = img_path.stem
        out_file = OUTPUT_DIR / f"{stem}.md"

        print(f"[{idx:02d}/{len(image_files):02d}] Xử lý: {img_path.name} ...", end=" ", flush=True)
        text, elapsed = process_image_with_vietocr(img_path, rapid, vietocr)

        meta_timings[stem] = {
            "processing_time_s": round(elapsed, 2),
            "char_count": len(text),
        }

        content = (
            f"# OCR Extraction: {stem}\n\n"
            f"**Engine**: Pipeline DBNet + VietOCR (VGG-Transformer)\n"
            f"**Processing Time**: {elapsed:.2f}s\n\n"
            f"---\n\n"
            f"{text}\n"
        )
        out_file.write_text(content, encoding="utf-8")
        print(f"XONG trong {elapsed:.2f}s ({len(text)} ký tự)", flush=True)

    meta_file = OUTPUT_DIR / "_vietocr_meta.json"
    meta_file.write_text(json.dumps(meta_timings, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 80)
    print(f"[+] Hoàn tất xử lý 20 tài liệu! Metadata lưu tại: {meta_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
