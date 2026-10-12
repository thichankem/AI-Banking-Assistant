#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run Tesseract OCR v5 with Vietnamese language pack (vie.traineddata) on the 20 rendered benchmark images.
Saves extracted text to benchmark/ocr/tesseract_output/
"""

import os
import subprocess
import time
from pathlib import Path

TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
IMAGES_DIR = BENCHMARK_DIR / "rendered_images"
OUTPUT_DIR = BENCHMARK_DIR / "tesseract_output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image_files = sorted([f for f in IMAGES_DIR.glob("*.png") if f.is_file()])

print(f"[*] Found {len(image_files)} benchmark images in {IMAGES_DIR}")

timings = {}

for idx, img_path in enumerate(image_files, 1):
    stem = img_path.stem
    out_file = OUTPUT_DIR / f"{stem}.md"
    print(f"[{idx:02d}/{len(image_files):02d}] Processing {img_path.name} with Tesseract OCR (vie)...", end=" ", flush=True)
    
    t0 = time.time()
    try:
        # Run tesseract with Vietnamese language and standard page segmentation
        # psm 1: Automatic page segmentation with OSD (good for tables and multi-column)
        # psm 3: Fully automatic page segmentation, but no OSD (default)
        cmd = [
            TESSERACT_EXE,
            str(img_path.resolve()),
            "stdout",
            "-l", "vie",
            "--psm", "3"
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        elapsed = time.time() - t0
        timings[stem] = elapsed
        
        extracted_text = result.stdout.strip()
        
        # Save markdown
        content = f"# OCR Extraction: {stem}\n\n**Engine**: Tesseract OCR v5 (vie.traineddata)\n**Processing Time**: {elapsed:.2f}s\n\n---\n\n{extracted_text}\n"
        out_file.write_text(content, encoding="utf-8")
        print(f"DONE in {elapsed:.2f}s ({len(extracted_text)} chars)")
        
    except Exception as e:
        print(f"FAILED: {e}")

print("\n[+] Tesseract OCR batch run completed!")
