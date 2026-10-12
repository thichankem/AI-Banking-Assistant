#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
RUNNER ĐÁNH GIÁ VÀ ĐỐI SOÁT TOÀN DIỆN OCR BENCHMARK (100% LOCAL)
Folder: benchmark/ocr/eval_script/
File: eval_benchmark.py
=============================================================================
Kịch bản quét toàn bộ tập mẫu Ground Truth, chạy các phép đo đa tầng và
xuất bảng so sánh đối đầu giữa các OCR Engine:
1. RapidOCR (ONNX - Baseline)
2. Tesseract OCR v5 (vie.traineddata)
3. Pipeline DBNet + VietOCR (VGG-Transformer)
=============================================================================
"""

import sys
import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Import metrics từ module cùng thư mục
try:
    from ocr_metrics import (
        clean_eval_text,
        evaluate_ocr_pair,
        calculate_cer,
        calculate_wer,
        calculate_ned,
        calculate_diacritic_metrics,
        calculate_gcer,
        calculate_numeric_metrics,
        normalize_vietnamese_text,
    )
except ImportError:
    from benchmark.ocr.eval_script.ocr_metrics import (
        clean_eval_text,
        evaluate_ocr_pair,
        calculate_cer,
        calculate_wer,
        calculate_ned,
        calculate_diacritic_metrics,
        calculate_gcer,
        calculate_numeric_metrics,
        normalize_vietnamese_text,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BENCHMARK_DIR = SCRIPT_DIR.parent
DEFAULT_GT_DIR = DEFAULT_BENCHMARK_DIR / "vision_groundtruth"
DEFAULT_TESSERACT_DIR = DEFAULT_BENCHMARK_DIR / "tesseract_output"
DEFAULT_RAPIDOCR_DIR = DEFAULT_BENCHMARK_DIR / "rapidocr_output"
DEFAULT_VIETOCR_DIR = DEFAULT_BENCHMARK_DIR / "vietocr_output"


def load_text(file_path: Path) -> str:
    """Đọc tệp văn bản với hỗ trợ UTF-8."""
    if not file_path.exists():
        return ""
    try:
        return file_path.read_text(encoding="utf-8")
    except Exception:
        try:
            return file_path.read_text(encoding="utf-8-sig")
        except Exception:
            return file_path.read_text(encoding="latin-1", errors="ignore")


def extract_latency_from_text(text: str) -> float:
    """Trích xuất thời gian xử lý nếu được ghi trong header markdown."""
    m = re.search(r"\*\*Processing Time\*\*:\s*([0-9.]+)s", text)
    if m:
        return float(m.group(1))
    m = re.search(r"Thời gian xử lý:\s*([0-9.]+)s", text)
    if m:
        return float(m.group(1))
    return 0.0


def evaluate_engine_directory(
    gt_dir: Path,
    pred_dir: Path,
    engine_name: str,
    file_suffix: str = ".md",
    pred_suffix: str = ".md",
    meta_json: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Đánh giá toàn bộ một thư mục kết quả của 1 engine so với thư mục Ground Truth.
    """
    gt_files = sorted(list(gt_dir.glob(f"*{file_suffix}")))
    if not gt_files:
        gt_files = sorted(list(gt_dir.glob("*.txt")))

    timing_meta = {}
    if meta_json and meta_json.exists():
        try:
            with open(meta_json, "r", encoding="utf-8") as f:
                timing_meta = json.load(f)
        except Exception:
            pass

    samples = []
    total_ref_chars = 0
    total_ref_words = 0
    total_ref_diacritics = 0
    total_ref_numbers = 0

    total_pred_chars = 0
    total_matched_numbers = 0
    total_matched_diacritics = 0

    macro_cer_list = []
    macro_wer_list = []
    macro_ned_list = []
    macro_gcer_list = []
    macro_diac_acc_list = []
    macro_diac_cer_list = []
    macro_num_acc_list = []
    latencies = []

    print(f"[*] Bắt đầu đánh giá Engine: {engine_name} ({len(gt_files)} mẫu)...", flush=True)

    for idx, gt_path in enumerate(gt_files, 1):
        stem = gt_path.stem
        base_stem = stem.replace("_vision", "").replace("_gt", "")

        gt_raw = load_text(gt_path)
        gt_clean = clean_eval_text(gt_raw)

        pred_candidates = [
            pred_dir / f"{base_stem}{pred_suffix}",
            pred_dir / f"{base_stem}_ocr{pred_suffix}",
            pred_dir / f"{base_stem}.md",
            pred_dir / f"{base_stem}.txt",
        ]
        pred_path = None
        for cand in pred_candidates:
            if cand.exists():
                pred_path = cand
                break

        pred_raw = load_text(pred_path) if pred_path else ""
        pred_clean = clean_eval_text(pred_raw)

        elapsed = 0.0
        if pred_raw:
            elapsed = extract_latency_from_text(pred_raw)
        if elapsed == 0.0 and base_stem in timing_meta:
            elapsed = timing_meta[base_stem].get("processing_time_s", 0.0)

        t0 = time.time()
        res = evaluate_ocr_pair(gt_clean, pred_clean, clean_markdown=False)
        calc_time = time.time() - t0

        res["sample_id"] = base_stem
        res["latency_s"] = elapsed
        latencies.append(elapsed)

        ref_chars = res["ref_char_count"]
        ref_words = res["ref_word_count"]
        ref_diacs = res["details"]["diacritics"]["ref_diacritics_count"]
        ref_nums = res["details"]["numerics"]["ref_numbers_count"]

        total_ref_chars += ref_chars
        total_ref_words += ref_words
        total_ref_diacritics += ref_diacs
        total_ref_numbers += ref_nums

        total_pred_chars += res["hyp_char_count"]
        total_matched_numbers += res["details"]["numerics"]["matched_numbers_count"]
        matched_diacs = min(ref_diacs, res["details"]["diacritics"]["hyp_diacritics_count"])
        total_matched_diacritics += matched_diacs

        macro_cer_list.append(res["cer"])
        macro_wer_list.append(res["wer"])
        macro_ned_list.append(res["ned"])
        macro_gcer_list.append(res["gcer"])
        macro_diac_acc_list.append(res["diacritic_accuracy"])
        macro_diac_cer_list.append(res["diacritic_cer"])
        macro_num_acc_list.append(res["numeric_accuracy"])

        samples.append(res)
        print(f"    [{idx:02d}/{len(gt_files):02d}] {base_stem:<32} -> CER: {res['cer']:5.1f}% | Dấu: {res['diacritic_accuracy']:5.1f}% | Số: {res['numeric_accuracy']:5.1f}% ({calc_time:.2f}s)", flush=True)

    n_samples = len(samples)
    if n_samples == 0:
        return {"engine_name": engine_name, "sample_count": 0}

    micro_diac_acc = (
        round((total_matched_diacritics / total_ref_diacritics) * 100.0, 2)
        if total_ref_diacritics > 0
        else 100.0
    )
    micro_num_acc = (
        round((total_matched_numbers / total_ref_numbers) * 100.0, 2)
        if total_ref_numbers > 0
        else 100.0
    )

    summary = {
        "engine_name": engine_name,
        "sample_count": n_samples,
        "total_ref_chars": total_ref_chars,
        "total_ref_words": total_ref_words,
        "total_ref_diacritics": total_ref_diacritics,
        "total_ref_numbers": total_ref_numbers,
        "averages": {
            "macro_cer": round(sum(macro_cer_list) / n_samples, 2),
            "macro_wer": round(sum(macro_wer_list) / n_samples, 2),
            "macro_ned": round(sum(macro_ned_list) / n_samples, 4),
            "macro_gcer": round(sum(macro_gcer_list) / n_samples, 2),
            "macro_diacritic_accuracy": round(sum(macro_diac_acc_list) / n_samples, 2),
            "macro_diacritic_cer": round(sum(macro_diac_cer_list) / n_samples, 2),
            "micro_diacritic_accuracy": micro_diac_acc,
            "macro_numeric_accuracy": round(sum(macro_num_acc_list) / n_samples, 2),
            "micro_numeric_accuracy": micro_num_acc,
            "avg_latency_s": round(sum(latencies) / n_samples, 2) if latencies else 0.0,
        },
        "samples": samples,
    }

    return summary


def print_multi_comparison_table(engine_results: List[Dict[str, Any]]):
    """In bảng so sánh đa engine đối đầu định lượng trên terminal."""
    if not engine_results:
        return

    print("\n" + "=" * 105, flush=True)
    print("       BẢNG XẾP HẠNG ĐỐI ĐẦU ĐỊNH LƯỢNG CÁC ENGINE OCR LOCAL CHO NGÂN HÀNG", flush=True)
    print("=" * 105, flush=True)

    header = f"{'Chỉ số Đánh Giá (Metric)':<38}"
    for eng in engine_results:
        header += f" | {eng['engine_name']:<18}"
    print(header, flush=True)
    print("-" * 105, flush=True)

    metrics_def = [
        ("Bảo Toàn Dấu Tiếng Việt (%)", "macro_diacritic_accuracy", "{:.2f}%"),
        ("Tỷ lệ Lỗi Dấu (Diacritic-CER %)", "macro_diacritic_cer", "{:.2f}%"),
        ("Tỷ lệ Lỗi Ký tự (Macro-CER %)", "macro_cer", "{:.2f}%"),
        ("Tỷ lệ Lỗi Từ (Macro-WER %)", "macro_wer", "{:.2f}%"),
        ("Độ Tương Đồng Chuẩn Hóa (NED)", "macro_ned", "{:.4f}"),
        ("Lỗi Cụm Ngữ Âm (GCER %)", "macro_gcer", "{:.2f}%"),
        ("Độ Chính Xác Số Liệu (%)", "macro_numeric_accuracy", "{:.2f}%"),
        ("Thời Gian Xử Lý / Trang (s)", "avg_latency_s", "{:.2f}s"),
    ]

    for label, key, fmt in metrics_def:
        row_str = f"{label:<38}"
        for eng in engine_results:
            val = eng["averages"].get(key, 0.0)
            val_str = fmt.format(val)
            row_str += f" | {val_str:<18}"
        print(row_str, flush=True)

    print("=" * 105, flush=True)


def generate_multi_markdown_report(engine_results: List[Dict[str, Any]], output_path: Path):
    """Xuất báo cáo Markdown tổng hợp chi tiết nhiều engine."""
    if not engine_results:
        return

    n_samples = engine_results[0]["sample_count"]
    md = f"""# BÁO CÁO ĐỐI SOÁT ĐỊNH LƯỢNG OCR LOCAL TOÀN DIỆN (OCR LEADERBOARD)

- **Ngày thực hiện**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Số lượng tài liệu khảo sát**: {n_samples} mẫu đại diện chính thức của Vietcombank
- **Môi trường**: 100% Offline Local (CPU / ONNX / Tesseract / VietOCR)

---

## 1. Bảng Xếp Hạng Hiệu Năng Tổng Thể (Overall Leaderboard)

| Chỉ số Đánh Giá (Metric) | """ + " | ".join([f"**{eng['engine_name']}**" for eng in engine_results]) + " | Ý Nghĩa Thực Tế |\n"
    md += "|---|" + "|".join(["---" for _ in engine_results]) + "|---|\n"

    metrics_def = [
        ("Bảo Toàn Dấu Tiếng Việt", "macro_diacritic_accuracy", "{:.2f}%", "Càng cao càng bảo toàn đúng ngữ nghĩa"),
        ("Tỷ lệ Lỗi Dấu (Diacritic-CER)", "macro_diacritic_cer", "{:.2f}%", "Đo lỗi riêng trên thanh điệu tiếng Việt"),
        ("Tỷ lệ Lỗi Ký Tự (Macro-CER)", "macro_cer", "{:.2f}%", "Chỉ số vàng đo độ chính xác toàn trang"),
        ("Tỷ lệ Lỗi Cấp Từ (Macro-WER)", "macro_wer", "{:.2f}%", "Đo lỗi ranh giới từ và dính từ"),
        ("Độ Tương Đồng Chuẩn Hóa (NED)", "macro_ned", "{:.4f}", "Chuẩn hóa về [0, 1], càng gần 1 càng tốt"),
        ("Lỗi Cụm Ngữ Âm (GCER)", "macro_gcer", "{:.2f}%", "Đo lỗi trên cụm ký tự gốc + dấu ghép"),
        ("Độ Chính Xác Số Liệu Tài Chính", "macro_numeric_accuracy", "{:.2f}%", "Bảo toàn số tiền, ngày tháng, tỷ lệ %"),
        ("Thời Gian Xử Lý / Trang", "avg_latency_s", "{:.2f}s", "Tốc độ xử lý thực tế trên CPU"),
    ]

    for label, key, fmt, meaning in metrics_def:
        row_vals = [fmt.format(eng["averages"].get(key, 0.0)) for eng in engine_results]
        md += f"| **{label}** | " + " | ".join(row_vals) + f" | {meaning} |\n"

    md += """
---

## 2. Bảng Chi Tiết CER Từng Tài Liệu Khảo Sát

| STT | Mã Tài Liệu | """ + " | ".join([f"CER {eng['engine_name']}" for eng in engine_results]) + " |\n"
    md += "|---|---|" + "|".join(["---" for _ in engine_results]) + "|\n"

    for idx in range(n_samples):
        stem = engine_results[0]["samples"][idx]["sample_id"]
        row = f"| {idx+1:02d} | `{stem}` | "
        vals = []
        for eng in engine_results:
            cer_val = eng["samples"][idx]["cer"]
            vals.append(f"{cer_val:.1f}%")
        row += " | ".join(vals) + " |\n"
        md += row

    output_path.write_text(md, encoding="utf-8")
    print(f"\n[+] Đã lưu báo cáo Markdown tại: {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Đánh giá và so sánh định lượng kết quả OCR cục bộ (Local OCR Benchmark)"
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default=str(DEFAULT_GT_DIR),
        help="Thư mục chứa văn bản Ground Truth",
    )
    parser.add_argument(
        "--tess_dir",
        type=str,
        default=str(DEFAULT_TESSERACT_DIR),
        help="Thư mục chứa kết quả Tesseract OCR",
    )
    parser.add_argument(
        "--rapid_dir",
        type=str,
        default=str(DEFAULT_RAPIDOCR_DIR),
        help="Thư mục chứa kết quả RapidOCR",
    )
    parser.add_argument(
        "--viet_dir",
        type=str,
        default=str(DEFAULT_VIETOCR_DIR),
        help="Thư mục chứa kết quả VietOCR",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default=str(SCRIPT_DIR / "eval_results.json"),
        help="Đường dẫn lưu kết quả JSON",
    )
    parser.add_argument(
        "--out_md",
        type=str,
        default=str(SCRIPT_DIR / "EVAL_SUMMARY_REPORT.md"),
        help="Đường dẫn lưu báo cáo Markdown",
    )
    args = parser.parse_args()

    gt_path = Path(args.gt_dir)
    tess_path = Path(args.tess_dir)
    rapid_path = Path(args.rapid_dir)
    viet_path = Path(args.viet_dir)

    print("=" * 90, flush=True)
    print("   HỆ THỐNG ĐÁNH GIÁ MA TRẬN ĐỘ ĐO OCR LOCAL CHO NGÂN HÀNG (TRIPLE-ENGINE LEADERBOARD)", flush=True)
    print(f"   - Thư mục Script: {SCRIPT_DIR}", flush=True)
    print(f"   - Ground Truth:   {gt_path}", flush=True)
    print(f"   - Candidate 1:    {rapid_path} (RapidOCR ONNX)", flush=True)
    print(f"   - Candidate 2:    {tess_path} (Tesseract v5 vie)", flush=True)
    print(f"   - Candidate 3:    {viet_path} (DBNet + VietOCR Transformer)", flush=True)
    print("=" * 90, flush=True)

    if not gt_path.exists():
        print(f"[-] Lỗi: Không tìm thấy thư mục Ground Truth tại {gt_path}", flush=True)
        sys.exit(1)

    engines = []

    # 1. RapidOCR
    if rapid_path.exists():
        rapid_meta = rapid_path / "_rapidocr_meta.json"
        res_rapid = evaluate_engine_directory(
            gt_dir=gt_path,
            pred_dir=rapid_path,
            engine_name="RapidOCR (ONNX)",
            pred_suffix="_ocr.md",
            meta_json=rapid_meta,
        )
        engines.append(res_rapid)

    # 2. Tesseract OCR
    if tess_path.exists():
        res_tess = evaluate_engine_directory(
            gt_dir=gt_path,
            pred_dir=tess_path,
            engine_name="Tesseract v5 (vie)",
            pred_suffix=".md",
        )
        engines.append(res_tess)

    # 3. VietOCR
    if viet_path.exists() and list(viet_path.glob("*.md")):
        viet_meta = viet_path / "_vietocr_meta.json"
        res_viet = evaluate_engine_directory(
            gt_dir=gt_path,
            pred_dir=viet_path,
            engine_name="VietOCR (Transformer)",
            pred_suffix=".md",
            meta_json=viet_meta,
        )
        engines.append(res_viet)

    # In kết quả đối đầu đa engine
    print_multi_comparison_table(engines)

    # Lưu JSON
    final_output = {
        "benchmark_timestamp": time.time(),
        "ground_truth_dir": str(gt_path),
        "engines": {eng["engine_name"]: eng for eng in engines},
    }
    out_json_path = Path(args.out_json)
    out_json_path.write_text(json.dumps(final_output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[+] Đã lưu kết quả JSON tại: {out_json_path}", flush=True)

    # Lưu Markdown
    out_md_path = Path(args.out_md)
    generate_multi_markdown_report(engines, out_md_path)


if __name__ == "__main__":
    main()
