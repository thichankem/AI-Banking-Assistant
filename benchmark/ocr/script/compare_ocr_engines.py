#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hệ thống So sánh Định lượng & Đối soát Toàn diện giữa các Engine OCR:
- Baseline: RapidOCR (PP-OCRv4 default model)
- Candidate: Tesseract OCR v5 (vie.traineddata)
- Chuẩn đối sánh: Antigravity Multimodal Vision Ground Truth

Tính toán:
1. CER (Character Error Rate)
2. WER (Word Error Rate)
3. Diacritics Accuracy (%)
4. Numeric Accuracy (%)
5. Độ trễ (Processing Latency)
"""

import sys
import re
import json
import time
from pathlib import Path
from typing import Dict, Any, List

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
VISION_DIR = BENCHMARK_DIR / "vision_groundtruth"
RAPIDOCR_DIR = BENCHMARK_DIR / "rapidocr_output"
TESSERACT_DIR = BENCHMARK_DIR / "tesseract_output"

VIETNAMESE_ACCENTED_CHARS = set(
    "áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ"
    "ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ"
)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Khoảng cách Levenshtein quy hoạch động 2 dòng."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def word_levenshtein_distance(w1: List[str], w2: List[str]) -> int:
    """Word Levenshtein distance."""
    if len(w1) < len(w2):
        return word_levenshtein_distance(w2, w1)
    if len(w2) == 0:
        return len(w1)

    previous_row = list(range(len(w2) + 1))
    for i, token1 in enumerate(w1):
        current_row = [i + 1]
        for j, token2 in enumerate(w2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (token1 != token2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def clean_text_for_eval(text: str) -> str:
    """Loại bỏ metadata header và chuẩn hóa khoảng trắng."""
    lines = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        if s.startswith("# ") or s.startswith("**Nguồn") or s.startswith("**Engine") or s.startswith("**Processing") or s.startswith("- **Thời gian") or s.startswith("- **Số"):
            continue
        if s.startswith("```"):
            continue
        if s == "---":
            continue
        s = s.replace("|", " ").replace("*", "").replace("-", " ")
        lines.append(s)

    clean_text = " ".join(lines)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    return clean_text


def evaluate_diacritics(ref: str, hyp: str) -> float:
    """Đánh giá tỷ lệ bảo toàn ký tự có dấu tiếng Việt."""
    ref_accents = [c for c in ref if c in VIETNAMESE_ACCENTED_CHARS]
    if not ref_accents:
        return 100.0

    hyp_accents = [c for c in hyp if c in VIETNAMESE_ACCENTED_CHARS]
    matched = min(len(hyp_accents), len(ref_accents))
    return round((matched / len(ref_accents)) * 100, 2)


def evaluate_numbers(ref: str, hyp: str) -> float:
    """Đánh giá độ chính xác số liệu."""
    ref_nums = re.findall(r"\d+", ref)
    if not ref_nums:
        return 100.0

    hyp_nums = re.findall(r"\d+", hyp)
    matched = sum(1 for n in ref_nums if n in hyp_nums)
    return round((matched / len(ref_nums)) * 100, 2)


def main():
    print("=" * 80)
    print("   BENCHMARK SO SÁNH TRỰC TIẾP: RAPIDOCR vs TESSERACT V5 (VIE) vs VISION GT")
    print("=" * 80)

    # Đọc metadata rapidocr
    rapid_meta_file = RAPIDOCR_DIR / "_rapidocr_meta.json"
    rapid_meta = {}
    if rapid_meta_file.exists():
        with open(rapid_meta_file, "r", encoding="utf-8") as f:
            rapid_meta = json.load(f)

    vision_files = sorted(list(VISION_DIR.glob("*_vision.md")))
    results = []

    for v_path in vision_files:
        stem = v_path.stem.replace("_vision", "")
        doc_type = "PDF" if stem.startswith("pdf_") else "DOCX"
        
        # Ground truth
        gt_raw = v_path.read_text(encoding="utf-8")
        gt_clean = clean_text_for_eval(gt_raw)
        gt_words = gt_clean.split()

        # 1. RapidOCR Output
        rapid_file = RAPIDOCR_DIR / f"{stem}_ocr.md"
        rapid_clean = ""
        rapid_time = rapid_meta.get(stem, {}).get("processing_time_s", 0.0)
        if rapid_file.exists():
            rapid_clean = clean_text_for_eval(rapid_file.read_text(encoding="utf-8"))

        # Rapid metrics
        rapid_cer = 100.0
        rapid_wer = 100.0
        rapid_diac = 0.0
        rapid_num = 0.0
        if gt_clean and rapid_clean:
            char_dist = levenshtein_distance(gt_clean, rapid_clean)
            rapid_cer = round((char_dist / max(len(gt_clean), 1)) * 100, 2)
            word_dist = word_levenshtein_distance(gt_words, rapid_clean.split())
            rapid_wer = round((word_dist / max(len(gt_words), 1)) * 100, 2)
            rapid_diac = evaluate_diacritics(gt_clean, rapid_clean)
            rapid_num = evaluate_numbers(gt_clean, rapid_clean)

        # 2. Tesseract Output
        tess_file = TESSERACT_DIR / f"{stem}.md"
        tess_clean = ""
        tess_time = 0.0
        if tess_file.exists():
            tess_raw = tess_file.read_text(encoding="utf-8")
            # extract time if present
            m = re.search(r"\*\*Processing Time\*\*:\s*([0-9.]+)s", tess_raw)
            if m:
                tess_time = float(m.group(1))
            tess_clean = clean_text_for_eval(tess_raw)

        # Tesseract metrics
        tess_cer = 100.0
        tess_wer = 100.0
        tess_diac = 0.0
        tess_num = 0.0
        if gt_clean and tess_clean:
            char_dist = levenshtein_distance(gt_clean, tess_clean)
            tess_cer = round((char_dist / max(len(gt_clean), 1)) * 100, 2)
            word_dist = word_levenshtein_distance(gt_words, tess_clean.split())
            tess_wer = round((word_dist / max(len(gt_words), 1)) * 100, 2)
            tess_diac = evaluate_diacritics(gt_clean, tess_clean)
            tess_num = evaluate_numbers(gt_clean, tess_clean)

        res = {
            "sample": stem,
            "doc_type": doc_type,
            "ref_length": len(gt_clean),
            "ref_words": len(gt_words),
            "rapidocr": {
                "cer": rapid_cer,
                "wer": rapid_wer,
                "diacritics_acc": rapid_diac,
                "numeric_acc": rapid_num,
                "time_s": rapid_time,
                "hyp_length": len(rapid_clean),
            },
            "tesseract_vie": {
                "cer": tess_cer,
                "wer": tess_wer,
                "diacritics_acc": tess_diac,
                "numeric_acc": tess_num,
                "time_s": tess_time,
                "hyp_length": len(tess_clean),
            },
            "improvements": {
                "cer_reduction": round(rapid_cer - tess_cer, 2),
                "wer_reduction": round(rapid_wer - tess_wer, 2),
                "diacritics_gain": round(tess_diac - rapid_diac, 2),
                "numeric_gain": round(tess_num - rapid_num, 2),
            }
        }
        results.append(res)
        print(f"[{doc_type}] {stem:<35} | CER: {rapid_cer:>5.1f}% -> {tess_cer:>5.1f}% (Δ {res['improvements']['cer_reduction']:>+5.1f}%) | Dấu: {rapid_diac:>4.1f}% -> {tess_diac:>5.1f}% (Δ {res['improvements']['diacritics_gain']:>+5.1f}%)")

    # Aggregate summaries
    def calc_mean(field_path):
        vals = []
        for r in results:
            keys = field_path.split(".")
            val = r
            for k in keys:
                val = val[k]
            vals.append(val)
        return round(sum(vals) / len(vals), 2)

    def calc_group_mean(field_path, doc_type):
        subset = [r for r in results if r["doc_type"] == doc_type]
        vals = []
        for r in subset:
            keys = field_path.split(".")
            val = r
            for k in keys:
                val = val[k]
            vals.append(val)
        return round(sum(vals) / len(vals), 2)

    summary = {
        "total_samples": len(results),
        "overall": {
            "rapidocr": {
                "cer": calc_mean("rapidocr.cer"),
                "wer": calc_mean("rapidocr.wer"),
                "diacritics_acc": calc_mean("rapidocr.diacritics_acc"),
                "numeric_acc": calc_mean("rapidocr.numeric_acc"),
                "avg_time_s": calc_mean("rapidocr.time_s"),
            },
            "tesseract_vie": {
                "cer": calc_mean("tesseract_vie.cer"),
                "wer": calc_mean("tesseract_vie.wer"),
                "diacritics_acc": calc_mean("tesseract_vie.diacritics_acc"),
                "numeric_acc": calc_mean("tesseract_vie.numeric_acc"),
                "avg_time_s": calc_mean("tesseract_vie.time_s"),
            },
            "overall_improvements": {
                "cer_reduction_pct_pts": round(calc_mean("rapidocr.cer") - calc_mean("tesseract_vie.cer"), 2),
                "wer_reduction_pct_pts": round(calc_mean("rapidocr.wer") - calc_mean("tesseract_vie.wer"), 2),
                "diacritics_gain_pct_pts": round(calc_mean("tesseract_vie.diacritics_acc") - calc_mean("rapidocr.diacritics_acc"), 2),
                "numeric_gain_pct_pts": round(calc_mean("tesseract_vie.numeric_acc") - calc_mean("rapidocr.numeric_acc"), 2),
            }
        },
        "pdf_subset": {
            "rapidocr": {
                "cer": calc_group_mean("rapidocr.cer", "PDF"),
                "wer": calc_group_mean("rapidocr.wer", "PDF"),
                "diacritics_acc": calc_group_mean("rapidocr.diacritics_acc", "PDF"),
                "numeric_acc": calc_group_mean("rapidocr.numeric_acc", "PDF"),
            },
            "tesseract_vie": {
                "cer": calc_group_mean("tesseract_vie.cer", "PDF"),
                "wer": calc_group_mean("tesseract_vie.wer", "PDF"),
                "diacritics_acc": calc_group_mean("tesseract_vie.diacritics_acc", "PDF"),
                "numeric_acc": calc_group_mean("tesseract_vie.numeric_acc", "PDF"),
            }
        },
        "docx_subset": {
            "rapidocr": {
                "cer": calc_group_mean("rapidocr.cer", "DOCX"),
                "wer": calc_group_mean("rapidocr.wer", "DOCX"),
                "diacritics_acc": calc_group_mean("rapidocr.diacritics_acc", "DOCX"),
                "numeric_acc": calc_group_mean("rapidocr.numeric_acc", "DOCX"),
            },
            "tesseract_vie": {
                "cer": calc_group_mean("tesseract_vie.cer", "DOCX"),
                "wer": calc_group_mean("tesseract_vie.wer", "DOCX"),
                "diacritics_acc": calc_group_mean("tesseract_vie.diacritics_acc", "DOCX"),
                "numeric_acc": calc_group_mean("tesseract_vie.numeric_acc", "DOCX"),
            }
        },
        "details": results
    }

    # Save to json
    out_json = BENCHMARK_DIR / "ocr_comparison_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("                         TỔNG HỢP KẾT QUẢ ĐỐI ĐẦU")
    print("=" * 80)
    print(f"Tổng số mẫu thử nghiệm: {len(results)} (10 PDF ngân hàng + 10 DOCX biểu phí/FAQ)")
    print(f"{'Chỉ số':<25} | {'RapidOCR (Baseline)':<20} | {'Tesseract v5 (vie)':<20} | {'Thay đổi (Δ)':<15}")
    print("-" * 85)
    print(f"{'CER (Lỗi ký tự)':<25} | {summary['overall']['rapidocr']['cer']:>19.2f}% | {summary['overall']['tesseract_vie']['cer']:>19.2f}% | {summary['overall']['overall_improvements']['cer_reduction_pct_pts']:>+14.2f}%")
    print(f"{'WER (Lỗi từ)':<25} | {summary['overall']['rapidocr']['wer']:>19.2f}% | {summary['overall']['tesseract_vie']['wer']:>19.2f}% | {summary['overall']['overall_improvements']['wer_reduction_pct_pts']:>+14.2f}%")
    print(f"{'Bảo toàn Dấu Tiếng Việt':<25} | {summary['overall']['rapidocr']['diacritics_acc']:>19.2f}% | {summary['overall']['tesseract_vie']['diacritics_acc']:>19.2f}% | {summary['overall']['overall_improvements']['diacritics_gain_pct_pts']:>+14.2f}%")
    print(f"{'Độ chính xác Số liệu':<25} | {summary['overall']['rapidocr']['numeric_acc']:>19.2f}% | {summary['overall']['tesseract_vie']['numeric_acc']:>19.2f}% | {summary['overall']['overall_improvements']['numeric_gain_pct_pts']:>+14.2f}%")
    print(f"{'Thời gian xử lý / trang':<25} | {summary['overall']['rapidocr']['avg_time_s']:>18.2f}s | {summary['overall']['tesseract_vie']['avg_time_s']:>18.2f}s |")
    print("=" * 80)


if __name__ == "__main__":
    main()
