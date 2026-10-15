"""
Module OCR Evaluator
====================
Đo lường chất lượng OCR trên tài liệu scan ngân hàng:
1. CER (Character Error Rate): Tỷ lệ sai số ký tự (Levenshtein Distance trên chuỗi ký tự).
2. WER (Word Error Rate): Tỷ lệ sai số từ (Levenshtein Distance trên chuỗi từ).
3. Đánh giá tác động của bước Lọc mộc đỏ ngân hàng HSV (Red-Stamp Filtered vs Raw Scan).
4. So sánh hiệu năng giữa các cấu hình hoặc OCR Engine.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Any, Callable


def _levenshtein_distance(seq1: list[Any] | str, seq2: list[Any] | str) -> int:
    """Tính khoảng cách Levenshtein giữa 2 chuỗi/danh sách bằng quy hoạch động 2 hàng O(min(m, n))."""
    m, n = len(seq1), len(seq2)
    if m < n:
        seq1, seq2 = seq2, seq1
        m, n = n, m

    current_row = list(range(n + 1))
    for i, c1 in enumerate(seq1):
        previous_row = current_row
        current_row = [i + 1] * (n + 1)
        for j, c2 in enumerate(seq2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row[j + 1] = min(insertions, deletions, substitutions)

    return current_row[n]


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Tính Character Error Rate (CER).
    CER = Levenshtein_Distance(ref_chars, hyp_chars) / len(ref_chars)
    """
    ref_chars = list(reference.strip())
    hyp_chars = list(hypothesis.strip())
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0

    dist = _levenshtein_distance(ref_chars, hyp_chars)
    return round(min(dist / len(ref_chars), 1.0), 4)


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Tính Word Error Rate (WER).
    WER = Levenshtein_Distance(ref_words, hyp_words) / len(ref_words)
    """
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    dist = _levenshtein_distance(ref_words, hyp_words)
    return round(min(dist / len(ref_words), 1.0), 4)


class OCREvaluator:
    """Bộ đo lường chất lượng OCR độc lập."""

    def __init__(self):
        pass

    def evaluate_text_pair(self, reference: str, hypothesis: str) -> dict[str, float]:
        """Đo lường CER và WER cho một cặp văn bản chuẩn và OCR."""
        cer = compute_cer(reference, hypothesis)
        wer = compute_wer(reference, hypothesis)
        return {
            "cer": cer,
            "wer": wer,
            "char_accuracy": round(1.0 - cer, 4),
            "word_accuracy": round(1.0 - wer, 4)
        }

    def evaluate_batch(self, references: list[str], hypotheses: list[str]) -> dict[str, float]:
        """Đo lường trung bình trên tập mẫu."""
        if not references or not hypotheses or len(references) != len(hypotheses):
            return {"mean_cer": 0.0, "mean_wer": 0.0, "sample_count": 0}

        cers, wers = [], []
        for ref, hyp in zip(references, hypotheses):
            cers.append(compute_cer(ref, hyp))
            wers.append(compute_wer(ref, hyp))

        return {
            "mean_cer": round(float(np.mean(cers)), 4),
            "mean_wer": round(float(np.mean(wers)), 4),
            "median_cer": round(float(np.median(cers)), 4),
            "median_wer": round(float(np.median(wers)), 4),
            "sample_count": len(references)
        }

    def evaluate_red_stamp_effect(
        self,
        image_input: str | Path | np.ndarray,
        ocr_extractor: Any,
        ground_truth: str
    ) -> dict[str, Any]:
        """
        So sánh kết quả OCR khi BẬT và TẮT bước lọc con dấu mộc đỏ HSV.
        """
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
        else:
            img = image_input

        if img is None:
            return {"error": "Không đọc được ảnh đầu vào"}

        # 1. OCR KHÔNG lọc mộc đỏ (Raw)
        raw_text, raw_conf = ocr_extractor.extract_text_and_confidence(img)
        # Tạm tắt filter trong ocr_extractor hoặc trích xuất thủ công
        raw_metrics = self.evaluate_text_pair(ground_truth, raw_text)

        # 2. OCR CÓ lọc mộc đỏ HSV
        cleaned_img = ocr_extractor.filter_red_stamp(img)
        filtered_text, filtered_conf = ocr_extractor.extract_text_and_confidence(cleaned_img)
        filtered_metrics = self.evaluate_text_pair(ground_truth, filtered_text)

        cer_improvement = round(raw_metrics["cer"] - filtered_metrics["cer"], 4)
        wer_improvement = round(raw_metrics["wer"] - filtered_metrics["wer"], 4)

        return {
            "without_stamp_filter": {
                "cer": raw_metrics["cer"],
                "wer": raw_metrics["wer"],
                "confidence": raw_conf
            },
            "with_stamp_filter": {
                "cer": filtered_metrics["cer"],
                "wer": filtered_metrics["wer"],
                "confidence": filtered_conf
            },
            "improvement": {
                "cer_delta": cer_improvement,
                "wer_delta": wer_improvement,
                "is_better": filtered_metrics["cer"] < raw_metrics["cer"]
            }
        }
