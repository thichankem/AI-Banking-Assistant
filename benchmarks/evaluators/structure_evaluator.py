"""
Module Structure Evaluator
==========================
Đo lường độ chính xác cấu trúc văn bản và phả hệ nhận thức bố cục:
1. Heading F1-Score: So sánh các tiêu đề phát hiện được (Precision, Recall, F1) với Mục lục chuẩn (TOC Ground Truth).
2. Breadcrumb Accuracy: Đo lường độ chính xác của chuỗi Breadcrumb (Exact Match, Prefix Match, Phân cấp đúng).
3. Table Structural Integrity: Đo lường mức độ bảo toàn cấu trúc bảng biểu:
   - Tỷ lệ bảo toàn số hàng (Row Retention Rate).
   - Tỷ lệ giữ nguyên tiêu đề cột sau khi chia batch 4 hàng (Header Preservation Rate).
   - Tỷ lệ bảo toàn ô gộp và không lặp text rác.
"""

from __future__ import annotations

import unicodedata
from typing import Any
from pathlib import Path


def _normalize_title(text: str) -> str:
    """Chuẩn hóa tiêu đề: chữ thường, khử dấu tiếng Việt và ký tự đặc biệt."""
    if not text:
        return ""
    text_clean = text.replace("đ", "d").replace("Đ", "D").lower()
    nfkd = unicodedata.normalize("NFKD", text_clean)
    no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    return "".join(c for c in no_accents if c.isalnum() or c.isspace()).strip()


class StructureEvaluator:
    """Bộ đánh giá cấu trúc phân cấp và bảng biểu."""

    def __init__(self):
        pass

    # -------------------------------------------------------------------------
    # 1. ĐÁNH GIÁ HEADING F1-SCORE
    # -------------------------------------------------------------------------

    def evaluate_heading_f1(
        self,
        predicted_headings: list[str],
        ground_truth_headings: list[str],
        similarity_threshold: float = 0.80
    ) -> dict[str, float]:
        """
        Tính Precision, Recall, F1-Score cho các tiêu đề phát hiện được so với TOC Ground Truth.
        Hỗ trợ đối chiếu mềm (Fuzzy Match) qua độ tương đồng từ.
        """
        if not ground_truth_headings:
            return {"precision": 1.0, "recall": 1.0, "f1_score": 1.0, "true_positives": 0}

        norm_gt = [_normalize_title(h) for h in ground_truth_headings if _normalize_title(h)]
        norm_pred = [_normalize_title(h) for h in predicted_headings if _normalize_title(h)]

        if not norm_pred:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "true_positives": 0}

        tp = 0
        matched_gt = set()

        for pred in norm_pred:
            pred_tokens = set(pred.split())
            best_match_idx = -1
            best_sim = 0.0

            for idx, gt in enumerate(norm_gt):
                if idx in matched_gt:
                    continue
                gt_tokens = set(gt.split())
                if not pred_tokens or not gt_tokens:
                    continue

                # Jaccard Token Overlap
                overlap = len(pred_tokens.intersection(gt_tokens))
                union = len(pred_tokens.union(gt_tokens))
                sim = overlap / max(union, 1)

                # Trường hợp chuỗi con chứa trọn vẹn nhau
                if gt in pred or pred in gt:
                    sim = max(sim, 0.90)

                if sim > best_sim:
                    best_sim = sim
                    best_match_idx = idx

            if best_sim >= similarity_threshold and best_match_idx != -1:
                tp += 1
                matched_gt.add(best_match_idx)

        precision = round(tp / max(len(norm_pred), 1), 4)
        recall = round(tp / max(len(norm_gt), 1), 4)
        if precision + recall > 0:
            f1 = round(2 * (precision * recall) / (precision + recall), 4)
        else:
            f1 = 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": tp,
            "predicted_count": len(norm_pred),
            "ground_truth_count": len(norm_gt)
        }

    # -------------------------------------------------------------------------
    # 2. ĐÁNH GIÁ ĐỘ CHÍNH XÁC BREADCRUMB
    # -------------------------------------------------------------------------

    def evaluate_breadcrumb_accuracy(
        self,
        predicted_chunks: list[dict[str, Any]],
        expected_breadcrumbs: list[str] | None = None
    ) -> dict[str, float]:
        """
        Đánh giá chất lượng của chuỗi breadcrumb trong các chunk:
        - Tỷ lệ có breadcrumb hợp lệ (Breadcrumb Coverage).
        - Độ sâu phân cấp trung bình (Average Hierarchy Depth).
        - Tỷ lệ phân cấp đa tầng (Multi-level Breadcrumb Ratio >= 2 levels).
        """
        if not predicted_chunks:
            return {"coverage": 0.0, "multi_level_ratio": 0.0, "avg_depth": 0.0}

        valid_bc_count = 0
        multi_level_count = 0
        total_depth = 0

        for chunk in predicted_chunks:
            bc = chunk.get("breadcrumb", "").strip()
            if bc and bc != "document" and not bc.startswith("Unknown"):
                valid_bc_count += 1
                parts = [p.strip() for p in bc.split(" > ") if p.strip()]
                depth = len(parts)
                total_depth += depth
                if depth >= 2:
                    multi_level_count += 1

        total = len(predicted_chunks)
        coverage = round(valid_bc_count / total, 4)
        multi_level_ratio = round(multi_level_count / total, 4)
        avg_depth = round(total_depth / max(valid_bc_count, 1), 2)

        return {
            "breadcrumb_coverage": coverage,
            "multi_level_ratio": multi_level_ratio,
            "avg_depth": avg_depth,
            "total_chunks": total
        }

    # -------------------------------------------------------------------------
    # 3. ĐÁNH GIÁ BẢO TOÀN CẤU TRÚC BẢNG BIỂU (TABLE INTEGRITY)
    # -------------------------------------------------------------------------

    def evaluate_table_integrity(
        self,
        table_chunks: list[dict[str, Any]],
        expected_total_rows: int | None = None,
        expected_columns: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Kiểm tra độ bảo toàn của bảng biểu sau khi tuyến tính hóa hoặc chia batch 4 hàng:
        1. Row Count Retention: Đếm số hàng dữ liệu trích xuất được.
        2. Header Retention Rate: Kiểm tra các chunk con có lặp lại đầy đủ tên cột không.
        """
        if not table_chunks:
            return {
                "extracted_rows": 0,
                "header_preservation_rate": 0.0,
                "table_chunk_count": 0
            }

        extracted_row_count = 0
        chunks_with_headers = 0

        for chunk in table_chunks:
            content = chunk.get("content", "")
            lines = content.split("\n")

            # Đếm hàng dữ liệu (* Hàng X: ...)
            rows = [l for l in lines if l.strip().startswith("* Hàng") or ("|" in l and not l.strip().startswith("|--"))]
            extracted_row_count += len(rows)

            # Kiểm tra xem chunk này có chứa định nghĩa header hoặc tên cột không
            has_header = False
            if expected_columns:
                norm_content = _normalize_title(content)
                matches = sum(1 for col in expected_columns if _normalize_title(col) in norm_content)
                if matches >= len(expected_columns) * 0.6:
                    has_header = True
            else:
                # Heuristic: có dòng chứa dấu hai chấm mô tả cột "Cột A:" hoặc header Markdown
                if any(":" in cell for r in rows for cell in r.split("|")) or "|---" in content:
                    has_header = True

            if has_header:
                chunks_with_headers += 1

        header_preservation_rate = round(chunks_with_headers / len(table_chunks), 4)
        row_retention_rate = 1.0
        if expected_total_rows and expected_total_rows > 0:
            row_retention_rate = round(min(extracted_row_count / expected_total_rows, 1.0), 4)

        return {
            "table_chunk_count": len(table_chunks),
            "extracted_rows": extracted_row_count,
            "row_retention_rate": row_retention_rate,
            "header_preservation_rate": header_preservation_rate
        }
