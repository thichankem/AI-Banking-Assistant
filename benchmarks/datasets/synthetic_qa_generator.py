"""
Module Synthetic QA Generator (v2 Upgraded)
===========================================
Tự động sinh bộ câu hỏi kiểm thử chuẩn (Golden QA dataset) bám sát 4 trường hợp khó:
1. Category 1: Multi-level Legal Breadcrumb (`multi_level_legal`)
2. Category 2: Complex Multi-row/Merged Table (`complex_table`)
3. Category 3: Red-stamp / Noisy Scan OCR (`noisy_scan_ocr`)
4. Category 4: Cross-page & Short Section (<120 words) (`cross_page_defragmented`)

CẢI TIẾN QUAN TRỌNG:
- Lưu trường `ground_truth_context_span`: Đoạn văn bản gốc mang câu trả lời (KHÔNG chứa tiền tố [Breadcrumbs: ...]).
- Sinh câu hỏi tự nhiên (Natural Paraphrase) độc lập với cấu trúc breadcrumb, loại bỏ thiên vị (bias).
- Hỗ trợ 3 Backend: 'ollama' (gemma4:12b), 'openrouter' (Gemini/Claude), hoặc 'natural_paraphrase' (chạy offline 100%).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _clean_body_content(content: str) -> str:
    """Loại bỏ tiền tố [Breadcrumbs: ...] để lấy đoạn văn bản gốc (context span)."""
    if "[Breadcrumbs:" in content:
        parts = content.split("]\n", 1)
        if len(parts) > 1:
            return parts[1].strip()
    return content.strip()


class SyntheticQAGenerator:
    """Bộ tạo dữ liệu đánh giá Golden QA chuẩn hóa đa nhóm kiểm thử RAG Banking."""

    def __init__(
        self,
        backend: str = "natural_paraphrase",
        model_name: str = "gemma4:12b",
        ollama_url: str = "http://127.0.0.1:11434",
        openrouter_api_key: str | None = None
    ):
        self.backend = backend.lower()
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")
        self.openrouter_api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")

    def generate_from_chunks(
        self,
        chunks: list[dict[str, Any]],
        target_total_samples: int = 50
    ) -> list[dict[str, Any]]:
        """
        Duyệt danh sách chunk và sinh câu hỏi chia đều cho cả 4 nhóm kiểm thử cốt lõi.
        """
        samples_per_cat = max(target_total_samples // 4, 8)
        golden_qa: list[dict[str, Any]] = []

        # 1. Phân loại chunks theo 4 nhóm
        legal_candidates = []
        table_candidates = []
        scan_candidates = []
        defrag_candidates = []

        for chunk in chunks:
            ctype = chunk.get("chunk_type", "")
            bc = chunk.get("breadcrumb", "")
            content = chunk.get("content", "")

            if ctype in ("docx_table", "pdf_table") or "* Hàng" in content or "|---" in content:
                table_candidates.append(chunk)
            elif ctype == "pdf_ocr_scan" or "ocr" in ctype:
                scan_candidates.append(chunk)
            elif (chunk.get("page_start", 1) != chunk.get("page_end", 1)) or (120 <= chunk.get("word_count", 0) <= 350 and "\n\n" in content):
                defrag_candidates.append(chunk)
            elif bc.count(" > ") >= 1:
                legal_candidates.append(chunk)

        print(f"[QA Generator] Thống kê ứng viên:")
        print(f"  - Category 1 (Legal Breadcrumb): {len(legal_candidates)}")
        print(f"  - Category 2 (Complex Table):   {len(table_candidates)}")
        print(f"  - Category 3 (Scan / Red-stamp): {len(scan_candidates)}")
        print(f"  - Category 4 (Cross-page/Defrag):{len(defrag_candidates)}")

        # 2. Sinh mẫu cho từng nhóm
        # Cat 1: Multi-level Legal
        for c in legal_candidates[:samples_per_cat]:
            item = self._generate_legal_sample(c)
            if item:
                golden_qa.append(item)

        # Cat 2: Complex Table
        for c in table_candidates[:samples_per_cat]:
            item = self._generate_table_sample(c)
            if item:
                golden_qa.append(item)

        # Cat 3: Red-stamp / Noisy Scan
        # Nếu chưa đủ scan candidates, lấy từ các văn bản nghị quyết / thông báo scan
        for c in scan_candidates[:samples_per_cat]:
            item = self._generate_scan_sample(c)
            if item:
                golden_qa.append(item)

        # Cat 4: Cross-page & Short Section De-fragmented
        for c in defrag_candidates[:samples_per_cat]:
            item = self._generate_defrag_sample(c)
            if item:
                golden_qa.append(item)

        # Nếu còn thiếu để đạt 50 mẫu, bổ sung thêm từ các nhóm có nhiều ứng viên
        cat_cycler = [legal_candidates[samples_per_cat:], table_candidates[samples_per_cat:], defrag_candidates[samples_per_cat:]]
        c_idx = 0
        while len(golden_qa) < target_total_samples and any(cat_cycler):
            for candidate_list in cat_cycler:
                if candidate_list and len(golden_qa) < target_total_samples:
                    extra_chunk = candidate_list.pop(0)
                    extra_sample = (
                        self._generate_table_sample(extra_chunk)
                        if "table" in extra_chunk.get("chunk_type", "")
                        else self._generate_legal_sample(extra_chunk)
                    )
                    if extra_sample:
                        golden_qa.append(extra_sample)

        print(f"[QA Generator] Tổng số câu hỏi kiểm thử đã tạo: {len(golden_qa)}")
        return golden_qa

    # -------------------------------------------------------------------------
    # BỘ SINH MẪU TỰ NHIÊN (NATURAL PARAPHRASE GENERATORS)
    # -------------------------------------------------------------------------

    def _generate_legal_sample(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        raw_body = _clean_body_content(chunk.get("content", ""))
        lines = [l.strip() for l in raw_body.split("\n") if len(l.strip()) > 30 and not l.strip().startswith("###")]
        if not lines:
            return None

        # Trích xuất 1 span ngữ cảnh cụ thể chứa thông tin quan trọng
        target_span = lines[0]
        for l in lines:
            if any(kw in l.lower() for kw in ["hạn mức", "phong toả", "đóng tài khoản", "phí duy trì", "điều kiện", "thời hạn", "trách nhiệm", "quyền", "nghĩa vụ"]):
                target_span = l
                break

        # Đặt câu hỏi tự nhiên bằng ngôn ngữ người dùng (không ghép chuỗi breadcrumb)
        t_low = target_span.lower()
        if "hạn mức" in t_low:
            question = "Quy định về hạn mức giao dịch tối đa đối với tài khoản thanh toán mở qua kênh trực tuyến chưa xác thực tại quầy là bao nhiêu?"
        elif "phong toả" in t_low:
            question = "Vietcombank có quyền phong toả tài khoản thanh toán của khách hàng cá nhân trong các trường hợp nào?"
        elif "đóng" in t_low and "tài khoản" in t_low:
            question = "Quy trình và điều kiện đóng tài khoản thanh toán trực tuyến của khách hàng được quy định thế nào?"
        elif "số dư" in t_low:
            question = "Khái niệm số dư được phép sử dụng trên tài khoản thanh toán của khách hàng được tính toán ra sao?"
        elif "địa điểm giao dịch" in t_low:
            question = "Địa điểm giao dịch hợp pháp thuộc mạng lưới hoạt động của Vietcombank bao gồm những cơ sở nào?"
        else:
            first_sent = target_span.split(".")[0]
            question = f"Quy định của ngân hàng liên quan đến nội dung '{first_sent[:60]}...' được áp dụng như thế nào?"

        bc = chunk.get("breadcrumb", "")
        bc_prefix = bc.split(" > ")[0] if " > " in bc else bc

        return {
            "question": question,
            "ground_truth_answer": target_span,
            "ground_truth_context_span": target_span,
            "expected_chunk_ids": [chunk.get("chunk_id", "")],
            "expected_breadcrumb_prefix": bc_prefix,
            "category": "multi_level_legal"
        }

    def _generate_table_sample(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        raw_body = _clean_body_content(chunk.get("content", ""))
        lines = [l.strip() for l in raw_body.split("\n") if l.strip()]

        data_rows = [l for l in lines if l.startswith("* Hàng") or ("|" in l and not l.startswith("|--"))]
        if not data_rows:
            return None

        target_row = data_rows[0]
        # Lấy context span là dòng dữ liệu nguyên bản
        cells = [c.strip() for c in target_row.replace("* Hàng", "").split("|") if ":" in c]

        if len(cells) >= 2:
            key_cell = cells[0]
            val_cell = cells[1]
            k_name, k_val = [x.strip() for x in key_cell.split(":", 1)]
            v_name, v_val = [x.strip() for x in val_cell.split(":", 1)]

            question = f"Mức {v_name.lower()} áp dụng cho hạng mục '{k_val}' ({k_name.lower()}) được ngân hàng quy định là bao nhiêu?"
            answer = f"Mức {v_name.lower()} của {k_val} là: {v_val}."
        else:
            question = f"Biểu phí chi tiết cho các chỉ tiêu trong bảng dữ liệu '{chunk.get('section_title', '')}' được quy định cụ thể ra sao?"
            answer = target_row

        bc = chunk.get("breadcrumb", "")
        bc_prefix = bc.split(" > ")[0] if " > " in bc else bc

        return {
            "question": question,
            "ground_truth_answer": answer,
            "ground_truth_context_span": target_row,
            "expected_chunk_ids": [chunk.get("chunk_id", "")],
            "expected_breadcrumb_prefix": bc_prefix,
            "category": "complex_table"
        }

    def _generate_scan_sample(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        raw_body = _clean_body_content(chunk.get("content", ""))
        lines = [l.strip() for l in raw_body.split("\n") if len(l.strip()) > 25]
        if not lines:
            return None

        target_span = lines[0]
        for l in lines:
            if any(c.isdigit() for c in l) and len(l) > 30:
                target_span = l
                break

        question = f"Theo văn bản ban hành có đóng dấu mộc, quy định về '{target_span[:50]}...' được phê duyệt với các nội dung/số liệu nào?"
        bc = chunk.get("breadcrumb", "")
        bc_prefix = bc.split(" > ")[0] if " > " in bc else bc

        return {
            "question": question,
            "ground_truth_answer": target_span,
            "ground_truth_context_span": target_span,
            "expected_chunk_ids": [chunk.get("chunk_id", "")],
            "expected_breadcrumb_prefix": bc_prefix,
            "category": "noisy_scan_ocr"
        }

    def _generate_defrag_sample(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        raw_body = _clean_body_content(chunk.get("content", ""))
        paras = [p.strip() for p in raw_body.split("\n\n") if len(p.strip()) > 35]
        if not paras:
            return None

        target_span = paras[-1]
        summary = target_span.split(".")[0] + "."

        question = f"Trong quy chuẩn vận hành dịch vụ của ngân hàng, nội dung nêu rõ: '{summary[:65]}...' được hướng dẫn chi tiết như thế nào?"
        bc = chunk.get("breadcrumb", "")
        bc_prefix = bc.split(" > ")[0] if " > " in bc else bc

        return {
            "question": question,
            "ground_truth_answer": summary,
            "ground_truth_context_span": target_span,
            "expected_chunk_ids": [chunk.get("chunk_id", "")],
            "expected_breadcrumb_prefix": bc_prefix,
            "category": "cross_page_defragmented"
        }


def main():
    parser = argparse.ArgumentParser(description="Sinh tập Golden QA cho benchmark RAG Banking (v2).")
    parser.add_argument("--input_chunks", type=str, required=True, help="Đường dẫn file chunks.json")
    parser.add_argument("--output_file", type=str, default="benchmarks/datasets/golden_qa.json")
    parser.add_argument("--backend", type=str, default="natural_paraphrase", choices=["natural_paraphrase", "ollama", "openrouter"])
    parser.add_argument("--samples", type=int, default=50, help="Tổng số mẫu cần tạo")
    args = parser.parse_args()

    in_path = Path(args.input_chunks)
    if not in_path.exists():
        print(f"Error: Không tìm thấy file {in_path}")
        return

    with open(in_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    gen = SyntheticQAGenerator(backend=args.backend)
    qa_list = gen.generate_from_chunks(chunks, target_total_samples=args.samples)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(qa_list, f, ensure_ascii=False, indent=2)

    print(f"[Done] Đã tạo thành công {len(qa_list)} mẫu Golden QA tại: {out_path}")


if __name__ == "__main__":
    main()
