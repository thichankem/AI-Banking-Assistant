"""
Worker độc lập (Subprocess Isolated) để sinh Chunks cho từng Cấu hình Benchmark.
Đảm bảo 100% không bị xung đột sys.modules giữa Legacy Worktree và Production Code.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Khởi tạo đường dẫn
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_WORKTREE = PROJECT_ROOT.parent / "legacy_pre_merge"

CORPUS_PATHS = [
    PROJECT_ROOT / "dataset" / "chunking_dataset" / "pdf" / "vcb_pdf_22_bieu_phi_the_tdqt_2026.pdf",
    PROJECT_ROOT / "dataset" / "chunking_dataset" / "pdf" / "vcb_pdf_27_dieu_kien_dieu_khoan_tktt_truc_tuyen.pdf",
    PROJECT_ROOT / "dataset" / "chunking_dataset" / "pdf" / "vcb_pdf_21_nq_hdqt_cctg_vcbs.pdf",
    PROJECT_ROOT / "dataset" / "chunking_dataset" / "docx" / "vcb_md_01__faq__the_tin_dung_va_the_ghi_no_ngan_hang.docx"
]


def run_config_0() -> tuple[list[dict[str, Any]], dict[str, str], str]:
    """Config 0: Naive Baseline Chunker (Fixed window 800 chars, overlap 100)."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from benchmarks.ablation_runners.run_ablation import BaselineNaiveChunker
    chunker = BaselineNaiveChunker(chunk_size=800, overlap=100)
    chunks = []
    for p in CORPUS_PATHS:
        chunks.extend(chunker.chunk_file(p))
    return chunks, {}, "benchmarks.ablation_runners.run_ablation.BaselineNaiveChunker"


def run_legacy_config_1() -> tuple[list[dict[str, Any]], dict[str, str], str]:
    """Config 1: Pipeline v1 từ legacy worktree (Structure + Breadcrumb, tắt bảng, tắt mộc đỏ)."""
    sys.path.insert(0, str(LEGACY_WORKTREE))
    sys.path.insert(1, str(PROJECT_ROOT))
    import rag_banking.chunking as rbc_legacy
    mod_path = str(rbc_legacy.__file__)
    chunker = rbc_legacy.LayoutAwareChunker(max_chunk_size=800, min_section_size=120, chunk_overlap=100)
    chunker.pdf_chunker.enable_vision_llm_tables = False
    chunker.ocr_extractor.filter_red_stamp = lambda img: img

    raw_chunks = []
    for p in CORPUS_PATHS:
        raw_chunks.extend(chunker.chunk_file(p))

    cleaned = []
    for c in raw_chunks:
        item = copy.deepcopy(c)
        if item.get("chunk_type") in ("docx_table", "pdf_table"):
            lines = item.get("content", "").split("\n")
            raw_rows = [l for l in lines if not l.startswith("[Breadcrumbs:")]
            item["content"] = "\n".join(raw_rows)
        cleaned.append(item)
    return cleaned, {}, mod_path


def run_legacy_config_2() -> tuple[list[dict[str, Any]], dict[str, str], str]:
    """Config 2: Pipeline v2 từ legacy worktree (Current Production cũ trước merge)."""
    sys.path.insert(0, str(LEGACY_WORKTREE))
    sys.path.insert(1, str(PROJECT_ROOT))
    import rag_banking.chunking as rbc_legacy
    mod_path = str(rbc_legacy.__file__)
    chunker = rbc_legacy.LayoutAwareChunker(max_chunk_size=800, min_section_size=120, chunk_overlap=100)
    chunker.pdf_chunker.enable_vision_llm_tables = False

    chunks = []
    for p in CORPUS_PATHS:
        chunks.extend(chunker.chunk_file(p))
    return chunks, {}, mod_path


def load_from_prod(module_name: str, rel_path: str):
    import importlib.util
    file_path = PROJECT_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_legacy_config_3() -> tuple[list[dict[str, Any]], dict[str, str], str]:
    """Config 3: Pipeline v2 từ legacy worktree + nâng cấp PaddleVietOCRAdapter & ScanTableDetector."""
    sys.path.insert(0, str(LEGACY_WORKTREE))
    sys.path.insert(1, str(PROJECT_ROOT))
    import rag_banking.chunking as rbc_legacy
    mod_path = str(rbc_legacy.__file__)
    chunker = rbc_legacy.LayoutAwareChunker(max_chunk_size=800, min_section_size=120, chunk_overlap=100)
    
    mod_paddle = load_from_prod("paddle_vietocr_adapter", "rag_banking/chunking/pdf_chunker/paddle_vietocr_adapter.py")
    mod_detector = load_from_prod("scan_table_detector", "rag_banking/chunking/pdf_chunker/scan_table_detector.py")
    PaddleVietOCRAdapter = mod_paddle.PaddleVietOCRAdapter
    ScanTableDetector = mod_detector.ScanTableDetector

    paddle_ocr = PaddleVietOCRAdapter()
    table_detector = ScanTableDetector()

    class Exp1OCRExtractorWrapper:
        def __init__(self, base_extractor: Any, paddle_adapter: Any, table_detector: Any):
            self.base_extractor = base_extractor
            self.paddle_adapter = paddle_adapter
            self.table_detector = table_detector

        def is_available(self) -> bool:
            return True

        def render_pdf_page_300dpi(self, pdf_path: str | Path, page_idx: int = 0):
            return self.base_extractor.render_pdf_page_300dpi(pdf_path, page_idx=page_idx)

        def extract_layout_lines(self, img_np: Any, apply_deskew: bool = True, apply_red_stamp_filter: bool = True):
            proc_img = img_np
            if apply_red_stamp_filter and hasattr(self.base_extractor, "filter_red_stamp"):
                proc_img = self.base_extractor.filter_red_stamp(img_np)
            tables = self.table_detector.detect_tables(proc_img)
            masked_img = self.table_detector.mask_tables_on_image(proc_img, tables)
            return self.paddle_adapter.extract_layout_lines(masked_img)

        def extract_text_from_pdf_page(self, pdf_path: str | Path, page_idx: int = 0):
            return self.base_extractor.extract_text_from_pdf_page(pdf_path, page_idx=page_idx)

        def filter_red_stamp(self, img_np: Any):
            return self.base_extractor.filter_red_stamp(img_np)

    wrapper = Exp1OCRExtractorWrapper(chunker.ocr_extractor, paddle_ocr, table_detector)
    chunker.ocr_extractor = wrapper
    chunker.pdf_chunker.ocr_extractor = wrapper

    chunks = []
    for p in CORPUS_PATHS:
        chunks.extend(chunker.chunk_file(p))
    return chunks, {}, mod_path


def run_legacy_config_4() -> tuple[list[dict[str, Any]], dict[str, str], str]:
    """Config 4: Pipeline Config 3 + TableFormatter Markdown Table (lặp Header)."""
    sys.path.insert(0, str(LEGACY_WORKTREE))
    sys.path.insert(1, str(PROJECT_ROOT))
    import rag_banking.chunking as rbc_legacy
    mod_path = str(rbc_legacy.__file__)
    chunker = rbc_legacy.LayoutAwareChunker(max_chunk_size=800, min_section_size=120, chunk_overlap=100)
    
    mod_paddle = load_from_prod("paddle_vietocr_adapter", "rag_banking/chunking/pdf_chunker/paddle_vietocr_adapter.py")
    mod_detector = load_from_prod("scan_table_detector", "rag_banking/chunking/pdf_chunker/scan_table_detector.py")
    mod_formatter = load_from_prod("table_formatter", "rag_banking/chunking/pdf_chunker/table_formatter.py")
    PaddleVietOCRAdapter = mod_paddle.PaddleVietOCRAdapter
    ScanTableDetector = mod_detector.ScanTableDetector
    TableFormatter = mod_formatter.TableFormatter
    TableFormatType = mod_formatter.TableFormatType

    paddle_ocr = PaddleVietOCRAdapter()
    table_detector = ScanTableDetector()

    class Exp1OCRExtractorWrapper:
        def __init__(self, base_extractor: Any, paddle_adapter: Any, table_detector: Any):
            self.base_extractor = base_extractor
            self.paddle_adapter = paddle_adapter
            self.table_detector = table_detector

        def is_available(self) -> bool:
            return True

        def render_pdf_page_300dpi(self, pdf_path: str | Path, page_idx: int = 0):
            return self.base_extractor.render_pdf_page_300dpi(pdf_path, page_idx=page_idx)

        def extract_layout_lines(self, img_np: Any, apply_deskew: bool = True, apply_red_stamp_filter: bool = True):
            proc_img = img_np
            if apply_red_stamp_filter and hasattr(self.base_extractor, "filter_red_stamp"):
                proc_img = self.base_extractor.filter_red_stamp(img_np)
            tables = self.table_detector.detect_tables(proc_img)
            masked_img = self.table_detector.mask_tables_on_image(proc_img, tables)
            return self.paddle_adapter.extract_layout_lines(masked_img)

        def extract_text_from_pdf_page(self, pdf_path: str | Path, page_idx: int = 0):
            return self.base_extractor.extract_text_from_pdf_page(pdf_path, page_idx=page_idx)

        def filter_red_stamp(self, img_np: Any):
            return self.base_extractor.filter_red_stamp(img_np)

    wrapper = Exp1OCRExtractorWrapper(chunker.ocr_extractor, paddle_ocr, table_detector)
    chunker.ocr_extractor = wrapper
    chunker.pdf_chunker.ocr_extractor = wrapper
    formatter = TableFormatter(default_batch_size=8)

    raw_chunks = []
    for p in CORPUS_PATHS:
        raw_chunks.extend(chunker.chunk_file(p))

    for c in raw_chunks:
        if c.get("chunk_type") in ("docx_table", "pdf_table"):
            content = c.get("content", "")
            lines = content.split("\n")
            bc_lines = [l for l in lines if l.startswith("[Breadcrumbs:")]
            raw_body = "\n".join(l for l in lines if not l.startswith("[Breadcrumbs:"))
            try:
                fmt_body = formatter.format(raw_body, format_type=TableFormatType.MARKDOWN_TABLE)
                c["content"] = (bc_lines[0] + "\n" if bc_lines else "") + fmt_body
            except Exception:
                pass
    return raw_chunks, {}, mod_path


def run_production_config_5(profile_type: str) -> tuple[list[dict[str, Any]], dict[str, str], str]:
    """Config 5A, 5B, 5C từ Production Repo."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from benchmarks.ablation_runners.run_ablation import (
        PROFILE_A,
        PROFILE_B,
        PROFILE_C,
        PipelineV3UltimateRunner
    )
    if profile_type == "5a":
        prof = PROFILE_A
    elif profile_type == "5b":
        prof = PROFILE_B
    elif profile_type == "5c":
        prof = PROFILE_C
    else:
        raise ValueError(f"Unknown profile: {profile_type}")

    runner = PipelineV3UltimateRunner(profile=prof)
    chunks = []
    for p in CORPUS_PATHS:
        chunks.extend(runner.chunk_file(p))
    parent_map = getattr(runner, "parent_map", {})
    return chunks, parent_map, "rag_banking.chunking.unified_chunk.layout_aware_chunker.LayoutAwareChunker"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, choices=["0", "1", "2", "3", "4", "5a", "5b", "5c"])
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    cfg = args.config.lower()
    if cfg == "0":
        chunks, parent_map, mod_path = run_config_0()
    elif cfg == "1":
        chunks, parent_map, mod_path = run_legacy_config_1()
    elif cfg == "2":
        chunks, parent_map, mod_path = run_legacy_config_2()
    elif cfg == "3":
        chunks, parent_map, mod_path = run_legacy_config_3()
    elif cfg == "4":
        chunks, parent_map, mod_path = run_legacy_config_4()
    elif cfg in ("5a", "5b", "5c"):
        chunks, parent_map, mod_path = run_production_config_5(cfg)
    else:
        raise ValueError(f"Invalid config: {cfg}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": cfg,
            "module_path": mod_path,
            "chunk_count": len(chunks),
            "chunks": chunks,
            "parent_map": parent_map
        }, f, ensure_ascii=False, indent=2)

    print(f"[Worker] Config {cfg} completed: {len(chunks)} chunks -> {out_path.name}")


if __name__ == "__main__":
    main()
