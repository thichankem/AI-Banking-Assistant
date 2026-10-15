"""
Module Layout-Aware Unified Chunker (DLA Chunker)
==================================================
Bộ phân đoạn hợp nhất nhận thức bố cục tài liệu ngân hàng chuyên biệt DOCX & PDF:
1. DOCX (.docx)  -> DOCXChunker (Heading AST, Table Atomicity, Section-Length-Aware Anti-Fragmentation)
2. PDF (.pdf)    -> PDFChunker (Digital Text + Tesseract OCR v5 + Legal Resolution + Page Anti-Fragmentation)

Hoàn toàn KHÔNG sử dụng biểu thức chính quy (0% Regex Engine).
Tuân thủ nghiêm ngặt Schema Contract:
- Bắt buộc có: chunk_id, source, chunk_type, breadcrumb, section_title, content, word_count.
- TUYỆT ĐỐI KHÔNG CÓ các trường: 'ref', 'legal_ref', 'raw_content', 'char_length', 'format'.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .canonical_resolver import (
    count_words,
    split_sentences_banking,
    group_sentences_banking
)


class LayoutAwareChunker:
    """
    Bộ phân đoạn hợp nhất nhận thức bố cục (DLA) cho hệ thống RAG Ngân hàng.
    Điều phối phân đoạn chuyên sâu xuyên suốt 2 định dạng cốt lõi: DOCX và PDF.
    """

    def __init__(
        self,
        max_chunk_size: int = 800,
        min_section_size: int = 120,
        chunk_overlap: int = 100,
        profile: Any | None = None
    ):
        self.max_chunk_size = max_chunk_size
        self.min_section_size = min_section_size
        self.chunk_overlap = chunk_overlap

        # Khởi tạo Profile âm tiết chuẩn hóa (Mặc định: Profile B 150-400 âm tiết)
        try:
            from .syllable_budget_manager import PROFILE_B, SyllableProfile
            from .unified_budget_chunker import UnifiedBudgetOrchestrator
            from ..pdf_chunker.paddle_vietocr_adapter import PaddleVietOCRAdapter
            from ..pdf_chunker.scan_table_detector import ScanTableDetector
            from ..pdf_chunker.pdf_chunker import PDFChunker, OCRExtractor
        except (ImportError, ValueError):
            from rag_banking.chunking.unified_chunk.syllable_budget_manager import PROFILE_B, SyllableProfile
            from rag_banking.chunking.unified_chunk.unified_budget_chunker import UnifiedBudgetOrchestrator
            from rag_banking.chunking.pdf_chunker.paddle_vietocr_adapter import PaddleVietOCRAdapter
            from rag_banking.chunking.pdf_chunker.scan_table_detector import ScanTableDetector
            from rag_banking.chunking.pdf_chunker.pdf_chunker import PDFChunker, OCRExtractor

        self.profile = profile or PROFILE_B
        self.budget_orch = UnifiedBudgetOrchestrator(profile=self.profile)

        # Cấu hình OCR Extractor: Ưu tiên PaddleVietOCRAdapter + ScanTableDetector
        try:
            self.paddle_ocr = PaddleVietOCRAdapter()
            self.table_detector = ScanTableDetector()
            self.ocr_extractor = self.paddle_ocr
        except Exception:
            self.ocr_extractor = OCRExtractor()

        self.docx_chunker = self.budget_orch.docx_chunker
        self.pdf_chunker = PDFChunker(
            ocr_extractor=self.ocr_extractor,
            max_chunk_size=self.profile.max_syllables * 4 if hasattr(self.profile, "max_syllables") else max_chunk_size,
            min_section_size=self.profile.min_syllables if hasattr(self.profile, "min_syllables") else min_section_size,
            chunk_overlap=self.profile.overlap_syllables * 4 if hasattr(self.profile, "overlap_syllables") else chunk_overlap,
            use_unified_stream=True
        )

    # -------------------------------------------------------------------------
    # 1. Các phương thức trích xuất từng định dạng chuyên biệt (DOCX, PDF, MD)
    # -------------------------------------------------------------------------

    def chunk_docx(self, file_path: str | Path) -> list[dict]:
        """Phân đoạn tệp Word DOCX theo ngân sách âm tiết Profile B."""
        return self.budget_orch.chunk_file(file_path)

    def chunk_pdf(self, file_path: str | Path, max_pages: int | None = None) -> list[dict]:
        """Phân đoạn tệp PDF theo luồng hình học 5D thống nhất không dùng ngưỡng 50%."""
        return self.pdf_chunker.chunk_pdf(file_path, max_pages=max_pages, use_unified_stream=True)

    # -------------------------------------------------------------------------
    # 2. Điều phối định dạng cốt lõi: chunk_file, chunk_dataset, chunk_directory
    # -------------------------------------------------------------------------

    def chunk_file(self, file_path: str | Path, max_pdf_pages: int | None = None) -> list[dict]:
        """
        Tự động phân đoạn một tệp dựa trên phần mở rộng (.docx, .pdf, .md):
        Tuân thủ nghiêm ngặt 100% Schema Contract 7 trường chuẩn.
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        raw_chunks: list[dict] = []
        if suffix == ".docx":
            raw_chunks = self.chunk_docx(file_path)
        elif suffix == ".pdf":
            raw_chunks = self.chunk_pdf(file_path, max_pages=max_pdf_pages)
        elif suffix in (".md", ".txt"):
            raw_chunks = self.budget_orch.chunk_file(file_path)
        else:
            print(f"[LayoutAwareChunker] Định dạng không được hỗ trợ: {suffix}")
            return []

        # Chuẩn hóa chính xác 100% theo Schema Contract (Đúng 7 trường chuẩn)
        cleaned_chunks: list[dict] = []
        for c in raw_chunks:
            c_id = c.get("chunk_id", "")
            content = c.get("content", "")
            cleaned_chunks.append({
                "chunk_id": c_id,
                "source": c.get("source", file_path.name),
                "chunk_type": c.get("chunk_type", "text"),
                "breadcrumb": c.get("breadcrumb", file_path.stem),
                "section_title": c.get("section_title", "N/A"),
                "content": content,
                "word_count": count_words(content)
            })

        return cleaned_chunks

    def chunk_dataset(
        self,
        dataset_dir: str | Path,
        output_dir: str | Path | None = None,
        max_pdf_pages: int | None = 3
    ) -> tuple[list[dict], dict]:
        """
        Xử lý toàn bộ dataset_chunking tập trung hoàn hảo vào 2 luồng cốt lõi: DOCX và PDF.
        """
        dataset_dir = Path(dataset_dir)
        start_time = time.time()

        docx_dir = dataset_dir / "docx"
        pdf_dir = dataset_dir / "pdf"

        docx_files = sorted(list(docx_dir.glob("*.docx"))) if docx_dir.exists() else []
        pdf_files = sorted(list(pdf_dir.glob("*.pdf"))) if pdf_dir.exists() else []

        all_chunks: list[dict] = []

        # Xử lý DOCX
        for docx_file in docx_files:
            try:
                chunks = self.chunk_docx(docx_file)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"[DLA Chunking] Lỗi file DOCX {docx_file.name}: {e}")

        # Xử lý PDF
        for pdf_file in pdf_files:
            try:
                chunks = self.chunk_pdf(pdf_file, max_pages=max_pdf_pages)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"[DLA Chunking] Lỗi file PDF {pdf_file.name}: {e}")

        elapsed_time = round(time.time() - start_time, 2)
        stats = {
            "total_chunks": len(all_chunks),
            "docx_files": len(docx_files),
            "pdf_files": len(pdf_files),
            "elapsed_time_seconds": elapsed_time
        }

        if output_dir:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            with open(out_p / "chunks.json", "w", encoding="utf-8") as f:
                json.dump(all_chunks, f, ensure_ascii=False, indent=2)
            with open(out_p / "chunking_stats.json", "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)

        return all_chunks, stats

    def chunk_directory(
        self,
        dir_path: str | Path,
        extensions: list[str] | None = None,
        max_files: int | None = None,
        max_pdf_pages: int | None = None
    ) -> list[dict]:
        """
        Duyệt và phân đoạn toàn bộ tệp trong một thư mục bất kỳ (.docx và .pdf).
        Hỗ trợ trực tiếp cho Ingestion Pipeline.
        """
        target_dir = Path(dir_path)
        if not target_dir.exists():
            print(f"[LayoutAwareChunker] Không tìm thấy thư mục: {target_dir}")
            return []

        default_exts = [".docx", ".pdf"]
        raw_exts = extensions or default_exts
        allowed_exts = set(ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in raw_exts)
        # Chỉ cho phép .docx và .pdf
        allowed_exts = allowed_exts.intersection({".docx", ".pdf"})

        files = [p for p in target_dir.rglob("*") if p.is_file() and p.suffix.lower() in allowed_exts]
        files.sort()
        if max_files:
            files = files[:max_files]

        all_chunks: list[dict] = []
        for file_p in files:
            try:
                chunks = self.chunk_file(file_p, max_pdf_pages=max_pdf_pages)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"[LayoutAwareChunker] Lỗi xử lý tệp {file_p.name}: {e}")

        return all_chunks


# Alias tương thích ngược
DLAChunker = LayoutAwareChunker
