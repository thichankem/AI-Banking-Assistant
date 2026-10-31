"""
Bộ điều phối phân đoạn tài liệu đa định dạng (LayoutAwareChunker / DLAChunker).
Thiết kế tối giản, trực quan, dễ hiểu:
Tự động nhận diện phần mở rộng tệp (.docx, .pdf) và định tuyến tới bộ phân đoạn chuyên trách.
"""

from __future__ import annotations
import json
import logging
import time
from pathlib import Path

from .syllable_budget_manager import build_7_field_chunk

logger = logging.getLogger("layout_aware_chunker")


class LayoutAwareChunker:
    """
    Bộ điều phối phân đoạn nhận thức bố cục (DLA) cho toàn bộ hệ thống RAG Banking.
    Chuyên trách phân đoạn 2 định dạng cốt lõi: .docx và .pdf.
    """

    def __init__(
        self,
        max_chunk_size: int = 800,
        min_section_size: int = 120,
        table_rows_per_chunk: int = 4,
        target_chunk_tokens: int = 400,
        max_chunk_words: int = 300,
        min_chunk_words: int = 30,
        config: object | None = None,
        **kwargs
    ):
        from ..docx_chunker.docx_chunker import DOCXChunker

        self.max_chunk_size = max_chunk_size
        self.min_section_size = min_section_size
        self.config = config

        # 1. Khởi tạo DOCX Chunker chuẩn hóa (Canonical Chunker)
        self.docx_chunker = DOCXChunker(
            max_chunk_size=max_chunk_size,
            min_section_size=min_section_size,
            **kwargs
        )

        # 2. Khởi tạo PDF Chunker thử nghiệm (độc lập tại experiments/pdf_chunker)
        self.pdf_chunker = None
        self.ocr_extractor = None
        try:
            from experiments.pdf_chunker.pdf_chunker import PDFChunker
            self.pdf_chunker = PDFChunker(
                max_chunk_size=max_chunk_size,
                min_section_size=min_section_size,
                table_rows_per_chunk=table_rows_per_chunk,
                config=config,
                **kwargs
            )
            self.ocr_extractor = getattr(self.pdf_chunker, "ocr_extractor", None)
        except Exception:
            pass

    def chunk_docx(self, file_path: str | Path) -> list[dict]:
        """Phân đoạn tệp Word DOCX chuẩn mực (Canonical Golden Document)."""
        return self.docx_chunker.chunk_docx(file_path)

    def chunk_pdf(self, file_path: str | Path, max_pages: int | None = None, **kwargs) -> list[dict]:
        """Phân đoạn tệp PDF (Module thử nghiệm R&D lưu tại experiments/pdf_chunker)."""
        if self.pdf_chunker is not None:
            return self.pdf_chunker.chunk_pdf(file_path, max_pages=max_pages, **kwargs)
        logger.info(
            f"[Canonical Ingestion Contract] Bỏ qua {Path(file_path).name}: "
            "Phân hệ PDF thử nghiệm được lưu tại experiments/pdf_chunker/. "
            "Lõi RAG Banking chuẩn hóa 100% đầu vào qua định dạng DOCX để bảo đảm độ chính xác tối đa."
        )
        return []

    def chunk_file(self, file_path: str | Path, max_pdf_pages: int | None = None, **kwargs) -> list[dict]:
        """
        Tự động phân đoạn tệp tài liệu:
        Mặc định chuẩn hóa 100% theo định dạng .docx (Canonical Contract).
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".docx":
            return self.chunk_docx(path)
        elif ext == ".pdf":
            # Chỉ kích hoạt PDF nếu người dùng chỉ định cờ thử nghiệm force_experimental_pdf=True
            if kwargs.get("force_experimental_pdf", False) and self.pdf_chunker is not None:
                return self.pdf_chunker.chunk_pdf(path, max_pages=max_pdf_pages, **kwargs)
            logger.info(
                f"[Canonical Ingestion Contract] Bỏ qua {path.name}: "
                "Lõi RAG Banking chuẩn hóa đầu vào duy nhất qua định dạng DOCX. "
                "Luồng PDF thử nghiệm đã được lưu trữ tại experiments/pdf_chunker/."
            )
            return []
        else:
            logger.warning(f"[LayoutAwareChunker] Bỏ qua định dạng chưa hỗ trợ: {ext}")
            return []

    def chunk_directory(
        self,
        dir_path: str | Path,
        extensions: list[str] | None = None,
        max_files: int | None = None,
        max_pdf_pages: int | None = None
    ) -> list[dict]:
        """
        Duyệt và phân đoạn toàn bộ tệp trong thư mục (mặc định chuẩn hóa 100% tệp .docx).
        """
        target_dir = Path(dir_path)
        if not target_dir.exists():
            return []

        allowed_exts = set(extensions or [".docx"])
        files = [p for p in target_dir.rglob("*") if p.is_file() and p.suffix.lower() in allowed_exts]
        files.sort()

        if max_files:
            files = files[:max_files]

        all_chunks: list[dict] = []
        for f in files:
            try:
                chunks = self.chunk_file(f, max_pdf_pages=max_pdf_pages)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"[LayoutAwareChunker] Lỗi xử lý tệp {f.name}: {e}")

        return all_chunks

    def chunk_dataset(
        self,
        dataset_dir: str | Path,
        output_dir: str | Path | None = None,
        max_pdf_pages: int | None = 3
    ) -> tuple[list[dict], dict]:
        """
        Phân đoạn trọn gói dataset ngân hàng, lưu kết quả và trả về thống kê.
        """
        start_t = time.time()
        dataset_path = Path(dataset_dir)
        all_chunks = self.chunk_directory(dataset_path, max_pdf_pages=max_pdf_pages)
        elapsed = round(time.time() - start_t, 2)

        stats = {
            "total_chunks": len(all_chunks),
            "elapsed_seconds": elapsed,
            "speed_chunks_per_sec": round(len(all_chunks) / max(elapsed, 0.001), 1)
        }

        if output_dir:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            with open(out_p / "chunks.json", "w", encoding="utf-8") as f:
                json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        return all_chunks, stats


# Bí danh tương thích ngược
DLAChunker = LayoutAwareChunker

__all__ = ["LayoutAwareChunker", "DLAChunker"]

