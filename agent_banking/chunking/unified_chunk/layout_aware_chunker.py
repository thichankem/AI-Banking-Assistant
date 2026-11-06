"""
Bộ điều phối phân đoạn tài liệu nhận thức bố cục (LayoutAwareChunker / DLAChunker).
Thiết kế tối giản, trực quan, chuyên sâu:
Chuẩn hóa 100% dữ liệu tài liệu Microsoft Word (.docx) ngân hàng qua OpenXML AST.
"""

from __future__ import annotations
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger("layout_aware_chunker")


class LayoutAwareChunker:
    """
    Bộ điều phối phân đoạn nhận thức bố cục OpenXML (DLA) cho hệ thống AI Banking Assistant.
    Chuyên trách phân đoạn tài liệu văn bản chuẩn mực: .docx.
    """

    def __init__(
        self,
        max_chunk_size: int | None = None,
        min_section_size: int = 120,
        table_rows_per_chunk: int = 4,
        target_chunk_tokens: int = 400,
        max_chunk_words: int = 300,
        min_chunk_words: int = 30,
        config: object | None = None,
        **kwargs
    ):
        try:
            from ..universal.universal_chunker import UniversalDocxChunker
        except (ImportError, ValueError):
            from agent_banking.chunking.universal.universal_chunker import UniversalDocxChunker
        self.docx_chunker = UniversalDocxChunker(
            max_chunk_words=max_chunk_words,
            target_chunk_tokens=target_chunk_tokens,
            table_rows_per_chunk=table_rows_per_chunk,
            max_chunk_chars=max_chunk_size,
            min_chunk_words=min_chunk_words,
            min_section_chars=min_section_size,
            **kwargs,
        )

        self.max_chunk_size = max_chunk_size
        self.min_section_size = min_section_size
        self.config = config

    def chunk_docx(self, file_path: str | Path, **kwargs) -> list[dict]:
        """Phân đoạn DOCX bằng OpenXML và LLM text theo chính sách ảnh hiện hành."""
        return self.docx_chunker.chunk_document(docx_path=file_path, **kwargs)

    @property
    def last_artifact(self):
        return self.docx_chunker.last_artifact

    def fingerprint(self):
        return self.docx_chunker.fingerprint()

    def close(self):
        self.docx_chunker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def chunk_file(self, file_path: str | Path, **kwargs) -> list[dict]:
        """
        Tự động phân đoạn tệp tài liệu:
        Chuẩn hóa 100% đầu vào qua định dạng .docx (Canonical Contract).
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".docx":
            return self.chunk_docx(path, **kwargs)
        else:
            logger.warning(f"[LayoutAwareChunker] Bỏ qua định dạng chưa hỗ trợ: {ext} ({path.name})")
            raise ValueError("Unsupported document format: " + ext)

    def chunk_directory(
        self,
        dir_path: str | Path,
        extensions: list[str] | None = None,
        max_files: int | None = None
    ) -> list[dict]:
        """
        Duyệt và phân đoạn toàn bộ tệp trong thư mục (chuẩn hóa 100% tệp .docx).
        """
        target_dir = Path(dir_path)
        if not target_dir.exists():
            return []

        allowed_exts = {extension.lower() for extension in (extensions or [".docx"])}
        files = [p for p in target_dir.rglob("*") if p.is_file() and p.suffix.lower() in allowed_exts and not p.name.startswith("~$")]
        files.sort()

        if max_files is not None:
            if max_files < 0:
                raise ValueError("max_files cannot be negative")
            files = files[:max_files]

        all_chunks: list[dict] = []
        for f in files:
            chunks = self.chunk_file(f, source_name=f.relative_to(target_dir).as_posix())
            all_chunks.extend(chunks)

        return all_chunks

    def chunk_dataset(
        self,
        dataset_dir: str | Path,
        output_dir: str | Path | None = None
    ) -> tuple[list[dict], dict]:
        """
        Phân đoạn trọn gói dataset ngân hàng, lưu kết quả và trả về thống kê.
        """
        start_t = time.time()
        dataset_path = Path(dataset_dir)
        all_chunks = self.chunk_directory(dataset_path)
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
