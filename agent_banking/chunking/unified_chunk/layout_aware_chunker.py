"""
Document Layout Aware Chunker (LayoutAwareChunker / DLAChunker).
Minimalist, intuitive, and domain-focused design:
Normalizes Microsoft Word (.docx) and PDF banking documents via OpenXML AST and document IR.
"""

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger("layout_aware_chunker")


class LayoutAwareChunker:
    """
    OpenXML Document Layout Aware (DLA) chunking orchestrator for AI Banking Assistant.
    Specialized for standardized document chunking: .docx and .pdf.
    """

    def __init__(
        self,
        max_chunk_size=None,
        min_section_size=120,
        table_rows_per_chunk=4,
        target_chunk_tokens=400,
        max_chunk_words=300,
        min_chunk_words=30,
        config=None,
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

    def chunk_docx(self, file_path, **kwargs):
        """Chunk DOCX using OpenXML and text models under active image policies."""
        return self.docx_chunker.chunk_document(docx_path=file_path, **kwargs)

    def chunk_pdf(self, file_path, **kwargs):
        return self.docx_chunker.chunk_pdf(file_path, **kwargs)

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

    def chunk_file(self, file_path, **kwargs):
        """
        Automatically chunk document file:
        Normalizes inputs via canonical format contract.
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".docx":
            return self.chunk_docx(path, **kwargs)
        elif ext == ".pdf":
            return self.chunk_pdf(path, **kwargs)
        else:
            logger.warning(f"[LayoutAwareChunker] Skipping unsupported format: {ext} ({path.name})")
            raise ValueError("Unsupported document format: " + ext)

    def chunk_directory(
        self,
        dir_path,
        extensions=None,
        max_files=None
    ):
        """
        Traverse and chunk all files in directory.
        """
        target_dir = Path(dir_path)
        if not target_dir.exists():
            return []

        allowed_exts = {extension.lower() for extension in (extensions or [".docx", ".pdf"])}
        files = [p for p in target_dir.rglob("*") if p.is_file() and p.suffix.lower() in allowed_exts and not p.name.startswith("~$")]
        files.sort()

        if max_files is not None:
            if max_files < 0:
                raise ValueError("max_files cannot be negative")
            files = files[:max_files]

        all_chunks = []
        for f in files:
            chunks = self.chunk_file(f, source_name=f.relative_to(target_dir).as_posix())
            all_chunks.extend(chunks)

        return all_chunks

    def chunk_dataset(
        self,
        dataset_dir,
        output_dir=None
    ):
        """
        Chunk complete banking dataset, save results and return statistics.
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


# Backward compatibility alias
DLAChunker = LayoutAwareChunker

__all__ = ["LayoutAwareChunker", "DLAChunker"]
