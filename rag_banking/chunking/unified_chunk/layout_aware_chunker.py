"""
Module Layout-Aware Unified Chunker (DLA Chunker)
==================================================
Bộ phân đoạn hợp nhất nhận thức bố cục tài liệu ngân hàng đa định dạng:
Điều phối tự động qua 4 luồng chuyên trách:
1. DOCX (.docx)  -> DOCXChunker (Heading AST, Table Atomicity, Section-Length-Aware Anti-Fragmentation)
2. PDF (.pdf)    -> PDFChunker (Digital Text + RapidOCR Scan + Legal Resolution + Page Anti-Fragmentation)
3. Excel (.xlsx) -> ExcelChunker (Ma trận đa sheet, Linearized Table Rows, Batch Row Chunking)
4. Markdown (.md)-> Markdown Layout Chunker (Header Markdown #, ##, ### + Tuyến tính hóa Table)

Tuân thủ nghiêm ngặt Schema Contract:
- Bắt buộc có: chunk_id, source, chunk_type, breadcrumb, section_title, content, word_count.
- TUYỆT ĐỐI KHÔNG CÓ các trường: 'ref', 'legal_ref', 'raw_content', 'char_length', 'format'.
"""

from __future__ import annotations

import os
import re
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

from .excel_chunker import ExcelChunker
from .canonical_resolver import (
    count_words,
    split_sentences_banking,
    group_sentences_banking
)


def _build_md_chunk(
    chunk_id: str,
    source: str,
    chunk_type: str,
    breadcrumb: str,
    section_title: str,
    content: str
) -> Dict[str, Any]:
    """Helper chuẩn hóa việc tạo đối tượng chunk Markdown/Text, đảm bảo Schema tinh gọn."""
    return {
        "chunk_id": chunk_id,
        "source": source,
        "chunk_type": chunk_type,
        "breadcrumb": breadcrumb,
        "section_title": section_title,
        "content": content,
        "word_count": count_words(content)
    }


class LayoutAwareChunker:
    """
    Bộ phân đoạn hợp nhất nhận thức bố cục (DLA) cho hệ thống RAG Ngân hàng.
    Điều phối phân đoạn chuyên sâu xuyên suốt 4 định dạng: DOCX, PDF, Excel, Markdown.
    """

    def __init__(
        self,
        max_chunk_size: int = 800,
        min_section_size: int = 120,
        chunk_overlap: int = 100,
        excel_rows_per_chunk: int = 3
    ):
        self.max_chunk_size = max_chunk_size
        self.min_section_size = min_section_size
        self.chunk_overlap = chunk_overlap

        # Khởi tạo các sub-chunkers chuyên biệt
        try:
            from ..docx_chunker import DOCXChunker
            from ..pdf_chunker import PDFChunker, OCRExtractor
        except (ImportError, ValueError):
            from rag_banking.chunking.docx_chunker import DOCXChunker
            from rag_banking.chunking.pdf_chunker import PDFChunker, OCRExtractor

        self.docx_chunker = DOCXChunker(
            max_chunk_size=max_chunk_size,
            min_section_size=min_section_size,
            chunk_overlap=chunk_overlap
        )
        self.ocr_extractor = OCRExtractor()
        self.pdf_chunker = PDFChunker(
            ocr_extractor=self.ocr_extractor,
            max_chunk_size=max_chunk_size,
            min_section_size=min_section_size,
            chunk_overlap=chunk_overlap
        )
        self.excel_chunker = ExcelChunker(
            rows_per_chunk=excel_rows_per_chunk
        )

    # -------------------------------------------------------------------------
    # 1. Các phương thức trích xuất từng định dạng chuyên biệt
    # -------------------------------------------------------------------------

    def chunk_docx(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Phân đoạn tệp Word DOCX theo DLA & Section-Length-Aware."""
        return self.docx_chunker.chunk_docx(file_path)

    def chunk_pdf(self, file_path: Union[str, Path], max_pages: Optional[int] = None) -> List[Dict[str, Any]]:
        """Phân đoạn tệp PDF (hỗ trợ digital và OCR scan + Anti-Fragmentation)."""
        return self.pdf_chunker.chunk_pdf(file_path, max_pages=max_pages)

    def chunk_excel(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Phân đoạn tệp bảng biểu Excel .xlsx đa sheet."""
        return self.excel_chunker.chunk_excel(file_path)

    def chunk_document(self, text: str, source_name: str = "document.md") -> List[Dict[str, Any]]:
        """
        Phân đoạn văn bản Markdown/Text theo bố cục tiêu đề và bảng biểu.
        """
        lines = text.split("\n")
        chunks: List[Dict[str, Any]] = []
        active_headers: Dict[int, str] = {}
        current_text_lines: List[str] = []
        chunk_counter = 0

        def get_current_breadcrumb() -> str:
            sorted_levels = sorted(active_headers.keys())
            titles = [active_headers[lvl] for lvl in sorted_levels if active_headers[lvl]]
            return " > ".join(titles) if titles else Path(source_name).stem

        def get_current_section_title() -> str:
            if not active_headers:
                return "Tổng quan"
            max_lvl = max(active_headers.keys())
            return active_headers[max_lvl]

        def flush_text_buffer(force_flush: bool = False) -> None:
            nonlocal chunk_counter
            if not current_text_lines:
                return

            raw_text = "\n".join(current_text_lines).strip()
            w_count = count_words(raw_text)

            # Cơ chế Section-Length-Aware: Nếu chưa đủ min_section_size và không ép buộc -> giữ lại gộp
            if not force_flush and w_count < self.min_section_size:
                return

            current_text_lines.clear()
            if not raw_text:
                return

            breadcrumb = get_current_breadcrumb()
            section = get_current_section_title()
            prefix = f"[Breadcrumbs: {breadcrumb}]"

            if len(raw_text) <= self.max_chunk_size:
                full_content = f"{prefix}\n{raw_text}"
                chunks.append(_build_md_chunk(
                    chunk_id=f"{Path(source_name).stem}_chunk_{chunk_counter}",
                    source=source_name,
                    chunk_type="md_text",
                    breadcrumb=breadcrumb,
                    section_title=section,
                    content=full_content
                ))
                chunk_counter += 1
            else:
                sentences = split_sentences_banking(raw_text)
                sub_parts = group_sentences_banking(sentences, self.max_chunk_size, self.chunk_overlap)
                for part in sub_parts:
                    full_content = f"{prefix}\n{part}"
                    chunks.append(_build_md_chunk(
                        chunk_id=f"{Path(source_name).stem}_chunk_{chunk_counter}",
                        source=source_name,
                        chunk_type="md_text",
                        breadcrumb=breadcrumb,
                        section_title=section,
                        content=full_content
                    ))
                    chunk_counter += 1

        in_table = False
        table_lines: List[str] = []

        for line in lines:
            stripped = line.strip()

            # Nhận diện dòng bảng Markdown
            if stripped.startswith("|") and stripped.endswith("|"):
                if not in_table:
                    flush_text_buffer(force_flush=True)
                    in_table = True
                    table_lines = [stripped]
                else:
                    table_lines.append(stripped)
                continue
            else:
                if in_table:
                    in_table = False
                    if len(table_lines) >= 3 and "---" in table_lines[1]:
                        table_chunk = self._process_markdown_table(
                            table_lines=table_lines,
                            source_name=source_name,
                            breadcrumb=get_current_breadcrumb(),
                            section=get_current_section_title(),
                            chunk_id=f"{Path(source_name).stem}_tbl_{chunk_counter}"
                        )
                        chunks.append(table_chunk)
                        chunk_counter += 1
                    else:
                        current_text_lines.extend(table_lines)
                    table_lines = []

            # Nhận diện tiêu đề Markdown
            header_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                if level == 1:
                    flush_text_buffer(force_flush=True)
                else:
                    current_words = count_words("\n".join(current_text_lines))
                    if current_words >= self.min_section_size:
                        flush_text_buffer(force_flush=True)
                    else:
                        if current_text_lines:
                            current_text_lines.append(f"\n### {title}")

                active_headers[level] = title
                levels_to_remove = [lvl for lvl in active_headers.keys() if lvl > level]
                for lvl in levels_to_remove:
                    del active_headers[lvl]
                continue

            if stripped:
                current_text_lines.append(stripped)

        # Flush bảng còn lại nếu có
        if in_table and len(table_lines) >= 3 and "---" in table_lines[1]:
            table_chunk = self._process_markdown_table(
                table_lines=table_lines,
                source_name=source_name,
                breadcrumb=get_current_breadcrumb(),
                section=get_current_section_title(),
                chunk_id=f"{Path(source_name).stem}_tbl_{chunk_counter}"
            )
            chunks.append(table_chunk)
            chunk_counter += 1

        flush_text_buffer(force_flush=True)
        return chunks

    def _process_markdown_table(
        self,
        table_lines: List[str],
        source_name: str,
        breadcrumb: str,
        section: str,
        chunk_id: str
    ) -> Dict[str, Any]:
        """Tuyến tính hóa bảng Markdown thành chuỗi có kèm tên cột."""
        raw_table = "\n".join(table_lines)
        header_cells = [c.strip() for c in table_lines[0].strip("|").split("|")]
        data_rows = []
        for line in table_lines[2:]:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if any(cells):
                row_str = " | ".join([
                    f"{header_cells[idx]}: {cells[idx]}"
                    for idx in range(min(len(header_cells), len(cells)))
                ])
                data_rows.append(row_str)

        linearized_content = "\n".join(data_rows)
        table_breadcrumb = f"{breadcrumb} > Bảng biểu" if breadcrumb else "Bảng biểu"

        header_prefix = f"[Breadcrumbs: {table_breadcrumb}]"
        full_content = (
            f"{header_prefix}\n"
            f"Cột dữ liệu: {', '.join(header_cells)}\n"
            f"Chi tiết từng hàng:\n{linearized_content}"
        )

        return _build_md_chunk(
            chunk_id=chunk_id,
            source=source_name,
            chunk_type="table",
            breadcrumb=table_breadcrumb,
            section_title=section,
            content=full_content
        )

    # -------------------------------------------------------------------------
    # 2. Điều phối đa định dạng: chunk_file & chunk_dataset
    # -------------------------------------------------------------------------

    def chunk_file(self, file_path: Union[str, Path], max_pdf_pages: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Tự động phân đoạn một tệp dựa trên phần mở rộng:
        - .docx -> DOCXChunker (Heading AST + Anti-fragmentation)
        - .pdf  -> PDFChunker (Digital + RapidOCR + Legal Ref)
        - .xlsx -> ExcelChunker
        - .md   -> Markdown Layout Chunker
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix == ".docx":
            return self.chunk_docx(file_path)
        elif suffix == ".pdf":
            return self.chunk_pdf(file_path, max_pages=max_pdf_pages)
        elif suffix in [".xlsx", ".xls"]:
            return self.chunk_excel(file_path)
        elif suffix in [".md", ".txt"]:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return self.chunk_document(content, source_name=file_path.name)
        else:
            print(f"[LayoutAwareChunker] Định dạng không được hỗ trợ: {suffix}")
            return []

    def chunk_dataset(
        self,
        dataset_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        max_pdf_pages: Optional[int] = 3,
        include_excel: bool = False,
        include_md: bool = False
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Xử lý toàn bộ dataset_chunking tập trung hoàn hảo vào 2 luồng cốt lõi: DOCX và PDF.
        """
        dataset_dir = Path(dataset_dir)
        start_time = time.time()

        docx_dir = dataset_dir / "docx"
        pdf_dir = dataset_dir / "pdf"
        excel_dir = dataset_dir / "excel"
        md_dir = dataset_dir / "md"

        docx_files = sorted(list(docx_dir.glob("*.docx"))) if docx_dir.exists() else []
        pdf_files = sorted(list(pdf_dir.glob("*.pdf"))) if pdf_dir.exists() else []
        excel_files = sorted(list(excel_dir.glob("*.xlsx"))) if (include_excel and excel_dir.exists()) else []
        md_files = sorted(list(md_dir.glob("*.md"))) if (include_md and md_dir.exists()) else []

        all_chunks: List[Dict[str, Any]] = []

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

        # Xử lý Excel nếu bật
        for excel_file in excel_files:
            try:
                chunks = self.chunk_excel(excel_file)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"[DLA Chunking] Lỗi file Excel {excel_file.name}: {e}")

        # Xử lý Markdown nếu bật
        for md_file in md_files:
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    c = f.read()
                chunks = self.chunk_document(c, source_name=md_file.name)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"[DLA Chunking] Lỗi file MD {md_file.name}: {e}")

        elapsed_time = round(time.time() - start_time, 2)
        stats = {
            "total_chunks": len(all_chunks),
            "docx_files": len(docx_files),
            "pdf_files": len(pdf_files),
            "excel_files": len(excel_files),
            "md_files": len(md_files),
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
        dir_path: Union[str, Path],
        extensions: Optional[List[str]] = None,
        max_files: Optional[int] = None,
        max_pdf_pages: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Duyệt và phân đoạn toàn bộ tệp trong một thư mục bất kỳ theo các đuôi mở rộng cho phép.
        Hỗ trợ trực tiếp cho Ingestion Pipeline (Phase 1).
        """
        target_dir = Path(dir_path)
        if not target_dir.exists():
            print(f"[LayoutAwareChunker] Không tìm thấy thư mục: {target_dir}")
            return []

        default_exts = [".docx", ".pdf", ".xlsx", ".md", ".txt"]
        raw_exts = extensions or default_exts
        allowed_exts = set(ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in raw_exts)

        files = [p for p in target_dir.rglob("*") if p.is_file() and p.suffix.lower() in allowed_exts]
        files.sort()
        if max_files:
            files = files[:max_files]

        all_chunks: List[Dict[str, Any]] = []
        for file_p in files:
            try:
                chunks = self.chunk_file(file_p, max_pdf_pages=max_pdf_pages)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"[LayoutAwareChunker] Lỗi xử lý tệp {file_p.name}: {e}")

        return all_chunks


# Alias tương thích ngược
DLAChunker = LayoutAwareChunker

