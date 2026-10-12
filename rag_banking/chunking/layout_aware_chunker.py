"""
Document Layout Analysis (DLA) & Unified Layout-Aware Chunker
Bộ phân đoạn hợp nhất nhận thức bố cục tài liệu ngân hàng đa định dạng:
Tập trung chuyên sâu vào 2 Luồng Cốt Lõi:
1. DOCX (.docx) -> DOCXChunker (Heading AST, Table Atomicity, Section-Length-Aware Anti-Fragmentation)
2. PDF (.pdf)   -> PDFChunker (Digital Text + RapidOCR Scan + Legal Resolution + Page Anti-Fragmentation)
Kèm hỗ trợ mở rộng cho Excel (.xlsx) và Markdown (.md).
"""

import os
import re
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from .ocr_extractor import OCRExtractor
from .docx_chunker import DOCXChunker
from .pdf_chunker import PDFChunker
from .excel_chunker import ExcelChunker
from .canonical_resolver import (
    count_words,
    count_syllables_vietnamese,
    extract_canonical_legal_ref,
    split_sentences_banking,
    group_sentences_banking
)


def _build_md_chunk(
    chunk_id: str,
    source: str,
    chunk_type: str,
    breadcrumb: str,
    section_title: str,
    content: str,
    raw_content: str,
    legal_ref: Optional[str] = None
) -> Dict[str, Any]:
    """Helper chuẩn hóa việc tạo đối tượng chunk Markdown/Text, đảm bảo Schema thống nhất."""
    return {
        "chunk_id": chunk_id,
        "source": source,
        "format": "md",
        "chunk_type": chunk_type,
        "breadcrumb": breadcrumb,
        "legal_ref": legal_ref,
        "section_title": section_title,
        "content": content,
        "raw_content": raw_content,
        "char_length": len(content),
        "word_count": count_words(content),
        "syllable_count": count_syllables_vietnamese(content)
    }


class LayoutAwareChunker:
    """
    Bộ phân đoạn hợp nhất nhận thức bố cục (DLA) cho hệ thống RAG Ngân hàng.
    Điều phối phân đoạn chuyên sâu xuyên suốt 2 luồng cốt lõi: DOCX và PDF.
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

    def chunk_docx(self, file_path: str | Path) -> List[Dict[str, Any]]:
        """Phân đoạn tệp Word DOCX theo DLA & Section-Length-Aware."""
        return self.docx_chunker.chunk_docx(file_path)

    def chunk_pdf(self, file_path: str | Path, max_pages: Optional[int] = None) -> List[Dict[str, Any]]:
        """Phân đoạn tệp PDF (hỗ trợ digital và OCR scan + Anti-Fragmentation)."""
        return self.pdf_chunker.chunk_pdf(file_path, max_pages=max_pages)

    def chunk_excel(self, file_path: str | Path) -> List[Dict[str, Any]]:
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

        def flush_text_buffer(force_flush: bool = False):
            nonlocal chunk_counter
            if not current_text_lines:
                return

            raw_text = "\n".join(current_text_lines).strip()
            w_count = count_words(raw_text)

            # Cơ chế Section-Length-Aware: Nếu chưa đủ min_section_size và không bị force_flush -> giữ lại gộp
            if not force_flush and w_count < self.min_section_size:
                return

            current_text_lines.clear()
            if not raw_text:
                return

            breadcrumb = get_current_breadcrumb()
            section = get_current_section_title()
            legal_ref = extract_canonical_legal_ref(breadcrumb, raw_text, source_name=source_name)

            prefix = f"[Breadcrumbs: {breadcrumb}]"
            if legal_ref:
                prefix += f"\n[Pháp lý: {legal_ref}]"

            if len(raw_text) <= self.max_chunk_size:
                full_content = f"{prefix}\n{raw_text}"
                chunks.append(_build_md_chunk(
                    chunk_id=f"{Path(source_name).stem}_chunk_{chunk_counter}",
                    source=source_name,
                    chunk_type="text",
                    breadcrumb=breadcrumb,
                    section_title=section,
                    content=full_content,
                    raw_content=raw_text,
                    legal_ref=legal_ref
                ))
                chunk_counter += 1
            else:
                sentences = split_sentences_banking(raw_text)
                sub_parts = group_sentences_banking(sentences, self.max_chunk_size, self.chunk_overlap)
                for part in sub_parts:
                    sub_legal = extract_canonical_legal_ref(breadcrumb, part, source_name=source_name) or legal_ref
                    sub_prefix = f"[Breadcrumbs: {breadcrumb}]"
                    if sub_legal:
                        sub_prefix += f"\n[Pháp lý: {sub_legal}]"
                    full_content = f"{sub_prefix}\n{part}"

                    chunks.append(_build_md_chunk(
                        chunk_id=f"{Path(source_name).stem}_chunk_{chunk_counter}",
                        source=source_name,
                        chunk_type="text",
                        breadcrumb=breadcrumb,
                        section_title=section,
                        content=full_content,
                        raw_content=part,
                        legal_ref=sub_legal
                    ))
                    chunk_counter += 1

        i = 0
        total_lines = len(lines)

        while i < total_lines:
            line = lines[i]
            stripped = line.strip()

            # 1. Tiêu đề Markdown (# H1, ## H2, ### H3, ...)
            header_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                if level == 1 or count_words("\n".join(current_text_lines)) >= self.min_section_size:
                    flush_text_buffer(force_flush=True)
                else:
                    if current_text_lines:
                        current_text_lines.append(f"\n### {title}")

                active_headers[level] = title
                levels_to_remove = [lvl for lvl in active_headers.keys() if lvl > level]
                for lvl in levels_to_remove:
                    del active_headers[lvl]
                i += 1
                continue

            # 2. Bảng biểu Markdown chuẩn (| Cột 1 | Cột 2 | ...)
            if stripped.startswith("|") and stripped.endswith("|") and i + 1 < total_lines:
                next_stripped = lines[i + 1].strip()
                if re.match(r"^\|(\s*:?-+:?\s*\|)+$", next_stripped):
                    flush_text_buffer(force_flush=True)
                    table_lines = [line]
                    i += 1
                    while i < total_lines and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                        table_lines.append(lines[i])
                        i += 1

                    table_chunk = self._extract_markdown_table(
                        table_lines=table_lines,
                        source_name=source_name,
                        breadcrumb=get_current_breadcrumb(),
                        section=get_current_section_title(),
                        chunk_id=f"{Path(source_name).stem}_chunk_{chunk_counter}"
                    )
                    chunks.append(table_chunk)
                    chunk_counter += 1
                    continue

            # 3. Phân cách đoạn
            if stripped in ["---", "***", "___"]:
                flush_text_buffer(force_flush=True)
            else:
                current_text_lines.append(line)

            i += 1

        flush_text_buffer(force_flush=True)
        return chunks

    def _extract_markdown_table(
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
        legal_ref = extract_canonical_legal_ref(table_breadcrumb, raw_table, source_name=source_name)

        header_prefix = f"[Breadcrumbs: {table_breadcrumb}]"
        if legal_ref:
            header_prefix += f"\n[Pháp lý: {legal_ref}]"

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
            content=full_content,
            raw_content=raw_table,
            legal_ref=legal_ref
        )

    # -------------------------------------------------------------------------
    # 2. Điều phối đa định dạng: chunk_file & chunk_dataset
    # -------------------------------------------------------------------------

    def chunk_file(self, file_path: str | Path, max_pdf_pages: Optional[int] = None) -> List[Dict[str, Any]]:
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
        dataset_dir: str | Path,
        output_dir: Optional[str | Path] = None,
        max_pdf_pages: Optional[int] = 3,
        include_excel: bool = False,
        include_md: bool = False
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Xử lý toàn bộ dataset_chunking tập trung hoàn hảo vào 2 luồng: DOCX và PDF.
        (Excel và Markdown tạm thời tắt mặc định theo chỉ đạo của người dùng).
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

        print("=" * 80)
        print(f"[*] KHỞI CHẠY DLA CHUNKING HOÀN HẢO 2 LUỒNG CỐT LÕI (PDF & DOCX): {dataset_dir.name}")
        print(f"    - Luồng 1 (PDF) : {len(pdf_files)} tệp (Scan RapidOCR + Digital)")
        print(f"    - Luồng 2 (DOCX): {len(docx_files)} tệp (Layout AST + Anti-Fragmentation)")
        if include_excel:
            print(f"    - Luồng 3 (Excel): {len(excel_files)} tệp")
        if include_md:
            print(f"    - Luồng 4 (MD)   : {len(md_files)} tệp")
        print("=" * 80)

        all_chunks: List[Dict[str, Any]] = []
        chunks_by_fmt: Dict[str, List[Dict[str, Any]]] = {
            "docx": [],
            "pdf": []
        }
        category_stats: Dict[str, Any] = {}

        # 1. Luồng DOCX
        if docx_files:
            print(f" -> [1/2] Đang xử lý {len(docx_files)} tệp Word DOCX...")
            t0 = time.time()
            for f in docx_files:
                chunks = self.chunk_docx(f)
                chunks_by_fmt["docx"].extend(chunks)
            dur = time.time() - t0
            all_chunks.extend(chunks_by_fmt["docx"])
            category_stats["docx"] = {
                "files": len(docx_files),
                "chunks": len(chunks_by_fmt["docx"]),
                "seconds": round(dur, 2)
            }
            print(f"    + Hoàn tất DOCX: {len(chunks_by_fmt['docx'])} chunks trong {dur:.2f}s")

        # 2. Luồng PDF
        if pdf_files:
            print(f" -> [2/2] Đang xử lý {len(pdf_files)} tệp PDF (tối đa {max_pdf_pages} trang/file)...")
            t0 = time.time()
            for idx, f in enumerate(pdf_files, start=1):
                chunks = self.chunk_pdf(f, max_pages=max_pdf_pages)
                chunks_by_fmt["pdf"].extend(chunks)
                if idx % 15 == 0 or idx == len(pdf_files):
                    print(f"    ... Đã xử lý {idx}/{len(pdf_files)} PDFs ({len(chunks_by_fmt['pdf'])} chunks)")
            dur = time.time() - t0
            all_chunks.extend(chunks_by_fmt["pdf"])
            category_stats["pdf"] = {
                "files": len(pdf_files),
                "chunks": len(chunks_by_fmt["pdf"]),
                "seconds": round(dur, 2)
            }
            print(f"    + Hoàn tất PDF: {len(chunks_by_fmt['pdf'])} chunks trong {dur:.2f}s")

        # 3. Excel (nếu bật)
        if include_excel and excel_files:
            chunks_by_fmt["excel"] = []
            for f in excel_files:
                chunks_by_fmt["excel"].extend(self.chunk_excel(f))
            all_chunks.extend(chunks_by_fmt["excel"])
            category_stats["excel"] = {"files": len(excel_files), "chunks": len(chunks_by_fmt["excel"])}

        # 4. Markdown (nếu bật)
        if include_md and md_files:
            chunks_by_fmt["md"] = []
            for f in md_files:
                with open(f, "r", encoding="utf-8") as mdf:
                    chunks_by_fmt["md"].extend(self.chunk_document(mdf.read(), source_name=f.name))
            all_chunks.extend(chunks_by_fmt["md"])
            category_stats["md"] = {"files": len(md_files), "chunks": len(chunks_by_fmt["md"])}

        total_time = time.time() - start_time

        # Thống kê phân loại & phân phối độ dài từ
        type_dist: Dict[str, int] = {}
        lengths: List[int] = []
        words_list: List[int] = []
        under_80 = 0
        under_120 = 0
        golden_range = 0
        over_450 = 0

        for c in all_chunks:
            ctype = c.get("chunk_type", "other")
            type_dist[ctype] = type_dist.get(ctype, 0) + 1
            l_chars = c.get("char_length", len(c.get("content", "")))
            w_count = c.get("word_count", count_words(c.get("content", "")))
            lengths.append(l_chars)
            words_list.append(w_count)

            if w_count < 80:
                under_80 += 1
            if w_count < 120:
                under_120 += 1
            if 150 <= w_count <= 450:
                golden_range += 1
            if w_count > 450:
                over_450 += 1

        lengths.sort()
        words_list.sort()
        total_count = max(len(all_chunks), 1)

        summary_stats = {
            "total_files": sum(s["files"] for s in category_stats.values()),
            "total_chunks": len(all_chunks),
            "duration_seconds": round(total_time, 2),
            "speed_chunks_per_sec": round(len(all_chunks) / max(total_time, 0.001), 1),
            "category_stats": category_stats,
            "type_distribution": type_dist,
            "quality_metrics": {
                "under_80_words_pct": round(under_80 / total_count * 100, 2),
                "under_120_words_pct": round(under_120 / total_count * 100, 2),
                "golden_range_150_450_pct": round(golden_range / total_count * 100, 2),
                "over_450_words_pct": round(over_450 / total_count * 100, 2),
            },
            "word_stats": {
                "min": words_list[0] if words_list else 0,
                "median": words_list[len(words_list) // 2] if words_list else 0,
                "avg": round(sum(words_list) / len(words_list), 1) if words_list else 0,
                "max": words_list[-1] if words_list else 0
            },
            "char_stats": {
                "min": lengths[0] if lengths else 0,
                "median": lengths[len(lengths) // 2] if lengths else 0,
                "avg": round(sum(lengths) / len(lengths), 1) if lengths else 0,
                "max": lengths[-1] if lengths else 0
            }
        }

        # Lưu trữ kết quả nếu chỉ định output_dir
        if output_dir:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            (out_p / "chunks_by_format").mkdir(parents=True, exist_ok=True)

            # Ghi all_chunks.json
            with open(out_p / "all_chunks.json", "w", encoding="utf-8") as f:
                json.dump(all_chunks, f, ensure_ascii=False, indent=2)

            # Ghi all_chunks.jsonl
            with open(out_p / "all_chunks.jsonl", "w", encoding="utf-8") as f:
                for c in all_chunks:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")

            # Ghi chunks_by_format
            for fmt, c_list in chunks_by_fmt.items():
                if c_list:
                    with open(out_p / "chunks_by_format" / f"{fmt}_chunks.json", "w", encoding="utf-8") as f:
                        json.dump(c_list, f, ensure_ascii=False, indent=2)

            # Ghi thống kê
            with open(out_p / "chunking_summary_stats.json", "w", encoding="utf-8") as f:
                json.dump(summary_stats, f, ensure_ascii=False, indent=2)

            print(f"[Lưu trữ] Đã lưu toàn bộ {len(all_chunks)} chunks vào: {out_p}")

        return all_chunks, summary_stats


DLAChunker = LayoutAwareChunker
