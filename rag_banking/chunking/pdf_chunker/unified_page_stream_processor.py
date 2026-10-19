"""
Module Unified Page Stream Processor (Cross-Page Stitching & Boundary Buffer)
==============================================================================
Bộ xử lý luồng trang thống nhất cho tài liệu PDF ngân hàng:
1. Duy trì bộ đệm nhìn trước 1 trang (1-page lookahead buffer) sau khi loại bỏ 6.5% lề đầu/cuối.
2. Hàn gắn bảng biểu xuyên trang (Cross-Page Table Stitching):
   - Bảng cuối trang N (trong 15% đáy active area) nối bảng đầu trang N+1 (trong 15% đỉnh active area).
   - Dung sai bề ngang ΔW <= 5%.
   - Ghép ảnh thị giác, gửi VLM trích xuất 1 lần duy nhất, bảo toàn phả hệ Breadcrumb.
3. Hàn gắn đoạn văn xuyên trang (Cross-Page Paragraph Stitching):
   - Đoạn văn cuối trang N không kết thúc bằng dấu chấm câu (. : ; ! ?).
   - Đoạn văn đầu trang N+1 có thụt lề khớp ΔX <= 2%.
   - Nối trực tiếp trước khi phân bổ ngân sách âm tiết.
4. Hợp đồng 7 trường (7-Field Schema Contract) và 0% REGEX LAW.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import pymupdf as fitz

from ..unified_chunk.syllable_budget_manager import build_7_field_chunk
from .vlm_table_extractor import (
    VLMTableExtractor,
    crop_pdf_table_image,
    prepare_cross_page_table_image
)
from .table_formatter import TableFormatter


class LayoutBlockType(str, Enum):
    HEADING = "heading"
    BODY = "body"
    TABLE = "table"


@dataclass
class LayoutBlock:
    """Đại diện cho 1 khối bố cục nội dung trên trang."""
    block_type: LayoutBlockType
    page_num: int
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    text: str = ""
    lines: list[dict] = field(default_factory=list)
    image_crop: object = None
    predicted_level: str = "body"


class BreadcrumbTracker:
    """Bộ theo dõi phả hệ tiêu đề phân cấp xuyên suốt các trang (0% Regex)."""

    def __init__(self, doc_title: str = "Tài liệu ngân hàng"):
        self.doc_title = doc_title
        self.h1: Optional[str] = None
        self.h2: Optional[str] = None
        self.h3: Optional[str] = None

    def update(self, level: str, text: str) -> None:
        clean_text = text.strip()
        if not clean_text:
            return

        lvl = level.lower()
        if lvl in ("title", "doc_title"):
            self.doc_title = clean_text
            self.h1 = None
            self.h2 = None
            self.h3 = None
        elif lvl == "h1":
            self.h1 = clean_text
            self.h2 = None
            self.h3 = None
        elif lvl == "h2":
            self.h2 = clean_text
            self.h3 = None
        elif lvl == "h3":
            self.h3 = clean_text

    def get_breadcrumb(self) -> str:
        parts = [self.doc_title]
        if self.h1:
            parts.append(self.h1)
        if self.h2:
            parts.append(self.h2)
        if self.h3:
            parts.append(self.h3)
        return " > ".join(parts)

    def get_section_title(self) -> str:
        if self.h3:
            return self.h3
        if self.h2:
            return self.h2
        if self.h1:
            return self.h1
        return self.doc_title


class UnifiedPageStreamProcessor:
    """
    Bộ xử lý luồng trang thống nhất tích hợp bộ đệm nhìn trước 1 trang.
    """

    def __init__(
        self,
        vlm_extractor: Optional[VLMTableExtractor] = None,
        table_formatter: Optional[TableFormatter] = None,
        ocr_extractor: object = None,
        header_margin_ratio: float = 0.065,
        footer_margin_ratio: float = 0.065,
        **kwargs
    ):
        self.vlm_extractor = vlm_extractor or VLMTableExtractor()
        self.table_formatter = table_formatter or TableFormatter()
        self.ocr_extractor = ocr_extractor
        self.header_margin_ratio = header_margin_ratio
        self.footer_margin_ratio = footer_margin_ratio

    def filter_active_page_blocks(
        self,
        blocks: list[LayoutBlock],
        page_h: float
    ) -> list[LayoutBlock]:
        """Loại bỏ các khối rơi vào vùng biên mép trang 6.5% (Header/Footer)."""
        active_top = page_h * self.header_margin_ratio
        active_bottom = page_h * (1.0 - self.footer_margin_ratio)

        valid_blocks: list[LayoutBlock] = []
        for b in blocks:
            _, y1, _, y2 = b.bbox
            # Nếu khối nằm hoàn toàn ngoài vùng active thì bỏ qua
            if y2 <= active_top or y1 >= active_bottom:
                continue
            valid_blocks.append(b)
        return valid_blocks

    def try_stitch_tables(
        self,
        block_a: LayoutBlock,
        block_b: LayoutBlock,
        page_w: float,
        page_h: float
    ) -> bool:
        """
        Kiểm tra điều kiện hàn gắn bảng biểu xuyên trang:
        1. Cả 2 đều là TABLE.
        2. block_a nằm trong 15% đáy active area trang N.
        3. block_b nằm trong 15% đỉnh active area trang N+1.
        4. Chênh lệch bề rộng ΔW <= 5% bề ngang trang.
        """
        if block_a.block_type != LayoutBlockType.TABLE or block_b.block_type != LayoutBlockType.TABLE:
            return False

        active_top = page_h * self.header_margin_ratio
        active_bottom = page_h * (1.0 - self.footer_margin_ratio)
        active_h = active_bottom - active_top
        if active_h <= 0:
            return False

        # Vị trí chuẩn hóa trong active area
        y2_a_rel = (block_a.bbox[3] - active_top) / active_h
        y1_b_rel = (block_b.bbox[1] - active_top) / active_h

        is_near_bottom_a = y2_a_rel >= 0.85
        is_near_top_b = y1_b_rel <= 0.15

        w_a = block_a.bbox[2] - block_a.bbox[0]
        w_b = block_b.bbox[2] - block_b.bbox[0]
        delta_w = abs(w_a - w_b) / max(page_w, 1.0)

        return is_near_bottom_a and is_near_top_b and (delta_w <= 0.05)

    def try_stitch_paragraphs(
        self,
        block_a: LayoutBlock,
        block_b: LayoutBlock,
        page_w: float
    ) -> bool:
        """
        Kiểm tra điều kiện hàn gắn đoạn văn xuyên trang:
        1. Cả 2 đều là BODY.
        2. block_a không kết thúc bằng dấu chấm câu kết thúc (. : ; ! ?).
        3. Khớp thụt lề trái ΔX <= 2% bề ngang trang.
        """
        if block_a.block_type != LayoutBlockType.BODY or block_b.block_type != LayoutBlockType.BODY:
            return False

        text_a = block_a.text.rstrip()
        if not text_a:
            return False

        last_char = text_a[-1]
        terminal_punctuations = {".", ":", ";", "!", "?"}
        if last_char in terminal_punctuations:
            return False

        delta_x = abs(block_a.bbox[0] - block_b.bbox[0]) / max(page_w, 1.0)
        return delta_x <= 0.02

    def process_page_stream(
        self,
        pages_blocks: list[list[LayoutBlock]],
        source: str,
        page_dimensions: list[tuple[float, float]],
        tracker: Optional[BreadcrumbTracker] = None,
        target_syllable_budget: int = 350
    ) -> list[dict]:
        """
        Xử lý toàn bộ luồng trang qua bộ đệm nhìn trước (Lookahead Buffer)
        và trả về danh sách các chunk đạt chuẩn 7 trường.
        """
        if tracker is None:
            doc_name = source.rsplit(".", 1)[0] if "." in source else source
            tracker = BreadcrumbTracker(doc_title=doc_name)

        all_stitched_blocks: list[LayoutBlock] = []
        num_pages = len(pages_blocks)

        # 1. Lọc vùng lề cho từng trang
        clean_pages: list[list[LayoutBlock]] = []
        for p_idx, page in enumerate(pages_blocks):
            pw, ph = page_dimensions[p_idx] if p_idx < len(page_dimensions) else (595.0, 842.0)
            clean_pages.append(self.filter_active_page_blocks(page, ph))

        # 2. Chạy bộ đệm nhìn trước 1 trang (Page-Boundary Lookahead Buffer)
        p_idx = 0
        while p_idx < num_pages:
            curr_blocks = clean_pages[p_idx]
            pw, ph = page_dimensions[p_idx] if p_idx < len(page_dimensions) else (595.0, 842.0)

            # Kiểm tra xem có trang kế tiếp N+1 không
            if p_idx + 1 < num_pages and curr_blocks and clean_pages[p_idx + 1]:
                next_blocks = clean_pages[p_idx + 1]
                last_block = curr_blocks[-1]
                first_block = next_blocks[0]

                # A. Thử hàn gắn Bảng biểu xuyên trang
                if self.try_stitch_tables(last_block, first_block, pw, ph):
                    stitched_crop = prepare_cross_page_table_image([
                        last_block.image_crop,
                        first_block.image_crop
                    ]) if (last_block.image_crop and first_block.image_crop) else None

                    stitched_table = LayoutBlock(
                        block_type=LayoutBlockType.TABLE,
                        page_num=last_block.page_num,
                        bbox=(
                            min(last_block.bbox[0], first_block.bbox[0]),
                            last_block.bbox[1],
                            max(last_block.bbox[2], first_block.bbox[2]),
                            first_block.bbox[3]
                        ),
                        text=f"{last_block.text}\n{first_block.text}".strip(),
                        lines=last_block.lines + first_block.lines,
                        image_crop=stitched_crop,
                        predicted_level="table"
                    )
                    # Thêm các block trước của trang N
                    all_stitched_blocks.extend(curr_blocks[:-1])
                    all_stitched_blocks.append(stitched_table)
                    # Bỏ block đầu tiên của trang N+1 vì đã được hàn gắn
                    clean_pages[p_idx + 1] = next_blocks[1:]
                    p_idx += 1
                    continue

                # B. Thử hàn gắn Đoạn văn xuyên trang
                elif self.try_stitch_paragraphs(last_block, first_block, pw):
                    stitched_body = LayoutBlock(
                        block_type=LayoutBlockType.BODY,
                        page_num=last_block.page_num,
                        bbox=(
                            min(last_block.bbox[0], first_block.bbox[0]),
                            last_block.bbox[1],
                            max(last_block.bbox[2], first_block.bbox[2]),
                            first_block.bbox[3]
                        ),
                        text=f"{last_block.text} {first_block.text}".strip(),
                        lines=last_block.lines + first_block.lines,
                        image_crop=None,
                        predicted_level="body"
                    )
                    all_stitched_blocks.extend(curr_blocks[:-1])
                    all_stitched_blocks.append(stitched_body)
                    clean_pages[p_idx + 1] = next_blocks[1:]
                    p_idx += 1
                    continue

            # Nếu không có hàn gắn xuyên trang
            all_stitched_blocks.extend(curr_blocks)
            p_idx += 1

        # 3. Chuyển đổi toàn bộ danh sách khối đã hàn gắn thành Chunks chuẩn 7 trường
        chunks: list[dict] = []
        chunk_counter = 0

        # Gom các khối BODY liên tiếp vào một bộ đệm văn bản
        body_buffer: list[LayoutBlock] = []

        def flush_body_buffer() -> None:
            nonlocal chunk_counter
            if not body_buffer:
                return

            full_text = "\n\n".join(b.text for b in body_buffer if b.text.strip()).strip()
            locations = [{"page": b.page_num, "bbox": list(b.bbox)} for b in body_buffer if b.page_num is not None]
            body_buffer.clear()
            if not full_text:
                return

            bc = tracker.get_breadcrumb()
            st = tracker.get_section_title()
            p_id = f"{source}_sec_{st.lower().replace(' ', '_')}"

            chunk_obj = build_7_field_chunk(
                chunk_id=f"{source}_c{chunk_counter}",
                source=source,
                chunk_type="pdf_text",
                breadcrumb=bc,
                section_title=st,
                content=full_text,
                original_text=full_text,
                parent_id=p_id,
                location=locations
            )
            chunks.append(chunk_obj)
            chunk_counter += 1

        for block in all_stitched_blocks:
            if block.block_type == LayoutBlockType.HEADING:
                flush_body_buffer()
                tracker.update(block.predicted_level, block.text)

            elif block.block_type == LayoutBlockType.TABLE:
                flush_body_buffer()
                bc = tracker.get_breadcrumb()
                st = tracker.get_section_title()

                # Gọi VLMTableExtractor
                if block.image_crop is not None:
                    vlm_res = self.vlm_extractor.extract_table(block.image_crop, raw_fallback_text=block.text)
                else:
                    vlm_res = self.vlm_extractor._deterministic_fallback(
                        image=None,
                        raw_fallback_text=block.text
                    )

                # Format thành chunks bằng TableFormatter
                table_chunks = self.table_formatter.format_vlm_result_to_chunks(
                    vlm_result=vlm_res,
                    source=source,
                    chunk_id_prefix=f"{source}_tbl",
                    breadcrumb=bc,
                    section_title=st,
                    chunk_type="pdf_table",
                    start_index=chunk_counter,
                    page_num=block.page_num,
                    bbox=list(block.bbox) if block.bbox else None
                )
                chunks.extend(table_chunks)
                chunk_counter += len(table_chunks)

            elif block.block_type == LayoutBlockType.BODY:
                body_buffer.append(block)
                # Nếu buffer gom được lượng từ tiệm cận ngân sách (>= 350 từ) thì flush
                approx_words = sum(len(b.text.split()) for b in body_buffer)
                if approx_words >= target_syllable_budget:
                    flush_body_buffer()

        flush_body_buffer()

        # Xác thực Hợp đồng trường cốt lõi
        for c in chunks:
            self._validate_7_field_schema(c)

        return chunks

    def process_pdf_document(
        self,
        pdf_path: str | Path,
        max_pages: Optional[int] = None
    ) -> tuple[list[dict], dict]:
        """
        Phân đoạn trọn vẹn một tệp PDF ngân hàng qua bộ đệm nhìn trước 1 trang
        và hàn gắn xuyên trang (Cross-Page Stitching).
        """
        path = Path(pdf_path)
        if not path.exists():
            return [], {"total_pages": 0, "total_chunks": 0, "elapsed_seconds": 0.0}

        t0 = time.time()
        try:
            doc = fitz.open(str(path))
        except Exception as e:
            return [], {"total_pages": 0, "total_chunks": 0, "error": str(e), "elapsed_seconds": 0.0}

        num_pages = len(doc)
        limit_pages = min(num_pages, max_pages) if max_pages else num_pages

        pages_blocks: list[list[LayoutBlock]] = []
        page_dimensions: list[tuple[float, float]] = []

        doc_title = path.stem.replace("_", " ").title()
        tracker = BreadcrumbTracker(doc_title=doc_title)

        for p_idx in range(limit_pages):
            page = doc[p_idx]
            pw, ph = page.rect.width, page.rect.height
            page_dimensions.append((pw, ph))

            page_blocks: list[LayoutBlock] = []

            # 1. Trích xuất bảng biểu
            try:
                tables = page.find_tables()
            except Exception:
                tables = []

            table_rects = []
            for tbl in (tables or []):
                t_bbox = tbl.bbox
                table_rects.append(t_bbox)
                page_blocks.append(LayoutBlock(
                    block_type=LayoutBlockType.TABLE,
                    page_num=p_idx + 1,
                    bbox=t_bbox,
                    text="",
                    image_crop=None,
                    predicted_level="table"
                ))

            # 2. Trích xuất text blocks bằng DLA Typography (Cỡ chữ, In đậm, Hình học không gian, 0% đọc từ)
            try:
                page_dict = page.get_text("dict")
                blocks_data = page_dict.get("blocks", [])
            except Exception:
                blocks_data = []

            # Đo lường cỡ chữ trung vị của trang (median font size)
            page_font_sizes: list[float] = []
            for b in blocks_data:
                if b.get("type") == 0:
                    for l in b.get("lines", []):
                        for s in l.get("spans", []):
                            stxt = s.get("text", "").strip()
                            if stxt:
                                page_font_sizes.extend([s.get("size", 10.0)] * min(len(stxt), 30))
            page_font_sizes.sort()
            median_size = page_font_sizes[len(page_font_sizes) // 2] if page_font_sizes else 11.0

            for b in blocks_data:
                if b.get("type") != 0:
                    continue
                bx0, by0, bx1, by1 = b.get("bbox", (0, 0, 0, 0))
                # Bỏ qua nếu khối nằm gọn trong vùng bảng
                if any((bx0 >= t[0] - 5 and by0 >= t[1] - 5 and bx1 <= t[2] + 5 and by1 <= t[3] + 5) for t in table_rects):
                    continue

                lines = b.get("lines", [])
                if not lines:
                    continue

                block_lines_text = []
                span_sizes = []
                span_bolds = []
                for l in lines:
                    l_spans = l.get("spans", [])
                    ltxt = "".join(s.get("text", "") for s in l_spans).strip()
                    if ltxt:
                        block_lines_text.append(ltxt)
                        for s in l_spans:
                            if s.get("text", "").strip():
                                span_sizes.append(s.get("size", 10.0))
                                span_bolds.append(bool((s.get("flags", 0) & 2) or "bold" in s.get("font", "").lower()))

                if not block_lines_text:
                    continue

                btext_clean = "\n".join(block_lines_text).strip()
                first_line = block_lines_text[0].strip()
                avg_size = (sum(span_sizes) / len(span_sizes)) if span_sizes else median_size
                is_bold = (sum(1 for bold in span_bolds if bold) > len(span_bolds) / 2) if span_bolds else False

                # Phân biệt Heading vs Body HOÀN TOÀN DỰA TRÊN TYPOGRAPHY & HÌNH HỌC (0% RULE-BASED ĐỌC TỪ)
                from .pdf_chunker import _is_heading_line
                is_heading = _is_heading_line(
                    line=first_line,
                    font_size=avg_size,
                    median_font_size=median_size,
                    is_bold=is_bold
                )

                b_type = LayoutBlockType.HEADING if is_heading else LayoutBlockType.BODY
                page_blocks.append(LayoutBlock(
                    block_type=b_type,
                    page_num=p_idx + 1,
                    bbox=(bx0, by0, bx1, by1),
                    text=btext_clean,
                    predicted_level="h2" if is_heading else "body"
                ))

            pages_blocks.append(page_blocks)

        doc.close()

        chunks = self.process_page_stream(
            pages_blocks=pages_blocks,
            source=path.name,
            page_dimensions=page_dimensions,
            tracker=tracker
        )

        elapsed = round(time.time() - t0, 3)
        stats = {
            "total_pages": limit_pages,
            "total_chunks": len(chunks),
            "elapsed_seconds": elapsed
        }
        return chunks, stats

    @staticmethod
    def _validate_7_field_schema(chunk: dict) -> None:
        """Đảm bảo mọi chunk chứa đúng và đủ các trường bắt buộc, loại bỏ rác legacy."""
        required_keys = {"chunk_id", "source", "chunk_type", "breadcrumb", "section_title", "content", "word_count"}
        for k in required_keys:
            if k not in chunk:
                chunk[k] = "" if k != "word_count" else 0

        # Loại bỏ các trường rác không thuộc chuẩn
        forbidden_keys = {"ref", "legal_ref", "raw_content", "char_length", "format"}
        for f in forbidden_keys:
            if f in chunk:
                del chunk[f]


__all__ = [
    "LayoutBlockType",
    "LayoutBlock",
    "BreadcrumbTracker",
    "UnifiedPageStreamProcessor"
]
