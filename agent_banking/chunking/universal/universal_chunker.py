# universal/universal_chunker.py
# Universal Multimodal DOCX Chunker Orchestrator
# Complies with GEMINI.md: 0% regex, no typing library, zero ternary, zero __init__.py

import os
import sys
import io
import hashlib
from pathlib import Path

_CURRENT_DIR = Path(__file__).resolve().parent
_CHUNKING_DIR = _CURRENT_DIR.parent
if str(_CHUNKING_DIR) not in sys.path:
    sys.path.insert(0, str(_CHUNKING_DIR))

# Import M1 foundation modules
try:
    from universal.renderer import render_docx_pages, get_docx_page_count
    from universal.vision_cache import PageVisionCache
    from universal.vision_client import QwenVisionClient
except ImportError:
    try:
        from renderer import render_docx_pages, get_docx_page_count
        from vision_cache import PageVisionCache
        from vision_client import QwenVisionClient
    except ImportError:
        render_docx_pages = None
        PageVisionCache = None
        QwenVisionClient = None

# Import M2 schema assembler
try:
    from universal.schema_assembler import (
        build_universal_chunk,
        link_chunk_sequence,
        infer_business_metadata,
        generate_chunk_id,
        REQUIRED_UNIVERSAL_SCHEMA_KEYS,
        VALID_CHUNK_TYPES,
    )
except ImportError:
    try:
        from schema_assembler import (
            build_universal_chunk,
            link_chunk_sequence,
            infer_business_metadata,
            generate_chunk_id,
            REQUIRED_UNIVERSAL_SCHEMA_KEYS,
            VALID_CHUNK_TYPES,
        )
    except ImportError:
        build_universal_chunk = None
        link_chunk_sequence = None
        infer_business_metadata = None
        generate_chunk_id = None
        REQUIRED_UNIVERSAL_SCHEMA_KEYS = None
        VALID_CHUNK_TYPES = None


def _parse_markdown_table_to_2d_list(md_text):
    """
    Parses a markdown table string into a 2D list of row cells using native string methods.
    Strictly adheres to 0% Regex Law and Zero Ternary Law.
    """
    if not md_text:
        return None

    lines = str(md_text).strip().split("\n")
    rows = []
    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            continue
        if clean_line.startswith("|"):
            is_separator = True
            chars_only = clean_line.replace("|", "").replace("-", "").replace(":", "").replace(" ", "")
            if len(chars_only) > 0:
                is_separator = False

            if not is_separator:
                raw_cells = clean_line.split("|")
                cells = []
                for idx in range(1, len(raw_cells) - 1):
                    cells.append(raw_cells[idx].strip())
                if cells:
                    rows.append(cells)

    if rows:
        return rows
    else:
        return None


class UniversalDocxChunker:
    """
    Universal Multimodal DOCX Chunker Orchestrator.

    Pipeline:
    1. In-RAM rasterization: PyMuPDF rasterizes DOCX pages to 150 DPI PNG images without Word/LibreOffice.
    2. Deterministic SHA-256 Hashing: Generates unique hash per rendered page.
    3. 2-Tier SQLite WAL Cache: Microsecond L1 memory + Sub-millisecond L2 SQLite WAL cache (< 1ms SLA).
    4. Vision Layout Extraction: Qwen2.5-VL-7B-Instruct extracts headings, paragraphs, tables, images.
    5. Pure Domain-Agnostic Chunk Assembly: Converts layout elements into Target Universal Schema chunks.
    6. Bidirectional Sequence Linking: Populates prev_chunk_id and next_chunk_id across all chunks.

    Complies with GEMINI.md:
    - 0% regex (string methods only)
    - Zero ternary (explicit if/else only)
    - No typing library
    - Zero __init__.py
    """

    def __init__(
        self,
        cache_db_path="cache/page_vision_cache.db",
        base_url=None,
        api_key=None,
        model=None,
        dpi=150,
        max_chunk_words=350,
        cache=None,
        vision_client=None,
    ):
        """
        Initialize UniversalDocxChunker.
        """
        self.dpi = dpi
        self.max_chunk_words = max_chunk_words

        if cache is not None:
            self.cache = cache
            self._owns_cache = False
        else:
            if PageVisionCache is not None:
                self.cache = PageVisionCache(db_path=cache_db_path)
            else:
                self.cache = None
            self._owns_cache = True

        self._explicit_vision_client = False
        if vision_client is not None:
            self.vision_client = vision_client
            self._explicit_vision_client = True
        else:
            if QwenVisionClient is not None and (base_url is not None or api_key is not None):
                self.vision_client = QwenVisionClient(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                )
            else:
                self.vision_client = None

    def chunk_docx(self, docx_path):
        """
        Backward-compatible alias for chunk_document. Follows Zero Ternary Law.
        """
        return self.chunk_document(docx_path=docx_path)

    def chunk_document(self, docx_path=None, docx_bytes=None, source_name=None):
        """
        Chunks a DOCX document into Target Universal Schema chunk dictionaries.
        """
        # 1. Resolve and Validate Document Input Source
        docx_source, resolved_source_name = self._resolve_source(
            docx_path=docx_path,
            docx_bytes=docx_bytes,
            source_name=source_name,
        )

        # 2. In-RAM Rasterization via PyMuPDF
        if render_docx_pages is None:
            raise RuntimeError("Renderer module render_docx_pages is not available.")

        rendered_pages = render_docx_pages(docx_source, dpi=self.dpi)
        if not rendered_pages:
            return []

        # 3. Dynamic Domain-Agnostic Business Metadata Inference
        business_metadata = infer_business_metadata(resolved_source_name)
        doc_stem = business_metadata.get("document_name", "doc")
        if not doc_stem:
            doc_stem = "doc"

        # 4. Multi-Page Vision Layout Extraction
        page_layouts = []
        has_vision_layout = False

        for page in rendered_pages:
            p_idx = page.get("page_index", 0)
            p_sha256 = page.get("sha256", "")
            img_bytes = page.get("image_bytes")
            w = page.get("width", 0)
            h = page.get("height", 0)

            layout_dict = None
            if self.cache is not None:
                layout_dict = self.cache.get(p_sha256)

            if layout_dict is None:
                if self.vision_client is not None:
                    try:
                        layout_dict = self.vision_client.analyze_page_layout(img_bytes)
                    except Exception:
                        layout_dict = None
                else:
                    layout_dict = None

                if self.cache is not None and layout_dict:
                    self.cache.put(p_sha256, layout_dict, width=w, height=h)

            if layout_dict is not None:
                has_elements = False
                if isinstance(layout_dict, dict):
                    if layout_dict.get("elements") or layout_dict.get("headings"):
                        has_elements = True
                if has_elements:
                    has_vision_layout = True
                    page_layouts.append((p_idx, layout_dict))
                else:
                    page_layouts.append((p_idx, {"headings": [], "elements": []}))
            else:
                page_layouts.append((p_idx, {"headings": [], "elements": []}))

        # 5. Assemble Structural Elements into Universal Chunks
        if not has_vision_layout and not self._explicit_vision_client:
            raw_chunks = self._extract_docx_chunks(
                docx_source=docx_source,
                source_name=resolved_source_name,
                doc_stem=doc_stem,
                business_metadata=business_metadata,
                rendered_pages=rendered_pages,
            )
        else:
            raw_chunks = self._assemble_elements_to_chunks(
                page_layouts=page_layouts,
                source_name=resolved_source_name,
                doc_stem=doc_stem,
                business_metadata=business_metadata,
            )

        # 6. Establish Bidirectional Sequence Linking
        linked_chunks = link_chunk_sequence(raw_chunks)
        return linked_chunks

    def _resolve_source(self, docx_path, docx_bytes, source_name):
        """
        Normalizes and validates input parameters for both file path and in-memory bytes.
        Follows Zero Ternary Law.
        """
        if isinstance(docx_path, (bytes, bytearray)):
            docx_bytes = docx_path
            docx_path = None

        if docx_bytes is not None:
            if not isinstance(docx_bytes, (bytes, bytearray)):
                raise TypeError("docx_bytes must be bytes or bytearray")
            if len(docx_bytes) == 0:
                raise ValueError("docx_bytes stream is empty")

            docx_source = docx_bytes
            if docx_path is not None:
                resolved_source = str(docx_path)
            else:
                if source_name is not None:
                    resolved_source = str(source_name)
                else:
                    resolved_source = "in_memory_document.docx"

            return docx_source, resolved_source

        if docx_path is not None:
            if hasattr(docx_path, "__fspath__"):
                path_str = str(docx_path)
            elif isinstance(docx_path, str):
                path_str = docx_path
            else:
                raise TypeError("docx_path must be a string or Path object")

            if not path_str.strip():
                raise ValueError("docx_path cannot be an empty string")

            if not os.path.exists(path_str):
                raise FileNotFoundError("DOCX file not found: " + path_str)

            docx_source = path_str
            resolved_source = path_str
            return docx_source, resolved_source

        raise ValueError("Either docx_path or docx_bytes must be provided and non-empty")

    def _extract_docx_chunks(
        self,
        docx_source,
        source_name,
        doc_stem,
        business_metadata,
        rendered_pages,
    ):
        """
        High-fidelity fallback extractor using python-docx when vision API is not active or empty.
        Converts docx blocks (headings, paragraphs, tables, captions) into Target Universal Schema chunks.
        """
        import docx

        if isinstance(docx_source, (bytes, bytearray)):
            doc_obj = docx.Document(io.BytesIO(docx_source))
        else:
            doc_obj = docx.Document(str(docx_source))

        raw_elements = []
        current_h1 = ""
        current_h2 = ""
        current_h3 = ""

        for block in doc_obj.element.body:
            tag_name = block.tag
            if tag_name.endswith("p"):
                p = docx.text.paragraph.Paragraph(block, doc_obj)
                txt = p.text.strip()
                if not txt:
                    continue

                style_name = ""
                if p.style is not None and p.style.name is not None:
                    style_name = p.style.name.lower()

                is_heading = False
                heading_level = 0
                if "heading 1" in style_name:
                    is_heading = True
                    heading_level = 1
                    current_h1 = txt
                    current_h2 = ""
                    current_h3 = ""
                else:
                    if "heading 2" in style_name:
                        is_heading = True
                        heading_level = 2
                        current_h2 = txt
                        current_h3 = ""
                    else:
                        if "heading 3" in style_name:
                            is_heading = True
                            heading_level = 3
                            current_h3 = txt

                bc_parts = []
                if current_h1 != "":
                    bc_parts.append(current_h1)
                if current_h2 != "":
                    bc_parts.append(current_h2)
                if current_h3 != "":
                    bc_parts.append(current_h3)

                if len(bc_parts) > 0:
                    breadcrumb = " > ".join(bc_parts)
                else:
                    breadcrumb = source_name

                sec_t = ""
                if current_h3 != "":
                    sec_t = current_h3
                else:
                    if current_h2 != "":
                        sec_t = current_h2
                    else:
                        if current_h1 != "":
                            sec_t = current_h1
                        else:
                            sec_t = source_name

                if is_heading:
                    raw_elements.append({
                        "type": "heading",
                        "level": heading_level,
                        "text": txt,
                        "breadcrumb": breadcrumb,
                        "section_title": txt,
                    })
                else:
                    is_caption = False
                    if "caption" in style_name or txt.startswith("Hình ") or txt.startswith("Figure "):
                        is_caption = True

                    if is_caption:
                        raw_elements.append({
                            "type": "caption",
                            "text": txt,
                            "breadcrumb": breadcrumb,
                            "section_title": sec_t,
                        })
                    else:
                        raw_elements.append({
                            "type": "paragraph",
                            "text": txt,
                            "breadcrumb": breadcrumb,
                            "section_title": sec_t,
                        })

            else:
                if tag_name.endswith("tbl"):
                    t = docx.table.Table(block, doc_obj)
                    headers = []
                    rows = []
                    for r_idx, row in enumerate(t.rows):
                        cell_texts = []
                        for cell in row.cells:
                            cell_texts.append(cell.text.strip())
                        if r_idx == 0:
                            headers = cell_texts
                        else:
                            rows.append(cell_texts)

                    md_lines = []
                    if len(headers) > 0:
                        md_lines.append("| " + " | ".join(headers) + " |")
                        md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                    for r in rows:
                        md_lines.append("| " + " | ".join(r) + " |")
                    table_md = "\n".join(md_lines)

                    bc_parts = []
                    if current_h1 != "":
                        bc_parts.append(current_h1)
                    if current_h2 != "":
                        bc_parts.append(current_h2)
                    if current_h3 != "":
                        bc_parts.append(current_h3)

                    if len(bc_parts) > 0:
                        breadcrumb = " > ".join(bc_parts)
                    else:
                        breadcrumb = source_name

                    sec_t = ""
                    if current_h3 != "":
                        sec_t = current_h3
                    else:
                        if current_h2 != "":
                            sec_t = current_h2
                        else:
                            if current_h1 != "":
                                sec_t = current_h1
                            else:
                                sec_t = source_name

                    raw_elements.append({
                        "type": "table",
                        "text": table_md,
                        "table_data": {"headers": headers, "rows": rows},
                        "breadcrumb": breadcrumb,
                        "section_title": sec_t,
                    })

        page_layouts = []
        num_pages = len(rendered_pages)

        if num_pages > 1:
            try:
                import fitz
            except ImportError:
                fitz = None

            page_texts = []
            if fitz is not None:
                doc_fitz = None
                try:
                    if isinstance(docx_source, (bytes, bytearray)):
                        doc_fitz = fitz.open(stream=docx_source, filetype="docx")
                    else:
                        doc_fitz = fitz.open(str(docx_source))
                    for p in doc_fitz:
                        page_texts.append(p.get_text())
                except Exception:
                    page_texts = []
                finally:
                    if doc_fitz is not None:
                        doc_fitz.close()

            if len(page_texts) == num_pages:
                page_elements_map = {}
                for pi in range(num_pages):
                    page_elements_map[pi] = []

                curr_p = 0
                for el in raw_elements:
                    el_txt = el.get("text", "").strip()
                    if curr_p + 1 < len(page_texts):
                        sample = el_txt[:50].strip()
                        if sample:
                            if sample in page_texts[curr_p + 1]:
                                curr_p += 1
                    page_elements_map[curr_p].append(el)

                for pi in range(num_pages):
                    p_sha = rendered_pages[pi]["sha256"]
                    p_raw = page_elements_map.get(pi, [])
                    cached_h = []
                    cached_el = []
                    for el in p_raw:
                        el_type = el["type"]
                        el_text = el["text"]
                        if el_type == "heading":
                            h_lvl = el.get("level", 1)
                            cached_h.append({"level": h_lvl, "text": el_text})
                            cached_el.append({
                                "type": el_type,
                                "level": h_lvl,
                                "text": el_text,
                            })
                        else:
                            if el_type == "table":
                                cached_el.append({
                                    "type": el_type,
                                    "text": el_text,
                                    "table_data": el.get("table_data"),
                                })
                            else:
                                cached_el.append({
                                    "type": el_type,
                                    "text": el_text,
                                })

                    p_layout = {
                        "headings": cached_h,
                        "elements": cached_el,
                        "page_number": pi + 1,
                        "source_type": "docx",
                    }
                    if self.cache is not None:
                        self.cache.put(p_sha, p_layout)
                    page_layouts.append((pi, p_layout))

        if not page_layouts:
            cached_headings = []
            cached_elements = []
            for el in raw_elements:
                el_type = el["type"]
                el_text = el["text"]
                if el_type == "heading":
                    h_lvl = el.get("level", 1)
                    cached_headings.append({"level": h_lvl, "text": el_text})
                    cached_elements.append({
                        "type": el_type,
                        "level": h_lvl,
                        "text": el_text,
                    })
                else:
                    if el_type == "table":
                        cached_elements.append({
                            "type": el_type,
                            "text": el_text,
                            "table_data": el.get("table_data"),
                        })
                    else:
                        cached_elements.append({
                            "type": el_type,
                            "text": el_text,
                        })

            layout_dict = {
                "headings": cached_headings,
                "elements": cached_elements,
                "page_number": 1,
                "source_type": "docx",
            }
            if self.cache is not None and len(rendered_pages) > 0:
                first_sha = rendered_pages[0]["sha256"]
                self.cache.put(first_sha, layout_dict)

            page_layouts = [(0, layout_dict)]

        return self._assemble_elements_to_chunks(
            page_layouts=page_layouts,
            source_name=source_name,
            doc_stem=doc_stem,
            business_metadata=business_metadata,
        )

    def _assemble_elements_to_chunks(self, page_layouts, source_name, doc_stem, business_metadata):
        """
        Stateful streaming accumulator that processes layout elements across all pages,
        maintains heading hierarchy, isolates tables and images from text swallowing,
        and generates universal chunks.
        """
        raw_chunks = []
        chunk_counter = 0
        section_counter = 0

        # Heading stack: list of tuples: (level, heading_text, section_id, parent_content_val, parent_sec_id)
        heading_stack = []
        text_buffer = []

        def flush_text_buffer():
            nonlocal chunk_counter
            if not text_buffer:
                return

            combined_text = "\n\n".join(text_buffer).strip()
            if not combined_text:
                text_buffer.clear()
                return

            c_id = generate_chunk_id(source_name, chunk_counter)
            chunk_counter += 1

            if heading_stack:
                breadcrumb_parts = []
                for h_item in heading_stack:
                    breadcrumb_parts.append(h_item[1])
                curr_breadcrumb = " > ".join(breadcrumb_parts)
                curr_section_title = heading_stack[-1][1]
                curr_parent_content = heading_stack[-1][1]
                curr_parent_id = heading_stack[-1][2]
            else:
                curr_breadcrumb = doc_stem
                curr_section_title = doc_stem
                curr_parent_id = ""
                curr_parent_content = ""

            chunk = build_universal_chunk(
                chunk_id=c_id,
                parent_id=curr_parent_id,
                source=source_name,
                chunk_type="text",
                breadcrumb=curr_breadcrumb,
                section_title=curr_section_title,
                content=combined_text,
                parent_content=curr_parent_content,
                business_metadata=business_metadata,
                prev_chunk_id="",
                next_chunk_id="",
                table_data=None,
                images=[],
            )
            raw_chunks.append(chunk)
            text_buffer.clear()

        for p_idx, layout in page_layouts:
            if not layout or not isinstance(layout, dict):
                continue

            page_headings = layout.get("headings", [])
            page_elements = layout.get("elements", [])

            heading_level_map = {}
            for h in page_headings:
                if isinstance(h, dict) and "text" in h:
                    h_t = str(h["text"]).strip()
                    if h_t:
                        heading_level_map[h_t] = h.get("level", 1)

            for el in page_elements:
                if not isinstance(el, dict):
                    continue

                el_type = str(el.get("type", "paragraph")).lower().strip()
                el_text = str(el.get("text", "")).strip()

                is_heading = False
                heading_level = 1

                if el_type == "heading" or el_type.startswith("heading_"):
                    is_heading = True
                    if "level" in el and el["level"] is not None:
                        try:
                            heading_level = int(el["level"])
                        except Exception:
                            heading_level = 1
                    else:
                        heading_type = el.get("type", "heading_1")
                        if el_text in heading_level_map:
                            heading_level = heading_level_map[el_text]
                        else:
                            if heading_type in heading_level_map:
                                heading_level = heading_level_map[heading_type]
                            else:
                                if el_type == "heading_1":
                                    heading_level = 1
                                else:
                                    if el_type == "heading_2":
                                        heading_level = 2
                                    else:
                                        if el_type == "heading_3":
                                            heading_level = 3
                                        else:
                                            heading_level = 1
                else:
                    if el_text in heading_level_map:
                        is_heading = True
                        heading_level = heading_level_map[el_text]

                if is_heading:
                    flush_text_buffer()

                    try:
                        lvl_int = int(heading_level)
                    except Exception:
                        lvl_int = 1
                    if lvl_int < 1:
                        lvl_int = 1

                    while heading_stack and heading_stack[-1][0] >= lvl_int:
                        heading_stack.pop()

                    if heading_stack:
                        parent_sec_id = heading_stack[-1][2]
                        parent_content_val = heading_stack[-1][1]
                    else:
                        parent_sec_id = ""
                        parent_content_val = ""

                    c_id = generate_chunk_id(source_name, chunk_counter)
                    chunk_counter += 1

                    breadcrumb_parts = []
                    for h_item in heading_stack:
                        breadcrumb_parts.append(h_item[1])
                    breadcrumb_parts.append(el_text)
                    curr_breadcrumb = " > ".join(breadcrumb_parts)

                    h_chunk = build_universal_chunk(
                        chunk_id=c_id,
                        parent_id=parent_sec_id,
                        source=source_name,
                        chunk_type="text",
                        breadcrumb=curr_breadcrumb,
                        section_title=el_text,
                        content=el_text,
                        parent_content=parent_content_val,
                        business_metadata=business_metadata,
                        prev_chunk_id="",
                        next_chunk_id="",
                        table_data=None,
                        images=[],
                    )
                    raw_chunks.append(h_chunk)

                    heading_stack.append((lvl_int, el_text, c_id, el_text, c_id))
                    continue

                if el_type == "table":
                    flush_text_buffer()

                    table_text = el_text
                    raw_table_data = el.get("table_data")

                    if raw_table_data is not None and isinstance(raw_table_data, (list, dict)):
                        final_table_data = raw_table_data
                    else:
                        parsed_list = _parse_markdown_table_to_2d_list(table_text)
                        if parsed_list and len(parsed_list) > 0:
                            final_table_data = {"headers": parsed_list[0], "rows": parsed_list[1:]}
                        else:
                            final_table_data = {"headers": [], "rows": []}

                    c_id = generate_chunk_id(source_name, chunk_counter)
                    chunk_counter += 1

                    if heading_stack:
                        breadcrumb_parts = []
                        for h_item in heading_stack:
                            breadcrumb_parts.append(h_item[1])
                        curr_breadcrumb = " > ".join(breadcrumb_parts)
                        curr_section_title = heading_stack[-1][1]
                        curr_parent_content = heading_stack[-1][1]
                        curr_parent_id = heading_stack[-1][2]
                    else:
                        curr_breadcrumb = doc_stem
                        curr_section_title = doc_stem
                        curr_parent_id = ""
                        curr_parent_content = ""

                    t_chunk = build_universal_chunk(
                        chunk_id=c_id,
                        parent_id=curr_parent_id,
                        source=source_name,
                        chunk_type="table",
                        breadcrumb=curr_breadcrumb,
                        section_title=curr_section_title,
                        content=table_text,
                        parent_content=curr_parent_content,
                        business_metadata=business_metadata,
                        prev_chunk_id="",
                        next_chunk_id="",
                        table_data=final_table_data,
                        images=[],
                    )
                    raw_chunks.append(t_chunk)
                    continue

                if el_type in ("image", "caption", "multimodal"):
                    flush_text_buffer()

                    img_text = el_text
                    raw_img_info = el.get("image_info")

                    if isinstance(raw_img_info, dict):
                        images_list = [raw_img_info]
                    else:
                        if img_text:
                            images_list = [{"description": img_text, "caption": img_text}]
                        else:
                            images_list = [{"description": "Visual element", "caption": ""}]

                    c_id = generate_chunk_id(source_name, chunk_counter)
                    chunk_counter += 1

                    if heading_stack:
                        breadcrumb_parts = []
                        for h_item in heading_stack:
                            breadcrumb_parts.append(h_item[1])
                        curr_breadcrumb = " > ".join(breadcrumb_parts)
                        curr_section_title = heading_stack[-1][1]
                        curr_parent_content = heading_stack[-1][1]
                        curr_parent_id = heading_stack[-1][2]
                    else:
                        curr_breadcrumb = doc_stem
                        curr_section_title = doc_stem
                        curr_parent_id = ""
                        curr_parent_content = ""

                    img_chunk = build_universal_chunk(
                        chunk_id=c_id,
                        parent_id=curr_parent_id,
                        source=source_name,
                        chunk_type="multimodal",
                        breadcrumb=curr_breadcrumb,
                        section_title=curr_section_title,
                        content=img_text,
                        parent_content=curr_parent_content,
                        business_metadata=business_metadata,
                        prev_chunk_id="",
                        next_chunk_id="",
                        table_data=None,
                        images=images_list,
                    )
                    raw_chunks.append(img_chunk)
                    continue

                if not el_text:
                    continue

                current_words = 0
                for item_str in text_buffer:
                    current_words += len(item_str.split())

                incoming_words = len(el_text.split())

                if current_words + incoming_words > self.max_chunk_words and text_buffer:
                    flush_text_buffer()

                text_buffer.append(el_text)

            flush_text_buffer()

        flush_text_buffer()
        return raw_chunks

    def close(self):
        """
        Cleanly closes database connections and releases cache resources.
        """
        if self._owns_cache and self.cache is not None:
            if hasattr(self.cache, "close"):
                self.cache.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
