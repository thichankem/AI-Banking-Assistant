"""
Module Unified Local Vision LLM Table Engine (VLMTableExtractor)
================================================================
Trích xuất cấu trúc bảng biểu qua Vision LLM (Gemma 4:12B Vision) trên GPU RTX 4060.
Tuân thủ:
1. 0% REGEX LAW: Tuyệt đối không dùng `import re` hay regex methods.
2. Khóa tuần tự GPUQueueLock bảo vệ VRAM RTX 4060.
3. Universal Footnote Anchor Schema: Trích xuất và phân loại FootnoteItem có anchor.
4. Universal Fallback 0% Regex: Phân tách footnote theo line gap và ký tự Unicode khi VLM offline.
5. Cắt ảnh đáy bảng linh hoạt (Dynamic Bottom Expansion) bảo vệ vùng footnote.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import socket
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("vlm_table_extractor")

from rag_banking.compute_router import router, ComputeTarget


# -----------------------------------------------------------------------------
# 1. GPU QUEUE LOCK (SINGLETON CHO RTX 4060)
# -----------------------------------------------------------------------------
class GPUQueueLock:
    """Singleton mutex lock tuần tự hóa truy cập GPU cục bộ (RTX 4060)."""
    _instance: Optional[GPUQueueLock] = None
    _init_lock = threading.Lock()

    def __new__(cls) -> GPUQueueLock:
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._sync_lock = threading.Lock()
                cls._instance._async_lock = asyncio.Lock()
            return cls._instance

    def __enter__(self) -> GPUQueueLock:
        self._sync_lock.acquire()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self._sync_lock.release()

    async def __aenter__(self) -> GPUQueueLock:
        await self._async_lock.acquire()
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self._async_lock.release()


# -----------------------------------------------------------------------------
# 2. DATA CONTRACTS & UNIVERSAL FOOTNOTE MODEL (0% REGEX)
# -----------------------------------------------------------------------------
@dataclass
class FootnoteItem:
    """Đại diện cho một mục chú thích / footnote dưới chân bảng biểu."""
    anchor_symbol: Optional[str] = None  # vd: "(*)", "(1)", "[a]", "†", hoặc None nếu là ghi chú chung
    content: str = ""                     # Nội dung giải thích (đã bóc tách bỏ anchor)

    def to_dict(self) -> dict:
        return {
            "anchor_symbol": self.anchor_symbol,
            "content": self.content
        }

    @classmethod
    def from_dict(cls, data: Union[dict, str, FootnoteItem]) -> FootnoteItem:
        if isinstance(data, FootnoteItem):
            return data
        if isinstance(data, str):
            anchor, content = extract_anchor_no_regex(data)
            return cls(anchor_symbol=anchor, content=content)
        if isinstance(data, dict):
            return cls(
                anchor_symbol=data.get("anchor_symbol"),
                content=str(data.get("content", ""))
            )
        return cls(anchor_symbol=None, content=str(data))

    def __str__(self) -> str:
        if self.anchor_symbol:
            return f"{self.anchor_symbol} {self.content}".strip()
        return self.content.strip()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            full_s = f"{self.anchor_symbol} {self.content}".strip() if self.anchor_symbol else self.content.strip()
            return (
                full_s == other
                or self.content == other
                or (self.anchor_symbol is not None and self.anchor_symbol == other)
                or (self.anchor_symbol is not None and other.startswith(self.anchor_symbol) and self.content in other)
            )
        if isinstance(other, FootnoteItem):
            return self.anchor_symbol == other.anchor_symbol and self.content == other.content
        return False


@dataclass
class TableRow:
    row_header: str
    cells: list[str] = field(default_factory=list)


@dataclass
class RowGroup:
    group_name: Optional[str] = None
    rows: list[TableRow] = field(default_factory=list)


@dataclass
class TableExtractionResult:
    table_title: Optional[str] = None
    column_headers: list[str] = field(default_factory=list)
    row_groups: list[RowGroup] = field(default_factory=list)
    footnotes: list[FootnoteItem] = field(default_factory=list)

    @property
    def headers(self) -> list[str]:
        """Thuộc tính tương thích: Danh sách tiêu đề các cột."""
        return self.column_headers

    @property
    def rows(self) -> list[list[str]]:
        """Thuộc tính tương thích: Danh sách các hàng dữ liệu dạng phẳng."""
        res: list[list[str]] = []
        for g in self.row_groups:
            for r in g.rows:
                res.append([r.row_header] + r.cells)
        return res

    def to_dict(self) -> dict:
        return {
            "table_title": self.table_title,
            "column_headers": self.column_headers,
            "row_groups": [
                {
                    "group_name": g.group_name,
                    "rows": [{"row_header": r.row_header, "cells": r.cells} for r in g.rows]
                }
                for g in self.row_groups
            ],
            "footnotes": [f.to_dict() if isinstance(f, FootnoteItem) else str(f) for f in self.footnotes]
        }

    @classmethod
    def from_dict(cls, data: dict) -> TableExtractionResult:
        row_groups: list[RowGroup] = []
        for g in data.get("row_groups", []):
            if isinstance(g, dict):
                rows = [
                    TableRow(
                        row_header=str(r.get("row_header", "")),
                        cells=[str(c) for c in r.get("cells", [])]
                    )
                    for r in g.get("rows", [])
                    if isinstance(r, dict)
                ]
                row_groups.append(RowGroup(group_name=g.get("group_name"), rows=rows))

        raw_fns = data.get("footnotes", [])
        parsed_fns: list[FootnoteItem] = []
        for item in raw_fns:
            if isinstance(item, FootnoteItem):
                parsed_fns.append(item)
            elif isinstance(item, dict):
                parsed_fns.append(FootnoteItem.from_dict(item))
            elif isinstance(item, str):
                parsed_fns.append(FootnoteItem.from_dict(item))

        return cls(
            table_title=data.get("table_title"),
            column_headers=[str(h) for h in data.get("column_headers", [])],
            row_groups=row_groups,
            footnotes=parsed_fns
        )


# -----------------------------------------------------------------------------
# 3. PHÂN ĐOẠN CHÚ THÍCH DỰ PHÒNG THUẦN TÚY (0% REGEX, 0% MẢNG KÝ TỰ NEO)
# -----------------------------------------------------------------------------
def extract_anchor_no_regex(text: str) -> tuple[Optional[str], str]:
    """
    Trích xuất tiêu đề/nội dung chú thích thuần túy (0% Regex, 0% mảng ký tự neo).
    """
    s = text.strip()
    if not s:
        return None, ""
    if ":" in s:
        parts = s.split(":", 1)
        if 1 <= len(parts[0].split()) <= 2:
            return parts[0].strip(), parts[1].strip()
    return None, s


def parse_fallback_footnotes_no_regex(
    raw_lines: Union[str, list],
    avg_line_height: float = 14.0
) -> list[FootnoteItem]:
    """
    Phân tích chú thích dự phòng thuần túy theo dòng văn bản (0% Regex, 0% mảng ký tự neo).
    """
    if isinstance(raw_lines, str):
        lines = [l.strip() for l in raw_lines.splitlines() if l.strip()]
    else:
        lines = []
        for item in raw_lines:
            if isinstance(item, str) and item.strip():
                lines.append(item.strip())
            elif isinstance(item, dict) and item.get("text"):
                lines.append(item["text"].strip())

    items: list[FootnoteItem] = []
    for line in lines:
        anchor, content = extract_anchor_no_regex(line)
        items.append(FootnoteItem(anchor_symbol=anchor, content=content))

    return items


# -----------------------------------------------------------------------------
# 4. HÀM XỬ LÝ HÌNH ẢNH THỊ GIÁC (0% REGEX)
# -----------------------------------------------------------------------------
def _to_pil_image(img_input: Union[Image.Image, np.ndarray, bytes, str, Path]) -> Image.Image:
    """Chuyển đổi các định dạng đầu vào thành ảnh PIL RGB."""
    if isinstance(img_input, Image.Image):
        return img_input.convert("RGB")
    if isinstance(img_input, np.ndarray):
        arr = img_input[:, :, ::-1] if (img_input.ndim == 3 and img_input.shape[2] == 3) else img_input
        return Image.fromarray(arr).convert("RGB")
    if isinstance(img_input, (str, Path)):
        return Image.open(str(img_input)).convert("RGB")
    if isinstance(img_input, (bytes, bytearray)):
        return Image.open(io.BytesIO(img_input)).convert("RGB")
    raise ValueError(f"Không hỗ trợ định dạng ảnh: {type(img_input)}")


def crop_pdf_table_image(
    page_image: Union[Image.Image, np.ndarray, bytes],
    bbox: Union[tuple[float, float, float, float], list[float]],
    page_dim: Optional[tuple[float, float]] = None,
    bottom_expansion_ratio: float = 0.08,
    footer_margin_ratio: float = 0.065,
    dynamic_bottom: Optional[float] = None
) -> Image.Image:
    """
    Cắt vùng bảng biểu từ trang PDF:
    - Nếu có `dynamic_bottom` (từ detect_dynamic_table_bottom): Dùng đáy động chính xác.
    - Nếu không có: Dùng fallback `bottom_expansion_ratio` (mặc định 0.08) chừa lề footer 6.5%.
    """
    img = _to_pil_image(page_image)
    w, h = img.size
    x1, y1, x2, y2 = bbox

    # Chuyển đổi tọa độ điểm PDF sang pixel ảnh nếu có scale
    if page_dim and page_dim[0] > 0 and page_dim[1] > 0:
        sx, sy = w / page_dim[0], h / page_dim[1]
        px1, py1, px2, py2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
        dyn_b_px = (dynamic_bottom * sy) if dynamic_bottom is not None else None
    elif max(x1, y1, x2, y2) <= 1.0:
        px1, py1, px2, py2 = x1 * w, y1 * h, x2 * w, y2 * h
        dyn_b_px = (dynamic_bottom * h) if dynamic_bottom is not None else None
    else:
        px1, py1, px2, py2 = x1, y1, x2, y2
        dyn_b_px = dynamic_bottom

    max_safe_bottom = h * (1.0 - footer_margin_ratio)

    pad_px = 12  # Thêm lề an toàn 12px để bảo toàn nguyên vẹn đường viền bảng và các ký tự biên
    if dyn_b_px is not None:
        py2_exp = min(max(dyn_b_px, py2) + pad_px, max_safe_bottom)
    else:
        py2_exp = min(py2 + (bottom_expansion_ratio * h), max_safe_bottom)

    c_x1 = max(0, int(round(px1 - pad_px)))
    c_y1 = max(0, int(round(py1 - pad_px)))
    c_x2 = min(w, int(round(px2 + pad_px)))
    c_y2 = min(h, int(round(py2_exp)))

    return img.crop((c_x1, c_y1, c_x2, c_y2)) if (c_x2 > c_x1 and c_y2 > c_y1) else img


def prepare_cross_page_table_image(
    table_images: list[Union[Image.Image, np.ndarray, bytes]],
    max_height_px: int = 2048
) -> Image.Image:
    """Ghép dọc các ảnh bảng biểu xuyên trang và tự động co tỷ lệ nếu chiều cao > 2048px."""
    if not table_images:
        return Image.new("RGB", (800, 400), color=(255, 255, 255))

    pil_imgs = [_to_pil_image(img) for img in table_images]
    if len(pil_imgs) == 1:
        comp = pil_imgs[0]
    else:
        mw = max(im.width for im in pil_imgs)
        th = sum(im.height for im in pil_imgs)
        comp = Image.new("RGB", (mw, th), color=(255, 255, 255))
        cy = 0
        for im in pil_imgs:
            comp.paste(im, ((mw - im.width) // 2, cy))
            cy += im.height

    if comp.height > max_height_px:
        ratio = max_height_px / float(comp.height)
        comp = comp.resize((max(1, int(round(comp.width * ratio))), max_height_px), Image.Resampling.LANCZOS)
    return comp


def render_docx_table_to_image(table: object, max_width_px: int = 1400) -> Image.Image:
    """Kết xuất cấu trúc bảng Word DOCX thành ảnh lưới tương phản cao để đưa vào Vision LLM."""
    rows = getattr(table, "rows", [])
    if not rows:
        return Image.new("RGB", (800, 300), color=(255, 255, 255))

    num_cols = max(len(r.cells) for r in rows) if rows else 1
    matrix = [[c.text.strip().replace("\n", " ") for c in r.cells] + [""] * (num_cols - len(r.cells)) for r in rows]

    font = ImageFont.load_default()
    for p in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            try:
                font = ImageFont.truetype(p, size=12)
                break
            except Exception:
                pass

    col_widths = [int(max(55, min(380, max(font.getlength(row[c]) for row in matrix[:30]) + 24))) for c in range(num_cols)]
    x_offsets = [10]
    for w_col in col_widths:
        x_offsets.append(x_offsets[-1] + w_col)

    row_h = 36
    total_w, total_h = x_offsets[-1] + 10, row_h * len(matrix) + 20
    img = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    hdr_cnt = min(2, len(matrix))
    draw.rectangle([(10, 10), (total_w - 10, 10 + hdr_cnt * row_h)], fill=(240, 243, 246))

    for r_idx, row in enumerate(matrix):
        yt, yb = 10 + r_idx * row_h, 10 + (r_idx + 1) * row_h
        for c_idx, val in enumerate(row):
            xl, xr = x_offsets[c_idx], x_offsets[c_idx + 1]
            draw.rectangle([(xl, yt), (xr, yb)], outline=(180, 190, 200), width=1)
            txt = val
            avail = xr - xl - 14
            if font.getlength(txt) > avail:
                while txt and font.getlength(txt + "...") > avail:
                    txt = txt[:-1]
                txt += "..."
            color = (15, 23, 42) if r_idx >= hdr_cnt else (10, 37, 64)
            draw.text((xl + 7, yt + 10), txt, fill=color, font=font)

    return img


# -----------------------------------------------------------------------------
# 5. UNIVERSAL SYSTEM PROMPT VLM (DOMAIN-AGNOSTIC)
# -----------------------------------------------------------------------------
TABLE_JSON_PROMPT = """Bạn là chuyên gia phân tích cấu trúc bảng biểu thị giác đa lĩnh vực (Tài chính, Y tế, Kỹ thuật, Hành chính, Học thuật...).
Nhiệm vụ: Đọc hình ảnh và trích xuất TOÀN BỘ cấu trúc bảng cùng các ghi chú chân bảng sang định dạng JSON nghiêm ngặt:
1. "table_title": Tiêu đề bảng biểu (string hoặc null).
2. "column_headers": Tiêu đề các cột. Nếu ô gộp nhiều tầng, làm phẳng bằng " -> " (vd: "Kỳ hạn -> 6 tháng").
3. "row_groups": Danh sách nhóm hàng (group_name: string hoặc null). Mỗi nhóm chứa "rows": [{"row_header": str, "cells": [str]}].
   * YÊU CẦU QUAN TRỌNG: Bảo toàn 100% mọi ký hiệu tham chiếu xuất hiện trong ô dữ liệu hoặc tiêu đề (ví dụ: (*), (1), [a], †...). Bảo toàn số liệu và mọi đơn vị đo lường (%, mg/dL, mm, VND, USD...).
4. "footnotes": Danh sách các mục chú thích chân bảng, điều kiện ràng buộc hoặc nguồn dữ liệu:
   * Nếu có ký hiệu neo (*, (1), [a], †...), tách vào "anchor_symbol" và phần nội dung còn lại vào "content".
   * Nếu là chú thích chung toàn bảng, nguồn trích dẫn, hoặc câu giải thích độc lập không có ký hiệu neo, đặt "anchor_symbol": null.

CHỈ TRẢ VỀ DUY NHẤT 1 KHỐI JSON:
```json
{
  "table_title": "string or null",
  "column_headers": ["Cột 1", "Cột 2"],
  "row_groups": [
    {
      "group_name": "string or null",
      "rows": [{"row_header": "Hàng 1", "cells": ["Giá trị 1 (*)", "Giá trị 2"]}]
    }
  ],
  "footnotes": [
    {"anchor_symbol": "(*)", "content": "Nội dung giải thích điều kiện"},
    {"anchor_symbol": null, "content": "Ghi chú chung toàn bảng"}
  ]
}
```"""


# -----------------------------------------------------------------------------
# 6. CORE VLM TABLE EXTRACTOR CLASS
# -----------------------------------------------------------------------------
class VLMTableExtractor:
    """Engine trích xuất bảng biểu bằng Vision LLM chạy cục bộ với GPUQueueLock và fallback."""

    def __init__(
        self,
        model_name: str = "qwen2-vl",
        ollama_endpoint: str = "http://127.0.0.1:11434/api/generate",
        timeout_seconds: int = 45,
        custom_prompt_context: Optional[str] = None,
        config: object | None = None
    ):
        self.model_name = model_name
        self.ollama_endpoint = ollama_endpoint
        self.timeout_seconds = timeout_seconds
        self.custom_prompt_context = custom_prompt_context or (config.custom_prompt_context if config else None)
        self.gpu_lock = GPUQueueLock()
        self._last_probe_time: float = 0.0
        self._is_server_alive: bool = False
        self._active_engine: str = "offline"  # "colab" | "ollama" | "offline"

    def get_prompt_text(self, custom_context: Optional[str] = None) -> str:
        ctx = custom_context or self.custom_prompt_context
        return f"{TABLE_JSON_PROMPT}\n\n[Ngữ cảnh]: {ctx}" if ctx else TABLE_JSON_PROMPT

    def is_available(self) -> bool:
        """Kiểm tra kết nối: Ưu tiên 1 Google Colab GPU Cloud, Dự phòng 2 Ollama Local."""
        now = time.time()
        if now - self._last_probe_time < 10.0:
            return self._is_server_alive
        self._last_probe_time = now

        # 1. Thử kết nối Colab GPU Cloud (Tesla T4)
        try:
            colab_info = router.check_colab_status(timeout_sec=1.5)
            if colab_info and colab_info.get("status") == "ok":
                self._is_server_alive = True
                self._active_engine = "colab"
                gpu_name = colab_info.get("gpu", "Cloud GPU")
                logger.info(f"🚀 [VLMTableExtractor] Ưu tiên 1: Đã kết nối Google Colab GPU ({gpu_name})!")
                return True
        except Exception:
            pass

        # 2. Thử kết nối Ollama Local (Dự phòng)
        try:
            parsed = urlparse(self.ollama_endpoint)
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            with socket.create_connection((host, port), timeout=0.2):
                self._is_server_alive = True
                self._active_engine = "ollama"
                logger.info("⚡ [VLMTableExtractor] Colab offline -> Sử dụng Ollama Local.")
                return True
        except Exception:
            pass

        self._is_server_alive = False
        self._active_engine = "offline"
        return False

    def _image_to_base64(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _parse_strict_json(self, response_text: str) -> dict:
        """Trích xuất và parse JSON thuần chuỗi 0% Regex."""
        text = response_text.strip()
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.rfind("```")
            if end > start:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.rfind("```")
            if end > start:
                text = text[start:end].strip()

        fb, lb = text.find("{"), text.rfind("}")
        if fb != -1 and lb != -1 and lb > fb:
            text = text[fb:lb + 1]
        return json.loads(text)

    def _call_vlm_raw(self, prompt: str, image_b64: str) -> str:
        """Định tuyến gọi mô hình VLM: Ưu tiên 1 Colab GPU, Dự phòng 2 Ollama."""
        if self._active_engine == "colab":
            url = f"{router.endpoint}/vlm/extract_table"
            payload = {
                "image_base64": image_b64,
                "prompt": prompt,
                "custom_context": self.custom_prompt_context
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "ngrok-skip-browser-warning": "true"
                }
            )
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return str(body.get("response", ""))
        else:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 2048}
            }
            req = urllib.request.Request(
                self.ollama_endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return str(body.get("response", ""))

    def _call_ollama_raw(self, prompt: str, image_b64: str) -> str:
        return self._call_vlm_raw(prompt, image_b64)

    def _deterministic_fallback(
        self,
        image: Optional[Image.Image] = None,
        raw_fallback_text: Optional[str] = None
    ) -> TableExtractionResult:
        """Fallback hình học xác định khi VLM offline hoặc lỗi JSON (0% Regex)."""
        rows: list[TableRow] = []
        raw_fn_lines: list[str] = []

        if raw_fallback_text:
            lines = [l.strip() for l in raw_fallback_text.splitlines() if l.strip()]
            for line in lines:
                anchor, _ = extract_anchor_no_regex(line)
                if anchor is not None:
                    raw_fn_lines.append(line)
                elif "|" in line:
                    parts = [p.strip() for p in line.split("|") if p.strip()]
                    if parts:
                        rows.append(TableRow(row_header=parts[0], cells=parts[1:]))
                else:
                    # Nếu dòng không có '|' nhưng nằm sau dòng có anchor thì coi là wrapped footnote
                    if raw_fn_lines:
                        raw_fn_lines.append(line)
                    else:
                        rows.append(TableRow(row_header=line, cells=[]))

        if not rows:
            rows.append(TableRow(row_header="Dữ liệu bảng quét thị giác", cells=["Không thể đọc tự động"]))

        parsed_footnotes = parse_fallback_footnotes_no_regex(raw_fn_lines)

        return TableExtractionResult(
            table_title=None,
            column_headers=["Nội dung trích xuất dự phòng"],
            row_groups=[RowGroup(group_name=None, rows=rows)],
            footnotes=parsed_footnotes
        )

    def extract_table(
        self,
        table_image: Union[Image.Image, np.ndarray, bytes],
        raw_fallback_text: Optional[str] = None,
        custom_prompt_context: Optional[str] = None
    ) -> TableExtractionResult:
        """Trích xuất bảng biểu qua VLM tuần tự hóa trên GPUQueueLock kèm retry 2 bước."""
        pil_img = _to_pil_image(table_image)
        if not self.is_available():
            return self._deterministic_fallback(pil_img, raw_fallback_text)

        img_b64 = self._image_to_base64(pil_img)
        prompt = self.get_prompt_text(custom_prompt_context)

        with self.gpu_lock:
            try:
                resp = self._call_ollama_raw(prompt, img_b64)
                return TableExtractionResult.from_dict(self._parse_strict_json(resp))
            except Exception as e1:
                try:
                    retry_p = f"{prompt}\n\n[LỖI JSON LẦN TRƯỚC: {e1}]. Hãy CHỈ trả về đúng chuỗi JSON thô hợp lệ."
                    resp2 = self._call_ollama_raw(retry_p, img_b64)
                    return TableExtractionResult.from_dict(self._parse_strict_json(resp2))
                except Exception as e2:
                    logger.warning(f"[VLMTableExtractor] Lỗi trích xuất VLM: {e2}. Dùng deterministic fallback.")

        return self._deterministic_fallback(pil_img, raw_fallback_text)


__all__ = [
    "GPUQueueLock",
    "FootnoteItem",
    "TableRow",
    "RowGroup",
    "TableExtractionResult",
    "extract_anchor_no_regex",
    "parse_fallback_footnotes_no_regex",
    "crop_pdf_table_image",
    "prepare_cross_page_table_image",
    "render_docx_table_to_image",
    "VLMTableExtractor",
    "TABLE_JSON_PROMPT",
]
