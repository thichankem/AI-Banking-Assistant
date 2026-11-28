"""Screen detectable text on a local copy before Cloud Vision receives an attachment."""

from collections import defaultdict
from difflib import SequenceMatcher
from io import BytesIO

import pytesseract
from PIL import Image, ImageDraw, ImageOps

_RAPID_OCR_ENGINE = None


def _get_rapidocr_engine():
    """Lazily initialize RapidOCR ONNX runtime when system Tesseract is unavailable."""
    global _RAPID_OCR_ENGINE
    if _RAPID_OCR_ENGINE is not None:
        return _RAPID_OCR_ENGINE
    try:
        from rapidocr_onnxruntime import RapidOCR
        _RAPID_OCR_ENGINE = RapidOCR()
    except (ImportError, OSError, RuntimeError, ValueError):
        return None
    return _RAPID_OCR_ENGINE


def _blackout_changed_span(draw, box_bounds, raw_text: str, screened_text: str):
    """Mask character spans inside box_bounds that changed after privacy screening."""
    if not raw_text or screened_text == raw_text:
        return
    left, top, right, bottom = box_bounds
    width = max(1, right - left)
    text_len = max(1, len(raw_text))
    opcodes = SequenceMatcher(None, raw_text, screened_text, autojunk=False).get_opcodes()
    for op, start, stop, _, _ in opcodes:
        if op == "equal":
            continue
        if start == 0 and stop >= text_len:
            draw.rectangle((left, top, right, bottom), fill="black")
            continue
        span_left = left + int((width * start) / text_len)
        span_right = left + int((width * stop) / text_len)
        if span_left > left:
            span_left = max(left, span_left - 2)
        if span_right < right:
            span_right = min(right, span_right + 2)
        if span_right <= span_left:
            span_right = min(right, span_left + 2)
        draw.rectangle((span_left, top, span_right, bottom), fill="black")


def _group_ocr_lines(entries: list[dict]) -> list[list[dict]]:
    """Group detected OCR boxes into horizontal lines sorted top-to-bottom, left-to-right."""
    if not entries:
        return []
    heights = sorted(item["h"] for item in entries)
    median_h = heights[len(heights) // 2]
    tol = max(4.0, 0.55 * median_h)
    ordered = sorted(entries, key=lambda item: (item["cy"], item["box"][0]))
    lines = []
    for item in ordered:
        if not lines:
            lines.append([item])
            continue
        cur_line = lines[-1]
        mean_cy = sum(elem["cy"] for elem in cur_line) / len(cur_line)
        if abs(item["cy"] - mean_cy) <= tol:
            cur_line.append(item)
        else:
            lines.append([item])
    for line in lines:
        line.sort(key=lambda item: item["box"][0])
    return lines


def _screen_with_tesseract(clean, guard_text):
    data = pytesseract.image_to_data(clean, lang="vie+eng", output_type=pytesseract.Output.DICT, timeout=5)
    lines = defaultdict(list)
    for i, value in enumerate(data["text"]):
        value = value.strip()
        if not value:
            continue
        key = tuple(data[field][i] for field in ("block_num", "par_num", "line_num"))
        lines[key].append((value, i))
    draw = ImageDraw.Draw(clean)
    for words in lines.values():
        original = " ".join(word for word, _ in words)
        screened = guard_text(original)
        changes = [(start, end) for op, start, end, _, _ in
                   SequenceMatcher(None, original, screened, autojunk=False).get_opcodes() if op != "equal"]
        cursor = 0
        for word, i in words:
            end = cursor + len(word)
            if any(start < end and cursor < stop for start, stop in changes):
                left, top = data["left"][i], data["top"][i]
                right = left + data["width"][i]
                bottom = top + data["height"][i]
                draw.rectangle((left, top, right, bottom), fill="black")
            cursor = end + 1


def _screen_with_rapidocr(clean, guard_text, engine):
    import numpy as np

    result, _ = engine(np.array(clean))
    if not result:
        return
    draw = ImageDraw.Draw(clean)
    entries = []
    for item in result:
        if not item or len(item) < 2:
            continue
        box = item[0]
        raw_text = str(item[1] or "").strip()
        if not raw_text or not box:
            continue
        xs = [int(round(pt[0])) for pt in box]
        ys = [int(round(pt[1])) for pt in box]
        left = max(0, min(xs))
        right = min(clean.width, max(xs))
        top = max(0, min(ys))
        bottom = min(clean.height, max(ys))
        if right <= left or bottom <= top:
            continue
        bounds = (left, top, right, bottom)
        _blackout_changed_span(draw, bounds, raw_text, guard_text(raw_text))
        entries.append({
            "box": bounds,
            "text": raw_text,
            "cy": (top + bottom) / 2.0,
            "h": max(1, bottom - top),
        })

    for line in _group_ocr_lines(entries):
        if len(line) < 2:
            continue
        spans = []
        cursor = 0
        parts = []
        for elem in line:
            txt = elem["text"]
            start = cursor
            end = start + len(txt)
            spans.append((start, end, elem["box"], txt))
            parts.append(txt)
            cursor = end + 1
        joined = " ".join(parts)
        screened = guard_text(joined)
        if screened == joined:
            continue
        opcodes = SequenceMatcher(None, joined, screened, autojunk=False).get_opcodes()
        for op, start, stop, _, _ in opcodes:
            if op == "equal":
                continue
            for b_start, b_end, bbox, txt in spans:
                if start < b_end and b_start < stop:
                    if start <= b_start and stop >= b_end:
                        draw.rectangle(bbox, fill="black")
                    else:
                        left, top, right, bottom = bbox
                        width = max(1, right - left)
                        txt_len = max(1, len(txt))
                        rel_start = max(0, start - b_start)
                        rel_stop = min(txt_len, stop - b_start)
                        s_left = left + int((width * rel_start) / txt_len)
                        s_right = left + int((width * rel_stop) / txt_len)
                        if s_left > left:
                            s_left = max(left, s_left - 2)
                        if s_right < right:
                            s_right = min(right, s_right + 2)
                        if s_right <= s_left:
                            s_right = min(right, s_left + 2)
                        draw.rectangle((s_left, top, s_right, bottom), fill="black")


def prepare_vision_image(image_bytes, guard_text, *, ocr_configure):
    with Image.open(BytesIO(image_bytes)) as image:
        if image.width * image.height > 16_000_000:
            raise ValueError("The attachment exceeds the privacy screen's image size limit")
        clean = ImageOps.exif_transpose(image).convert("RGB")
    if ocr_configure():
        _screen_with_tesseract(clean, guard_text)
    else:
        rapid_engine = _get_rapidocr_engine()
        if rapid_engine is None:
            raise RuntimeError("Local attachment privacy screening is unavailable")
        _screen_with_rapidocr(clean, guard_text, rapid_engine)
    output = BytesIO()
    clean.save(output, format="PNG")
    return output.getvalue()
