"""Screen detectable text on a local copy before Cloud Vision receives an attachment."""

from collections import defaultdict
from difflib import SequenceMatcher
from io import BytesIO

import pytesseract
from PIL import Image, ImageDraw, ImageOps


def prepare_vision_image(image_bytes, guard_text, *, ocr_configure):
    with Image.open(BytesIO(image_bytes)) as image:
        if image.width * image.height > 16_000_000:
            raise ValueError("The attachment exceeds the privacy screen's image size limit")
        clean = ImageOps.exif_transpose(image).convert("RGB")
    if not ocr_configure():
        raise RuntimeError("Local attachment privacy screening is unavailable")
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
    output = BytesIO()
    clean.save(output, format="PNG")
    return output.getvalue()
