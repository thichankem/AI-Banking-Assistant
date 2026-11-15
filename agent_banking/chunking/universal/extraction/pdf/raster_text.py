"""Identify raster regions with text-like ink before requesting page OCR."""

from itertools import groupby
import math


def _runs(values):
    index = 0
    for occupied, group in groupby(values):
        end = index + sum(1 for _ in group)
        if occupied:
            yield index, end
        index = end


def _has_glyph_row(image):
    width, height = image.size
    samples = image.tobytes()
    rows = [samples[y * width:(y + 1) * width] for y in range(height)]
    occupied = [3 <= row.count(255) <= width * 0.60 for row in rows]
    for top, bottom in _runs(occupied):
        line_height = bottom - top
        if not 3 <= line_height <= height * 0.45:
            continue
        columns = (any(column) for column in zip(*rows[top:bottom]))
        glyphs = [(left, right) for left, right in _runs(columns) if right - left <= line_height * 2]
        if len(glyphs) >= 4 and glyphs[-1][1] - glyphs[0][0] >= line_height * 2:
            return True
    return False


def region_may_contain_text(page, box, visible_elements):
    """Look for rows of separated glyph-like components without transcribing them."""
    from PIL import Image, ImageDraw, ImageStat
    from pymupdf import Matrix, Rect

    region = Rect(box)
    if region.width < 18 or region.height < 8:
        return False
    larger_side = max(region.width, region.height)
    scale = min(3, 900 / larger_side)
    pixmap = page.get_pixmap(matrix=Matrix(scale, scale), clip=region, alpha=False, annots=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples).convert("L")
    background = ImageStat.Stat(image).median[0]
    draw = ImageDraw.Draw(image)
    for element in visible_elements:
        x0, y0, x1, y1 = element["bbox"]
        span = Rect(page.rect.x0 + x0 * page.rect.width, page.rect.y0 + y0 * page.rect.height,
                    page.rect.x0 + x1 * page.rect.width, page.rect.y0 + y1 * page.rect.height)
        clipped = span & region
        if not clipped.is_empty:
            draw.rectangle((math.floor(clipped.x0 * scale) - pixmap.x,
                            math.floor(clipped.y0 * scale) - pixmap.y,
                            math.ceil(clipped.x1 * scale) - pixmap.x,
                            math.ceil(clipped.y1 * scale) - pixmap.y), fill=background)
    light_background = background >= 128
    cutoff = background - 36 if light_background else background + 36
    mask = image.point([255 if (shade <= cutoff if light_background else shade >= cutoff) else 0 for shade in range(256)])
    return _has_glyph_row(mask) or _has_glyph_row(mask.transpose(Image.Transpose.ROTATE_90))
