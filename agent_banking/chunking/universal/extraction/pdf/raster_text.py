"""Identify raster regions with text-like ink before requesting page OCR."""

from itertools import groupby
import math

from agent_banking.chunking.config.settings import get_config_section

_RASTER_CFG = get_config_section("extraction")["pdf"]["raster_text"]


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
    min_ink = _RASTER_CFG["row_min_ink"]
    max_ink = width * _RASTER_CFG["row_max_ink_ratio"]
    min_lh = _RASTER_CFG["min_line_height"]
    max_lh = height * _RASTER_CFG["max_line_height_ratio"]
    min_wide = _RASTER_CFG["min_glyph_count_wide"]
    short_min_g = _RASTER_CFG["short_label_min_glyphs"]
    short_max_g = _RASTER_CFG["short_label_max_glyphs"]
    short_min_lh = _RASTER_CFG["short_label_min_line_height"]
    short_ext_ratio = _RASTER_CFG["short_label_max_extent_ratio"]
    short_sub_div = _RASTER_CFG["short_label_substantial_divisor"]
    short_min_sub = _RASTER_CFG["short_label_min_substantial"]
    occupied = [min_ink <= row.count(255) <= max_ink for row in rows]
    for top, bottom in _runs(occupied):
        line_height = bottom - top
        if not min_lh <= line_height <= max_lh:
            continue
        columns = (any(column) for column in zip(*rows[top:bottom]))
        glyphs = [(left, right) for left, right in _runs(columns) if right - left <= line_height * 2]
        if len(glyphs) >= min_wide and glyphs[-1][1] - glyphs[0][0] >= line_height * 2:
            return True
        if short_min_g <= len(glyphs) <= short_max_g and line_height >= short_min_lh:
            extent = glyphs[-1][1] - glyphs[0][0]
            # A compact short label can consist of a digit and a percent sign.
            substantial = sum(right - left >= max(2, line_height // short_sub_div) for left, right in glyphs)
            if substantial >= short_min_sub and line_height <= extent <= line_height * short_ext_ratio:
                return True
    return False


def region_may_contain_text(page, box, visible_elements):
    """Look for rows of separated glyph-like components without transcribing them."""
    from PIL import Image, ImageDraw, ImageStat
    from pymupdf import Matrix, Rect

    region = Rect(box)
    if region.width < _RASTER_CFG["min_region_width"] or region.height < _RASTER_CFG["min_region_height"]:
        return False
    larger_side = max(region.width, region.height)
    scale = min(_RASTER_CFG["max_scale"], _RASTER_CFG["target_side"] / larger_side)
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
    light_background = background >= _RASTER_CFG["luminance_midpoint"]
    delta = _RASTER_CFG["contrast_delta"]
    if light_background:
        cutoff = background - delta
        lut = []
        for shade in range(256):
            if shade <= cutoff:
                lut.append(255)
            else:
                lut.append(0)
    else:
        cutoff = background + delta
        lut = []
        for shade in range(256):
            if shade >= cutoff:
                lut.append(255)
            else:
                lut.append(0)
    mask = image.point(lut)
    return _has_glyph_row(mask) or _has_glyph_row(mask.transpose(Image.Transpose.ROTATE_90))
