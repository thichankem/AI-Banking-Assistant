"""Extract deterministic visual crops (app screens, tables, figures) from rendered pages.

RULES:
- 0% REGEX LAW: Strictly string methods only (no re).
- ZERO TERNARY LAW: Explicit if/else statements only.
- NO TYPING LIBRARY: Native Python types only.
"""
import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image

from agent_banking.chunking.publication_io import atomic_bytes

_MIN_PAGE_DIMENSION = 400
_MAX_CROPS_PER_PAGE = 4

_FIGURE_KEYWORDS = (
    "hình ",
    "sơ đồ",
    "biểu đồ",
    "lưu đồ",
    "figure",
    "cơ cấu",
    "kiến trúc",
    "quy trình",
    "minh họa",
    "trụ cột",
    "kpi",
)
_TABLE_KEYWORDS = (
    "bảng ",
    "biểu phí",
    "hạn mức",
    "tỷ lệ",
    "ma trận",
    "lịch trả nợ",
    "chỉ tiêu",
)
_VISUAL_FALLBACK_KEYWORDS = _FIGURE_KEYWORDS + _TABLE_KEYWORDS


def _crop_and_save(pil_image, box, asset_dir, metadata):
    left, top, right, bottom = box
    cropped = pil_image.crop((left, top, right, bottom))
    stream = io.BytesIO()
    cropped.save(stream, format="PNG")
    raw_bytes = stream.getvalue()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    target_dir = Path(asset_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / (digest + ".png")
    if not target_path.is_file() or hashlib.sha256(target_path.read_bytes()).hexdigest() != digest:
        atomic_bytes(target_path, raw_bytes)
    crop_w, crop_h = cropped.size
    item = dict(metadata)
    item["path"] = str(target_path.resolve())
    item["sha256"] = digest
    item["width"] = int(crop_w)
    item["height"] = int(crop_h)
    return item


def _true_intervals(mask):
    spans = []
    in_span = False
    start_idx = 0
    length = len(mask)
    for idx in range(length):
        if bool(mask[idx]) and not in_span:
            in_span = True
            start_idx = idx
        elif not bool(mask[idx]) and in_span:
            in_span = False
            spans.append([start_idx, idx])
    if in_span:
        spans.append([start_idx, length])
    return spans


def _merge_intervals(spans, max_gap):
    merged = []
    for span in spans:
        if merged and span[0] - merged[-1][1] <= max_gap:
            if span[1] > merged[-1][1]:
                merged[-1][1] = span[1]
        else:
            merged.append([span[0], span[1]])
    return merged


def _is_dark_slide_page(arr):
    h, w, _ = arr.shape
    if w <= h:
        return False
    y0 = int(h * 0.2)
    y1 = int(h * 0.8)
    edge_w = max(4, int(w * 0.035))
    left_edge = arr[y0:y1, :edge_w]
    right_edge = arr[y0:y1, w - edge_w:]
    margins = np.concatenate([left_edge, right_edge], axis=1)
    med = np.median(margins.reshape(-1, 3), axis=0)
    return bool(med[0] < 65 and med[1] < 85 and med[2] < 130)


def _extract_dark_slide_crops(pil_image, arr, page_index, page_sha256, asset_dir):
    h, w, _ = arr.shape
    y_start = int(h * 0.095)
    y_end = int(h * 0.915)
    body = arr[y_start:y_end].astype(np.int32)
    edge_w = max(4, int(w * 0.03))
    margins = np.concatenate([body[:, :edge_w], body[:, w - edge_w:]], axis=1)
    row_bg = np.median(margins, axis=1).astype(np.int32)
    diff = np.max(np.abs(body - row_bg[:, None, :]), axis=2)
    fg = diff > 28

    col_ratio = fg.mean(axis=0)
    raw_cols = _true_intervals(col_ratio > 0.04)
    merged_cols = _merge_intervals(raw_cols, int(w * 0.025))
    valid_cols = []
    min_col_w = int(w * 0.10)
    max_col_w = int(w * 0.85)
    for col_start, col_end in merged_cols:
        col_w = col_end - col_start
        if col_w >= min_col_w and col_w <= max_col_w:
            valid_cols.append((col_start, col_end))

    candidates = []
    pad = 18
    if 1 <= len(valid_cols) <= 6:
        for col_start, col_end in valid_cols:
            sub_fg = fg[:, col_start:col_end]
            row_ratio = sub_fg.mean(axis=1)
            active_rows = np.where(row_ratio > 0.015)[0]
            if len(active_rows) == 0:
                continue
            top_rel = int(active_rows[0])
            bottom_rel = int(active_rows[-1] + 1)
            if bottom_rel - top_rel < int(h * 0.22):
                continue
            active_xs = np.where(sub_fg[top_rel:bottom_rel].any(axis=0))[0]
            if len(active_xs) == 0:
                continue
            x0 = max(0, col_start + int(active_xs[0]) - pad)
            x1 = min(w, col_start + int(active_xs[-1] + 1) + pad)
            y0 = max(0, y_start + top_rel - pad)
            y1 = min(h, y_start + bottom_rel + pad)
            candidates.append(("app_step", x0, y0, x1, y1))

    if not candidates and float(fg.mean()) > 0.06:
        ys, xs = np.where(fg)
        if len(xs) > 0 and len(ys) > 0:
            x0 = max(0, int(xs.min()) - pad)
            x1 = min(w, int(xs.max() + 1) + pad)
            y0 = max(0, y_start + int(ys.min()) - pad)
            y1 = min(h, y_start + int(ys.max() + 1) + pad)
            if (x1 - x0) >= int(w * 0.25) and (y1 - y0) >= int(h * 0.25):
                candidates.append(("slide_visual", x0, y0, x1, y1))

    total = len(candidates)
    crops = []
    for idx, item in enumerate(candidates):
        region_type, x0, y0, x1, y1 = item
        if region_type == "app_step":
            if total > 1:
                display_caption = f"Minh họa thao tác ứng dụng - Trang {page_index + 1} (Khung {idx + 1}/{total})"
            else:
                display_caption = f"Minh họa thao tác ứng dụng - Trang {page_index + 1}"
        else:
            display_caption = f"Tổng quan hướng dẫn thao tác - Trang {page_index + 1}"
        meta = {
            "kind": "cropped_region",
            "region_type": region_type,
            "page_index": int(page_index),
            "page_number": int(page_index + 1),
            "region_index": idx,
            "region_count": total,
            "bbox": [round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4)],
            "source_page_sha256": page_sha256,
            "display_caption": display_caption,
        }
        crops.append(_crop_and_save(pil_image, (x0, y0, x1, y1), asset_dir, meta))
    return crops


def _classify_a4_region(sub_lum, sub_sat, reg_hrule, reg_bg, height, width, internal_hrule_rows=0):
    top_slice_h = max(4, min(int(height * 0.22), sub_lum.shape[0]))
    top_lum = sub_lum[:top_slice_h]
    top_dark_rows = int(((top_lum < 115).mean(axis=1) >= 0.72).sum())

    body_y0 = min(sub_lum.shape[0] - 1, max(1, int(height * 0.24)))
    body_y1 = max(body_y0 + 1, int(height * 0.90))
    body_sat = float(((sub_sat[body_y0:body_y1] > 40) & (sub_lum[body_y0:body_y1] < 245)).mean())

    if top_dark_rows >= 8 and body_sat < 0.08:
        return "table"
    if body_sat < 0.025 and internal_hrule_rows >= 3 and reg_bg >= 0.25 and height >= int(width * 0.14):
        return "table"
    if height < int(width * 0.22) and reg_bg >= 0.50 and reg_hrule <= 12:
        # Check for horizontal white gutters splitting side-by-side KPI cards
        mid_y0 = sub_lum.shape[0] // 4
        mid_y1 = max(mid_y0 + 1, (sub_lum.shape[0] * 3) // 4)
        col_white = (sub_lum[mid_y0:mid_y1] > 250).all(axis=0)
        white_runs = _true_intervals(col_white)
        internal_gutters = 0
        for g_start, g_end in white_runs:
            if g_start > int(width * 0.08) and g_end < int(width * 0.92) and (g_end - g_start) >= 12:
                internal_gutters += 1
        if internal_gutters >= 1:
            return "visual_card"
    return "figure"


def _extract_light_a4_crops(pil_image, arr, page_index, page_sha256, asset_dir):
    h, w, _ = arr.shape
    y_start = int(h * 0.072)
    y_end = int(h * 0.925)
    body = arr[y_start:y_end].astype(np.int32)
    r = body[:, :, 0]
    g = body[:, :, 1]
    b = body[:, :, 2]
    lum = (299 * r + 587 * g + 114 * b) // 1000
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)

    non_white = lum < 235
    win = max(20, int(w * 0.25))
    csum = np.cumsum(non_white.astype(np.int32), axis=1)
    runs = csum[:, win:] - csum[:, :-win]
    has_hrule = np.max(runs, axis=1) >= int(win * 0.97)

    color_px = (sat > 40) & (lum < 245)
    has_color = color_px.mean(axis=1) >= 0.055

    x_l = int(w * 0.07)
    x_r = int(w * 0.93)
    bg_px = (lum[:, x_l:x_r] >= 205) & (lum[:, x_l:x_r] <= 251)
    has_box_bg = bg_px.mean(axis=1) >= 0.35

    vwin = max(12, int(h * 0.025))
    vcsum = np.cumsum(non_white.astype(np.int32), axis=0)
    vruns = vcsum[vwin:, :] - vcsum[:-vwin, :]
    vborder_px = np.zeros_like(non_white, dtype=bool)
    half_v = vwin // 2
    vborder_px[half_v:half_v + vruns.shape[0], :] = vruns >= int(vwin * 0.92)
    has_vborder = vborder_px.sum(axis=1) >= 2

    visual = has_hrule | has_color | has_box_bg | has_vborder
    idx_true = np.where(visual)[0]
    if len(idx_true) == 0:
        return []

    spans = []
    cur_s = int(idx_true[0])
    cur_e = int(idx_true[0])
    gap_max = max(6, int(h * 0.008))
    for idx_val in idx_true[1:]:
        idx = int(idx_val)
        if idx - cur_e <= gap_max:
            cur_e = idx
        else:
            spans.append((cur_s, cur_e + 1))
            cur_s = idx
            cur_e = idx
    spans.append((cur_s, cur_e + 1))

    candidates = []
    min_h = int(h * 0.055)
    min_w = int(w * 0.28)
    for y0, y1 in spans:
        region_h = y1 - y0
        if region_h < min_h:
            continue
        sub_lum = lum[y0:y1]
        sub_sat = sat[y0:y1]
        xs = np.where((sub_lum < 248).any(axis=0))[0]
        if len(xs) == 0:
            continue
        x0 = int(xs[0])
        x1 = int(xs[-1] + 1)
        region_w = x1 - x0
        if region_w < min_w:
            continue
        reg_sat = float(((sub_sat[:, x0:x1] > 40) & (sub_lum[:, x0:x1] < 245)).mean())
        reg_bg = float(((sub_lum[:, x0:x1] >= 205) & (sub_lum[:, x0:x1] <= 251)).mean())
        reg_hrule = int(has_hrule[y0:y1].sum())
        mid_s = y0 + max(1, int(region_h * 0.12))
        mid_e = max(mid_s, y1 - max(1, int(region_h * 0.12)))
        internal_hrule_rows = int(has_hrule[mid_s:mid_e].sum())
        reg_vb = int(has_vborder[y0:y1].sum())
        if reg_sat < 0.08 and reg_hrule < 2 and reg_bg < 0.18 and reg_vb < int(region_h * 0.40):
            continue
        region_type = _classify_a4_region(
            sub_lum[:, x0:x1],
            sub_sat[:, x0:x1],
            reg_hrule,
            reg_bg,
            region_h,
            region_w,
            internal_hrule_rows=internal_hrule_rows,
        )
        if region_type == "table":
            priority = 0
        elif region_type == "figure":
            priority = 1
        else:
            priority = 2
        area = region_w * region_h
        candidates.append((priority, -area, y0, y1, x0, x1, region_type))

    if len(candidates) > _MAX_CROPS_PER_PAGE:
        candidates.sort()
        candidates = candidates[:_MAX_CROPS_PER_PAGE]
    # Present retained crops in natural top-to-bottom reading order.
    candidates.sort(key=lambda item: (item[2], item[4]))

    total = len(candidates)
    pad = 14
    crops = []
    for idx, item in enumerate(candidates):
        _, _, y0, y1, x0, x1, region_type = item
        crop_x0 = max(0, x0 - pad)
        crop_x1 = min(w, x1 + pad)
        crop_y0 = max(0, y_start + y0 - pad)
        crop_y1 = min(h, y_start + y1 + pad)
        if region_type == "table":
            label_prefix = "Bảng biểu tài liệu"
        elif region_type == "visual_card":
            label_prefix = "Thẻ chỉ số trực quan"
        else:
            label_prefix = "Sơ đồ / Biểu đồ minh họa"
        display_caption = f"{label_prefix} - Trang {page_index + 1}"
        meta = {
            "kind": "cropped_region",
            "region_type": region_type,
            "page_index": int(page_index),
            "page_number": int(page_index + 1),
            "region_index": idx,
            "region_count": total,
            "bbox": [round(crop_x0 / w, 4), round(crop_y0 / h, 4), round(crop_x1 / w, 4), round(crop_y1 / h, 4)],
            "source_page_sha256": page_sha256,
            "display_caption": display_caption,
        }
        crops.append(_crop_and_save(pil_image, (crop_x0, crop_y0, crop_x1, crop_y1), asset_dir, meta))
    return crops


def extract_page_visual_crops(pixels, page_index, page_sha256, asset_dir, source_name=""):
    """Detect and save cropped visual regions (app steps, tables, figures) from a rendered page."""
    if not pixels or asset_dir is None:
        return []
    with Image.open(io.BytesIO(pixels)) as opened:
        rgb_image = opened.convert("RGB")
        width, height = rgb_image.size
        if width < _MIN_PAGE_DIMENSION or height < _MIN_PAGE_DIMENSION:
            return []
        arr = np.asarray(rgb_image)
        if _is_dark_slide_page(arr):
            return _extract_dark_slide_crops(rgb_image, arr, page_index, page_sha256, asset_dir)
        return _extract_light_a4_crops(rgb_image, arr, page_index, page_sha256, asset_dir)


def extract_step_numbers_from_text(text):
    """Extract ordered step integers following 'bước' or 'buoc' using string scanning only."""
    if not text:
        return []
    folded = str(text).casefold()
    found = []
    seen = set()
    for marker in ("bước ", "buoc ", "bước:", "buoc:"):
        cursor = 0
        marker_len = len(marker)
        while True:
            pos = folded.find(marker, cursor)
            if pos < 0:
                break
            idx = pos + marker_len
            text_len = len(folded)
            while idx < text_len and folded[idx] in " :.-":
                idx += 1
            num_start = idx
            while idx < text_len and folded[idx].isdigit():
                idx += 1
            if idx > num_start:
                step_val = int(folded[num_start:idx])
                if step_val not in seen and 1 <= step_val <= 50:
                    seen.add(step_val)
                    found.append(step_val)
            cursor = pos + marker_len
    return found


def collect_page_step_numbers(blocks):
    """Build {page_index: sorted_unique_step_numbers} from document blocks."""
    by_page = {}
    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        prov = block.get("provenance")
        if not isinstance(prov, dict):
            continue
        page_idx = prov.get("page_index")
        if type(page_idx) is not int:
            continue
        steps = extract_step_numbers_from_text(block.get("text", ""))
        if not steps:
            continue
        bucket = by_page.setdefault(page_idx, set())
        for step in steps:
            bucket.add(step)
    return {page_idx: sorted(values) for page_idx, values in by_page.items()}


def _build_cropped_entry(asset, doc_label, match_strength):
    sha = str(asset.get("sha256") or "")
    caption = str(asset.get("display_caption") or "").strip()
    if not caption:
        if doc_label:
            caption = doc_label
        else:
            caption = "Hình ảnh minh họa tài liệu"
    return {
        "asset_id": sha,
        "sha256": sha,
        "path": str(asset.get("path") or ""),
        "kind": str(asset.get("kind") or "cropped_region"),
        "region_type": str(asset.get("region_type") or "figure"),
        "page_index": asset.get("page_index"),
        "page_number": asset.get("page_number"),
        "region_index": asset.get("region_index", 0),
        "region_count": asset.get("region_count", 1),
        "bbox": list(asset.get("bbox") or []),
        "width": int(asset.get("width") or 0),
        "height": int(asset.get("height") or 0),
        "match_strength": match_strength,
        "display_caption": caption,
    }


def _match_page_crops(chunk, page_crops, page_steps, doc_label):
    if not page_crops:
        return []
    app_crops = [crop for crop in page_crops if crop.get("region_type") in ("app_step", "slide_visual")]
    if app_crops:
        step_crops = [crop for crop in app_crops if crop.get("region_type") == "app_step"]
        if len(step_crops) > 1:
            chunk_text = str(chunk.get("section_title") or "") + "\n" + str(chunk.get("content") or "")
            chunk_steps = extract_step_numbers_from_text(chunk_text)
            if len(chunk_steps) == 1 and page_steps and len(page_steps) == len(step_crops):
                target_step = chunk_steps[0]
                if target_step in page_steps:
                    pos = page_steps.index(target_step)
                    return [_build_cropped_entry(step_crops[pos], doc_label, "direct")]
        return [_build_cropped_entry(crop, doc_label, "direct") for crop in app_crops]

    chunk_type = str(chunk.get("chunk_type") or "text")
    combined_lower = (str(chunk.get("section_title") or "") + " " + str(chunk.get("content") or "")).lower()
    wants_table = chunk_type == "table" or any(kw in combined_lower for kw in _TABLE_KEYWORDS)
    wants_figure = any(kw in combined_lower for kw in _FIGURE_KEYWORDS)

    selected = []
    if wants_table and not wants_figure:
        tables = [crop for crop in page_crops if crop.get("region_type") == "table"]
        if tables:
            selected = [_build_cropped_entry(crop, doc_label, "direct") for crop in tables]
    elif wants_figure and not wants_table:
        figures = [crop for crop in page_crops if crop.get("region_type") in ("figure", "visual_card")]
        if figures:
            selected = [_build_cropped_entry(crop, doc_label, "direct") for crop in figures]
    elif wants_table and wants_figure:
        selected = [_build_cropped_entry(crop, doc_label, "direct") for crop in page_crops]

    if selected:
        return selected[:3]

    # Attach limited page context when the page holds structured figures or tables.
    meaningful = [crop for crop in page_crops if crop.get("region_type") in ("table", "figure")]
    if meaningful:
        return [_build_cropped_entry(crop, doc_label, "page_context") for crop in meaningful[:2]]
    return []


def match_chunk_visual_assets(chunk, ir_assets, page_step_numbers=None):
    """Resolve and deduplicate visual assets associated with a chunk without altering BM25 keys."""
    existing = chunk.get("images")
    if isinstance(existing, list):
        resolved = [dict(item) for item in existing if isinstance(item, dict)]
    else:
        resolved = []

    seen_shas = set()
    for item in resolved:
        sha = item.get("sha256") or item.get("asset_id")
        if sha:
            seen_shas.add(str(sha))

    if not isinstance(ir_assets, dict) or not ir_assets:
        return resolved

    meta = chunk.get("business_metadata")
    if not isinstance(meta, dict):
        meta = {}
    prov = meta.get("provenance")
    if not isinstance(prov, dict):
        prov = {}
    locations = prov.get("locations")
    if not isinstance(locations, list):
        locations = []

    doc_label = str(chunk.get("section_title") or chunk.get("breadcrumb") or "").strip()

    # 1. DOCX embedded images referenced directly by image_asset_sha256
    for loc in locations:
        if not isinstance(loc, dict):
            continue
        sha = loc.get("image_asset_sha256")
        if not sha or str(sha) in seen_shas:
            continue
        asset = ir_assets.get(sha)
        if not isinstance(asset, dict) or not asset.get("path"):
            continue
        seen_shas.add(str(sha))
        if doc_label:
            caption = doc_label
        else:
            caption = "Hình minh họa tài liệu"
        resolved.append({
            "asset_id": str(sha),
            "sha256": str(sha),
            "path": str(asset["path"]),
            "kind": str(asset.get("kind") or "embedded_image"),
            "region_type": "embedded_figure",
            "width": int(asset.get("width") or 0),
            "height": int(asset.get("height") or 0),
            "match_strength": "direct",
            "display_caption": caption,
        })

    # 2. PDF cropped regions indexed by page_index
    crops_by_page = {}
    for asset in ir_assets.values():
        if not isinstance(asset, dict):
            continue
        if asset.get("kind") != "cropped_region":
            continue
        page_idx = asset.get("page_index")
        if type(page_idx) is int:
            crops_by_page.setdefault(page_idx, []).append(asset)
    for page_idx in crops_by_page:
        crops_by_page[page_idx].sort(key=lambda item: int(item.get("region_index", 0)))

    if crops_by_page:
        page_indices = []
        seen_pages = set()
        for loc in locations:
            if not isinstance(loc, dict):
                continue
            page_idx = loc.get("page_index")
            if type(page_idx) is int and page_idx not in seen_pages:
                seen_pages.add(page_idx)
                page_indices.append(page_idx)
        step_map = page_step_numbers or {}
        for page_idx in page_indices:
            page_crops = crops_by_page.get(page_idx)
            if not page_crops:
                continue
            matched = _match_page_crops(chunk, page_crops, step_map.get(page_idx), doc_label)
            for item in matched:
                sha = item.get("sha256")
                if not sha or sha in seen_shas:
                    continue
                seen_shas.add(sha)
                resolved.append(item)
                if len(resolved) >= 6:
                    return resolved

    return resolved
