"""Tests for PDF visual cropping, chunk visual matching, agent citation image filtering, and asset serving."""

import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from agent_banking.chunking.universal.extraction.pdf.visual_cropper import (
    collect_page_step_numbers,
    extract_page_visual_crops,
    extract_step_numbers_from_text,
    match_chunk_visual_assets,
)


def _png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_dark_slide_extracts_two_ordered_app_step_crops(tmp_path):
    w, h = 1200, 800
    img = Image.new("RGB", (w, h), (19, 50, 88))
    draw = ImageDraw.Draw(img)
    # Header and footer bands
    draw.rectangle((40, 20, 1160, 60), fill=(255, 255, 255))
    draw.rectangle((40, 745, 1160, 775), fill=(160, 180, 200))
    # Left phone card (step 1) and right phone card (step 2)
    draw.rounded_rectangle((180, 120, 480, 690), radius=24, fill=(245, 248, 252), outline=(210, 170, 50), width=4)
    draw.rounded_rectangle((700, 120, 1000, 690), radius=24, fill=(245, 248, 252), outline=(210, 170, 50), width=4)

    pixels = _png_bytes(img)
    sha = hashlib.sha256(pixels).hexdigest()
    crops = extract_page_visual_crops(pixels, 0, sha, tmp_path, source_name="HDSD 03.pdf")

    assert len(crops) == 2
    assert [c["region_type"] for c in crops] == ["app_step", "app_step"]
    assert crops[0]["bbox"][0] < crops[1]["bbox"][0]
    for crop in crops:
        saved = Path(crop["path"])
        assert saved.is_file()
        assert hashlib.sha256(saved.read_bytes()).hexdigest() == crop["sha256"]


def test_light_a4_extracts_table_and_figure_and_skips_plain_text(tmp_path):
    w, h = 800, 1100
    # 1. Pure text page -> 0 crops
    plain = Image.new("RGB", (w, h), (255, 255, 255))
    plain_draw = ImageDraw.Draw(plain)
    for y in range(140, 900, 32):
        plain_draw.rectangle((70, y, 520, y + 8), fill=(25, 25, 25))
    assert extract_page_visual_crops(_png_bytes(plain), 0, "a" * 64, tmp_path) == []

    # 2. A4 page with structured table and colored figure box
    rich = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(rich)
    # Table (dark header + bordered rows)
    draw.rectangle((70, 180, 730, 225), fill=(15, 34, 64))
    draw.rectangle((70, 225, 730, 390), fill=(242, 246, 250), outline=(180, 190, 205), width=2)
    draw.line((70, 280, 730, 280), fill=(180, 190, 205), width=2)
    draw.line((70, 335, 730, 335), fill=(180, 190, 205), width=2)
    # Colored diagram/figure box below
    draw.rectangle((70, 520, 730, 760), fill=(240, 247, 255), outline=(59, 130, 246), width=3)
    draw.rectangle((140, 565, 340, 640), fill=(37, 99, 235))
    draw.rectangle((450, 565, 650, 640), fill=(217, 119, 6))

    crops = extract_page_visual_crops(_png_bytes(rich), 1, "b" * 64, tmp_path)
    assert len(crops) == 2
    assert crops[0]["region_type"] == "table"
    assert crops[1]["region_type"] == "figure"


def test_small_image_guard_returns_empty(tmp_path):
    small = Image.new("RGB", (200, 200), (19, 50, 88))
    assert extract_page_visual_crops(_png_bytes(small), 0, "c" * 64, tmp_path) == []


def test_step_extraction_and_chunk_matching_for_docx_and_pdf():
    assert extract_step_numbers_from_text("Bước 3: Chọn Face ID và Bước 4 xác nhận") == [3, 4]
    blocks = [
        {"text": "Bước 3: Mở bảo mật", "provenance": {"page_index": 1}},
        {"text": "Bước 4: Nhập Smart OTP", "provenance": {"page_index": 1}},
    ]
    step_map = collect_page_step_numbers(blocks)
    assert step_map == {1: [3, 4]}

    sha_docx = "1" * 64
    sha_step3 = "2" * 64
    sha_step4 = "3" * 64
    ir_assets = {
        sha_docx: {"path": f"assets/{sha_docx}.png", "sha256": sha_docx, "kind": "embedded_image", "width": 600, "height": 300},
        sha_step3: {
            "path": f"assets/{sha_step3}.png", "sha256": sha_step3, "kind": "cropped_region",
            "region_type": "app_step", "page_index": 1, "page_number": 2, "region_index": 0, "region_count": 2,
            "bbox": [0.1, 0.2, 0.4, 0.8], "width": 400, "height": 900, "display_caption": "Bước 3",
        },
        sha_step4: {
            "path": f"assets/{sha_step4}.png", "sha256": sha_step4, "kind": "cropped_region",
            "region_type": "app_step", "page_index": 1, "page_number": 2, "region_index": 1, "region_count": 2,
            "bbox": [0.5, 0.2, 0.8, 0.8], "width": 400, "height": 900, "display_caption": "Bước 4",
        },
    }

    docx_chunk = {
        "chunk_id": "c1", "chunk_type": "text", "section_title": "Sơ đồ quy trình", "content": "Chi tiết quy trình",
        "business_metadata": {"provenance": {"locations": [{"image_asset_sha256": sha_docx}]}}, "images": [],
    }
    matched_docx = match_chunk_visual_assets(docx_chunk, ir_assets)
    assert [m["sha256"] for m in matched_docx] == [sha_docx]
    assert "display_caption" in matched_docx[0]
    assert "caption" not in matched_docx[0]

    pdf_step4_chunk = {
        "chunk_id": "c2", "chunk_type": "text", "section_title": "Bước 4: Xác thực OTP",
        "content": "Nhập mã Smart OTP để hoàn tất.",
        "business_metadata": {"provenance": {"locations": [{"page_index": 1}]}}, "images": [],
    }
    matched_pdf = match_chunk_visual_assets(pdf_step4_chunk, ir_assets, page_step_numbers=step_map)
    assert [m["sha256"] for m in matched_pdf] == [sha_step4]


def test_resolve_cited_images_filters_and_deduplicates():
    from agent_banking.agent.deep import _resolve_cited_images

    sha_a = "a" * 64
    sha_b = "b" * 64
    chunks = [
        {
            "chunk_id": "c1",
            "business_metadata": {"document_name": "HDSD 03"},
            "images": [
                {"asset_id": sha_a, "sha256": sha_a, "region_type": "app_step", "page_number": 2,
                 "match_strength": "direct", "display_caption": "Bước 4", "width": 400, "height": 800},
            ],
        },
        {
            "chunk_id": "c2",
            "business_metadata": {"document_name": "BH 01"},
            "images": [
                {"asset_id": sha_b, "sha256": sha_b, "region_type": "table", "page_number": 5,
                 "match_strength": "direct", "display_caption": "Bảng quyền lợi", "width": 900, "height": 500},
            ],
        },
    ]
    citations = [{"index": 1, "label": "[Nguồn 1]", "source": "HDSD 03.pdf", "chunk_id": "c1", "section_title": "Bước 4"}]

    assert _resolve_cited_images(citations, chunks, "BLOCKED") == []
    images = _resolve_cited_images(citations, chunks, "PASS")
    assert len(images) == 1
    assert images[0]["asset_id"] == sha_a
    assert images[0]["url"] == "/api/assets/" + sha_a
    assert images[0]["citation_label"] == "[Nguồn 1]"


def test_fastapi_asset_endpoint_validation_and_serving(monkeypatch):
    import main as web_main

    with TemporaryDirectory() as tmp:
        shared = Path(tmp) / "shared"
        shared.mkdir()
        valid_id = "d" * 64
        img_bytes = _png_bytes(Image.new("RGB", (24, 24), (10, 20, 30)))
        (shared / (valid_id + ".png")).write_bytes(img_bytes)

        monkeypatch.setattr(web_main, "_SHARED_ASSETS_DIR", shared.resolve())
        monkeypatch.setattr(web_main, "_PUBLICATION_KNOWLEDGE_DIR", Path(tmp) / "missing_knowledge")
        monkeypatch.setattr(web_main, "_ASSET_INDEX_CACHE", {})

        client = TestClient(web_main.app)
        bad_resp = client.get("/api/assets/..%2F..%2Fetc%2Fpasswd")
        assert bad_resp.status_code in (400, 404)
        non_hex = client.get("/api/assets/" + ("z" * 64))
        assert non_hex.status_code == 400
        missing = client.get("/api/assets/" + ("e" * 64))
        assert missing.status_code == 404
        ok = client.get("/api/assets/" + valid_id)
        assert ok.status_code == 200
        assert ok.headers["content-type"] == "image/png"
        assert ok.content == img_bytes
