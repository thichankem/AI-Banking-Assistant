from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import urllib.request

import pymupdf
import pytest
from docx import Document
from PIL import Image, ImageDraw

from agent_banking.chunking.golden_suite import build_corpus
from agent_banking.chunking.universal.models import preflight, text_client
from agent_banking.chunking.universal.pipeline import UniversalDocumentChunker
from agent_banking.core.fallback import ModelEndpoint
from agent_banking.core.models import chain_fingerprint, model_primary


def test_profile_uses_environment_and_invalidates_cache(monkeypatch):
    monkeypatch.setenv("CHUNKING_VISION_MODEL", "some-unfamiliar-vision-model")
    monkeypatch.setenv("CHUNKING_VISION_BASE_URL", "https://test.invalid/v1")
    monkeypatch.setenv("CHUNKING_VISION_API_KEY", "fixture")
    monkeypatch.setenv("CHUNKING_VISION_TEMPERATURE", "0.25")
    monkeypatch.setenv("CHUNKING_VISION_TOKEN_PARAMETER", "max_completion_tokens")
    monkeypatch.setenv("CHUNKING_VISION_MAX_OUTPUT_TOKENS", "320")
    monkeypatch.setenv("CHUNKING_VISION_JSON_MODE", "off")
    endpoint = model_primary("chunking.vision")
    assert endpoint.model == "some-unfamiliar-vision-model"
    initial = chain_fingerprint("chunking.vision")
    request = {"max_tokens": 1024, "temperature": 0, "response_format": {"type": "json_object"}, "messages": []}
    limit = text_client.apply_chat_profile(request, endpoint, 4096)
    assert limit == 320
    assert request == {"max_completion_tokens": 320, "temperature": 0.25, "messages": []}
    monkeypatch.setenv("CHUNKING_VISION_TEMPERATURE", "0.4")
    assert chain_fingerprint("chunking.vision") != initial
    changed = chain_fingerprint("chunking.vision")
    monkeypatch.setenv("CHUNKING_VISION_API_KEY", "new-fixture")
    assert chain_fingerprint("chunking.vision") == changed


def test_preflight_reports_candidates_without_using_fallback(monkeypatch):
    endpoint = ModelEndpoint("first", "custom-vision", "https://test.invalid", "secret-fixture", max_output_tokens=4000)
    bad = ModelEndpoint("second", "disabled-credential", "https://test.invalid", "", max_output_tokens=4000)
    chains = {"text": [endpoint], "vision": [endpoint], "vision_review": [bad], "decision": []}
    observed = []
    monkeypatch.setattr(preflight, "get_model_chain", lambda name: chains[name.split(".")[1]])
    monkeypatch.setattr(preflight, "_probe", lambda role, candidate: observed.append((role, candidate.model)))
    results = preflight.check_models()
    assert observed == [("text", endpoint.model), ("vision", endpoint.model)]
    assert [result["status"] for result in results] == ["ok", "ok", "failed", "disabled"]
    assert "secret-fixture" not in repr(results)


def test_multilingual_default_and_opt_in_docx_image_source(tmp_path, monkeypatch):
    def unexpected_network(*args, **kwargs):
        raise AssertionError("A diagnostic with an injected vision stub must not call the network")

    monkeypatch.setattr(urllib.request, "urlopen", unexpected_network)
    source = Document()
    source.add_paragraph("Reimbursement requires approval.")
    figure = BytesIO()
    with Image.new("RGB", (210, 80), "white") as image:
        ImageDraw.Draw(image).text((8, 10), "Torque 25 Nm", fill="black")
        image.save(figure, format="PNG")
    source.add_picture(BytesIO(figure.getvalue()))
    source.add_table(rows=1, cols=1).cell(0, 0).paragraphs[0].add_run().add_picture(BytesIO(figure.getvalue()))
    source.add_paragraph("審査期間は五日です。")
    data = BytesIO()
    source.save(data)
    calls = []
    def cloud_read(pixels, instructions):
        calls.append(pixels)
        return {"elements": [{"type": "caption", "text": "Torque 25 Nm", "bbox": [0.1, 0.1, 0.9, 0.8]}],
                "review": {"verdict": "accepted"}, "uncertainties": []}

    stub = SimpleNamespace(model="fixture-vision", analyze_page_layout=cloud_read)
    options = dict(cache_db_path=tmp_path / "cache.db", asset_dir=tmp_path / "assets",
                   use_llm=False, use_decision=False, vision_client=stub)
    with UniversalDocumentChunker(**options) as chunker:
        baseline = chunker.fingerprint()
        chunks = chunker.chunk_docx(docx_bytes=data.getvalue(), source_name="manual.docx")
        assert "Reimbursement" in " ".join(c["content"] for c in chunks)
        assert "審査期間" in " ".join(c["content"] for c in chunks)
        assert not calls
    with UniversalDocumentChunker(**options, ocr_docx_images=True) as chunker:
        assert chunker.fingerprint() != baseline
        chunks = chunker.chunk_docx(docx_bytes=data.getvalue(), source_name="manual.docx")
        assert "Torque 25 Nm" in " ".join(c["content"] for c in chunks)
        assert all(chunk["business_metadata"]["citation"] == {"file_name": "manual.docx"}
                   for chunk in chunks)
        evidence = chunker.last_artifact["document"]["docx_image_reviews"]
        assert len(calls) == 1 and len(evidence) == 2
        assert evidence[0]["asset_sha256"] == evidence[1]["asset_sha256"]
        assert evidence[0]["xpath"] != evidence[1]["xpath"]


def test_golden_corpus_is_multilingual_and_covers_authored_and_scanned_pages(tmp_path):
    expected = build_corpus(tmp_path)
    assert len(expected) == 24
    assert len({Path(source).parts[0] for source in expected}) == 6
    with pymupdf.open(tmp_path / "economics/economics-3.pdf") as pdf:
        assert len(pdf) == 2
        assert "Final-48" in pdf[1].get_text()
    with pymupdf.open(tmp_path / "legal/legal-4.pdf") as scan:
        assert not scan[0].get_text()
        assert scan[0].get_images()
    japanese = Document(tmp_path / "academic/academic-2.docx")
    assert any("調査期間" in paragraph.text for paragraph in japanese.paragraphs)


def test_docx_preserves_referenced_footnote_and_source_anchor(tmp_path):
    from docx.opc.constants import RELATIONSHIP_TYPE
    from docx.opc.packuri import PackURI
    from docx.opc.part import Part
    from docx.oxml import OxmlElement
    from docx.oxml.ns import nsdecls, qn

    document = Document()
    paragraph = document.add_paragraph("Article 7 preserves the appeal period.")
    reference = OxmlElement("w:footnoteReference")
    reference.set(qn("w:id"), "5")
    paragraph.add_run()._r.append(reference)
    notes = ('<w:footnotes ' + nsdecls("w") + '><w:footnote w:id="5"><w:p><w:r><w:t>'
             'Except for public holidays.</w:t></w:r></w:p></w:footnote></w:footnotes>')
    part = Part(PackURI("/word/footnotes.xml"),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml",
                notes.encode("utf-8"), document.part.package)
    document.part.relate_to(part, RELATIONSHIP_TYPE.FOOTNOTES)
    data = BytesIO()
    document.save(data)

    with UniversalDocumentChunker(asset_dir=tmp_path, cache_db_path=":memory:", use_llm=False,
                                  use_decision=False, search_language="all") as chunker:
        chunks = chunker.chunk_docx(docx_bytes=data.getvalue(), source_name="regulation.docx")
        assert "Except for public holidays." in " ".join(c["content"] for c in chunks)
        note = chunker.last_artifact["document"]["blocks"][-1]
        assert note["provenance"]["part"] == "word/footnotes.xml"
        assert note["provenance"]["note_references"][0].endswith("/w:footnoteReference")


def test_unreadable_image_only_docx_does_not_publish_as_empty(tmp_path):
    from agent_banking.chunking.universal.extraction.pdf.page_review import CloudOCRUnavailable

    source = Document()
    figure = BytesIO()
    with Image.new("RGB", (50, 50), "black") as image:
        image.save(figure, format="PNG")
    figure.seek(0)
    source.add_picture(figure)
    data = BytesIO()
    source.save(data)

    def unavailable(*args):
        raise RuntimeError("Vision provider unavailable")

    stub = SimpleNamespace(model="fixture-vision", analyze_page_layout=unavailable)
    with UniversalDocumentChunker(asset_dir=tmp_path, cache_db_path=":memory:", use_llm=False, use_decision=False,
                                  vision_client=stub, ocr_docx_images=True) as chunker:
        with pytest.raises(CloudOCRUnavailable, match="Image-only DOCX"):
            chunker.chunk_docx(docx_bytes=data.getvalue(), source_name="scan.docx")
        assert chunker.last_artifact is None


def test_common_provider_verdict_keeps_uncertainty_separate():
    from agent_banking.chunking.universal.extraction.pdf.page_review import _prepare_layout

    reading = {"elements": [{"type": "paragraph", "text": "Voltage 48 V DC", "bbox": [0.1, 0.1, 0.9, 0.9]}],
               "review": {"verdict": "ok"}, "uncertainties": []}
    clean = _prepare_layout(reading)["visual_review"]
    assert clean["provider_verdict"] == "ok"
    assert clean["issues"] == [] and not clean["incomplete"]
    reading["uncertainties"] = [{"text": "48 V"}]
    partial = _prepare_layout(reading)["visual_review"]
    assert partial["issues"] == ["model_uncertainty"] and partial["incomplete"]
