"""Build and score the reproducible PDF/DOCX chunking benchmark."""

import argparse
from io import BytesIO
import json
from pathlib import Path
import shutil
import tempfile
import unicodedata
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from docx import Document
from docx.oxml import OxmlElement
import pymupdf

from .config.settings import get_golden_suite_settings
from .run_chunking import load_published_chunks, run_auto_chunking

_GOLDEN = get_golden_suite_settings()
_RENDER = _GOLDEN["render"]

SOURCE_CASES = {
    domain: tuple(tuple(case) for case in cases)
    for domain, cases in _GOLDEN["source_cases"].items()
}
SOURCE_REVISION = _GOLDEN["source_revision"]
TABLE_FACTS = tuple(_GOLDEN["table_facts"])
FIGURE_LABEL = _GOLDEN["figure_label"]
REAL_CASES = dict(_GOLDEN["real_cases"])


def _word(path, title, facts, structured):
    document = Document()
    document.add_heading(title, 0)
    document.add_heading("Source requirements", 1)
    document.add_paragraph(facts[0])
    if structured:
        table = document.add_table(rows=3, cols=2)
        header = table.cell(0, 0).merge(table.cell(0, 1))
        header.text = _RENDER["word_table_header"]
        row_labels = _RENDER["word_table_rows"]
        table.cell(1, 0).text = row_labels[0]
        table.cell(1, 1).text = TABLE_FACTS[0]
        table.cell(2, 0).text = row_labels[1]
        table.cell(2, 1).text = TABLE_FACTS[1]
        math = OxmlElement("m:oMath")
        run = OxmlElement("m:r")
        text = OxmlElement("m:t")
        text.text = _RENDER["word_equation"]
        run.append(text)
        math.append(run)
        document.add_paragraph()._p.append(math)
        image = BytesIO()
        with pymupdf.open() as figure:
            page = figure.new_page(width=_RENDER["word_figure_width"], height=_RENDER["word_figure_height"])
            page.insert_text(tuple(_RENDER["word_figure_pos"]), FIGURE_LABEL, fontsize=_RENDER["word_figure_fontsize"])
            image.write(page.get_pixmap().tobytes("png"))
        image.seek(0)
        document.add_picture(image)
        document.add_paragraph(_RENDER["pdf_caption_text"], style="Caption")
    document.add_paragraph(facts[1])
    package = BytesIO()
    document.save(package)
    with ZipFile(package) as original, ZipFile(path, "w", compression=ZIP_DEFLATED) as frozen:
        for name in original.namelist():
            frozen.writestr(ZipInfo(name), original.read(name), compress_type=ZIP_DEFLATED)


def _place(page, text, box, fontsize=None):
    if fontsize is None:
        fontsize = _RENDER["pdf_default_fontsize"]
    font = pymupdf.Font(_RENDER["pdf_font"])
    writer = pymupdf.TextWriter(page.rect)
    if writer.fill_textbox(pymupdf.Rect(*box), text, font=font, fontsize=fontsize):
        raise ValueError("Benchmark text exceeded its page box")
    writer.write_text(page)


def _pdf(path, title, facts, scanned, repeated_table):
    with pymupdf.open() as authored:
        page = authored.new_page()
        _place(page, title, tuple(_RENDER["pdf_title_box"]), _RENDER["pdf_title_fontsize"])
        if scanned:
            _place(page, facts[0], (42, 110, 277, 240))
            _place(page, facts[1], (312, 110, 553, 240))
            page.draw_rect((315, 250, 547, 351), color=(0, 0, 0))
            _place(page, FIGURE_LABEL, (326, 270, 530, 342))
            _place(page, _RENDER["pdf_caption_text"], (322, 355, 550, 420), 12)
        else:
            _place(page, facts[0], (42, 110, 553, 190))
            _place(page, facts[1], (42, 210, 553, 300))
        if repeated_table:
            for index, fact in enumerate(TABLE_FACTS):
                if index:
                    page = authored.new_page()
                if index:
                    y = 100
                else:
                    y = 400
                for x in (42, 260):
                    for top in (y, y + 36):
                        page.draw_rect((x, top, x + 218, top + 36), color=(0, 0, 0))
                for col, label in enumerate(_RENDER["pdf_table_headers"]):
                    x = 50 + col * 218
                    _place(page, label, (x, y + 6, x + 200, y + 34), 12)
                _place(page, str(index + 1), (50, y + 42, 243, y + 70), 12)
                _place(page, fact, (271, y + 42, 471, y + 70), 12)
        if not scanned:
            authored.save(path, no_new_id=True)
        else:
            with pymupdf.open() as raster:
                for original in authored:
                    scanned_page = raster.new_page(width=original.rect.width, height=original.rect.height)
                    scanned_page.insert_image(scanned_page.rect, stream=original.get_pixmap(dpi=_RENDER["pdf_dpi"]).tobytes("png"))
                raster.save(path, no_new_id=True)


def build_corpus(incoming, regression_input=None):
    expected = {}
    for domain, cases in SOURCE_CASES.items():
        folder = incoming / domain
        folder.mkdir(parents=True, exist_ok=True)
        for variant, (title, *facts) in enumerate(cases):
            is_pdf = variant >= 2
            if is_pdf:
                ext = ".pdf"
            else:
                ext = ".docx"
            path = folder / (domain + "-" + str(variant + 1) + ext)
            probes = list(facts)
            if is_pdf:
                table = domain == "economics" and variant == 2
                _pdf(path, title, facts, scanned=variant == 3, repeated_table=table)
                if table:
                    probes.extend(TABLE_FACTS)
                if variant == 3:
                    probes.append(FIGURE_LABEL)
            else:
                structured = variant == 1
                _word(path, title, facts, structured)
                if structured:
                    probes.extend((*TABLE_FACTS, FIGURE_LABEL))
            expected[path.relative_to(incoming).as_posix()] = probes
    if regression_input is not None:
        for prefix, probes in REAL_CASES.items():
            matches = sorted(regression_input.rglob(prefix + " *.pdf"))
            if len(matches) != 1:
                raise ValueError("Expected exactly one regression source for " + prefix)
            path = incoming / "regressions" / matches[0].name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(matches[0], path)
            expected[path.relative_to(incoming).as_posix()] = probes
    return expected


def _normalized(value):
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def score(publication, expected, *, strict=True):
    chunks, manifest = load_published_chunks(publication, return_manifest=True)
    searchable = {}
    heading_only = 0
    parent_eq_child = 0
    empty_table_headers = 0
    for chunk in chunks:
        metadata = chunk["business_metadata"]
        if metadata["retrieval"].get("role") == "attachment_only":
            continue
        if not metadata.get("provenance", {}).get("locations"):
            continue
        searchable.setdefault(chunk["source"], []).append(_normalized(chunk["content"]))
        section_title = chunk.get("section_title", "").strip()
        content_text = chunk.get("content", "").strip()
        if section_title and content_text == section_title:
            heading_only += 1
        if chunk.get("parent_content", "").strip() == content_text:
            parent_eq_child += 1
        table_data = chunk.get("table_data")
        if isinstance(table_data, dict):
            headers = table_data.get("headers") or []
            if headers and all(not str(label).strip() for label in headers):
                empty_table_headers += 1
    documents = []
    for source, probes in expected.items():
        readings = searchable.get(source, [])
        missing = [probe for probe in probes if not any(_normalized(probe) in value for value in readings)]
        documents.append({"source": source, "checks": len(probes), "missing": missing})
    count = sum(document["checks"] for document in documents)
    missing = sum(len(document["missing"]) for document in documents)
    index_ready = manifest.get("index_ready") is True
    ocr_partial_pages = manifest.get("ocr_partial_pages", 0)
    model_warnings = manifest.get("model_warning_count", 0)
    passed = not missing and not manifest["failed_files"] and not manifest["ocr_failed_pages"] and index_ready
    if strict:
        passed = passed and not ocr_partial_pages and not empty_table_headers
    return {
        "revision": SOURCE_REVISION,
        "pipeline_fingerprint": manifest["pipeline_fingerprint"],
        "files": len(expected),
        "failed_files": manifest["failed_files"],
        "chunks": len(chunks),
        "critical_checks": count,
        "passed_critical_checks": count - missing,
        "index_ready": index_ready,
        "ocr_failed_pages": manifest["ocr_failed_pages"],
        "ocr_partial_pages": ocr_partial_pages,
        "model_warning_count": model_warnings,
        "empty_table_headers": empty_table_headers,
        "heading_only_chunks": heading_only,
        "parent_eq_child_chunks": parent_eq_child,
        "passed": passed,
        "documents": documents,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Isolated publication directory")
    parser.add_argument("--cache", type=Path, help="Model cache to share across benchmark configurations")
    parser.add_argument("--regression-input", type=Path, help="Existing input root containing HDSD 05 and HDSD 36")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="chunking-golden-") as directory:
        incoming = Path(directory)
        expected = build_corpus(incoming, args.regression_input)
        options = {}
        if args.cache is not None:
            options["cache_db_path"] = args.cache
        run_auto_chunking(incoming, args.output, search_language="all", ocr_docx_images=True,
                          retry_delays=(), **options)
        result = score(args.output, expected)
    report = args.output / "golden_report.json"
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "documents"}, ensure_ascii=False, indent=2))
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
