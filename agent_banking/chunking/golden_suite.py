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

from .run_chunking import load_published_chunks, run_auto_chunking


# These are authored source strings. Keep assertions independent of model output.
SOURCE_CASES = {
    "legal": (
        ("Notice period", "Article 8: Appeals must be filed within 17 days.", "Exception: official holidays are excluded."),
        ("Điều khoản chuyển tiếp", "Điều 12 áp dụng từ ngày 01/04/2030.", "Ngoại lệ: hồ sơ cũ áp dụng Điều 7."),
        ("Retention schedule", "Section 9: Retain the archive for 27 months.", "Exception: a preservation order suspends deletion."),
        ("Appeal rules", "Article 14: Submit the appeal within 21 days.", "See Article 6 for the only exception."),
    ),
    "economics": (
        ("Price index", "The reference year is 2020.", "All observations use the same reference year."),
        ("Báo cáo giá song ngữ", "Mức thuế là 7,25%.", "The tax does not apply to samples."),
        ("Regional accounts", "Amounts are shown in thousands of euros.", "Negative balances are carried forward."),
        ("Capacity statistics", "Survey coverage reached 82.4%.", "Percentages are weighted by output."),
    ),
    "internal": (
        ("Access requests", "Approval requires two reviewers.", "Do not reuse expired access codes."),
        ("Quy trình nội bộ", "Bước 2: Chọn Xác nhận.", "Chỉ quản trị viên được đặt lại quyền."),
        ("Incident runbook", "Notify the on-call lead within 12 minutes.", "Use incident channel OPS-42."),
        ("Warehouse instructions", "Set the scanner to inventory mode.", "Step 3: Select Confirm inventory."),
    ),
    "technical": (
        ("Calibration", "Maximum deviation is 0.08 mm.", "Recalibrate after any mechanical adjustment."),
        ("Electrical safety", "Do not exceed 48 V DC.", "Disconnect power before opening the panel."),
        ("Cooling specification", "The inlet temperature must remain below 38 °C.", "Valve B must remain open."),
        ("Pressure limits", "Maintain at least 2.5 bar.", "Stop the pump if the relief valve opens."),
    ),
    "academic": (
        ("Sampling method", "The study retained 312 observations.", "Appendix C explains the exclusion criteria."),
        ("研究データ", "調査期間は30日です。", "対照群は20人です。"),
        ("Methods and results", "The confidence level is 95%.", "The null hypothesis was retained."),
        ("Laboratory report", "The specimen mass is 8.25 g.", "Report mass in grams."),
    ),
    "banking": (
        ("Loan schedule", "Interest accrues at 6.75% per annum.", "Fee waivers apply to the first payment only."),
        ("Quy định phí", "Phí xử lý là 35.000 đồng.", "Khoản phí áp dụng một lần cho mỗi yêu cầu."),
        ("Transfer help", "The daily transfer limit is 250,000 VND.", "Select Review transfer before confirming."),
        ("Account controls", "Choose Lock card for this device.", "Enter the six-digit confirmation code."),
    ),
}

SOURCE_REVISION = 1
TABLE_FACTS = ("Baseline-27", "Final-48")
FIGURE_LABEL = "Sensor channel R24"
REAL_CASES = {
    "HDSD 05": ["Khóa toàn bộ dịch vụ Ngân hàng số", "Nhập mã xác nhận 6 số"],
    "HDSD 36": ["Đề nghị đóng và hủy hạn mức thấu chi", "Hạn mức thấu chi đã đóng thành công"],
}


def _word(path, title, facts, structured):
    document = Document()
    document.add_heading(title, 0)
    document.add_heading("Source requirements", 1)
    document.add_paragraph(facts[0])
    if structured:
        table = document.add_table(rows=3, cols=2)
        header = table.cell(0, 0).merge(table.cell(0, 1))
        header.text = "Reported measurement"
        table.cell(1, 0).text = "Initial"
        table.cell(1, 1).text = TABLE_FACTS[0]
        table.cell(2, 0).text = "Final"
        table.cell(2, 1).text = TABLE_FACTS[1]
        math = OxmlElement("m:oMath")
        run = OxmlElement("m:r")
        text = OxmlElement("m:t")
        text.text = "x+y=17"
        run.append(text)
        math.append(run)
        document.add_paragraph()._p.append(math)
        image = BytesIO()
        with pymupdf.open() as figure:
            page = figure.new_page(width=400, height=80)
            page.insert_text((16, 45), FIGURE_LABEL, fontsize=22)
            image.write(page.get_pixmap().tobytes("png"))
        image.seek(0)
        document.add_picture(image)
        document.add_paragraph("Figure 1. Instrument panel.", style="Caption")
    document.add_paragraph(facts[1])
    package = BytesIO()
    document.save(package)
    with ZipFile(package) as original, ZipFile(path, "w", compression=ZIP_DEFLATED) as frozen:
        for name in original.namelist():
            frozen.writestr(ZipInfo(name), original.read(name), compress_type=ZIP_DEFLATED)


def _place(page, text, box, fontsize=15):
    font = pymupdf.Font("china-ss")
    writer = pymupdf.TextWriter(page.rect)
    if writer.fill_textbox(pymupdf.Rect(*box), text, font=font, fontsize=fontsize):
        raise ValueError("Benchmark text exceeded its page box")
    writer.write_text(page)


def _pdf(path, title, facts, scanned, repeated_table):
    with pymupdf.open() as authored:
        page = authored.new_page()
        _place(page, title, (42, 30, 553, 90), 22)
        if scanned:
            _place(page, facts[0], (42, 110, 277, 240))
            _place(page, facts[1], (312, 110, 553, 240))
            page.draw_rect((315, 250, 547, 351), color=(0, 0, 0))
            _place(page, FIGURE_LABEL, (326, 270, 530, 342))
            _place(page, "Figure 1. Instrument panel.", (322, 355, 550, 420), 12)
        else:
            _place(page, facts[0], (42, 110, 553, 190))
            _place(page, facts[1], (42, 210, 553, 300))
        if repeated_table:
            for index, fact in enumerate(TABLE_FACTS):
                if index:
                    page = authored.new_page()
                y = 400 if index == 0 else 100
                for x in (42, 260):
                    for top in (y, y + 36):
                        page.draw_rect((x, top, x + 218, top + 36), color=(0, 0, 0))
                for col, label in enumerate(("Period", "Observed result")):
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
                    scanned_page.insert_image(scanned_page.rect, stream=original.get_pixmap(dpi=144).tobytes("png"))
                raster.save(path, no_new_id=True)


def build_corpus(incoming, regression_input=None):
    expected = {}
    for domain, cases in SOURCE_CASES.items():
        folder = incoming / domain
        folder.mkdir(parents=True, exist_ok=True)
        for variant, (title, *facts) in enumerate(cases):
            is_pdf = variant >= 2
            path = folder / (domain + "-" + str(variant + 1) + (".pdf" if is_pdf else ".docx"))
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


def score(publication, expected):
    chunks, manifest = load_published_chunks(publication, return_manifest=True)
    searchable = {}
    for chunk in chunks:
        metadata = chunk["business_metadata"]
        if metadata["retrieval"].get("role") == "attachment_only":
            continue
        if not metadata.get("provenance", {}).get("locations"):
            continue
        searchable.setdefault(chunk["source"], []).append(_normalized(chunk["content"]))
    documents = []
    for source, probes in expected.items():
        readings = searchable.get(source, [])
        missing = [probe for probe in probes if not any(_normalized(probe) in value for value in readings)]
        documents.append({"source": source, "checks": len(probes), "missing": missing})
    count = sum(document["checks"] for document in documents)
    missing = sum(len(document["missing"]) for document in documents)
    return {"revision": SOURCE_REVISION, "pipeline_fingerprint": manifest["pipeline_fingerprint"],
            "files": len(expected), "failed_files": manifest["failed_files"], "chunks": len(chunks),
            "critical_checks": count, "passed_critical_checks": count - missing,
            "ocr_failed_pages": manifest["ocr_failed_pages"], "ocr_partial_pages": manifest["ocr_partial_pages"],
            "passed": not missing and not manifest["failed_files"] and not manifest["ocr_failed_pages"],
            "documents": documents}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Isolated publication directory")
    parser.add_argument("--cache", type=Path, help="Model cache to share across benchmark configurations")
    parser.add_argument("--regression-input", type=Path, help="Existing input root containing HDSD 05 and HDSD 36")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="chunking-golden-") as directory:
        incoming = Path(directory)
        expected = build_corpus(incoming, args.regression_input)
        options = {"cache_db_path": args.cache} if args.cache is not None else {}
        run_auto_chunking(incoming, args.output, search_language="all", ocr_docx_images=True,
                          retry_delays=(), **options)
        result = score(args.output, expected)
    report = args.output / "golden_report.json"
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "documents"}, ensure_ascii=False, indent=2))
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
