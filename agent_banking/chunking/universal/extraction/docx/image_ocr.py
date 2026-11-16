"""Send one embedded raster image at a time to the PDF cloud OCR adapter."""

from hashlib import sha256
from io import BytesIO
from warnings import catch_warnings, simplefilter

from PIL import Image

from agent_banking.chunking.publication_io import atomic_bytes, file_hash
from ...enrichment.model_warnings import record_model_warning
from ..pdf.page_review import CloudOCRUnavailable, PdfVisualReviewer


def _raster_image(blob, max_pixels):
    if len(blob) > 16_000_000:
        raise ValueError("Embedded image exceeds the input size limit")
    with catch_warnings():
        simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(BytesIO(blob)) as src:
            width, height = src.size
            if width * height > max_pixels or getattr(src, "n_frames", 1) != 1:
                raise ValueError("Embedded image exceeds the pixel or frame limit")
            stream = BytesIO()
            rgba = src.convert("RGBA")
            with Image.new("RGB", src.size, "white") as background:
                background.paste(rgba, mask=rgba.getchannel("A"))
                background.save(stream, format="PNG")
    if len(stream.getbuffer()) > 16_000_000:
        raise ValueError("Rendered embedded image exceeds the input size limit")
    return stream.getvalue(), width, height


class DocxImageOCR:
    def __init__(self, asset_dir, cache, vision_client, review_client, max_pixels):
        self.asset_dir = asset_dir
        self.reader = PdfVisualReviewer(asset_dir, cache, vision_client, review_client, False)
        self.max_pixels = max_pixels

    def read(self, ir, relation, xpath):
        reviews = ir.setdefault("docx_image_reviews", [])
        evidence = {"xpath": xpath, "method": "docx_cloud_image_ocr"}
        reviews.append(evidence)
        if relation is None or relation.is_external or not relation.reltype.endswith("/image"):
            evidence["issues"] = ["unsupported_image_reference"]
            return [], None
        try:
            pixels, width, height = _raster_image(relation.target_part.blob, self.max_pixels)
        except (ValueError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
            evidence["issues"] = ["unsupported_raster_image"]
            record_model_warning(ir, "docx_image_input", type(error).__name__)
            return [], None
        digest = sha256(pixels).hexdigest()
        path = self.asset_dir / (digest + ".png")
        if not path.is_file() or file_hash(path) != digest:
            atomic_bytes(path, pixels)
        ir["assets"].setdefault(digest, {"path": str(path.resolve()), "sha256": digest, "kind": "embedded_image",
                                        "width": width, "height": height})
        evidence["asset_sha256"] = digest
        try:
            layout = self.reader.read(pixels, [], width, height)
        except CloudOCRUnavailable as error:
            evidence["issues"] = ["ocr_unavailable"]
            record_model_warning(ir, "docx_image_ocr", type(error).__name__)
            return [], digest
        evidence.update(layout["visual_review"])
        texts = []
        for element in layout["elements"]:
            text = element.get("text", "")
            if element["type"] == "table":
                table = element["table_data"]
                text = "\n".join(" | ".join(row) for row in table["header_grid"] + table["rows"])
            if text.strip():
                texts.append(text)
        return texts, digest
