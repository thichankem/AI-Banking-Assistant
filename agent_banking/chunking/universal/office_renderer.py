"""Optional native Word rendering for Office drawings unsupported by MuPDF."""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

from .renderer import fitz


def render_with_word(document_bytes, dpi=200):
    if os.name != "nt":
        raise RuntimeError("Native Word rendering requires Windows and Microsoft Word")
    # The script is constant; source filenames are process environment data,
    # never interpolated into executable PowerShell code. No user document is edited.
    script = r'''
$ErrorActionPreference = 'Stop'
$word = $null
$document = $null
$discardChanges = 0
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3
    $word.Options.UpdateLinksAtOpen = $false
    $document = $word.Documents.Open($env:CHUNKING_WORD_INPUT, $false, $true, $false)
    $document.ExportAsFixedFormat($env:CHUNKING_WORD_OUTPUT, 17)
} finally {
    if ($null -ne $document) {
        $document.Close([ref]$discardChanges)
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($document)
    }
    if ($null -ne $word) {
        $word.Quit([ref]$discardChanges)
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($word)
    }
}
'''
    with tempfile.TemporaryDirectory(prefix="chunking_word_") as directory:
        incoming = Path(directory) / "drawing.docx"
        outgoing = Path(directory) / "drawing.pdf"
        incoming.write_bytes(document_bytes)
        environment = os.environ.copy()
        environment["CHUNKING_WORD_INPUT"] = str(incoming)
        environment["CHUNKING_WORD_OUTPUT"] = str(outgoing)
        result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                                env=environment, capture_output=True, timeout=120,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode or not outgoing.is_file():
            raise RuntimeError("Microsoft Word could not render the native Office figure")
        pages = []
        with fitz.open(str(outgoing)) as document:
            for index, page in enumerate(document):
                pixmap = page.get_pixmap(dpi=dpi, alpha=False)
                image = pixmap.tobytes("png")
                pages.append({"page_index": index, "image_bytes": image,
                              "sha256": hashlib.sha256(image).hexdigest(),
                              "width": pixmap.width, "height": pixmap.height})
        return pages
