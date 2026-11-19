"""Build bounded native excerpts for model-assisted interpretation."""
from agent_banking.chunking.config.settings import get_config_section
from ..text.language import foreign_language

_EVIDENCE_CFG = get_config_section("enrichment")["evidence"]
OMISSION = _EVIDENCE_CFG["omission_marker"]
_DEFAULT_ROWS = _EVIDENCE_CFG["table_preview_rows"]
_DEFAULT_COLS = _EVIDENCE_CFG["table_preview_cols"]
_CELL_CHARS = _EVIDENCE_CFG["table_preview_cell_chars"]


def evidence_preview(text, maximum):
    if len(text) <= maximum:
        return text
    if maximum <= len(OMISSION) + 1:
        return text[:maximum]
    left = (maximum - len(OMISSION)) // 2
    right = maximum - len(OMISSION) - left
    return text[:left] + OMISSION + text[-right:]


def table_preview(grid, rows=_DEFAULT_ROWS, *, native=False):
    result = []
    for row in grid[:rows]:
        preview_row = []
        for cell in row[:_DEFAULT_COLS]:
            if not native and foreign_language(cell):
                preview_row.append("")
            else:
                preview_row.append(evidence_preview(cell, _CELL_CHARS))
        result.append(preview_row)
    return result
