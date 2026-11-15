"""Build bounded native excerpts for model-assisted interpretation."""
from .language import foreign_language


OMISSION = "\n[... omitted from this request ...]\n"


def evidence_preview(text, maximum):
    if len(text) <= maximum:
        return text
    if maximum <= len(OMISSION) + 1:
        return text[:maximum]
    left = (maximum - len(OMISSION)) // 2
    right = maximum - len(OMISSION) - left
    return text[:left] + OMISSION + text[-right:]


def table_preview(grid, rows=8, *, native=False):
    return [["" if not native and foreign_language(cell) else evidence_preview(cell, 120)
             for cell in row[:8]] for row in grid[:rows]]
