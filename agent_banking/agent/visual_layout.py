"""Describe the UI tree carried in an exported visualization file."""

from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints

_IDENT_FIRST = set("abcdefghijklmnopqrstuvwxyz")
_IDENT_REST = set("abcdefghijklmnopqrstuvwxyz0123456789_")


def check_identifier(value: str) -> str:
    if not (1 <= len(value) <= 24) or value[0] not in _IDENT_FIRST or any(ch not in _IDENT_REST for ch in value[1:]):
        raise ValueError("Invalid identifier")
    return value


BlockKind = Literal[
    "column",
    "row",
    "grid",
    "heading",
    "text",
    "input",
    "result",
    "formulas",
    "assumptions",
    "reset",
    "chart",
    "diagram",
    "table",
]
_CONTAINERS = {"column", "row", "grid"}


class LayoutBlock(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    type: BlockKind
    children: list["LayoutBlock"] = Field(default_factory=list, max_length=12)
    columns: Annotated[int, Field(ge=1, le=4)] | None = None
    text: (
        Annotated[
            str, StringConstraints(strip_whitespace=True, min_length=1, max_length=180)
        ]
        | None
    ) = None
    ref: Annotated[str, Field(min_length=1, max_length=24), AfterValidator(check_identifier)] | None = (
        None
    )
    tone: Literal["plain", "card", "accent"] = "plain"


def check_layout(
    layout: LayoutBlock | None, allowed: set[tuple], required: set[tuple], guard=None
):
    if layout is None:
        return  # Previously saved artifacts still have a compatible renderer.
    pending = [(layout, 1)]
    used = set()
    count = 0
    while pending:
        block, depth = pending.pop()
        count += 1
        if count > 40 or depth > 6:
            raise ValueError("Layout is too large or deeply nested")
        if block.text is not None and guard is not None:
            guard(block.text)
        if block.type in _CONTAINERS:
            if not block.children or block.ref is not None or block.text is not None:
                raise ValueError("A layout container requires children")
            if block.columns is not None and block.type != "grid":
                raise ValueError("Only grid blocks accept columns")
            pending.extend((child, depth + 1) for child in block.children)
            continue
        if block.children or block.columns is not None:
            raise ValueError("Leaf blocks cannot contain other blocks")
        if block.type in ("heading", "text"):
            if block.text is None or block.ref is not None:
                raise ValueError("Text blocks need text without a reference")
            continue
        # Blocks backed by the whole artifact can refer to themselves explicitly.
        # Normalize the reference so duplicate singleton blocks are still rejected.
        if block.type not in ("input", "result") and block.ref == block.type:
            ref = None
        else:
            ref = block.ref
        key = (block.type, ref)
        if key not in allowed or key in used:
            raise ValueError("Unknown or repeated layout reference")
        used.add(key)
    if not required <= used:
        raise ValueError("Layout is missing a declared control or renderer")
