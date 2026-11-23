"""Describe the UI tree carried in an exported visualization file."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

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
    ref: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,23}$")] | None = (
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
        ref = (
            None
            if block.type not in ("input", "result") and block.ref == block.type
            else block.ref
        )
        key = (block.type, ref)
        if key not in allowed or key in used:
            raise ValueError("Unknown or repeated layout reference")
        used.add(key)
    if not required <= used:
        raise ValueError("Layout is missing a declared control or renderer")
