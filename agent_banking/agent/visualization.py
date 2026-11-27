"""Validate declarative chart and diagram data, styling and layouts."""

from typing import Annotated, Literal
from uuid import uuid4

from jsonschema import Draft202012Validator
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    model_validator,
)

from agent_banking.agent.visual_layout import LayoutBlock, check_layout

SKILL_ID = "visualization"
MAX_RECENT = 2
MAX_SAFE_INTEGER = 2**53 - 1
MAX_ROWS = 60
MAX_COLUMNS = 8
_HEX_LOWER = set("0123456789abcdef")
_HEX_ALL = set("0123456789abcdefABCDEF")
_SCHEME_BODY = set("abcdefghijklmnopqrstuvwxyz0123456789+.-")
_FLOWCHART_HEADERS = (
    "flowchart TB",
    "flowchart TD",
    "flowchart BT",
    "flowchart RL",
    "flowchart LR",
    "graph TB",
    "graph TD",
    "graph BT",
    "graph RL",
    "graph LR",
)
_EXACT_HEADERS = {
    "sequence": "sequenceDiagram",
    "class": "classDiagram",
    "er": "erDiagram",
    "mindmap": "mindmap",
    "architecture": "architecture-beta",
}
_UNSAFE_DIRECTIVES = {
    "click",
    "href",
    "link",
    "links",
    "callback",
    "call",
    "style",
    "classdef",
    "cssclass",
    "linkstyle",
    "classname",
    "acctitle",
    "accdescr",
}
_ALLOWED_ARCHITECTURE_ICONS = ("cloud", "database", "disk", "internet", "server")


def _check_visual_id(value: str) -> str:
    if len(value) != 32 or any(ch not in _HEX_LOWER for ch in value):
        raise ValueError("Invalid visual identifier")
    return value


def _check_chart_color(value: str) -> str:
    if len(value) != 7 or not value.startswith("#") or any(ch not in _HEX_ALL for ch in value[1:]):
        raise ValueError("Invalid hex color")
    return value


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def _has_control_text(text: str) -> bool:
    for ch in text:
        code = ord(ch)
        if code < 0x20 or code == 0x7F or 0x202A <= code <= 0x202E or 0x2066 <= code <= 0x2069:
            return True
    return False


def _skip_spaces(text: str, pos: int) -> int:
    length = len(text)
    while pos < length and text[pos].isspace():
        pos += 1
    return pos


def _has_unsafe_mermaid(source: str) -> bool:
    for token in ("%%", "@{", "`", "\\", ";"):
        if token in source:
            return True
    length = len(source)
    idx = source.find(":::")
    while idx != -1:
        pos = _skip_spaces(source, idx + 3)
        if pos < length and _is_word_char(source[pos]):
            return True
        idx = source.find(":::", idx + 3)

    low = source.lower()
    idx = low.find("<")
    while idx != -1:
        pos = _skip_spaces(low, idx + 1)
        if pos < length:
            nxt = low[pos]
            if nxt in ("!", "?", "/") or ("a" <= nxt <= "z"):
                return True
        idx = low.find("<", idx + 1)

    idx = low.find("://")
    while idx != -1:
        cursor = idx - 1
        has_alpha = False
        while cursor >= 0 and low[cursor] in _SCHEME_BODY:
            if "a" <= low[cursor] <= "z":
                has_alpha = True
            cursor -= 1
        if has_alpha:
            return True
        idx = low.find("://", idx + 3)

    for kw, suffix in (("url", "("), ("javascript", ":"), ("data", ":"), ("file", ":")):
        idx = low.find(kw)
        while idx != -1:
            if idx == 0 or not _is_word_char(low[idx - 1]):
                pos = _skip_spaces(low, idx + len(kw))
                if pos < length and low[pos] == suffix:
                    return True
            idx = low.find(kw, idx + len(kw))

    for line in source.split("\n"):
        stripped = line.strip()
        if stripped == "---":
            return True
        word_chars = []
        for ch in stripped:
            if _is_word_char(ch):
                word_chars.append(ch)
            else:
                break
        if word_chars and "".join(word_chars).lower() in _UNSAFE_DIRECTIVES:
            return True

    idx = source.find("::icon")
    while idx != -1:
        end = idx + 6
        if end == length or not _is_word_char(source[end]):
            return True
        idx = source.find("::icon", idx + 6)
    return False


def _has_custom_architecture_icon(source: str) -> bool:
    for line in source.split("\n"):
        stripped = line.lstrip()
        prefix_len = 0
        if stripped.startswith("service") and len(stripped) > 7 and stripped[7].isspace():
            prefix_len = 7
        elif stripped.startswith("group") and len(stripped) > 5 and stripped[5].isspace():
            prefix_len = 5
        if prefix_len == 0:
            continue
        pos = _skip_spaces(stripped, prefix_len)
        ident_start = pos
        while pos < len(stripped) and (_is_word_char(stripped[pos]) or stripped[pos] == "-"):
            pos += 1
        if pos == ident_start:
            continue
        pos = _skip_spaces(stripped, pos)
        if pos >= len(stripped) or stripped[pos] != "(":
            continue
        after = stripped[pos + 1:]
        matched_allowed = False
        for icon in _ALLOWED_ARCHITECTURE_ICONS:
            if after.startswith(icon):
                close_pos = _skip_spaces(after, len(icon))
                if close_pos < len(after) and after[close_pos] == ")":
                    matched_allowed = True
                    break
        if not matched_allowed:
            return True
    return False


Label = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=90)
]
VisualId = Annotated[str, Field(min_length=32, max_length=32), AfterValidator(_check_visual_id)]
ChartNumber = Annotated[
    float, Field(allow_inf_nan=False, ge=-MAX_SAFE_INTEGER, le=MAX_SAFE_INTEGER)
]
ChartColor = Annotated[str, Field(min_length=7, max_length=7), AfterValidator(_check_chart_color)]
ChartKind = Literal["bar", "stacked_bar", "line", "area", "pie", "donut", "scatter"]


class _Selection(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    title: Label
    update_of: VisualId | None = None
    layout: LayoutBlock | None = None


class ChartSeries(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    name: Label
    values: list[ChartNumber] = Field(min_length=1, max_length=MAX_ROWS)


class ChartConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    colors: list[ChartColor] = Field(default_factory=list, max_length=12)
    y_name: Label | None = None
    legend: bool | None = None
    show_points: bool | None = None
    smooth: bool = False
    area_opacity: Annotated[float, Field(ge=0, le=1)] | None = None
    donut_hole: Annotated[int, Field(ge=10, le=62)] | None = None


class ChartSelection(_Selection):
    engine: Literal["echarts"]
    kind: ChartKind
    x_name: Label
    x: list[Label | ChartNumber] = Field(min_length=1, max_length=MAX_ROWS)
    series: list[ChartSeries] = Field(min_length=1, max_length=MAX_COLUMNS - 1)
    config: ChartConfig = Field(default_factory=ChartConfig)

    @model_validator(mode="after")
    def check_columns(self):
        check_layout(self.layout, {("chart", None), ("table", None)}, {("chart", None)})
        if len(self.series) != len({item.name for item in self.series}):
            raise ValueError("Chart series names must be distinct")
        category = self.kind != "scatter"
        if any(isinstance(value, str) != category for value in self.x):
            raise ValueError(
                "Use labels for categories and numbers for scatter coordinates"
            )
        if (category and len(set(self.x)) != len(self.x)) or any(
            len(item.values) != len(self.x) for item in self.series
        ):
            raise ValueError("Chart data lengths or categories are inconsistent")
        if self.kind in ("pie", "donut", "scatter") and len(self.series) != 1:
            raise ValueError("Expected one value column")
        if self.kind == "stacked_bar" and len(self.series) < 2:
            raise ValueError("Expected at least two value columns")
        if self.kind in ("pie", "donut"):
            values = self.series[0].values
            if len(values) > 12 or min(values) < 0 or not any(values):
                raise ValueError("Invalid part-to-whole series")
        return self


class ChartPresentation(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    kind: ChartKind | None = None
    title: Label | None = None
    x_name: Label | None = None
    config: ChartConfig | None = None
    layout: LayoutBlock | None = None

    @model_validator(mode="after")
    def check_change(self):
        if not self.model_dump(exclude_none=True):
            raise ValueError("A presentation update must change a field")
        return self


class ChartUpdate(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    engine: Literal["echarts"]
    update_of: VisualId
    presentation: ChartPresentation


class DiagramSelection(_Selection):
    engine: Literal["mermaid"]
    kind: Literal["flowchart", "sequence", "class", "er", "mindmap", "architecture"]
    source: str = Field(min_length=8, max_length=3000)
    theme: Literal["dark", "default", "neutral", "forest"] = "dark"

    @model_validator(mode="after")
    def check_source(self):
        check_layout(self.layout, {("diagram", None)}, {("diagram", None)})
        self.source = self.source.replace("\r\n", "\n").strip()
        lines = self.source.split("\n")
        if self.kind == "flowchart":
            valid_header = lines[0] in _FLOWCHART_HEADERS
        else:
            valid_header = lines[0] == _EXACT_HEADERS.get(self.kind)
        if not 2 <= len(lines) <= 48 or not valid_header:
            raise ValueError("Invalid Mermaid header or line count")
        if any(len(line) > 180 or _has_control_text(line) for line in lines):
            raise ValueError("Unsupported Mermaid characters or line length")
        custom_icon = self.kind == "architecture" and _has_custom_architecture_icon(
            self.source
        )
        if _has_unsafe_mermaid(self.source) or custom_icon:
            raise ValueError(
                "Mermaid includes an unsupported directive or external resource"
            )
        return self


_SELECTION = TypeAdapter(ChartSelection | ChartUpdate | DiagramSelection)
VISUALIZATION_SCHEMA = _SELECTION.json_schema()
_VALIDATOR = Draft202012Validator(VISUALIZATION_SCHEMA)


def validate_selection(raw):
    if not _VALIDATOR.is_valid(raw):
        raise ValueError("Invalid chart or diagram specification")
    return _SELECTION.validate_python(raw)


def safe_visualization_label(label: str, guard) -> str:
    if (
        any(char in label for char in "<>`")
        or _has_control_text(label)
        or guard(label) != label
        or "[REDACTED" in label.upper()
    ):
        raise ValueError("Unsafe visualization label")
    return label


def prepare_visualization(raw, recent: list[dict], guard) -> dict | None:
    """Return a public artifact and its session-only data, or omit an invalid artifact."""
    try:
        selection = validate_selection(raw)
        previous = next(
            (
                entry
                for entry in recent
                if entry["visualization"]["id"] == selection.update_of
            ),
            None,
        )
        if selection.update_of and previous is None:
            return None
        if previous:
            prior = previous["visualization"]
        else:
            prior = None
        if isinstance(selection, ChartUpdate):
            if prior["engine"] != "echarts":
                return None
            changes = selection.presentation.model_dump(
                exclude_unset=True, exclude_none=True
            )
            base = {
                key: prior[key] for key in ChartSelection.model_fields if key in prior
            }
            if "config" in changes:
                changes["config"] = {**(base.get("config") or {}), **changes["config"]}
            selection = ChartSelection.model_validate({**base, **changes})
        if isinstance(selection, ChartSelection):
            primary = ("chart", None)
        else:
            primary = ("diagram", None)
        if primary[0] == "chart":
            allowed_blocks = {primary, ("table", None)}
        else:
            allowed_blocks = {primary}
        check_layout(
            selection.layout,
            allowed_blocks,
            {primary},
            lambda label: safe_visualization_label(label, guard),
        )
        if prior:
            vis_id = prior["id"]
            revision = prior["revision"] + 1
        else:
            vis_id = uuid4().hex
            revision = 1
        visual = {
            **selection.model_dump(exclude={"update_of"}),
            "schema_version": 1,
            "title": safe_visualization_label(selection.title, guard),
            "id": vis_id,
            "revision": revision,
        }
        if isinstance(selection, ChartSelection):
            labels = [
                selection.x_name,
                *(row.name for row in selection.series),
                *(value for value in selection.x if isinstance(value, str)),
            ]
            if selection.config.y_name:
                labels.append(selection.config.y_name)
            for label in labels:
                safe_visualization_label(label, guard)
        else:
            if (
                guard(selection.source) != selection.source
                or "[REDACTED" in selection.source.upper()
            ):
                return None
            visual["source"] = selection.source
        return {"visualization": visual}
    except (ValueError, KeyError, TypeError, IndexError):
        return None


def recent_visualizations(recent: list[dict]) -> list[dict]:
    """Give the planner scoped IDs and the data needed for a revision."""
    return [entry["visualization"] for entry in recent]
