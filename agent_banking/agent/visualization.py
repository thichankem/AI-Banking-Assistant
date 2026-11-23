"""Validate declarative chart and diagram data, styling and layouts."""

import re
from typing import Annotated, Literal
from uuid import uuid4

from jsonschema import Draft202012Validator
from pydantic import (
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
Label = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=90)
]
VisualId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]

_HEADERS = {
    "flowchart": r"(?:flowchart|graph) (?:TB|TD|BT|RL|LR)",
    "sequence": r"sequenceDiagram",
    "class": r"classDiagram",
    "er": r"erDiagram",
    "mindmap": r"mindmap",
    "architecture": r"architecture-beta",
}
_UNSAFE_MERMAID = re.compile(
    r"%%|@\{|`|\\|;|:::(?:\s*\w)|<\s*(?:/?\s*[a-z]|[!?/])|"
    r"(?:[a-z][a-z0-9+.-]*://|\b(?:url\s*\(|(?:javascript|data|file)\s*:))|"
    r"(?m:^\s*(?:click|href|link|links|callback|call|style|classDef|cssClass|linkStyle|"
    r"className|accTitle|accDescr)\b)|^\s*---\s*$",
    re.IGNORECASE,
)
_CUSTOM_ARCHITECTURE_ICON = re.compile(
    r"^\s*(?:service|group)\s+[\w-]+\s*\((?!(?:cloud|database|disk|internet|server)\s*\))",
    re.MULTILINE,
)
_CONTROL_TEXT = re.compile(r"[\x00-\x1f\x7f\u202a-\u202e\u2066-\u2069]")
ChartNumber = Annotated[
    float, Field(allow_inf_nan=False, ge=-MAX_SAFE_INTEGER, le=MAX_SAFE_INTEGER)
]
ChartColor = Annotated[str, StringConstraints(pattern=r"^#[0-9a-fA-F]{6}$")]
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
        if not 2 <= len(lines) <= 48 or not re.fullmatch(_HEADERS[self.kind], lines[0]):
            raise ValueError("Invalid Mermaid header or line count")
        if any(len(line) > 180 or _CONTROL_TEXT.search(line) for line in lines):
            raise ValueError("Unsupported Mermaid characters or line length")
        custom_icon = self.kind == "architecture" and _CUSTOM_ARCHITECTURE_ICON.search(
            self.source
        )
        if (
            _UNSAFE_MERMAID.search(self.source)
            or re.search(r"::icon\b", self.source)
            or custom_icon
        ):
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
        or _CONTROL_TEXT.search(label)
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
        prior = previous["visualization"] if previous else None
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
        primary = (
            "chart" if isinstance(selection, ChartSelection) else "diagram",
            None,
        )
        check_layout(
            selection.layout,
            {primary, ("table", None)} if primary[0] == "chart" else {primary},
            {primary},
            lambda label: safe_visualization_label(label, guard),
        )
        visual = {
            **selection.model_dump(exclude={"update_of"}),
            "schema_version": 1,
            "title": safe_visualization_label(selection.title, guard),
            "id": prior["id"] if prior else uuid4().hex,
            "revision": prior["revision"] + 1 if prior else 1,
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
