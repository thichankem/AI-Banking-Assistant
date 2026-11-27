"""Validate bounded calculator definitions and check that their defaults compute."""

import math
import operator
from functools import reduce
from typing import Annotated, Literal
from uuid import uuid4

from jsonschema import Draft202012Validator
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StringConstraints,
    TypeAdapter,
    model_validator,
)

from agent_banking.agent.visual_layout import LayoutBlock, check_identifier, check_layout
from agent_banking.agent.visualization import Label, VisualId, safe_visualization_label

SKILL_ID = "interactive-simulator"
MAX_MAGNITUDE = 100_000_000_000_000
Number = Annotated[
    float, Field(allow_inf_nan=False, ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)
]
Identifier = Annotated[str, Field(min_length=1, max_length=24), AfterValidator(check_identifier)]
Unit = Annotated[str, StringConstraints(strip_whitespace=True, max_length=28)]
Assumption = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)
]
Operation = Literal[
    "add",
    "subtract",
    "multiply",
    "divide",
    "power",
    "min",
    "max",
    "pmt",
    "declining_interest",
]
_LOAN_OPERATIONS = ("pmt", "declining_interest")
_BINARY_OPERATIONS = ("subtract", "divide", "power")


def _loan_period_count(principal: float, rate: float, periods: float) -> int:
    # Normalizing the tolerance also supports converting a year count to months.
    count = round(periods)
    if (
        principal < 0
        or not 0 <= rate <= 1
        or not 1 <= count <= 1200
        or abs(periods - count) > 1e-9
    ):
        raise ValueError("Invalid payment parameters")
    return count


def _payment(principal: float, rate: float, periods: float) -> float:
    """Constant end-of-period payment with a zero balance after the final period."""
    count = _loan_period_count(principal, rate, periods)
    if rate == 0:
        return principal / count
    return principal * (rate / -math.expm1(-count * math.log1p(rate)))


_OPERATIONS = {
    "add": lambda *values: reduce(operator.add, values),
    "subtract": operator.sub,
    "multiply": lambda *values: reduce(operator.mul, values),
    "divide": operator.truediv,
    "power": math.pow,
    "min": min,
    "max": max,
    "pmt": _payment,
    "declining_interest": lambda p, r, n: p * r * (_loan_period_count(p, r, n) + 1) / 2,
}


class _Fields(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class SimulatorInput(_Fields):
    id: Identifier
    label: Label
    unit: Unit = ""
    value: Number
    min: Number
    max: Number
    step: Annotated[Number, Field(ge=0.000001)]
    whole: bool = False
    control: Literal["both", "number", "slider"] = "both"

    @model_validator(mode="after")
    def check_range(self):
        if not self.min <= self.value <= self.max or self.step > self.max - self.min:
            raise ValueError("Invalid slider range or default")
        if self.min + self.step == self.min or self.max - self.step == self.max:
            raise ValueError("Slider step is smaller than floating-point resolution")
        if self.whole and any(
            not value.is_integer()
            for value in (self.min, self.max, self.step, self.value)
        ):
            raise ValueError(
                "Integer control requires integer bounds, step and default"
            )
        return self


class SimulatorCalculation(_Fields):
    model_config = ConfigDict(
        json_schema_extra={
            "if": {"properties": {"show": {"const": False}}, "required": ["show"]},
            "else": {"properties": {"unit": {"minLength": 1}}, "required": ["unit"]},
            "allOf": [
                {
                    "if": {"properties": {"op": {"enum": list(operations)}}},
                    "then": {
                        "properties": {"args": {"minItems": size, "maxItems": size}}
                    },
                }
                for operations, size in ((_LOAN_OPERATIONS, 3), (_BINARY_OPERATIONS, 2))
            ],
        }
    )

    id: Identifier
    label: Label
    unit: Unit = ""
    op: Operation
    args: list[Identifier | Number] = Field(min_length=2, max_length=4)
    show: bool = True
    decimals: Annotated[int, Field(ge=0, le=6)] | None = None


class SimulatorSelection(_Fields):
    engine: Literal["simulator"]
    title: Label = Field(
        description="Stable title without values from the adjustable controls"
    )
    inputs: list[SimulatorInput] = Field(min_length=1, max_length=8)
    calculations: list[SimulatorCalculation] = Field(min_length=1, max_length=12)
    assumptions: list[Assumption] = Field(
        min_length=1,
        max_length=4,
        description="Assumptions that hold as the controls change",
    )
    update_of: VisualId | None = None
    layout: LayoutBlock | None = None
    _default_results: dict[str, float] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def check_calculations(self):
        self.check_layout()
        values = {item.id: item.value for item in self.inputs}
        units = {item.id: item.unit for item in self.inputs}
        dependencies = {item.id: {item.id} for item in self.inputs}
        used_inputs = set()
        if len(values) != len(self.inputs) or not any(
            item.show for item in self.calculations
        ):
            raise ValueError("Calculator requires unique inputs and a visible result")
        for item in self.calculations:
            if item.id in values:
                raise ValueError("Duplicate calculation ID")
            if (
                item.op in _LOAN_OPERATIONS
                and len(item.args) != 3
                or item.op in _BINARY_OPERATIONS
                and len(item.args) != 2
            ):
                raise ValueError("Incorrect argument count")
            if item.show and not item.unit:
                raise ValueError("Every visible result requires an explicit unit")
            if item.op in _LOAN_OPERATIONS and isinstance(item.args[1], str):
                rate_unit = units.get(item.args[1], "").casefold()
                if any(marker in rate_unit for marker in ("%", "phần trăm", "percent")):
                    raise ValueError("Loan operations require a decimal rate")
            try:
                args = []
                for arg in item.args:
                    if isinstance(arg, str):
                        args.append(values[arg])
                    else:
                        args.append(arg)
                result = _OPERATIONS[item.op](*args)
                if not math.isfinite(result) or abs(result) > MAX_MAGNITUDE:
                    raise ValueError("Calculation exceeds supported range")
            except (ArithmeticError, KeyError, ValueError) as error:
                raise ValueError(
                    "Invalid result or undefined reference at defaults"
                ) from error
            values[item.id] = result
            units[item.id] = item.unit
            dependencies[item.id] = set().union(
                *(dependencies[arg] for arg in item.args if isinstance(arg, str))
            )
            if item.show:
                used_inputs.update(dependencies[item.id])
                self._default_results[item.id] = result
        if used_inputs != {item.id for item in self.inputs}:
            raise ValueError("Every input must affect a visible result")
        return self

    def check_layout(self, guard=None):
        required = {("input", item.id) for item in self.inputs} | {
            ("result", item.id) for item in self.calculations if item.show
        }
        optional = {("formulas", None), ("assumptions", None), ("reset", None)}
        check_layout(self.layout, required | optional, required, guard)


class SimulatorDefaultUpdate(_Fields):
    engine: Literal["simulator"]
    update_of: VisualId
    defaults: dict[Identifier, Number] = Field(min_length=1, max_length=8)


_SELECTION = TypeAdapter(SimulatorSelection | SimulatorDefaultUpdate)
SIMULATOR_SCHEMA = _SELECTION.json_schema()
_VALIDATOR = Draft202012Validator(SIMULATOR_SCHEMA)


def validate_selection(raw):
    if not _VALIDATOR.is_valid(raw):
        raise ValueError("Invalid calculator specification")
    return _SELECTION.validate_python(raw)


def prepare_simulator(raw: dict, recent: list[dict], guard) -> dict | None:
    """Bind a checked calculator to the existing chat artifact transport."""
    try:
        selection = validate_selection(raw)
        prior = next(
            (
                entry["visualization"]
                for entry in recent
                if entry["visualization"]["id"] == selection.update_of
            ),
            None,
        )
        if selection.update_of and (prior is None or prior["engine"] != "simulator"):
            return None
        if isinstance(selection, SimulatorDefaultUpdate):
            if not selection.defaults.keys() <= {
                item["id"] for item in prior["inputs"]
            }:
                return None
            selection = SimulatorSelection.model_validate(
                {
                    **{
                        key: prior[key]
                        for key in ("engine", "title", "calculations", "assumptions")
                    },
                    "layout": prior.get("layout"),
                    "inputs": [
                        {
                            **item,
                            "value": selection.defaults.get(item["id"], item["value"]),
                        }
                        for item in prior["inputs"]
                    ],
                }
            )
        for value in (selection.title, *selection.assumptions):
            safe_visualization_label(value, guard)
        selection.check_layout(lambda label: safe_visualization_label(label, guard))
        for item in [*selection.inputs, *selection.calculations]:
            safe_visualization_label(item.label, guard)
            if item.unit:
                safe_visualization_label(item.unit, guard)
        if prior:
            vis_id = prior["id"]
            revision = prior["revision"] + 1
        else:
            vis_id = uuid4().hex
            revision = 1
        return {
            "simulator_defaults": selection._default_results,
            "visualization": {
                **selection.model_dump(exclude={"update_of"}),
                "schema_version": 1,
                "kind": "calculator",
                "id": vis_id,
                "revision": revision,
            },
        }
    except ValueError:
        return None
