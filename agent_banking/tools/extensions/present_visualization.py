"""Validate a visualization before the agent submits it for review."""

import json

from agent_banking.agent import simulator, visualization

TOOL_NAME = "present_visualization"
MAX_ARTIFACT_BYTES = 64 * 1024


def _module(artifact):
    return simulator if artifact.get("engine") == "simulator" else visualization


def present_visualization(artifact: dict) -> dict:
    if (
        len(json.dumps(artifact, ensure_ascii=False, allow_nan=False).encode("utf-8"))
        > MAX_ARTIFACT_BYTES
    ):
        raise ValueError("Visualization is too large")
    if not artifact.get("layout") and not (
        {"defaults", "presentation"} & artifact.keys()
    ):
        raise ValueError("New visualizations require an authored layout")
    _module(artifact).validate_selection(artifact)
    return {"artifact": artifact}


def prepare_presentation(artifact, recent, guard):
    module = _module(artifact)
    prepare = (
        module.prepare_simulator
        if module is simulator
        else module.prepare_visualization
    )
    return prepare(artifact, recent, guard)
