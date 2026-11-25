"""Shared dataset validator and exporter for DS-01 through DS-07 suites."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill

REQUIRED_FIELDS = (
    "id",
    "dataset",
    "group",
    "capability",
    "difficulty",
    "priority",
    "question",
    "turns",
    "reference_answer",
    "rubric",
    "checks",
    "evidence",
)


def _cell_text(value: Any, max_len: int = 32000) -> str:
    """Convert any metadata value to Excel-safe plain text."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def validate_dataset_items(items: list[dict[str, Any]], dataset_code: str) -> dict[str, Any]:
    """Validate uniqueness, required keys, rubric structure, and compute coverage."""
    seen_ids: set[str] = set()
    seen_prompts: set[tuple[str, str, str, str]] = set()
    by_group: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}
    by_priority: dict[str, int] = {}
    attachment_count = 0

    for idx, item in enumerate(items):
        for field in REQUIRED_FIELDS:
            if field not in item:
                raise ValueError(f"{dataset_code} item {idx} missing field: {field}")

        item_id = str(item["id"]).strip()
        if not item_id:
            raise ValueError(f"{dataset_code} item {idx} has empty id")
        if item_id in seen_ids:
            raise ValueError(f"{dataset_code} duplicate id: {item_id}")
        seen_ids.add(item_id)

        turns = item.get("turns")
        if not isinstance(turns, list) or len(turns) == 0:
            raise ValueError(f"{dataset_code} item {item_id} needs non-empty turns")
        question = str(item.get("question", "")).strip()
        if not question:
            raise ValueError(f"{dataset_code} item {item_id} has empty question")

        profile = item.get("profile")
        if not isinstance(profile, dict):
            profile = {}
        atts = item.get("attachments")
        att_sig = ""
        if isinstance(atts, list):
            attachment_count += len(atts)
            att_sig = "|".join(str(x) for x in atts)
        prompt_sig = (
            " || ".join(str(t).strip() for t in turns),
            str(profile.get("persona", "")),
            str(profile.get("tone", "")),
            str(profile.get("format_type", "")) + "::" + att_sig,
        )
        if prompt_sig in seen_prompts:
            raise ValueError(f"{dataset_code} duplicate scenario signature at {item_id}")
        seen_prompts.add(prompt_sig)

        rubric = item.get("rubric")
        if not isinstance(rubric, dict) or not rubric.get("must_include"):
            raise ValueError(f"{dataset_code} item {item_id} missing rubric.must_include")

        grp = str(item["group"])
        diff = str(item["difficulty"])
        prio = str(item["priority"])
        by_group[grp] = by_group.get(grp, 0) + 1
        by_difficulty[diff] = by_difficulty.get(diff, 0) + 1
        by_priority[prio] = by_priority.get(prio, 0) + 1

    return {
        "dataset": dataset_code,
        "total_cases": len(items),
        "total_attachments": attachment_count,
        "by_group": by_group,
        "by_difficulty": by_difficulty,
        "by_priority": by_priority,
    }


def select_stratified_canary(items: list[dict[str, Any]], target_size: int) -> list[dict[str, Any]]:
    """Select an even stratified canary subset across all groups."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        grp = str(item["group"])
        grouped.setdefault(grp, []).append(item)

    canary: list[dict[str, Any]] = []
    group_names = sorted(grouped.keys())
    if not group_names:
        return canary

    cursors = {grp: 0 for grp in group_names}
    while len(canary) < target_size:
        added_any = False
        for grp in group_names:
            if len(canary) >= target_size:
                break
            idx = cursors[grp]
            bucket = grouped[grp]
            if idx < len(bucket):
                canary.append(bucket[idx])
                cursors[grp] = idx + 1
                added_any = True
        if not added_any:
            break
    return canary


def write_dataset_excel(items: list[dict[str, Any]], excel_path: Path, sheet_name: str) -> None:
    """Export structured suite cases to a formatted Excel workbook."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_name[:31]

    headers = [
        "STT",
        "ID",
        "Dataset",
        "Group",
        "Capability",
        "Difficulty",
        "Priority",
        "Question",
        "Turns JSON",
        "Profile JSON",
        "Attachments JSON",
        "Reference Answer",
        "Rubric JSON",
        "Checks JSON",
        "Evidence JSON",
    ]
    ws.append(headers)

    header_fill = PatternFill(start_color="0F2240", end_color="0F2240", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True, size=10)
    for col_idx in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for idx, item in enumerate(items, start=1):
        row_values = [
            idx,
            _cell_text(item.get("id")),
            _cell_text(item.get("dataset")),
            _cell_text(item.get("group")),
            _cell_text(item.get("capability")),
            _cell_text(item.get("difficulty")),
            _cell_text(item.get("priority")),
            _cell_text(item.get("question")),
            _cell_text(item.get("turns")),
            _cell_text(item.get("profile")),
            _cell_text(item.get("attachments")),
            _cell_text(item.get("reference_answer")),
            _cell_text(item.get("rubric")),
            _cell_text(item.get("checks")),
            _cell_text(item.get("evidence")),
        ]
        ws.append(row_values)

    widths = [6, 14, 10, 26, 20, 10, 10, 44, 36, 24, 24, 52, 34, 36, 36]
    for col_idx, width in enumerate(widths, start=1):
        col_letter = openpyxl.utils.get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = width

    ws.freeze_panes = "A2"
    wb.save(excel_path)


def export_suite_bundle(
    items: list[dict[str, Any]],
    output_dir: Path,
    slug: str,
    dataset_code: str,
    canary_size: int = 30,
) -> dict[str, Any]:
    """Validate and write JSON, JSONL, XLSX, coverage JSON, and canary JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage = validate_dataset_items(items, dataset_code)

    json_path = output_dir / f"a_bank_{slug}.json"
    jsonl_path = output_dir / f"a_bank_{slug}.jsonl"
    xlsx_path = output_dir / f"a_bank_{slug}.xlsx"
    cov_path = output_dir / f"coverage_{slug}.json"
    canary_path = output_dir / f"canary_{slug}.json"

    json_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for item in items:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")

    write_dataset_excel(items, xlsx_path, dataset_code)
    canary = select_stratified_canary(items, canary_size)
    canary_path.write_text(json.dumps(canary, ensure_ascii=False, indent=2), encoding="utf-8")

    coverage["canary_count"] = len(canary)
    coverage["files"] = {
        "json": str(json_path).replace("\\", "/"),
        "jsonl": str(jsonl_path).replace("\\", "/"),
        "xlsx": str(xlsx_path).replace("\\", "/"),
        "canary": str(canary_path).replace("\\", "/"),
    }
    cov_path.write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
    return coverage
