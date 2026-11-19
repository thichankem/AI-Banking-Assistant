"""Thin loader for chunking prompts from local config/prompts/ and config/prompts.py."""

from copy import deepcopy
from importlib import import_module
from pathlib import Path
import yaml


_local_name = __package__ + ".prompts"
try:
    _local = import_module(_local_name)
except ModuleNotFoundError as error:
    if error.name != _local_name:
        raise
    _local = None

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load_yaml_file(relative_path):
    path = _PROMPTS_DIR / relative_path
    if path.is_file():
        value = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        if isinstance(value, dict):
            return value
    return {}


def _prompt(name, relative_path):
    value = getattr(_local, name, "")
    if isinstance(value, str) and value.strip():
        return value.strip()
    path = _PROMPTS_DIR / relative_path
    if path.is_file():
        value = path.read_text(encoding="utf-8-sig").strip()
        if value:
            return value
    return ""


def _yaml_prompt(name, relative_path):
    value = getattr(_local, name, None)
    if isinstance(value, dict) and value:
        return deepcopy(value)
    return _load_yaml_file(relative_path)


def _version(name, key):
    value = getattr(_local, name, "")
    if isinstance(value, str) and value.strip():
        return value.strip()
    versions = _load_yaml_file("versions.yaml")
    return str(versions.get(key, ""))


def render_prompt(template, **values):
    """Substitute named placeholders safely without interpreting JSON braces."""
    rendered = str(template)
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


PROMPT_VERSION = _version("PROMPT_VERSION", "prompt_version")
PDF_PROMPT_VERSION = _version("PDF_PROMPT_VERSION", "pdf_prompt_version")
ENGINE_VERSION = _version("ENGINE_VERSION", "engine_version")
PDF_SCHEMA_VERSION = _version("PDF_SCHEMA_VERSION", "pdf_schema_version")
DECISION_ENGINE_VERSION = _version("DECISION_ENGINE_VERSION", "decision_engine_version")
BOUNDARY_VERSION = _version("BOUNDARY_VERSION", "boundary_version")
ATOM_SCORE_VERSION = _version("ATOM_SCORE_VERSION", "atom_score_version")

PAGE_PROMPT = _prompt("PAGE_PROMPT", "pdf_extractor/page.md")
CLOUD_OCR_PAGE_PROMPT = _prompt("CLOUD_OCR_PAGE_PROMPT", "pdf_extractor/cloud_page.md")
PDF_PAGE_REPAIR_PROMPT = _prompt("PDF_PAGE_REPAIR_PROMPT", "pdf_extractor/page_repair.md")
PDF_REGION_CROP_INSTRUCTION = _prompt("PDF_REGION_CROP_INSTRUCTION", "pdf_extractor/region_crop_instruction.md")

START_PROMPT = _prompt("START_PROMPT", "semantic_boundaries/start.md")
BOUNDARY_PROMPT = _prompt("BOUNDARY_PROMPT", "semantic_boundaries/boundary.md")
START_REPAIR_PROMPT = _prompt("START_REPAIR_PROMPT", "semantic_boundaries/start_repair.md")
BOUNDARY_REPAIR_PROMPT = _prompt("BOUNDARY_REPAIR_PROMPT", "semantic_boundaries/boundary_repair.md")

STRUCTURE_SYSTEM_PROMPT = _prompt("STRUCTURE_SYSTEM_PROMPT", "structure/system.md")
STRUCTURE_TASK_BASE = _prompt("STRUCTURE_TASK_BASE", "structure/task_base.md")
STRUCTURE_TASK_OCR_ADDITION = _prompt("STRUCTURE_TASK_OCR_ADDITION", "structure/task_ocr_addition.md")
STRUCTURE_TASK_EDITABLE_SUFFIX = _prompt("STRUCTURE_TASK_EDITABLE_SUFFIX", "structure/task_editable_suffix.md")
STRUCTURE_REPAIR_PROMPT = _prompt("STRUCTURE_REPAIR_PROMPT", "structure/repair.md")

OCR_PROMPT = _prompt("OCR_PROMPT", "ocr/transcription.md")

QWEN_VISION_SYSTEM_PROMPT = _prompt("QWEN_VISION_SYSTEM_PROMPT", "vision/qwen_system.md")
QWEN_VISION_SCHEMA_REPAIR_PROMPT = _prompt("QWEN_VISION_SCHEMA_REPAIR_PROMPT", "vision/qwen_schema_repair.md")
DOCUMENT_VISION_SYSTEM_PROMPT = _prompt("DOCUMENT_VISION_SYSTEM_PROMPT", "vision/document_system.md")
DOCUMENT_VISION_SCHEMA_REPAIR_PROMPT = _prompt("DOCUMENT_VISION_SCHEMA_REPAIR_PROMPT", "vision/document_schema_repair.md")
VISION_DEFAULT_INSTRUCTIONS = _prompt("VISION_DEFAULT_INSTRUCTIONS", "vision/default_instructions.md")
VISION_DEFAULT_ASK = _prompt("VISION_DEFAULT_ASK", "vision/default_ask.md")

TABLE_REPAIR_SYSTEM_PROMPT = _prompt("TABLE_REPAIR_SYSTEM_PROMPT", "table_review/repair_system.md")
TABLE_REPAIR_RETRY_PROMPT = _prompt("TABLE_REPAIR_RETRY_PROMPT", "table_review/repair_retry.md")
TABLE_PIXEL_AUDIT_PROMPT = _prompt("TABLE_PIXEL_AUDIT_PROMPT", "table_review/pixel_audit.md")
TABLE_ADJUDICATION_QUESTION = _yaml_prompt("TABLE_ADJUDICATION_QUESTION", "table_review/adjudication.yaml")

IMAGE_CONTEXT_TASK_PROMPT = _prompt("IMAGE_CONTEXT_TASK_PROMPT", "enrichment/image_context_task.md")
IMAGE_CONTEXT_REPAIR_PROMPT = _prompt("IMAGE_CONTEXT_REPAIR_PROMPT", "enrichment/image_context_repair.md")
TABLE_HEADER_TASK_PROMPT = _prompt("TABLE_HEADER_TASK_PROMPT", "enrichment/table_header_task.md")
TABLE_HEADER_REPAIR_PROMPT = _prompt("TABLE_HEADER_REPAIR_PROMPT", "enrichment/table_header_repair.md")
TABLE_HEADER_CONFIRM_QUESTION = _yaml_prompt("TABLE_HEADER_CONFIRM_QUESTION", "enrichment/table_header_confirm.yaml")

HEADINGS_REVIEW_QUESTION = _yaml_prompt("HEADINGS_REVIEW_QUESTION", "review/headings_question.yaml")
EXTRACTION_ROUTE_QUESTION = _yaml_prompt("EXTRACTION_ROUTE_QUESTION", "review/extraction_route.yaml")
ADJUDICATE_REPAIR_TASK = _prompt("ADJUDICATE_REPAIR_TASK", "review/adjudicate_task.md")
ADJUDICATE_REPAIR_TASK_OCR = _prompt("ADJUDICATE_REPAIR_TASK_OCR", "review/adjudicate_task_ocr.md")
ADJUDICATE_REPAIR_RETRY_PROMPT = _prompt("ADJUDICATE_REPAIR_RETRY_PROMPT", "review/adjudicate_repair.md")
REPAIR_CONFIRM_QUESTION = _yaml_prompt("REPAIR_CONFIRM_QUESTION", "review/repair_confirm.yaml")

BREADCRUMB_HIERARCHY_PROMPT = _yaml_prompt("BREADCRUMB_HIERARCHY_PROMPT", "decisions/breadcrumb_hierarchy.yaml")
PROMPT_PAIRS_PROMPT = _yaml_prompt("PROMPT_PAIRS_PROMPT", "decisions/prompt_pairs.yaml")
PERIOD_BOUNDARIES_PROMPT = _yaml_prompt("PERIOD_BOUNDARIES_PROMPT", "decisions/period_boundaries.yaml")
CAPTION_ATTACHMENT_PROMPT = _yaml_prompt("CAPTION_ATTACHMENT_PROMPT", "decisions/caption_attachment.yaml")
IMAGE_TITLE_CONTEXT_PROMPT = _yaml_prompt("IMAGE_TITLE_CONTEXT_PROMPT", "decisions/image_title_and_context.yaml")
TABLE_STRUCTURE_TITLE_PROMPT = _yaml_prompt("TABLE_STRUCTURE_TITLE_PROMPT", "decisions/table_structure_and_title.yaml")
TABLE_CONTINUATION_PROMPT = _yaml_prompt("TABLE_CONTINUATION_PROMPT", "decisions/table_continuation.yaml")

PREFLIGHT_CANARIES = _yaml_prompt("PREFLIGHT_CANARIES", "preflight/canaries.yaml")
