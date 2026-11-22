# -*- coding: utf-8 -*-
"""
Evaluator Settings & Prompt Loader.
Loads configuration from settings/config.yaml and settings/excel.yaml.
Complies with 0% Regex Law, Zero Ternary Law.
"""

from importlib import import_module
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
_SETTINGS_DIR = Path(__file__).resolve().parent / "settings"
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def resolve_repo_path(relpath, root=None):
    if root is None:
        base = REPO_ROOT
    else:
        base = Path(root)
    target = base
    for part in str(relpath).replace("\\", "/").split("/"):
        if part:
            target = target / part
    return target


def render_prompt(template, values):
    parts = []
    pos = 0
    while pos < len(template):
        start = template.find("{", pos)
        if start == -1:
            parts.append(template[pos:])
            break
        end = template.find("}", start + 1)
        if end == -1:
            parts.append(template[pos:])
            break
        key = template[start + 1:end]
        if key in values:
            parts.append(template[pos:start])
            parts.append(str(values[key]))
            pos = end + 1
        else:
            parts.append(template[pos:start + 1])
            pos = start + 1
    return "".join(parts)


def _load_yaml_file(path):
    if path.is_file():
        with open(path, "r", encoding="utf-8-sig") as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict):
                return data
    return {}


def load_evaluator_settings():
    yaml_path = _SETTINGS_DIR / "config.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Missing evaluator settings file: {yaml_path}")
    data = _load_yaml_file(yaml_path)
    if not data:
        raise ValueError(f"Evaluator settings file must contain a mapping: {yaml_path}")
    return data


def load_excel_settings():
    yaml_path = _SETTINGS_DIR / "excel.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Missing evaluator excel settings file: {yaml_path}")
    data = _load_yaml_file(yaml_path)
    if not data:
        raise ValueError(f"Evaluator excel settings file must contain a mapping: {yaml_path}")
    return data


def _load_prompts_module():
    try:
        return import_module("agent_banking.evaluator.config.prompts")
    except ModuleNotFoundError as exc:
        if exc.name != "agent_banking.evaluator.config.prompts":
            raise
        return None


def get_evaluator_prompt(attr_name, filename):
    mod = _load_prompts_module()
    if mod is not None:
        val = getattr(mod, attr_name, "")
        if isinstance(val, str) and val.strip():
            return val.strip()
    path = _PROMPTS_DIR / filename
    if path.is_file():
        return path.read_text(encoding="utf-8-sig").strip()
    raise FileNotFoundError(f"Missing evaluator prompt file: {path}")


def get_evaluator_messages(section):
    mod = _load_prompts_module()
    attr_map = {
        "feedback": "EVAL_FEEDBACK",
        "excel": "EVAL_EXCEL_LABELS",
    }
    if mod is not None and section in attr_map:
        val = getattr(mod, attr_map[section], None)
        if isinstance(val, dict):
            return dict(val)
    data = _load_yaml_file(_PROMPTS_DIR / "eval_messages.yaml")
    sub = data.get(section)
    if isinstance(sub, dict):
        return dict(sub)
    return {}


_cfg = load_evaluator_settings()
_excel_cfg = load_excel_settings()


class EvaluatorConfig:
    """Evaluator settings loaded from YAML configuration."""

    DATASETS = dict(_cfg["datasets"])
    CACHES_AND_OUTPUTS = dict(_cfg["caches_and_outputs"])
    HARNESS = dict(_cfg["harness"])
    GOLDEN_EVALUATOR = dict(_cfg["golden_evaluator"])
    JUDGE_785 = dict(_cfg["judge_785"])
    REFERENCE_ALIGNMENT = dict(_cfg["reference_alignment"])
    TUNING_EVAL = dict(_cfg["tuning_eval"])
    RETRIEVAL_SWEEP = dict(_cfg["retrieval_sweep"])
    KNOWLEDGE_BUILD = dict(_cfg["knowledge_build"])
    EXCEL = dict(_excel_cfg)

    DATASET_GROUPS = tuple(DATASETS["groups"])
    EXPECTED_TOTAL_CASES = int(DATASETS["expected_total_cases"])
    GROUP_PATH_TEMPLATE = str(DATASETS["group_path_template"])
    TUNING_CODE_PATHS = list(TUNING_EVAL["code_paths"])
