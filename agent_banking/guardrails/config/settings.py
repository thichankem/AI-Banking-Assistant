# -*- coding: utf-8 -*-
"""Guardrails configuration loading from YAML settings."""
import os
from pathlib import Path
import yaml


def _load_yaml_settings():
    yaml_path = Path(__file__).resolve().parent / "settings" / "config.yaml"
    if yaml_path.exists():
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}


_cfg = _load_yaml_settings()


class GuardrailsConfig:
    ENABLE_SEMANTIC_GUARDRAIL = os.getenv("ENABLE_SEMANTIC_GUARDRAIL", str(_cfg.get("enable_semantic_guardrail", True))).lower() == "true"
    SEMANTIC_SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", str(_cfg.get("semantic_similarity_threshold", 0.88))))
    ENABLE_INDIRECT_INJECTION_SCAN = os.getenv("ENABLE_INDIRECT_INJECTION_SCAN", str(_cfg.get("enable_indirect_injection_scan", True))).lower() == "true"
    MAX_LOAN_AMOUNT_VND = int(os.getenv("MAX_LOAN_AMOUNT_VND", str(_cfg.get("max_loan_amount_vnd", 50000000000))))
    MAX_LOAN_TENURE_MONTHS = int(os.getenv("MAX_LOAN_TENURE_MONTHS", str(_cfg.get("max_loan_tenure_months", 360))))
    MIN_SAVINGS_AMOUNT_VND = int(os.getenv("MIN_SAVINGS_AMOUNT_VND", str(_cfg.get("min_savings_amount_vnd", 1000000))))
