# -*- coding: utf-8 -*-
"""
Generator prompt configuration wrapper.
Follows the chunking prompt_config convention: resolves config/prompts.py first,
then falls back to markdown files under config/prompts/ and safe defaults.
Complies with GEMINI.md section 4 (no typing, no regex, no ternary).
"""
from pathlib import Path

from agent_banking.generator.config import prompts as _prompts_mod


_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
PROMPT_VERSION = "banking-generator-v2.0"


def _read_prompt(relative_path, default_text=""):
    path = _PROMPTS_DIR / relative_path
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8-sig") as handle:
                text = handle.read().strip()
                if text:
                    return text
        except Exception:
            pass
    return default_text


ProfileBundle = _prompts_mod.ProfileBundle
load_prompt_profiles = _prompts_mod.load_prompt_profiles
PERSONA_PROFILES = _prompts_mod.PERSONA_PROFILES
TONE_PROFILES = _prompts_mod.TONE_PROFILES
FORMAT_PROFILES = _prompts_mod.FORMAT_PROFILES
SYSTEM_GUARDRAILS = _prompts_mod.SYSTEM_GUARDRAILS
PROMPT_LEAK_SIGNATURES = _prompts_mod.PROMPT_LEAK_SIGNATURES
build_banking_system_prompt = _prompts_mod.build_banking_system_prompt
