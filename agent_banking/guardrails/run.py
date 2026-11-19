"""
Direct runner for Guardrails Module.
Executes input-to-output processing with state deduplication.
"""
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.guardrails.config.settings import GuardrailsConfig, load_guardrail_prompts
from agent_banking.guardrails.run_guardrails import run_auto_guardrails


if __name__ == "__main__":
    _runner_cfg = dict(GuardrailsConfig.RUNNER_SETTINGS or {})
    _prompts = load_guardrail_prompts()
    _runner_msgs = dict(getattr(_prompts, "RUNNER_MESSAGES", {}) or {})
    _input_candidates = list(_runner_cfg.get("input_candidates") or [])
    if _input_candidates:
        _default_input_name = _input_candidates[0]
    else:
        _default_input_name = ""

    parser = argparse.ArgumentParser(description=str(_runner_msgs.get("cli_description") or ""))
    parser.add_argument("--force", action="store_true", help="Force re-run from scratch")
    args = parser.parse_args()

    module_dir = Path(__file__).resolve().parent
    in_file = module_dir / str(_runner_cfg.get("input_dir_name") or "input") / _default_input_name
    out_dir = module_dir / str(_runner_cfg.get("output_dir_name") or "output")
    run_auto_guardrails(input_file=in_file, output_dir=out_dir, force=args.force)
