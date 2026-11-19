"""
Direct runner for Tracing Module.
Executes input-to-output tracing pipeline with deduplication.
"""
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.tracing.config.settings import TracingConfig, load_tracing_prompts
from agent_banking.tracing.run_tracing import run_auto_tracing


if __name__ == "__main__":
    _runner_cfg = dict(TracingConfig.RUNNER_SETTINGS or {})
    _prompts = load_tracing_prompts()
    _runner_msgs = dict(getattr(_prompts, "RUNNER_MESSAGES", {}) or {})

    parser = argparse.ArgumentParser(description=str(_runner_msgs.get("cli_description") or ""))
    parser.add_argument("--force", action="store_true", help="Force re-run from scratch")
    args = parser.parse_args()

    module_dir = Path(__file__).resolve().parent
    in_file = module_dir / str(_runner_cfg.get("input_dir_name") or "input") / str(_runner_cfg.get("input_filename") or "")
    out_dir = module_dir / str(_runner_cfg.get("output_dir_name") or "output")
    run_auto_tracing(input_file=in_file, output_dir=out_dir, force=args.force)
