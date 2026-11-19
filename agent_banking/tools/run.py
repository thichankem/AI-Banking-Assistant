"""
Direct Runner for Module 12: Tools.
Runs independently from input/ to output/ with deduplication.
"""
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.tools.config.settings import get_tools_config
from agent_banking.tools.run_tools import run_auto_tools

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tools Node (Input -> Output)")
    parser.add_argument("--force", action="store_true", help="Rerun all tools from scratch")
    args = parser.parse_args()

    runner_cfg = get_tools_config()["runner"]
    default_in_name = runner_cfg["default_input_file"]
    module_dir = Path(__file__).resolve().parent
    in_file = module_dir / "input" / default_in_name
    out_dir = module_dir / "output"
    run_auto_tools(input_file=in_file, output_dir=out_dir, force=args.force)
