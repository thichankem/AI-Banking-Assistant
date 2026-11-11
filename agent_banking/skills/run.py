"""
Direct Runner for Module 11: Skills.
Runs independently from input/ to output/ with deduplication.
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.skills.run_skills import run_auto_skills

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Skills Node (Input -> Output)")
    parser.add_argument("--force", action="store_true", help="Rerun all skills from scratch")
    args = parser.parse_args()

    module_dir = Path(__file__).resolve().parent
    in_file = module_dir / "input" / "sample_skill_inputs.json"
    out_dir = module_dir / "output"
    run_auto_skills(input_file=in_file, output_dir=out_dir, force=args.force)
