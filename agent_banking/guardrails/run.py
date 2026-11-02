"""
Runner truc tiep cho Module 8: Guardrails.
Chay doc lap tu input/ sang output/ voi co che chong trung lap.
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.guardrails.run_guardrails import run_auto_guardrails

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Guardrails Node (Input -> Output)")
    parser.add_argument("--force", action="store_true", help="Chay lai toan bo tu dau")
    args = parser.parse_args()

    module_dir = Path(__file__).resolve().parent
    in_file = module_dir / "input" / "sample_queries.json"
    out_dir = module_dir / "output"
    run_auto_guardrails(input_file=in_file, output_dir=out_dir, force=args.force)
