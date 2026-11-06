"""
Runner truc tiep cho Module 7: Generator.
Chay doc lap tu input/ sang output/ voi co che chong trung lap.
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.generator.run_generator import run_auto_generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Generator Node (Input -> Output)")
    parser.add_argument("--force", action="store_true", help="Chay lai toan bo tu dau")
    parser.add_argument("--input", "-i", type=str, default=None, help="Duong dan tap tin input JSON")
    parser.add_argument("--output", "-o", type=str, default=None, help="Thu muc xuat ket qua")
    args = parser.parse_args()

    module_dir = Path(__file__).resolve().parent

    if args.input:
        in_file = Path(args.input)
    else:
        in_file = module_dir / "input" / "top_chunks.json"

    if args.output:
        out_dir = Path(args.output)
    else:
        out_dir = module_dir / "output"

    run_auto_generator(input_file=in_file, output_dir=out_dir, force=args.force)
