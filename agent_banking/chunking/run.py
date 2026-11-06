"""
Runner truc tiep cho Module 1: Chunking.
Chay doc lap tu input/ sang output/ voi co che chong trung lap.
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.chunking.run_chunking import run_auto_chunking

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Run Chunking Node (Input -> Output)")
    parser.add_argument("--force", action="store_true", help="Chay lai toan bo tu dau")
    parser.add_argument("--max-files", type=int, default=None, help="Gioi han so file DOCX")
    parser.add_argument("--profile", choices=["accuracy", "native"], default="accuracy", help="Cloud models or explicit structural diagnostics")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    module_dir = Path(__file__).resolve().parent
    in_dir = args.input or module_dir / "input"
    out_dir = args.output or module_dir / "output"
    run_auto_chunking(input_dir=in_dir, output_dir=out_dir, force=args.force, max_files=args.max_files, profile=args.profile)
