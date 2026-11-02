"""
Runner truc tiep cho Module 2: Embedding.
Chay doc lap tu input/ sang output/ voi co che chong trung lap.
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.embedding.run_embedding import run_auto_embedding

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Embedding Node (Input -> Output)")
    parser.add_argument("--force", action="store_true", help="Chay lai toan bo tu dau")
    parser.add_argument("--batch-size", type=int, default=32, help="Kich thuoc batch")
    args = parser.parse_args()

    module_dir = Path(__file__).resolve().parent
    in_file = module_dir / "input" / "chunks.json"
    out_dir = module_dir / "output"
    run_auto_embedding(input_file=in_file, output_dir=out_dir, force=args.force, batch_size=args.batch_size)
