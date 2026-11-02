"""
Runner truc tiep cho Module 5: Retrieval.
Chay doc lap tu input/ sang output/ voi co che chong trung lap.
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.retrieval.run_retrieval import run_auto_retrieval

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Retrieval Node (Input -> Output)")
    parser.add_argument("--force", action="store_true", help="Chay lai toan bo tu dau")
    parser.add_argument("--top-k", type=int, default=10, help="So luong tai lieu trich xuat")
    parser.add_argument("--use-memory", action="store_true", help="Dung RAM storage cho Qdrant")
    args = parser.parse_args()

    module_dir = Path(__file__).resolve().parent
    in_file = module_dir / "input" / "parsed_queries.json"
    out_dir = module_dir / "output"
    run_auto_retrieval(input_file=in_file, output_dir=out_dir, top_k=args.top_k, force=args.force, use_memory=args.use_memory)
