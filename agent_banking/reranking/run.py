"""
Direct runner for Module 6: Reranking.
Runs independently from input/ to output/ with StateTracker deduplication.
"""
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.reranking.run_reranking import run_auto_reranking

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Reranking Node (Input -> Output)")
    parser.add_argument("--force", action="store_true", help="Re-run everything from scratch")
    parser.add_argument("--top-k", type=int, default=3, help="Number of reranked candidates to retain")
    args = parser.parse_args()

    module_dir = Path(__file__).resolve().parent
    in_file = module_dir / "input" / "candidates.json"
    out_dir = module_dir / "output"
    run_auto_reranking(input_file=in_file, output_dir=out_dir, top_k=args.top_k, force=args.force)
