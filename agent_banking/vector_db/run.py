"""
Direct runner for Vector DB & BM25 Module.
Executes input-to-output indexing with state deduplication.
"""
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.vector_db.run_vector_db import run_auto_indexing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Vector DB Node (Input -> Output)")
    parser.add_argument("--force", action="store_true", help="Force rebuild from scratch")
    parser.add_argument("--use-memory", action="store_true", help="Use in-memory storage for Qdrant")
    args = parser.parse_args()

    module_dir = Path(__file__).resolve().parent
    in_file = module_dir / "input" / "embedded_chunks.json"
    out_dir = module_dir / "output"
    run_auto_indexing(input_file=in_file, output_dir=out_dir, force=args.force, use_memory=args.use_memory)
