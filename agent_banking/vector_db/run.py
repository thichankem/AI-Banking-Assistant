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
    parser.add_argument("--input", type=str, default=None, help="Optional custom embedded_chunks.json input path")
    parser.add_argument("--output", type=str, default=None, help="Optional custom output directory path")
    args = parser.parse_args()

    run_auto_indexing(
        input_file=args.input,
        output_dir=args.output,
        force=args.force,
        use_memory=args.use_memory
    )
