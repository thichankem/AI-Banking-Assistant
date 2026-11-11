"""
Direct runner for Module 2: Embedding.
Runs independently from input to output with deduplication mechanisms.
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
    parser.add_argument("--force", action="store_true", help="Re-run everything from scratch")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding calls")
    parser.add_argument("--input", type=str, default=None, help="Input chunks file path")
    parser.add_argument("--output", type=str, default=None, help="Output directory path")
    args = parser.parse_args()

    run_auto_embedding(
        input_file=args.input,
        output_dir=args.output,
        force=args.force,
        batch_size=args.batch_size
    )
