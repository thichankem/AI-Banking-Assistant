"""
Direct runner for Module 4: Query Parser.
Executes query parsing pipeline from input/ to output/ with StateTracker deduplication.

MANDATORY RULES (GEMINI.md & AGENTS.md):
1. NO __INIT__.PY.
2. NO TYPING LIBRARY.
3. ZERO TERNARY: explicit if/else statements.
4. 0% REGEX LAW: string methods only.
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.query_parser.run_query_parser import run_auto_query_parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Query Parser Node (Input -> Output)")
    parser.add_argument("--force", action="store_true", help="Re-run everything from scratch")
    parser.add_argument("--input", type=str, default=None, help="Input queries file path")
    parser.add_argument("--output", type=str, default=None, help="Output directory path")
    parser.add_argument("--query", type=str, default=None, help="Parse a single query directly")
    args = parser.parse_args()

    module_dir = Path(__file__).resolve().parent
    if args.output is not None:
        out_dir = Path(args.output)
    else:
        out_dir = module_dir / "output"

    run_auto_query_parser(
        input_file=args.input,
        output_dir=out_dir,
        force=args.force,
        single_query=args.query
    )
