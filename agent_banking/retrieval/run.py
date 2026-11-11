"""
Direct runner for Hybrid Retrieval Module.
Supports single query CLI execution (--query) and batch execution (--force).

INVARIANTS:
1. 0% REGEX LAW: No import re.
2. ZERO TERNARY: Explicit if/else statements.
3. NO TYPING LIBRARY: No typing library imports.
4. NO __INIT__.PY: Direct module imports.
"""

import sys
import argparse
from pathlib import Path

# Safe UTF-8 console setup for Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.retrieval.run_retrieval import run_retrieval


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Retrieval (Dual-Index Dense + BM25 + RRF k=60)"
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Quick test query from command line"
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of Top-K retrieved documents (default: 5)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-retrieval from scratch, ignoring cached state"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Custom path to parsed queries JSON"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Custom path to output directory"
    )

    args = parser.parse_args()

    custom_input = None
    if args.input:
        custom_input = Path(args.input)

    custom_output = None
    if args.output:
        custom_output = Path(args.output)

    run_retrieval(
        input_path=custom_input,
        output_dir=custom_output,
        force=args.force,
        top_k=args.top_k,
        single_query=args.query
    )


if __name__ == "__main__":
    main()
