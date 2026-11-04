"""
Runner truc tiep cho Module 5: Hybrid Retrieval.
Ho tro ca Single Query tu CLI (--query) va Batch Execution (--force).

Quy chuan:
1. 0% REGEX LAW: Khong import re.
2. ZERO TERNARY: if/else tuong minh.
3. NO TYPING LIBRARY: Khong import typing hay from __future__ import annotations.
4. NO __INIT__.PY: Import truc tiep module.
"""

import sys
import argparse
from pathlib import Path

# Dam bao UTF-8 an toan tren Windows console
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
        description="Module 5: Hybrid Retrieval (Dual-Index Dense 1024-dim + BM25 + RRF k=60)"
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Chay kiem thu nhanh mot cau hoi don le tu terminal"
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="So luong tai lieu trich xuat Top-K (default: 5)"
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Chay lai toan bo bo qua cache StateTracker"
    )
    parser.add_argument(
        "--use-memory",
        action="store_true",
        help="Dung RAM storage cho Qdrant (in-memory mode)"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Duong dan tuy chinh toi parsed_queries.json"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Thu muc dau ra tuy chinh (default: retrieval/output)"
    )
    args = parser.parse_args()

    module_dir = Path(__file__).resolve().parent

    if args.input is not None:
        in_file = Path(args.input)
    else:
        in_file = module_dir / "input" / "parsed_queries.json"

    if args.output is not None:
        out_dir = Path(args.output)
    else:
        out_dir = module_dir / "output"

    run_retrieval(
        query=args.query,
        input_file=in_file,
        output_dir=out_dir,
        top_k=args.top_k,
        force=args.force,
        use_memory=args.use_memory
    )


if __name__ == "__main__":
    main()
