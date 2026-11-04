"""
Runner trực tiếp cho Module 4: Query Parser.
Chạy độc lập từ input/ sang output/ với cơ chế chống trùng lặp StateTracker.

NGUYÊN TẮC BẮT BUỘC (GEMINI.md & AGENTS.md):
1. KHÔNG DÙNG __INIT__.PY.
2. KHÔNG DÙNG THƯ VIỆN TYPING.
3. ZERO TERNARY: if/else tường minh từng dòng.
4. 0% REGEX LAW.
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
    parser.add_argument("--force", action="store_true", help="Chạy lại toàn bộ từ đầu")
    parser.add_argument("--input", type=str, default=None, help="Đường dẫn tệp queries đầu vào")
    parser.add_argument("--output", type=str, default=None, help="Thư mục xuất kết quả")
    parser.add_argument("--query", type=str, default=None, help="Phân tích trực tiếp một câu hỏi cụ thể")
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
