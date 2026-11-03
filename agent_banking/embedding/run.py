"""
Runner trực tiếp cho Module 2: Embedding.
Chạy độc lập từ input sang output với cơ chế chống trùng lặp.
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
    parser.add_argument("--force", action="store_true", help="Chạy lại toàn bộ từ đầu")
    parser.add_argument("--batch-size", type=int, default=32, help="Kích thước batch")
    parser.add_argument("--input", type=str, default=None, help="Đường dẫn tệp chunks đầu vào")
    parser.add_argument("--output", type=str, default=None, help="Thư mục xuất kết quả")
    args = parser.parse_args()

    run_auto_embedding(
        input_file=args.input,
        output_dir=args.output,
        force=args.force,
        batch_size=args.batch_size
    )
