"""
Module Task 0b Reproduce Pinned (Tái Lập Ghim Phiên Bản & Đối Soát Khoa Học)
=============================================================================
Đáp ứng đầy đủ 5 yêu cầu cốt lõi của Task 0b:
2.1 Ghim phiên bản: Config 1-4 nạp từ worktree pre-v3-merge (d1aaf5cf), Config 5A-5C nạp từ v3-production (007d9d0e).
    In ra module path + git hash trong log.
2.2 Test cô lập cấu hình: Hash (SHA256) danh sách chunk trên cùng tài liệu; các config khác thiết kế phải khác hash.
2.3 Bảng so sánh Task 0b: Config | code_ref | Recall@5 cũ | mới | Δ | MRR cũ | mới | Δ | Tái lập? (Y/N).
2.4 Điều tra chuyên sâu Config 2, 3, 4: So sánh chunk differences (file, trang, loại chunk), map với câu hỏi Golden QA.
2.5 Kiểm tra tính xác định (Determinism) của 5B qua 2 lần chạy độc lập.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_WORKTREE = PROJECT_ROOT.parent / "legacy_pre_merge"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.evaluators.retrieval_evaluator import (
    RetrievalEvaluator,
    compute_token_containment,
    _tokenize
)


def get_git_hash(repo_dir: Path) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_dir))
        return out.decode("utf-8").strip()[:8]
    except Exception:
        return "unknown"


def compute_chunk_list_hash(chunks: list[dict[str, Any]]) -> str:
    """Tính chuỗi hash SHA256 đặc trưng cho một danh sách chunk."""
    hasher = hashlib.sha256()
    for c in chunks:
        cid = str(c.get("chunk_id", ""))
        content = str(c.get("content", ""))
        ctype = str(c.get("chunk_type", ""))
        hasher.update(cid.encode("utf-8"))
        hasher.update(content.encode("utf-8"))
        hasher.update(ctype.encode("utf-8"))
    return hasher.hexdigest()[:16]


def load_legacy_chunker():
    """Nạp LayoutAwareChunker từ legacy worktree pre-v3-merge."""
    if not LEGACY_WORKTREE.exists():
        raise RuntimeError(f"Không tìm thấy legacy worktree tại {LEGACY_WORKTREE}")
    legacy_sys_path = str(LEGACY_WORKTREE)
    if legacy_sys_path not in sys.path:
        sys.path.insert(0, legacy_sys_path)
    # Nạp module từ legacy
    import importlib
    spec = importlib.util.spec_from_file_location(
        "legacy_layout_aware_chunker",
        LEGACY_WORKTREE / "rag_banking" / "chunking" / "unified_chunk" / "layout_aware_chunker.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LayoutAwareChunker, mod.__file__


def load_production_chunker():
    """Nạp LayoutAwareChunker từ v3 production repo hiện tại."""
    import importlib
    spec = importlib.util.spec_from_file_location(
        "prod_layout_aware_chunker",
        PROJECT_ROOT / "rag_banking" / "chunking" / "unified_chunk" / "layout_aware_chunker.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LayoutAwareChunker, mod.__file__
