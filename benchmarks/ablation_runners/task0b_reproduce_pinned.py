"""
Task 0b: Tái lập Kết quả Master Ablation với Cơ chế Ghim Phiên bản (Version-Pinned Reproduction).
================================================================================================
Ràng buộc nghiêm ngặt:
- Config 0: Baseline Naive Fixed Window 800.
- Config 1 - 4: Chạy từ git worktree legacy_pre_merge (commit d1aaf5cf) qua subprocess độc lập.
- Config 5A - 5C: Chạy từ git main (v3-production d409e36) qua subprocess độc lập.
- Kiểm tra tính xác định (Determinism) cho Config 5B qua 2 lần chạy độc lập.
- Kiểm tra cô lập hash chunk SHA256 (Hash Isolation).
- Điều tra nguyên nhân kỹ thuật Config 2, 3, 4 có điểm Retrieval tương đương.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_WORKTREE = PROJECT_ROOT.parent / "legacy_pre_merge"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.evaluators.retrieval_evaluator import (
    RetrievalEvaluator,
    compute_token_containment
)


def get_git_hash(repo_dir: Path) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_dir))
        return out.decode("utf-8").strip()[:8]
    except Exception:
        return "unknown"


def compute_chunk_hash(chunks: list[dict[str, Any]]) -> str:
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


def run_chunk_worker(config_key: str, out_json: Path) -> tuple[list[dict[str, Any]], dict[str, str], str]:
    """Chạy worker trong tiến trình con độc lập để cách ly hoàn toàn môi trường import."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "benchmarks" / "ablation_runners" / "chunk_corpus_worker.py"),
        "--config", config_key,
        "--out", str(out_json)
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), env=env, encoding="utf-8")
    if res.returncode != 0:
        raise RuntimeError(f"Lỗi khi chạy worker cho config {config_key}:\n{res.stderr}\n{res.stdout}")

    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["chunks"], data.get("parent_map", {}), data.get("module_path", "")


def run_task0b_benchmark():
    print(f"\n{'='*88}")
    print("TASK 0b: TÁI LẬP KẾT QUẢ VỚI GHIM PHIÊN BẢN (VERSION-PINNED REPRODUCIBILITY)")
    print(f"Main Repo Commit:   {get_git_hash(PROJECT_ROOT)} (v3-production)")
    print(f"Legacy Worktree:    {get_git_hash(LEGACY_WORKTREE)} at {LEGACY_WORKTREE} (pre-v3-merge)")
    print(f"{'='*88}\n")

    # 1. Đọc Golden QA
    qa_path = PROJECT_ROOT / "benchmarks" / "datasets" / "golden_qa.json"
    with open(qa_path, "r", encoding="utf-8") as f:
        golden_qa = json.load(f)

    target_baselines = {
        "Config 0 - Baseline (Naive)":        {"r5": 0.6200, "mrr": 0.4045, "code_ref": "pre-v3-merge (d1aaf5cf)", "worker_cfg": "0"},
        "Config 1 - Pipeline v1":              {"r5": 0.7600, "mrr": 0.5933, "code_ref": "pre-v3-merge (d1aaf5cf)", "worker_cfg": "1"},
        "Config 2 - Pipeline v2 (Prod cũ)":    {"r5": 0.8400, "mrr": 0.6733, "code_ref": "pre-v3-merge (d1aaf5cf)", "worker_cfg": "2"},
        "Config 3 - Upgrade OCR":              {"r5": 0.8400, "mrr": 0.6733, "code_ref": "pre-v3-merge + exp1 (d1aaf5cf)", "worker_cfg": "3"},
        "Config 4 - Fast Vision":              {"r5": 0.8400, "mrr": 0.6733, "code_ref": "pre-v3-merge + exp1+exp2 (d1aaf5cf)", "worker_cfg": "4"},
        "Config 5A - Profile A":               {"r5": 0.7800, "mrr": 0.5365, "code_ref": "v3-production (d409e36)", "worker_cfg": "5a"},
        "Config 5B - Profile B (Ult)":         {"r5": 0.9000, "mrr": 0.7403, "code_ref": "v3-production (d409e36)", "worker_cfg": "5b"},
        "Config 5C - Profile C":               {"r5": 0.7800, "mrr": 0.5327, "code_ref": "v3-production (d409e36)", "worker_cfg": "5c"}
    }

    evaluator = RetrievalEvaluator(containment_threshold=0.65)
    results_dir = PROJECT_ROOT / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    task0b_rows = []
    chunk_outputs_by_config: dict[str, list[dict[str, Any]]] = {}
    chunk_hashes: dict[str, str] = {}

    for name, meta in target_baselines.items():
        w_cfg = meta["worker_cfg"]
        code_ref = meta["code_ref"]
        slug = name.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").lower()
        chunk_file = results_dir / f"task0b_chunks_{slug}.json"

        print(f">>> Đang chạy sinh chunk: [{name}] (code_ref: {code_ref})")
        chunks, parent_map, mod_path = run_chunk_worker(w_cfg, chunk_file)
        
        c_hash = compute_chunk_hash(chunks)
        chunk_outputs_by_config[name] = chunks
        chunk_hashes[name] = c_hash
        print(f"    Chunks: {len(chunks)} | Hash: {c_hash} | Module: {mod_path}")

        # Đánh giá truy xuất
        evaluator.retriever.index(chunks)
        recalls_5, mrrs = [], []
        per_q_records = []

        for q_idx, sample in enumerate(golden_qa):
            query = sample.get("question", "")
            target_ids = set(sample.get("expected_chunk_ids", []))
            target_span = sample.get("ground_truth_context_span") or sample.get("ground_truth_answer", "")
            target_bc = sample.get("expected_breadcrumb_prefix", "")
            cat = sample.get("category", "general")

            top10_docs = evaluator.retriever.search(query, top_k=10)
            hits = [
                1 if evaluator.is_chunk_relevant(
                    retrieved_chunk=doc,
                    expected_chunk_ids=target_ids,
                    expected_context_span=target_span,
                    expected_bc_prefix=target_bc,
                    parent_map=parent_map
                ) else 0
                for doc, _ in top10_docs
            ]

            r5 = 1.0 if any(hits[:5]) else 0.0
            recalls_5.append(r5)

            mrr_val = 0.0
            first_rank = None
            for r_pos, h in enumerate(hits[:10], start=1):
                if h == 1:
                    mrr_val = 1.0 / r_pos
                    first_rank = r_pos
                    break
            mrrs.append(mrr_val)

            per_q_records.append({
                "question_idx": q_idx,
                "question": query,
                "category": cat,
                "first_hit_rank": first_rank,
                "hit_at_5": bool(r5 > 0),
                "mrr_at_10": round(mrr_val, 4)
            })

        jsonl_path = results_dir / f"task0b_{slug}_per_question.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in per_q_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        measured_r5 = round(sum(recalls_5) / max(len(recalls_5), 1), 4)
        measured_mrr = round(sum(mrrs) / max(len(mrrs), 1), 4)

        target_r5 = meta["r5"]
        target_mrr = meta["mrr"]

        diff_r5 = round(measured_r5 - target_r5, 4)
        diff_mrr = round(measured_mrr - target_mrr, 4)

        # Sai số cho phép ±0.005
        is_reproduced = (abs(diff_r5) <= 0.005) and (abs(diff_mrr) <= 0.005)
        status_str = "Y" if is_reproduced else "N"

        task0b_rows.append({
            "config": name,
            "code_ref": code_ref,
            "r5_old": target_r5,
            "r5_new": measured_r5,
            "delta_r5": diff_r5,
            "mrr_old": target_mrr,
            "mrr_new": measured_mrr,
            "delta_mrr": diff_mrr,
            "reproduced": status_str,
            "chunks": len(chunks),
            "hash": c_hash
        })

    # -------------------------------------------------------------------------
    # 2.2 TEST CÔ LẬP CẤU HÌNH (HASH ISOLATION TEST)
    # -------------------------------------------------------------------------
    print(f"\n{'='*88}")
    print("2.2 TEST CÔ LẬP CẤU HÌNH (HASH ISOLATION TEST)")
    print(f"{'='*88}")
    hash_collisions = []
    names = list(chunk_hashes.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            if chunk_hashes[n1] == chunk_hashes[n2]:
                hash_collisions.append((n1, n2, chunk_hashes[n1]))

    if hash_collisions:
        for n1, n2, h in hash_collisions:
            print(f"  [CẢNH BÁO VA CHẠM HASH] {n1} trùng hash {n2}: {h}")
    else:
        print("  [PASS] 100% Cấu hình khác nhau có Hash SHA256 độc lập hoàn toàn!")

    # -------------------------------------------------------------------------
    # 2.4 ĐIỀU TRA CONFIG 2, 3, 4 (CHUNK DIFFERENCES VÀ MAP VỚI GOLDEN QA)
    # -------------------------------------------------------------------------
    print(f"\n{'='*88}")
    print("2.4 ĐIỀU TRA KỸ THUẬT: TẠI SAO CONFIG 2, 3, 4 CÙNG ĐẠT RECALL@5 = 0.8400")
    print(f"{'='*88}")
    chunks_c2 = chunk_outputs_by_config["Config 2 - Pipeline v2 (Prod cũ)"]
    chunks_c3 = chunk_outputs_by_config["Config 3 - Upgrade OCR"]
    chunks_c4 = chunk_outputs_by_config["Config 4 - Fast Vision"]

    def analyze_chunk_differences(list_a, list_b, label_a, label_b):
        content_set_a = {c.get("content", "").strip() for c in list_a}
        content_set_b = {c.get("content", "").strip() for c in list_b}
        diff_in_b = [c for c in list_b if c.get("content", "").strip() not in content_set_a]
        diff_in_a = [c for c in list_a if c.get("content", "").strip() not in content_set_b]

        print(f"\nSo sánh [{label_a}] ({len(list_a)} chunks) vs [{label_b}] ({len(list_b)} chunks):")
        print(f"  - Số chunk có nội dung khác nhau: {len(diff_in_b)} (trong {label_b}) vs {len(diff_in_a)} (trong {label_a})")
        diff_files = sorted(list({c.get("source", "") for c in diff_in_b}))
        diff_types = sorted(list({c.get("chunk_type", "") for c in diff_in_b}))
        print(f"  - Tệp bị ảnh hưởng: {diff_files}")
        print(f"  - Loại chunk bị ảnh hưởng: {diff_types}")
        return diff_in_b

    diff_2_3 = analyze_chunk_differences(chunks_c2, chunks_c3, "Config 2", "Config 3")
    diff_3_4 = analyze_chunk_differences(chunks_c3, chunks_c4, "Config 3", "Config 4")

    # Map diff chunks với Golden QA
    impacted_queries_2_3 = []
    for q in golden_qa:
        span = q.get("ground_truth_context_span", "").lower()
        q_text = q.get("question", "")
        cat = q.get("category", "")
        for c in diff_2_3:
            if compute_token_containment(c.get("content", "").lower(), span) >= 0.65:
                impacted_queries_2_3.append((q_text, cat, c.get("source", "")))
                break

    print(f"\nSố câu hỏi Golden QA chạm vào phần chunk khác biệt giữa Config 2 và Config 3: {len(impacted_queries_2_3)}/50")
    for q_text, cat, src in impacted_queries_2_3[:5]:
        print(f"  * [{cat}] {q_text[:70]}... (File: {src})")

    # -------------------------------------------------------------------------
    # 2.5 KIỂM TRA TÍNH XÁC ĐỊNH (DETERMINISM TEST) CỦA CONFIG 5B
    # -------------------------------------------------------------------------
    print(f"\n{'='*88}")
    print("2.5 KIỂM TRA TÍNH XÁC ĐỊNH (DETERMINISM) CỦA CONFIG 5B (2 RUNS ĐỘC LẬP)")
    print(f"{'='*88}")
    run1_file = results_dir / "task0b_5b_run1.json"
    run2_file = results_dir / "task0b_5b_run2.json"
    chunks_5b_run1, _, _ = run_chunk_worker("5b", run1_file)
    chunks_5b_run2, _, _ = run_chunk_worker("5b", run2_file)

    hash_5b_run1 = compute_chunk_hash(chunks_5b_run1)
    hash_5b_run2 = compute_chunk_hash(chunks_5b_run2)

    is_deterministic = (hash_5b_run1 == hash_5b_run2)
    print(f"  Run 1: {len(chunks_5b_run1)} chunks | Hash: {hash_5b_run1}")
    print(f"  Run 2: {len(chunks_5b_run2)} chunks | Hash: {hash_5b_run2}")
    print(f"  Kết quả: {'PASS - Hoàn toàn xác định 100%' if is_deterministic else 'FAIL - Không xác định!'}")

    # -------------------------------------------------------------------------
    # XUẤT BẢNG TỔNG HỢP TASK 0b
    # -------------------------------------------------------------------------
    print(f"\n{'='*105}")
    print("BẢNG TỔNG HỢP TASK 0b (VERSION-PINNED REPRODUCIBILITY TABLE)")
    print(f"{'='*105}")
    print(f"{'Cấu hình':<32} | {'code_ref':<28} | {'R@5 cũ':<7} | {'mới':<7} | {'Δ':<7} | {'MRR cũ':<7} | {'mới':<7} | {'Δ':<7} | {'Tái lập?'}")
    print(f"{'-'*32}-|-{'-'*28}-|-{'-'*7}-|-{'-'*7}-|-{'-'*7}-|-{'-'*7}-|-{'-'*7}-|-{'-'*7}-|-{'-'*9}")

    for row in task0b_rows:
        cfg = row["config"]
        ref = row["code_ref"]
        r5_o = f"{row['r5_old']:.4f}"
        r5_n = f"{row['r5_new']:.4f}"
        d_r5 = f"{row['delta_r5']:+.4f}"
        mrr_o = f"{row['mrr_old']:.4f}"
        mrr_n = f"{row['mrr_new']:.4f}"
        d_mrr = f"{row['delta_mrr']:+.4f}"
        rep = row["reproduced"]
        print(f"{cfg:<32} | {ref:<28} | {r5_o:<7} | {r5_n:<7} | {d_r5:<7} | {mrr_o:<7} | {mrr_n:<7} | {d_mrr:<7} | {rep}")

    print(f"{'='*105}\n")

    summary_file = results_dir / "task0b_reproduction_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "task0b_rows": task0b_rows,
            "hash_isolation": chunk_hashes,
            "determinism_5b": is_deterministic,
            "config_2_3_diff_count": len(diff_2_3),
            "config_3_4_diff_count": len(diff_3_4),
            "impacted_golden_qa_count": len(impacted_queries_2_3)
        }, f, ensure_ascii=False, indent=2)

    print(f"Đã lưu kết quả Task 0b ra: {summary_file}")
    return task0b_rows


if __name__ == "__main__":
    run_task0b_benchmark()
