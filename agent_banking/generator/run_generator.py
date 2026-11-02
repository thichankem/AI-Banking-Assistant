"""
Module Tu Dong Sinh Cau Tra Loi LLM (Auto Generator Node with Deduplication).
=============================================================================
Dac diem:
1. Doc tu generator/input/top_chunks.json (hoac tu dong lay tu reranking/output).
2. Tinh toan SHA-256 hash cua cau hoi kem top_chunks.
3. Kiem tra StateTracker: Cau hoi nao da sinh tra loi roi thi tai su dung (SKIP).
4. Chi goi LLM API (DeepSeek-V4-Flash) sinh cau tra loi cho nhung truy van moi.
5. Xuat ket qua: answers.json, answers.md, latest_answer.json.

NGUYEN TAC BAT BUOC:
1. KHONG DUNG __INIT__.PY.
2. KHONG DUNG THU VIEN TYPING.
3. ZERO TERNARY: if/else tuong minh.
4. 0% REGEX LAW.
"""

from __future__ import annotations
import sys
import json
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.config import Config
from agent_banking.pipeline import BankingRAGPipeline
from agent_banking.state_tracker import StateTracker


def run_auto_generator(
    input_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    force: bool = False
) -> list[dict]:
    """
    Tu dong dong goi ngu canh va sinh cau tra loi tu input sang output kem co che chong trung lap.
    """
    cfg = Config()
    in_dir = PROJECT_ROOT / "agent_banking" / "generator" / "input"
    in_dir.mkdir(parents=True, exist_ok=True)

    if output_dir is not None:
        out_path = Path(output_dir)
    else:
        out_path = PROJECT_ROOT / "agent_banking" / "generator" / "output"
    out_path.mkdir(parents=True, exist_ok=True)

    state_file = out_path / ".state_tracker.json"
    tracker = StateTracker(state_file)
    if force:
        tracker.clear()

    # 1. Tim nguon top chunks
    target_in_file = None
    if input_file is not None:
        target_in_file = Path(input_file)
    else:
        top_in = in_dir / "top_chunks.json"
        if top_in.exists():
            target_in_file = top_in
        else:
            rerank_out = PROJECT_ROOT / "agent_banking" / "reranking" / "output" / "reranked_chunks.json"
            if rerank_out.exists():
                target_in_file = rerank_out
            else:
                rag_rerank_out = PROJECT_ROOT / "rag_banking" / "reranking" / "output" / "reranked_chunks.json"
                if rag_rerank_out.exists():
                    target_in_file = rag_rerank_out

    if target_in_file is None or not target_in_file.exists():
        print(f"⚠️ Khong tim thay tap tin top_chunks.json trong: {in_dir}")
        print("Vui long bo file 'top_chunks.json' vao agent_banking/generator/input/")
        return []

    print(f"\n{'='*75}")
    print(f"▶ [MODULE 7: GENERATOR & LLM] Input: {target_in_file.name} -> Output: {out_path}")
    print(f"{'='*75}")

    with open(target_in_file, "r", encoding="utf-8") as fp:
        raw_reranked = json.load(fp)

    if not raw_reranked:
        print("⚠️ Danh sach top_chunks rong.")
        return []

    # Dong bo file vao in_dir
    in_copy = in_dir / "top_chunks.json"
    if target_in_file != in_copy:
        try:
            with open(in_copy, "w", encoding="utf-8") as fp:
                json.dump(raw_reranked, fp, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # 2. Doc cau tra loi cu de tai su dung
    prev_answers = {}
    prev_file = out_path / "answers.json"
    if prev_file.exists() and not force:
        try:
            with open(prev_file, "r", encoding="utf-8") as fp:
                prev_list = json.load(fp)
                for item in prev_list:
                    q = item.get("query", "")
                    if q:
                        prev_answers[q] = item
            if len(prev_answers) >= len(raw_reranked) and len(raw_reranked) > 0:
                q_hash_map = {}
                for item in raw_reranked:
                    q = item.get("query", "")
                    top_chunks = item.get("top_chunks", [])
                    chunk_ids = [c.get("chunk_id", "") for c in top_chunks]
                    c_str = ",".join(chunk_ids)
                    q_hash_map[q] = tracker.compute_text_hash(f"{q}:{c_str}")
                if tracker.are_all_items_processed(q_hash_map):
                    print(f"  ⏭️ [SKIP ALL] Tat ca {len(raw_reranked)} queries da duoc sinh cau tra loi truoc do. Bo qua toan bo trong 0.0s!")
                    print(f"✅ [Generator Hoan Tat] Da xuat {len(prev_list)} cau tra loi vao: {prev_file} trong 0.0s")
                    return prev_list
        except Exception:
            prev_answers = {}

    pipeline = BankingRAGPipeline(config=cfg, use_memory=True)
    final_answers: list[dict] = []
    markdown_reports: list[str] = [
        "# 📋 BÁO CÁO KẾT QUẢ SINH CÂU TRẢ LỜI (MODULE 7 - LLM SERVING)\n",
        f"> Thời gian: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    ]

    skipped_count = 0
    new_count = 0
    start_t = time.time()

    for item in raw_reranked:
        q = item.get("query", "")
        q_id = item.get("query_id", 1)
        top_chunks = item.get("top_chunks", [])

        # Tinh hash dua tren query va danh sach chunk_id
        chunk_ids = [c.get("chunk_id", "") for c in top_chunks]
        c_str = ",".join(chunk_ids)
        q_hash = tracker.compute_text_hash(f"{q}:{c_str}")

        is_done = tracker.is_processed(q, q_hash)

        if is_done and q in prev_answers:
            cached_ans = dict(prev_answers[q])
            cached_ans["query_id"] = q_id
            final_answers.append(cached_ans)
            skipped_count += 1

            markdown_reports.append(f"## ❓ Câu hỏi {q_id}: {q} (Tái sử dụng)")
            markdown_reports.append(f"**💡 Câu trả lời:**\n{cached_ans.get('answer', '')}\n")
            cits = cached_ans.get("legal_citations", [])
            if cits:
                markdown_reports.append(f"* **Tham chiếu:** {', '.join(cits)}")
            markdown_reports.append("---\n")
            print(f"  [{q_id:02d}] [SKIP] '{q[:40]}...' (Da sinh cau tra loi)")
        else:
            t0 = time.time()
            synthesized_context = pipeline.step4_context_synthesizer(top_chunks)
            serving_result = pipeline.step5_llm_serving(q, synthesized_context)
            dt_ms = round((time.time() - t0) * 1000, 1)

            ans_text = serving_result.get("answer", "")
            citations = serving_result.get("citations", [])
            sources = serving_result.get("sources", [])

            record = {
                "query_id": q_id,
                "query": q,
                "answer": ans_text,
                "legal_citations": citations,
                "sources_used": sources,
                "chunks_used": len(top_chunks),
                "duration_ms": dt_ms
            }
            final_answers.append(record)
            tracker.record_processed(q, q_hash, {"citations_count": len(citations)})
            new_count += 1

            markdown_reports.append(f"## ❓ Câu hỏi {q_id}: {q}")
            markdown_reports.append(f"**💡 Câu trả lời:**\n{ans_text}\n")
            if citations:
                markdown_reports.append(f"* **Tham chiếu pháp lý:** {', '.join(citations)}")
            if sources:
                markdown_reports.append(f"* **Tài liệu nguồn:** {', '.join(sources)}")
            markdown_reports.append(f"* **Thời gian sinh:** {dt_ms}ms\n---\n")

            print(f"  [{q_id:02d}] [NEW]  '{q[:40]}...' -> Hoan tat sinh ({dt_ms}ms)")

    tracker.save()
    total_elapsed = round(time.time() - start_t, 2)

    # 3. Xuat ket qua
    json_out = out_path / "answers.json"
    with open(json_out, "w", encoding="utf-8") as fp:
        json.dump(final_answers, fp, ensure_ascii=False, indent=2)

    md_out = out_path / "answers.md"
    with open(md_out, "w", encoding="utf-8") as fp:
        fp.write("\n".join(markdown_reports))

    if final_answers:
        latest_file = out_path / "latest_answer.json"
        with open(latest_file, "w", encoding="utf-8") as fp:
            json.dump(final_answers[-1], fp, ensure_ascii=False, indent=2)

    print(f"✅ [Generator Hoan Tat] Da xuat cau tra loi vao:\n    - JSON: {json_out}\n    - MD:   {md_out}\n    (Moi: {new_count}, Bo qua: {skipped_count}) trong {total_elapsed}s")
    return final_answers


if __name__ == "__main__":
    run_auto_generator()
