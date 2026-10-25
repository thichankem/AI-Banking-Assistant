"""
Kiểm thử toàn trình 5 bước AI Banking RAG Pipeline (End-to-End Regression Test):
Kiểm tra luồng dữ liệu liên hoàn:
Query thô -> Step 1: Query Parser -> Step 2: Hybrid Retrieval (Qdrant + BM25 + RRF)
          -> Step 3: Reranker (Cross-Encoder) -> Step 4: Context Synthesizer (Markdown + Pháp lý)
          -> Step 5: LLM Serving (DeepSeek-V4-Flash / Banking Guardrails).
"""

import sys
import json
import time
from pathlib import Path

# Đảm bảo UTF-8 trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent_banking.config import Config
from agent_banking.pipeline import BankingRAGPipeline


def main():
    print("=" * 80)
    print("▶ 1. KHỞI TẠO PIPELINE TOÀN TRÌNH (CHẾ ĐỘ TEST IN-MEMORY)")
    print("=" * 80)

    config = Config()
    pipeline = BankingRAGPipeline(config=config, use_memory=True)

    # 2. Nạp dữ liệu mẫu thực tế từ FAQ và Pháp lý ngân hàng
    cand_dirs = [
        ROOT_DIR / "agent_banking" / "chunking" / "output" / "chunks_per_file",
        ROOT_DIR / "agent_banking" / "chunking" / "chunking_dataset" / "dataset_1" / "chunks_per_file",
        ROOT_DIR / "dataset" / "dataset_1" / "chunks_per_file",
    ]
    all_test_chunks = []
    for d in cand_dirs:
        if d.exists():
            for f in d.glob("*.json"):
                try:
                    with open(f, "r", encoding="utf-8") as fp:
                        data = json.load(fp)
                        if isinstance(data, list):
                            all_test_chunks.extend(data[:15])
                except Exception:
                    pass
                if len(all_test_chunks) >= 30:
                    break
        if len(all_test_chunks) >= 30:
            break

    print(f"▶ Nạp {len(all_test_chunks)} chunks mẫu vào Pipeline...")
    pipeline.ingest_chunks(all_test_chunks)

    test_queries = [
        "Tôi bị nuốt thẻ ATM tại cây ATM thì cần làm gì để nhận lại thẻ?",
        "Điều kiện mở tài khoản thanh toán trực tuyến qua eKYC và xác thực CCCD gắn chip là gì?"
    ]

    for idx, query in enumerate(test_queries, start=1):
        print("\n" + "=" * 80)
        print(f"▶ 2.{idx} TEST TOÀN TRÌNH CÂU HỎI: '{query}'")
        print("=" * 80)

        res = pipeline.run_pipeline(query)

        # Kiểm tra Step 1: Query Parser
        pq = res["step1_parsed_query"]
        print(f"\n[Bước 1: Query Parser]:")
        print(f"  * Normalized Query : {pq.normalized_query}")
        print(f"  * Keywords Str     : {pq.keywords_str}")
        print(f"  * Detected Acronyms: {pq.detected_acronyms}")

        # Kiểm tra Step 2: Hybrid Retrieval
        candidates = res["step2_candidates_top5"]
        print(f"\n[Bước 2: Hybrid Retrieval (Top-{len(candidates)})]:")
        for c in candidates:
            chunk = c["chunk"]
            title = chunk.get("section_title", "")
            print(f"  Top {c['candidate_rank']} [RRF: {c['rrf_score']:.6f} | DenseRank: {c['dense_rank']} | BM25Rank: {c['bm25_rank']}]: {title[:65]}...")

        # Kiểm tra Step 3: Reranker
        top3 = res["step3_top3_chunks"]
        print(f"\n[Bước 3: Reranker Cross-Encoder (Top-{len(top3)})]:")
        for item in top3:
            chunk = item.get("chunk", item)
            score = item.get("relevance_score", 0.0)
            rank = item.get("rerank_rank", 1)
            title = chunk.get("section_title", "")
            print(f"  ★ Top {rank} [Score: {score:.4f}]: {title[:70]}...")

        # Kiểm tra Step 4: Context Synthesizer
        ctx = res["step4_synthesized_context"]
        print(f"\n[Bước 4: Context Synthesizer]:")
        print(f"  * Tổng số chunks đóng gói : {ctx.chunks_included}")
        print(f"  * Ước tính token ngữ cảnh  : {ctx.total_tokens} tokens")
        print(f"  * Thẻ pháp lý nhận diện    : {ctx.legal_citations}")
        print(f"  * Nguồn tài liệu tham chiếu: {ctx.sources_used}")

        # Kiểm tra Step 5: LLM Serving
        print(f"\n[Bước 5: LLM Serving (DeepSeek-V4-Flash)]: ")
        print("-" * 60)
        print(res["answer"])
        print("-" * 60)

        # Thống kê độ trễ
        print(f"\n[Thống kê độ trễ toàn trình]:")
        for step_name, ms in res["latency"].items():
            print(f"  - {step_name:<28}: {ms:>8.1f} ms")

        # Khẳng định chất lượng (Assertions)
        assert len(candidates) > 0, "Hybrid Retrieval phải trả về ứng viên!"
        assert len(top3) > 0, "Reranker phải trả về kết quả tinh lọc!"
        assert len(res["answer"]) > 20, "LLM phải sinh câu trả lời có nghĩa!"

    print("\n" + "=" * 80)
    print("🎉 TOÀN BỘ 5 BƯỚC PIPELINE AI BANKING RAG ĐÃ HOẠT ĐỘNG HOÀN HẢO 100%!")
    print("=" * 80)

    pipeline.close()


if __name__ == "__main__":
    main()
