#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Folder-Driven Node Pipeline (Kiến trúc RAG Hóa Folder-to-Folder)
======================================================================
Mô đun kết nối liên hoàn toàn bộ các Node trong hệ thống AI Banking Assistant:
Mỗi Node đều có thư mục Đầu vào (Input Folder) và thư mục Đầu ra (Output Folder) rõ ràng,
cho phép người dùng dễ dàng theo dõi (track), đối soát và kiểm thử độc lập từng tầng:

- Node 1 (Chunking Node):
    * Input : Thư mục chứa tài liệu chuẩn hóa (.docx)
    * Output: `chunks_per_file/`, `all_chunks.json`, `chunking_manifest.json`
- Node 2 (Embedding Node):
    * Input : Chunks từ Node 1
    * Output: `embeddings_manifest.json`, `sample_embeddings.json` (vector 1024-dim, SQLite cache)
- Node 3 (Vector DB & BM25 Indexing Node):
    * Input : Chunks & Embeddings từ Node 1 & 2
    * Output: `indexing_manifest.json` (Trạng thái chỉ mục kép Qdrant Dense & Okapi BM25)
- Node 4 (Query Parser Node):
    * Input : Câu hỏi người dùng (File `queries.json` hoặc danh sách truy vấn)
    * Output: `parsed_queries.json` (Chuẩn hóa từ viết tắt ngân hàng, từ khóa, thực thể)
- Node 5 (Hybrid Retrieval Node):
    * Input : Parsed Queries từ Node 4 + Qdrant / BM25
    * Output: `retrieval_candidates.json` (Top ứng viên kết hợp RRF k=60, Dense rank, BM25 rank)
- Node 6 (Reranking Node):
    * Input : Candidates từ Node 5
    * Output: `reranked_chunks.json` (Top-3 chunks tinh lọc qua Cross-Encoder bge-reranker-v2-m3)
- Node 7 (Context Synthesizer & LLM Generator Node):
    * Input : Top chunks từ Node 6
    * Output: `answers.json` & `answers.md` (Câu trả lời trích dẫn điều khoản, guardrails, độ trễ)

NGUYÊN TẮC:
1. 0% REGEX LAW: Không dùng regex trong phân tích logic.
2. ZERO TERNARY: Ưu tiên if/else tường minh, dễ đọc.
3. MINH BẠCH DỮ LIỆU: Mọi node đều lưu file JSON có cấu trúc để track dễ dàng.
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
from pathlib import Path

# UTF-8 trên Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_banking.config import Config
from rag_banking.pipeline import BankingRAGPipeline
from rag_banking.query_parser.rag_query_parser import ParsedQuery


class NodePipeline:
    """
    Bộ điều phối đường ống dữ liệu dạng Node (Folder-to-Folder Data Pipeline).
    """

    def __init__(self, config: Config | None = None, use_memory: bool = False):
        if config is not None:
            self.config = config
        else:
            self.config = Config()
        self.use_memory = use_memory
        self.pipeline = BankingRAGPipeline(config=self.config, use_memory=self.use_memory)

    # =========================================================================
    # NODE 1: CHUNKING NODE (Tài liệu thô -> Chunks phân cấp)
    # =========================================================================
    def execute_chunking_node(
        self,
        input_docs_dir: Path | str,
        output_chunks_dir: Path | str,
        extensions: list[str] | None = None,
        max_files: int | None = None,
        max_pdf_pages: int | None = 5
    ) -> list[dict]:
        """
        Node 1: Phân đoạn tài liệu thô bằng DLA Chunker bảo toàn Breadcrumbs.
        - Đọc tệp từ: input_docs_dir (quét đệ quy toàn bộ thư mục con)
        - Ghi ra: file JSON của từng tài liệu ngay tại output_chunks_dir/{stem}.json,
                  và tổng hợp tại all_chunks.json, chunking_manifest.json
        """
        in_path = Path(input_docs_dir)
        out_path = Path(output_chunks_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        chunks_per_file_dir = out_path / "chunks_per_file"
        chunks_per_file_dir.mkdir(parents=True, exist_ok=True)

        if not in_path.exists():
            raise FileNotFoundError(f"[Node 1] Không tìm thấy thư mục input: {in_path}")

        if extensions is None:
            extensions = [".docx"]

        print(f"\n{'='*75}")
        print(f"▶ [NODE 1: CHUNKING] Đọc từ: {in_path} -> Xuất ra: {out_path}")
        print(f"{'='*75}")

        files = []
        for ext in extensions:
            files.extend(list(in_path.rglob(f"*{ext}")))
            files.extend(list(in_path.rglob(f"*{ext.upper()}")))
        files = sorted(list(set(files)))

        if max_files is not None and max_files > 0:
            files = files[:max_files]

        print(f"[*] Tìm thấy {len(files)} tệp văn bản hợp lệ trong toàn bộ các thư mục input.")

        all_chunks: list[dict] = []
        files_manifest: list[dict] = []
        t0 = time.time()

        for idx, f in enumerate(files, start=1):
            t_f0 = time.time()
            try:
                if f.suffix.lower() == ".pdf":
                    print(f"  [Canonical Notice] Bỏ qua {f.name}: Phân hệ PDF thử nghiệm đã lưu tại experiments/pdf_chunker/. Lõi RAG Banking chuẩn hóa 100% đầu vào qua định dạng DOCX.")
                    continue
                else:
                    file_chunks = self.pipeline.dla_chunker.chunk_file(f)

                all_chunks.extend(file_chunks)
                dt_f = round(time.time() - t_f0, 2)

                # 1. Lưu file JSON chứa các chunk của từng tài liệu TRỰC TIẾP tại output_chunks_dir
                doc_json_path = out_path / f"{f.stem}.json"
                with open(doc_json_path, "w", encoding="utf-8") as fp:
                    json.dump(file_chunks, fp, ensure_ascii=False, indent=2)

                # 2. Đồng thời lưu vào chunks_per_file/ để đảm bảo tương thích
                single_json_path = chunks_per_file_dir / f"{f.stem}.json"
                with open(single_json_path, "w", encoding="utf-8") as fp:
                    json.dump(file_chunks, fp, ensure_ascii=False, indent=2)

                word_count = sum(c.get("word_count", len(c.get("content", "").split())) for c in file_chunks)
                files_manifest.append({
                    "file_name": f.name,
                    "rel_path": str(f.relative_to(in_path)).replace("\\", "/"),
                    "extension": f.suffix.lower(),
                    "chunk_count": len(file_chunks),
                    "word_count": word_count,
                    "duration_sec": dt_f,
                    "json_output": f"{f.stem}.json",
                    "status": "SUCCESS"
                })
                print(f"  [{idx:02d}/{len(files)}] {f.name[:45]:<45} -> {len(file_chunks):>3} chunks ({word_count:>5} từ, {dt_f}s)")
            except Exception as e:
                print(f"  [{idx:02d}/{len(files)}] {f.name} -> LỖI: {e}")
                files_manifest.append({
                    "file_name": f.name,
                    "rel_path": str(f.relative_to(in_path)).replace("\\", "/"),
                    "extension": f.suffix.lower(),
                    "chunk_count": 0,
                    "word_count": 0,
                    "duration_sec": round(time.time() - t_f0, 2),
                    "status": f"FAILED: {e}"
                })

        total_duration = round(time.time() - t0, 2)

        # Lưu all_chunks.json
        all_chunks_file = out_path / "all_chunks.json"
        with open(all_chunks_file, "w", encoding="utf-8") as fp:
            json.dump(all_chunks, fp, ensure_ascii=False, indent=2)

        # Lưu manifest.json
        manifest_file = out_path / "chunking_manifest.json"
        manifest_data = {
            "node_name": "Node_1_Chunking",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_dir": str(in_path),
            "output_dir": str(out_path),
            "total_files": len(files),
            "total_chunks": len(all_chunks),
            "total_words": sum(item["word_count"] for item in files_manifest),
            "total_duration_sec": total_duration,
            "files": files_manifest
        }
        with open(manifest_file, "w", encoding="utf-8") as fp:
            json.dump(manifest_data, fp, ensure_ascii=False, indent=2)

        print(f"✅ [Node 1 Hoàn Tất] {len(all_chunks)} chunks đã lưu vào: {out_path} ({total_duration}s)")
        return all_chunks

    # =========================================================================
    # NODE 2: EMBEDDING NODE (Chunks -> Vector 1024-dim & Cache)
    # =========================================================================
    def execute_embedding_node(
        self,
        input_chunks_or_path: list[dict] | Path | str,
        output_embeddings_dir: Path | str,
        batch_size: int = 32
    ) -> list[dict]:
        """
        Node 2: Nhúng vector 1024 chiều có bộ nhớ đệm SQLite.
        - Đọc từ: input_chunks_or_path (hoặc file all_chunks.json)
        - Ghi ra: output_embeddings_dir/embeddings_manifest.json và sample_embeddings.json
        """
        out_path = Path(output_embeddings_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        chunks: list[dict] = []
        if isinstance(input_chunks_or_path, (str, Path)):
            p = Path(input_chunks_or_path)
            if p.is_dir():
                p = p / "all_chunks.json"
            if not p.exists():
                raise FileNotFoundError(f"[Node 2] Không tìm thấy tệp chunk đầu vào: {p}")
            with open(p, "r", encoding="utf-8") as fp:
                chunks = json.load(fp)
        else:
            chunks = input_chunks_or_path

        print(f"\n{'='*75}")
        print(f"▶ [NODE 2: EMBEDDING] Đang tạo vector cho {len(chunks)} chunks -> Xuất ra: {out_path}")
        print(f"{'='*75}")

        t0 = time.time()
        embedded_chunks = self.pipeline.embedder.embed_chunks(chunks, batch_size=batch_size)
        total_duration = round(time.time() - t0, 2)

        cache_hits = getattr(self.pipeline.embedder, "last_cache_hits", 0)

        # Lưu mẫu vector (10 chunks đầu) để người dùng track kích thước/tọa độ mà không làm phình đĩa
        sample_vectors = []
        for c in embedded_chunks[:10]:
            vec = c.get("vector", c.get("embedding", []))
            sample_vectors.append({
                "chunk_id": c.get("chunk_id"),
                "source": c.get("source"),
                "section_title": c.get("section_title"),
                "dimension": len(vec),
                "vector_head_5": vec[:5] if vec else [],
                "vector_tail_5": vec[-5:] if vec else []
            })

        sample_file = out_path / "sample_embeddings.json"
        with open(sample_file, "w", encoding="utf-8") as fp:
            json.dump(sample_vectors, fp, ensure_ascii=False, indent=2)

        manifest_file = out_path / "embeddings_manifest.json"
        manifest_data = {
            "node_name": "Node_2_Embedding",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": self.config.EMBEDDING_MODEL_1024,
            "dimension": self.config.EMBEDDING_DIM,
            "total_chunks_embedded": len(embedded_chunks),
            "cache_hits": cache_hits,
            "cache_db_path": str(getattr(self.pipeline.embedder, "cache_path", "")),
            "total_duration_sec": total_duration,
            "sample_file": str(sample_file.name)
        }
        with open(manifest_file, "w", encoding="utf-8") as fp:
            json.dump(manifest_data, fp, ensure_ascii=False, indent=2)

        print(f"✅ [Node 2 Hoàn Tất] Đã nhúng {len(embedded_chunks)} vector (Cache hits: {cache_hits}, {total_duration}s)")
        return embedded_chunks

    # =========================================================================
    # NODE 3: VECTOR DB & BM25 INDEXING NODE (Nạp vào Qdrant & BM25 Inverted Index)
    # =========================================================================
    def execute_indexing_node(
        self,
        input_embedded_chunks: list[dict] | Path | str,
        output_vectordb_dir: Path | str
    ) -> dict:
        """
        Node 3: Lập chỉ mục kép song song (Dual-Indexing):
        - Dense Vector HNSW 1024-dim vào Qdrant
        - Sparse Okapi BM25 vào Keyword Inverted Index
        """
        out_path = Path(output_vectordb_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        chunks: list[dict] = []
        if isinstance(input_embedded_chunks, (str, Path)):
            p = Path(input_embedded_chunks)
            if p.is_dir():
                p = p / "all_chunks.json"
            if not p.exists():
                raise FileNotFoundError(f"[Node 3] Không tìm thấy chunks đầu vào: {p}")
            with open(p, "r", encoding="utf-8") as fp:
                chunks = json.load(fp)
        else:
            chunks = input_embedded_chunks

        print(f"\n{'='*75}")
        print(f"▶ [NODE 3: VECTOR DB & BM25] Lập chỉ mục {len(chunks)} chunks -> Xuất ra: {out_path}")
        print(f"{'='*75}")

        t0 = time.time()
        self.pipeline.vector_store.add_chunks(chunks)
        total_duration = round(time.time() - t0, 2)

        total_count = self.pipeline.vector_store.count()
        manifest_file = out_path / "indexing_manifest.json"
        manifest_data = {
            "node_name": "Node_3_VectorDB_BM25",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "collection_name": self.config.QDRANT_COLLECTION,
            "dimension": self.config.EMBEDDING_DIM,
            "total_indexed_chunks": len(chunks),
            "collection_total_points": total_count,
            "total_duration_sec": total_duration,
            "status": "READY_FOR_RETRIEVAL"
        }
        with open(manifest_file, "w", encoding="utf-8") as fp:
            json.dump(manifest_data, fp, ensure_ascii=False, indent=2)

        print(f"✅ [Node 3 Hoàn Tất] Chỉ mục kép Qdrant & BM25 đã sẵn sàng ({total_duration}s)")
        return manifest_data

    # =========================================================================
    # NODE 4: QUERY PARSER NODE (Chuẩn hóa câu hỏi, thực thể, từ viết tắt)
    # =========================================================================
    def execute_query_parser_node(
        self,
        queries: list[str] | str | Path,
        output_query_dir: Path | str
    ) -> list[dict]:
        """
        Node 4: Phân tích và chuẩn hóa câu hỏi:
        - Chuẩn hóa từ viết tắt ngân hàng (eKYC, TKTT, Smart OTP, VCB Digibank...)
        - Trích xuất từ khóa, thực thể nghiệp vụ.
        """
        out_path = Path(output_query_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        query_list: list[str] = []
        if isinstance(queries, (str, Path)) and Path(str(queries)).is_file():
            p = Path(str(queries))
            with open(p, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                if isinstance(data, list):
                    query_list = [item if isinstance(item, str) else item.get("question", "") for item in data]
                elif isinstance(data, dict):
                    query_list = data.get("queries", [str(data)])
        elif isinstance(queries, str):
            query_list = [queries]
        else:
            query_list = list(queries)

        print(f"\n{'='*75}")
        print(f"▶ [NODE 4: QUERY PARSER] Phân tích {len(query_list)} câu hỏi -> Xuất ra: {out_path}")
        print(f"{'='*75}")

        parsed_results: list[dict] = []
        for idx, q in enumerate(query_list, start=1):
            parsed: ParsedQuery = self.pipeline.step1_query_parser(q)
            res = {
                "id": idx,
                "raw_query": parsed.raw_query,
                "normalized_query": parsed.normalized_query,
                "detected_acronyms": parsed.detected_acronyms,
                "keywords": parsed.keywords,
                "keywords_str": parsed.keywords_str
            }
            parsed_results.append(res)
            print(f"  [{idx:02d}] '{q}' -> Từ khóa: {parsed.keywords_str}")

        out_file = out_path / "parsed_queries.json"
        with open(out_file, "w", encoding="utf-8") as fp:
            json.dump(parsed_results, fp, ensure_ascii=False, indent=2)

        print(f"✅ [Node 4 Hoàn Tất] Đã lưu {len(parsed_results)} parsed queries vào: {out_file}")
        return parsed_results

    # =========================================================================
    # NODE 5: HYBRID RETRIEVAL NODE (Dense + BM25 + RRF k=60)
    # =========================================================================
    def execute_retrieval_node(
        self,
        parsed_queries: list[dict] | Path | str,
        output_retrieval_dir: Path | str,
        top_k: int = 5
    ) -> list[dict]:
        """
        Node 5: Truy vấn đồng thời Dense Vector HNSW và Okapi BM25, hợp nhất bằng RRF.
        """
        out_path = Path(output_retrieval_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        p_queries: list[dict] = []
        if isinstance(parsed_queries, (str, Path)):
            p = Path(parsed_queries)
            if p.is_dir():
                p = p / "parsed_queries.json"
            with open(p, "r", encoding="utf-8") as fp:
                p_queries = json.load(fp)
        else:
            p_queries = parsed_queries

        print(f"\n{'='*75}")
        print(f"▶ [NODE 5: HYBRID RETRIEVAL] Tìm Top-{top_k} ứng viên cho {len(p_queries)} queries -> Xuất ra: {out_path}")
        print(f"{'='*75}")

        retrieval_results: list[dict] = []
        for item in p_queries:
            q_text = item.get("normalized_query", item.get("raw_query", ""))
            q_id = item.get("id", 1)

            parsed_obj = ParsedQuery(
                raw_query=item.get("raw_query", q_text),
                normalized_query=q_text,
                detected_acronyms=item.get("detected_acronyms", {}),
                keywords=item.get("keywords", []),
                keywords_str=item.get("keywords_str", "")
            )

            t0 = time.time()
            candidates = self.pipeline.step2_hybrid_retrieval(parsed_obj, top_k=top_k)
            dt = round((time.time() - t0) * 1000, 1)

            retrieval_results.append({
                "query_id": q_id,
                "query": item.get("raw_query", q_text),
                "duration_ms": dt,
                "candidates_count": len(candidates),
                "candidates": candidates
            })
            print(f"  [{q_id:02d}] '{q_text[:40]}...' -> Lấy được {len(candidates)} candidates ({dt}ms)")

        out_file = out_path / "retrieval_candidates.json"
        with open(out_file, "w", encoding="utf-8") as fp:
            json.dump(retrieval_results, fp, ensure_ascii=False, indent=2)

        print(f"✅ [Node 5 Hoàn Tất] Đã lưu kết quả Retrieval vào: {out_file}")
        return retrieval_results

    # =========================================================================
    # NODE 6: RERANKING NODE (Cross-Encoder bge-reranker-v2-m3)
    # =========================================================================
    def execute_reranking_node(
        self,
        retrieval_results: list[dict] | Path | str,
        output_reranking_dir: Path | str,
        top_k: int = 3
    ) -> list[dict]:
        """
        Node 6: Tinh lọc ứng viên bằng Cross-Encoder bge-reranker-v2-m3.
        """
        out_path = Path(output_reranking_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        ret_items: list[dict] = []
        if isinstance(retrieval_results, (str, Path)):
            p = Path(retrieval_results)
            if p.is_dir():
                p = p / "retrieval_candidates.json"
            with open(p, "r", encoding="utf-8") as fp:
                ret_items = json.load(fp)
        else:
            ret_items = retrieval_results

        print(f"\n{'='*75}")
        print(f"▶ [NODE 6: RERANKING] Cross-Encoder Rerank Top-{top_k} cho {len(ret_items)} queries -> Xuất ra: {out_path}")
        print(f"{'='*75}")

        reranked_results: list[dict] = []
        for item in ret_items:
            q = item.get("query", "")
            q_id = item.get("query_id", 1)
            candidates = item.get("candidates", [])

            t0 = time.time()
            top_reranked = self.pipeline.step3_reranking(q, candidates, top_k=top_k)
            dt = round((time.time() - t0) * 1000, 1)

            reranked_results.append({
                "query_id": q_id,
                "query": q,
                "duration_ms": dt,
                "reranked_count": len(top_reranked),
                "top_chunks": top_reranked
            })
            print(f"  [{q_id:02d}] Rerank xong: {len(top_reranked)} chunks chuẩn nhất ({dt}ms)")

        out_file = out_path / "reranked_chunks.json"
        with open(out_file, "w", encoding="utf-8") as fp:
            json.dump(reranked_results, fp, ensure_ascii=False, indent=2)

        print(f"✅ [Node 6 Hoàn Tất] Đã lưu kết quả Reranking vào: {out_file}")
        return reranked_results

    # =========================================================================
    # NODE 7: CONTEXT SYNTHESIZER & GENERATOR NODE (LLM Serving & Guardrails)
    # =========================================================================
    def execute_generation_node(
        self,
        reranked_results: list[dict] | Path | str,
        output_generation_dir: Path | str
    ) -> list[dict]:
        """
        Node 7: Đóng gói ngữ cảnh Markdown + Sinh câu trả lời qua DeepSeek/Gemini kèm Guardrails.
        """
        out_path = Path(output_generation_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        rerank_items: list[dict] = []
        if isinstance(reranked_results, (str, Path)):
            p = Path(reranked_results)
            if p.is_dir():
                p = p / "reranked_chunks.json"
            with open(p, "r", encoding="utf-8") as fp:
                rerank_items = json.load(fp)
        else:
            rerank_items = reranked_results

        print(f"\n{'='*75}")
        print(f"▶ [NODE 7: GENERATOR] Sinh câu trả lời cho {len(rerank_items)} queries -> Xuất ra: {out_path}")
        print(f"{'='*75}")

        final_answers: list[dict] = []
        markdown_reports: list[str] = [
            "# 📋 BÁO CÁO KẾT QUẢ SINH CÂU TRẢ LỜI (NODE 7 - LLM SERVING)\n",
            f"> Thời gian khởi tạo: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        ]

        for item in rerank_items:
            q = item.get("query", "")
            q_id = item.get("query_id", 1)
            top_chunks = item.get("top_chunks", [])

            t0 = time.time()
            # Bước 4: Synthesize context
            synthesized_context = self.pipeline.step4_context_synthesizer(top_chunks)
            # Bước 5: Generate answer
            serving_result = self.pipeline.step5_llm_serving(q, synthesized_context)
            dt = round((time.time() - t0) * 1000, 1)

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
                "duration_ms": dt
            }
            final_answers.append(record)

            markdown_reports.append(f"## ❓ Câu hỏi {q_id}: {q}")
            markdown_reports.append(f"**💡 Câu trả lời:**\n{ans_text}\n")
            if citations:
                markdown_reports.append(f"* **Tham chiếu pháp lý:** {', '.join(citations)}")
            if sources:
                markdown_reports.append(f"* **Tài liệu nguồn:** {', '.join(sources)}")
            markdown_reports.append(f"* **Thời gian sinh:** {dt}ms\n---\n")

            print(f"  [{q_id:02d}] Hoàn tất sinh câu trả lời ({dt}ms)")

        json_out = out_path / "answers.json"
        with open(json_out, "w", encoding="utf-8") as fp:
            json.dump(final_answers, fp, ensure_ascii=False, indent=2)

        md_out = out_path / "answers.md"
        with open(md_out, "w", encoding="utf-8") as fp:
            fp.write("\n".join(markdown_reports))

        print(f"✅ [Node 7 Hoàn Tất] Đã lưu kết quả câu trả lời vào:\n    - JSON: {json_out}\n    - Markdown: {md_out}")
        return final_answers

    # =========================================================================
    # QUẢN LÝ VÀ CHẠY PIPELINE TỪ FOLDER CỦA TỪNG NODE (rag_banking/<node>/)
    # =========================================================================
    @staticmethod
    def get_node_folders() -> dict[str, dict[str, Path]]:
        """
        Trả về đường dẫn thư mục input và output cố định tại từng node trong rag_banking.
        """
        nodes = ["chunking", "embedding", "vector_db", "query_parser", "retrieval", "reranking", "generator"]
        folders: dict[str, dict[str, Path]] = {}
        for n in nodes:
            in_p = PROJECT_ROOT / "rag_banking" / n / "input"
            out_p = PROJECT_ROOT / "rag_banking" / n / "output"
            in_p.mkdir(parents=True, exist_ok=True)
            out_p.mkdir(parents=True, exist_ok=True)
            folders[n] = {"input": in_p, "output": out_p}
        return folders

    def run_node_directories_pipeline(self, max_files: int | None = None) -> dict:
        """
        Chạy toàn trình pipeline bằng cách đọc trực tiếp từ thư mục input của từng Node
        và ghi kết quả sang thư mục output của từng Node:
        1. rag_banking/chunking/input -> rag_banking/chunking/output
        2. Chuyển chunks -> rag_banking/embedding/input -> rag_banking/embedding/output
        3. Chuyển embedded_chunks -> rag_banking/vector_db/input -> rag_banking/vector_db/output
        4. rag_banking/query_parser/input -> rag_banking/query_parser/output
        5. Chuyển parsed_queries -> rag_banking/retrieval/input -> rag_banking/retrieval/output
        6. Chuyển candidates -> rag_banking/reranking/input -> rag_banking/reranking/output
        7. Chuyển top_chunks -> rag_banking/generator/input -> rag_banking/generator/output
        """
        f = self.get_node_folders()

        # 1. Node 1: Chunking
        chunks = self.execute_chunking_node(f["chunking"]["input"], f["chunking"]["output"], max_files=max_files)

        # Chuyển tiếp kết quả Node 1 sang Input của Node 2
        embed_in_file = f["embedding"]["input"] / "chunks.json"
        with open(embed_in_file, "w", encoding="utf-8") as fp:
            json.dump(chunks, fp, ensure_ascii=False, indent=2)

        # 2. Node 2: Embedding
        embedded_chunks = self.execute_embedding_node(chunks, f["embedding"]["output"])

        # Chuyển tiếp kết quả Node 2 sang Input của Node 3
        vdb_in_file = f["vector_db"]["input"] / "embedded_chunks.json"
        with open(vdb_in_file, "w", encoding="utf-8") as fp:
            json.dump(embedded_chunks, fp, ensure_ascii=False, indent=2)

        # 3. Node 3: Vector DB & BM25
        self.execute_indexing_node(embedded_chunks, f["vector_db"]["output"])

        # 4. Node 4: Query Parser
        q_file = f["query_parser"]["input"] / "queries.json"
        queries: list[str] = []
        if q_file.exists():
            try:
                with open(q_file, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    if isinstance(data, list):
                        queries.extend([str(item) for item in data])
                    elif isinstance(data, dict) and "queries" in data:
                        queries.extend([str(item) for item in data["queries"]])
            except Exception:
                pass
        if not queries:
            queries = [
                "Hạn mức giao dịch mở tài khoản eKYC tối đa bao nhiêu một ngày?",
                "Bị nuốt thẻ ATM tại cây ATM thì liên hệ tổng đài số nào?"
            ]

        parsed_queries = self.execute_query_parser_node(queries, f["query_parser"]["output"])

        # Chuyển tiếp sang Node 5: Retrieval
        ret_in_file = f["retrieval"]["input"] / "parsed_queries.json"
        with open(ret_in_file, "w", encoding="utf-8") as fp:
            json.dump(parsed_queries, fp, ensure_ascii=False, indent=2)

        # 5. Node 5: Retrieval
        candidates = self.execute_retrieval_node(parsed_queries, f["retrieval"]["output"], top_k=self.config.HYBRID_TOP_K)

        # Chuyển tiếp sang Node 6: Reranking
        rerank_in_file = f["reranking"]["input"] / "candidates.json"
        with open(rerank_in_file, "w", encoding="utf-8") as fp:
            json.dump(candidates, fp, ensure_ascii=False, indent=2)

        # 6. Node 6: Reranking
        reranked = self.execute_reranking_node(candidates, f["reranking"]["output"], top_k=self.config.RERANK_TOP_K)

        # Chuyển tiếp sang Node 7: Generator
        gen_in_file = f["generator"]["input"] / "top_chunks.json"
        with open(gen_in_file, "w", encoding="utf-8") as fp:
            json.dump(reranked, fp, ensure_ascii=False, indent=2)

        # 7. Node 7: Generator
        answers = self.execute_generation_node(reranked, f["generator"]["output"])

        print("\n" + "=" * 80)
        print("  🎉 ĐÃ HOÀN TẤT ĐỒNG BỘ TOÀN BỘ CÁC THƯ MỤC INPUT/OUTPUT CỦA TỪNG NODE!")
        print("=" * 80 + "\n")
        return {
            "status": "success",
            "total_chunks": len(chunks),
            "total_queries": len(answers),
            "node_folders": {k: {"input": str(v["input"]), "output": str(v["output"])} for k, v in f.items()}
        }

    def run_single_query_with_node_trace(self, query_text: str) -> dict:
        """
        Thực thi một câu hỏi qua toàn bộ 7 nodes, đồng thời ghi nhận chi tiết
        Input & Output của từng node và lưu file vết vào folder input/output của node đó.
        """
        t0 = time.time()
        f = self.get_node_folders()

        # 1. Trạng thái Node 1 (Chunking) & Node 3 (Vector DB)
        total_chunks_stored = self.pipeline.vector_store.count()

        # 2. Node 4: Query Parser
        t_parse_0 = time.time()
        parsed_q = self.pipeline.query_parser.parse(query_text)
        t_parse_ms = round((time.time() - t_parse_0) * 1000, 1)

        # Lưu vết Node 4
        with open(f["query_parser"]["input"] / "latest_query.json", "w", encoding="utf-8") as fp:
            json.dump({"query": query_text, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, fp, ensure_ascii=False, indent=2)
        with open(f["query_parser"]["output"] / "latest_parsed.json", "w", encoding="utf-8") as fp:
            json.dump(parsed_q.to_dict(), fp, ensure_ascii=False, indent=2)

        # 3. Node 2: Embedding query
        t_embed_0 = time.time()
        q_vec = self.pipeline.embedder.embed_query(parsed_q.normalized_query)
        t_embed_ms = round((time.time() - t_embed_0) * 1000, 1)

        # Lưu vết Node 2
        with open(f["embedding"]["input"] / "latest_text.json", "w", encoding="utf-8") as fp:
            json.dump({"text": parsed_q.normalized_query}, fp, ensure_ascii=False, indent=2)
        with open(f["embedding"]["output"] / "latest_vector.json", "w", encoding="utf-8") as fp:
            json.dump({
                "dimension": len(q_vec),
                "model": self.config.EMBEDDING_MODEL_1024,
                "first_5_values": [round(float(v), 5) for v in q_vec[:5]],
                "latency_ms": t_embed_ms
            }, fp, ensure_ascii=False, indent=2)

        # 4. Node 5: Hybrid Retrieval (Dense + BM25 với RRF)
        t_ret_0 = time.time()
        candidates = self.pipeline.retriever.retrieve(
            query=parsed_q,
            top_k=self.config.HYBRID_TOP_K
        )
        t_ret_ms = round((time.time() - t_ret_0) * 1000, 1)

        # Lưu vết Node 5
        cand_dicts = candidates
        with open(f["retrieval"]["input"] / "latest_parsed.json", "w", encoding="utf-8") as fp:
            json.dump(parsed_q.to_dict(), fp, ensure_ascii=False, indent=2)
        with open(f["retrieval"]["output"] / "latest_candidates.json", "w", encoding="utf-8") as fp:
            json.dump(cand_dicts, fp, ensure_ascii=False, indent=2)

        # 5. Node 6: Reranking
        t_rerank_0 = time.time()
        reranked = self.pipeline.reranker.rerank(
            query=query_text,
            candidates=candidates,
            top_k=self.config.RERANK_TOP_K
        )
        t_rerank_ms = round((time.time() - t_rerank_0) * 1000, 1)

        # Lưu vết Node 6
        rerank_dicts = reranked
        with open(f["reranking"]["input"] / "latest_candidates.json", "w", encoding="utf-8") as fp:
            json.dump(cand_dicts, fp, ensure_ascii=False, indent=2)
        with open(f["reranking"]["output"] / "latest_reranked.json", "w", encoding="utf-8") as fp:
            json.dump(rerank_dicts, fp, ensure_ascii=False, indent=2)

        # 6. Node 7: Generator & Guardrails
        t_gen_0 = time.time()
        synthesized_context = self.pipeline.step4_context_synthesizer(reranked)
        serving_result = self.pipeline.step5_llm_serving(query_text, synthesized_context)
        final_answer = serving_result.get("answer", "")
        t_gen_ms = round((time.time() - t_gen_0) * 1000, 1)

        # Lưu vết Node 7
        with open(f["generator"]["input"] / "latest_context.json", "w", encoding="utf-8") as fp:
            json.dump(rerank_dicts, fp, ensure_ascii=False, indent=2)
        with open(f["generator"]["output"] / "latest_answer.json", "w", encoding="utf-8") as fp:
            json.dump({
                "query": query_text,
                "answer": final_answer,
                "latency_ms": t_gen_ms
            }, fp, ensure_ascii=False, indent=2)

        total_lat_ms = round((time.time() - t0) * 1000, 1)

        # Trích dẫn nguồn
        citations = serving_result.get("legal_citations", [])
        sources = serving_result.get("sources_used", [])
        if not citations:
            for rc in reranked:
                c = rc.get("chunk", rc) if isinstance(rc, dict) else rc
                if isinstance(c, dict):
                    bc = c.get("breadcrumb", "")
                    if bc and bc not in citations:
                        citations.append(bc)
        if not sources:
            for rc in reranked:
                c = rc.get("chunk", rc) if isinstance(rc, dict) else rc
                if isinstance(c, dict):
                    src = c.get("source", "")
                    if src and src not in sources:
                        sources.append(src)

        return {
            "reply": final_answer,
            "latency_ms": total_lat_ms,
            "citations": citations,
            "sources": sources,
            "nodes": {
                "01_chunking": {
                    "name": "Node 1: Chunking (DLA Chunker)",
                    "input_folder": "rag_banking/chunking/input/",
                    "output_folder": "rag_banking/chunking/output/",
                    "status": "ready",
                    "total_chunks_in_store": total_chunks_stored,
                    "description": "Bảo toàn cấu trúc phân cấp Header, Bảng biểu và Breadcrumbs"
                },
                "02_embedding": {
                    "name": "Node 2: Dense Embedding (1024-dim)",
                    "input_folder": "rag_banking/embedding/input/",
                    "output_folder": "rag_banking/embedding/output/",
                    "model": self.config.EMBEDDING_MODEL_1024,
                    "dimension": len(q_vec),
                    "vector_sample": [round(float(v), 5) for v in q_vec[:6]],
                    "latency_ms": t_embed_ms
                },
                "03_vectordb": {
                    "name": "Node 3: Vector DB & BM25 Store",
                    "input_folder": "rag_banking/vector_db/input/",
                    "output_folder": "rag_banking/vector_db/output/",
                    "qdrant_collection": self.config.QDRANT_COLLECTION,
                    "qdrant_points_count": total_chunks_stored,
                    "bm25_indexed_chunks": total_chunks_stored
                },
                "04_query_parser": {
                    "name": "Node 4: Banking Query Parser",
                    "input_folder": "rag_banking/query_parser/input/",
                    "output_folder": "rag_banking/query_parser/output/",
                    "input_query": query_text,
                    "normalized_query": parsed_q.normalized_query,
                    "keywords": parsed_q.keywords,
                    "entities": parsed_q.entities,
                    "latency_ms": t_parse_ms
                },
                "05_retrieval": {
                    "name": "Node 5: Hybrid Retrieval (RRF k=60)",
                    "input_folder": "rag_banking/retrieval/input/",
                    "output_folder": "rag_banking/retrieval/output/",
                    "candidates_count": len(candidates),
                    "candidates": [
                        {
                            "rank": c.get("candidate_rank", i + 1),
                            "chunk_id": (c.get("chunk", {}) if isinstance(c.get("chunk"), dict) else {}).get("chunk_id", ""),
                            "source": (c.get("chunk", {}) if isinstance(c.get("chunk"), dict) else {}).get("source", ""),
                            "breadcrumb": (c.get("chunk", {}) if isinstance(c.get("chunk"), dict) else {}).get("breadcrumb", ""),
                            "rrf_score": round(float(c.get("rrf_score", 0.0)), 6),
                            "dense_score": round(float(c.get("dense_score", 0.0)), 4),
                            "bm25_score": round(float(c.get("bm25_score", 0.0)), 4),
                            "preview": (c.get("chunk", {}) if isinstance(c.get("chunk"), dict) else {}).get("content", "")[:200]
                        }
                        for i, c in enumerate(candidates[:5])
                    ],
                    "latency_ms": t_ret_ms
                },
                "06_reranking": {
                    "name": "Node 6: Cross-Encoder Reranker",
                    "input_folder": "rag_banking/reranking/input/",
                    "output_folder": "rag_banking/reranking/output/",
                    "model": self.config.RERANKER_MODEL,
                    "top_chunks": [
                        {
                            "rerank_rank": rc.get("rerank_rank", i + 1),
                            "chunk_id": (rc.get("chunk", {}) if isinstance(rc.get("chunk"), dict) else rc).get("chunk_id", ""),
                            "source": (rc.get("chunk", {}) if isinstance(rc.get("chunk"), dict) else rc).get("source", ""),
                            "breadcrumb": (rc.get("chunk", {}) if isinstance(rc.get("chunk"), dict) else rc).get("breadcrumb", ""),
                            "relevance_score": round(float(rc.get("relevance_score", 0.0)), 4),
                            "preview": (rc.get("chunk", {}) if isinstance(rc.get("chunk"), dict) else rc).get("content", "")[:250]
                        }
                        for i, rc in enumerate(reranked)
                    ],
                    "latency_ms": t_rerank_ms
                },
                "07_generator": {
                    "name": "Node 7: Generator & Guardrails",
                    "input_folder": "rag_banking/generator/input/",
                    "output_folder": "rag_banking/generator/output/",
                    "model": self.config.LLM_MODEL,
                    "answer": final_answer,
                    "citations": citations,
                    "latency_ms": t_gen_ms
                }
            }
        }


def main():
    parser = argparse.ArgumentParser(description="AI Banking Assistant - Folder-Driven Node Pipeline")
    parser.add_argument("--demo", action="store_true", help="Chạy kịch bản demo mẫu kiểm thử liên hoàn các node")
    parser.add_argument("--input-docs", type=str, default=None, help="Đường dẫn thư mục chứa tài liệu chuẩn hóa (.docx). Mặc định: data/input/docs")
    parser.add_argument("--output-dir", type=str, default=None, help="Đường dẫn thư mục lưu kết quả các node. Mặc định: data/output")
    parser.add_argument("--query", type=str, default=None, help="Câu hỏi kiểm thử đơn lẻ")
    parser.add_argument("--queries-file", type=str, default=None, help="File JSON hoặc TXT chứa danh sách câu hỏi kiểm thử")
    parser.add_argument("--max-files", type=int, default=None, help="Số lượng tệp tối đa khi thử nghiệm")
    args = parser.parse_args()

    node_pipe = NodePipeline(use_memory=True)

    if args.demo:
        demo_input = PROJECT_ROOT / "rag_banking" / "chunking" / "chunking_dataset" / "docx"
        if not demo_input.exists():
            demo_input = PROJECT_ROOT / "dataset" / "dataset_1" / "docx"

        demo_queries = [
            "Hạn mức giao dịch mở tài khoản eKYC tối đa bao nhiêu một ngày?",
            "Bị nuốt thẻ ATM tại cây ATM thì liên hệ tổng đài số nào?"
        ]
        demo_out = PROJECT_ROOT / "data" / "pipeline_io" / "demo_run"
        node_pipe.run_full_pipeline(
            input_docs_dir=demo_input,
            queries=demo_queries,
            output_base_dir=demo_out,
            max_files=args.max_files or 2
        )
        return

    # Xác định thư mục đầu vào
    if args.input_docs:
        in_docs = Path(args.input_docs)
    else:
        in_docs = PROJECT_ROOT / "data" / "input" / "docs"

    # Xác định thư mục đầu ra
    if args.output_dir:
        out_base = Path(args.output_dir)
    else:
        out_base = PROJECT_ROOT / "data" / "output"

    # Xác định danh sách queries
    queries: list[str] = []
    if args.query:
        queries.append(args.query)
    elif args.queries_file and Path(args.queries_file).exists():
        q_path = Path(args.queries_file)
        if q_path.suffix.lower() == ".json":
            with open(q_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    queries.extend([str(item) for item in data])
                elif isinstance(data, dict) and "queries" in data:
                    queries.extend([str(item) for item in data["queries"]])
        else:
            with open(q_path, "r", encoding="utf-8") as f:
                queries.extend([line.strip() for line in f if line.strip()])
    else:
        default_q_json = PROJECT_ROOT / "data" / "input" / "queries.json"
        default_q_txt = PROJECT_ROOT / "data" / "input" / "queries.txt"
        if default_q_json.exists():
            with open(default_q_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    queries.extend([str(item) for item in data])
                elif isinstance(data, dict) and "queries" in data:
                    queries.extend([str(item) for item in data["queries"]])
        elif default_q_txt.exists():
            with open(default_q_txt, "r", encoding="utf-8") as f:
                queries.extend([line.strip() for line in f if line.strip()])
        else:
            queries.append("Quy định và hạn mức giao dịch mở tài khoản eKYC trực tuyến?")

    if in_docs.exists():
        node_pipe.run_full_pipeline(
            input_docs_dir=in_docs,
            queries=queries,
            output_base_dir=out_base,
            max_files=args.max_files
        )
    else:
        print(f"[Cảnh báo] Thư mục đầu vào không tồn tại: {in_docs}")
        print("Vui lòng tạo thư mục và bỏ tài liệu chuẩn hóa (.docx) vào: data/input/docs/")
        print("Hoặc chỉ định đường dẫn qua cờ: --input-docs <đường_dẫn_folder>")
        parser.print_help()


if __name__ == "__main__":
    main()
