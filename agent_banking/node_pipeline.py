"""
Automated Node-to-Node Pipeline Orchestrator.
==================================================================================
Features:
1. End-to-end orchestration across 7 RAG Nodes + 5 Agentic Modules.
2. Integrated StateTracker for incremental processing and deduplication:
   - Already processed files, queries, and chunks are skipped immediately.
   - Computes vectors and invokes LLM only for new or modified data.
3. Automated handover: Output of upstream nodes serves as Input for downstream nodes.
4. Execution modes:
   - Full pipeline execution: --auto
   - Independent single node execution: --node <node_name>
   - Continuous directory watcher daemon: --watch
   - Force reset cache and re-run all: --force

MANDATORY RULES:
1. NO __INIT__.PY: Direct module imports.
2. NO TYPING LIBRARY: Use Python native types only.
3. ZERO TERNARY: Explicit if/else blocks only (no ?: or conditional expressions).
4. 0% REGEX LAW: Strictly string methods only (in, split, startswith, endswith, strip, lower, replace, find).
"""

from __future__ import annotations
import sys
import json
import time
import argparse
from pathlib import Path

# Ensure UTF-8 on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.core.settings import Config
from agent_banking.state_tracker import StateTracker
from agent_banking.chunking.run_chunking import run_auto_chunking
from agent_banking.embedding.run_embedding import run_auto_embedding
from agent_banking.vector_db.run_vector_db import run_auto_indexing
from agent_banking.query_parser.run_query_parser import run_auto_query_parser
from agent_banking.retrieval.run_retrieval import run_auto_retrieval
from agent_banking.reranking.run_reranking import run_auto_reranking
from agent_banking.generator.run_generator import run_auto_generator
from agent_banking.guardrails.run_guardrails import run_auto_guardrails
from agent_banking.tracing.run_tracing import run_auto_tracing
from agent_banking.tasks.run_tasks import run_auto_tasks
from agent_banking.skills.run_skills import run_auto_skills
from agent_banking.tools.run_tools import run_auto_tools


class NodePipeline:
    """
    Orchestrator for automated node execution from Input to Output with deduplication.
    """

    def __init__(
        self,
        config: Config | None = None,
        base_dir: Path | None = None,
        rag_pipeline=None
    ):
        if config is not None:
            self.config = config
        else:
            self.config = Config()

        if base_dir is not None:
            self.base_dir = Path(base_dir)
        else:
            self.base_dir = PROJECT_ROOT / "agent_banking"

        self._rag_pipeline = rag_pipeline

    @property
    def rag_pipeline(self):
        if self._rag_pipeline is None:
            from agent_banking.pipeline import BankingRAGPipeline
            self._rag_pipeline = BankingRAGPipeline(config=self.config)
        return self._rag_pipeline

    def run_single_query_with_node_trace(self, query: str, persona_config: dict | None = None) -> dict:
        """
        Run a single query across all pipeline nodes and return detailed step execution traces.
        """
        return self.rag_pipeline.run_pipeline(query=query, persona_config=persona_config)

    def get_node_folders(self) -> dict:
        """Get input and output directories for all pipeline nodes."""
        nodes = [
            "chunking", "embedding", "vector_db", "query_parser",
            "retrieval", "reranking", "generator", "guardrails",
            "tracing", "tasks", "skills", "tools"
        ]
        res = {}
        for n in nodes:
            in_p = self.base_dir / n / "input"
            out_p = self.base_dir / n / "output"
            in_p.mkdir(parents=True, exist_ok=True)
            out_p.mkdir(parents=True, exist_ok=True)
            res[n] = {"input": in_p, "output": out_p}
        return res

    def run_node(self, node_name: str, force: bool = False) -> dict:
        """Run an isolated single node from input to output with deduplication."""
        f = self.get_node_folders()
        clean_name = node_name.strip().lower()

        if clean_name in ("1", "chunking"):
            chunks = run_auto_chunking(f["chunking"]["input"], f["chunking"]["output"], force=force)
            return {"node": "chunking", "count": len(chunks)}
        elif clean_name in ("2", "embedding"):
            embeds = run_auto_embedding(output_dir=f["embedding"]["output"], force=force)
            return {"node": "embedding", "count": len(embeds)}
        elif clean_name in ("3", "vector_db", "indexing"):
            vdb = run_auto_indexing(output_dir=f["vector_db"]["output"], force=force)
            return {"node": "vector_db", "manifest": vdb}
        elif clean_name in ("4", "query_parser"):
            qp = run_auto_query_parser(output_dir=f["query_parser"]["output"], force=force)
            return {"node": "query_parser", "count": len(qp)}
        elif clean_name in ("5", "retrieval"):
            ret = run_auto_retrieval(output_dir=f["retrieval"]["output"], force=force)
            return {"node": "retrieval", "count": len(ret)}
        elif clean_name in ("6", "reranking"):
            rerank = run_auto_reranking(output_dir=f["reranking"]["output"], force=force)
            return {"node": "reranking", "count": len(rerank)}
        elif clean_name in ("7", "generator"):
            gen = run_auto_generator(output_dir=f["generator"]["output"], force=force)
            return {"node": "generator", "count": len(gen)}
        elif clean_name in ("8", "guardrails"):
            guards = run_auto_guardrails(output_dir=f["guardrails"]["output"], force=force)
            return {"node": "guardrails", "count": len(guards)}
        elif clean_name in ("9", "tracing"):
            traces = run_auto_tracing(output_dir=f["tracing"]["output"], force=force)
            return {"node": "tracing", "count": len(traces)}
        elif clean_name in ("10", "tasks"):
            tasks = run_auto_tasks(output_dir=f["tasks"]["output"], force=force)
            return {"node": "tasks", "count": len(tasks)}
        elif clean_name in ("11", "skills"):
            skills = run_auto_skills(output_dir=f["skills"]["output"], force=force)
            return {"node": "skills", "count": len(skills)}
        elif clean_name in ("12", "tools"):
            tools = run_auto_tools(output_dir=f["tools"]["output"], force=force)
            return {"node": "tools", "count": len(tools)}
        else:
            print(f"[NodePipeline] Unrecognized node: '{node_name}'. Valid options: chunking, embedding, vector_db, query_parser, retrieval, reranking, generator, guardrails, tracing, tasks, skills, tools")
            return {"error": "UNKNOWN_NODE"}

    def run_full_pipeline(self, force: bool = False, max_files: int | None = None) -> dict:
        """
        Execute end-to-end pipeline across 7 RAG Nodes + 5 Agentic Modules:
        Node 1 (Chunking) -> Node 2 (Embedding) -> Node 3 (Vector DB & BM25)
        -> Node 4 (Query Parser) -> Node 5 (Retrieval) -> Node 6 (Reranking)
        -> Node 7 (Generator) + (Guardrails, Tracing, Tasks, Skills, Tools).

        Smart skipping:
        If a node has already finished and input data has not changed, the node
        is skipped immediately.
        """
        start_all = time.time()
        print("\n" + "#" * 80)
        print("  STARTING END-TO-END FOLDER PIPELINE (WITH INCREMENTAL DEDUPLICATION)")
        print("#" * 80)

        f = self.get_node_folders()

        # ---------------------------------------------------------------------
        # 1. Node 1: Chunking
        # ---------------------------------------------------------------------
        chunk_file_src = f["chunking"]["output"] / "all_chunks.json"
        tracker_c = StateTracker(f["chunking"]["output"] / ".state_tracker.json")
        docx_files = []
        for ext in (".docx", ".DOCX"):
            docx_files.extend(list(f["chunking"]["input"].rglob(f"*{ext}")))
        docx_files = sorted(list(set(docx_files)))

        is_chunking_done = False
        if not force and chunk_file_src.exists() and docx_files:
            c_hash_map = {}
            for fp in docx_files:
                rel = str(fp.relative_to(f["chunking"]["input"])).replace("\\", "/")
                c_hash_map[rel] = tracker_c.compute_file_hash(fp)
            if tracker_c.are_all_items_processed(c_hash_map):
                is_chunking_done = True

        if is_chunking_done:
            print("\n[PIPELINE: NODE 1 - CHUNKING SKIPPED] All .docx files were previously chunked in all_chunks.json. Skipping node.")
            try:
                with open(chunk_file_src, "r", encoding="utf-8") as fp:
                    chunks = json.load(fp)
            except Exception:
                chunks = run_auto_chunking(input_dir=f["chunking"]["input"], output_dir=f["chunking"]["output"], force=force, max_files=max_files)
        else:
            chunks = run_auto_chunking(
                input_dir=f["chunking"]["input"],
                output_dir=f["chunking"]["output"],
                force=force,
                max_files=max_files
            )

        # Handover all_chunks.json to Node 2: Embedding Input
        if chunk_file_src.exists():
            embed_in = f["embedding"]["input"] / "chunks.json"
            try:
                with open(chunk_file_src, "r", encoding="utf-8") as s_fp:
                    data = json.load(s_fp)
                with open(embed_in, "w", encoding="utf-8") as d_fp:
                    json.dump(data, d_fp, ensure_ascii=False, indent=2)
            except Exception:
                pass

        # ---------------------------------------------------------------------
        # 2. Node 2: Embedding
        # ---------------------------------------------------------------------
        embed_src = f["embedding"]["output"] / "embedded_chunks.json"
        is_embedding_done = False
        if not force and embed_src.exists() and len(chunks) > 0:
            try:
                with open(embed_src, "r", encoding="utf-8") as fp:
                    embedded_chunks = json.load(fp)
                if len(embedded_chunks) >= len(chunks):
                    is_embedding_done = True
            except Exception:
                is_embedding_done = False

        if is_embedding_done:
            print(f"\n[PIPELINE: NODE 2 - EMBEDDING SKIPPED] All {len(chunks)} chunks already have embeddings in output/embedded_chunks.json. Skipping node.")
        else:
            embedded_chunks = run_auto_embedding(
                output_dir=f["embedding"]["output"],
                force=force
            )

        # Handover embedded_chunks.json to Node 3: Vector DB Input
        if embed_src.exists():
            vdb_in = f["vector_db"]["input"] / "embedded_chunks.json"
            try:
                with open(embed_src, "r", encoding="utf-8") as s_fp:
                    data = json.load(s_fp)
                with open(vdb_in, "w", encoding="utf-8") as d_fp:
                    json.dump(data, d_fp, ensure_ascii=False, indent=2)
            except Exception:
                pass

        # ---------------------------------------------------------------------
        # 3. Node 3: Vector DB & BM25 Indexing
        # ---------------------------------------------------------------------
        manifest_vdb_src = f["vector_db"]["output"] / "indexing_manifest.json"
        is_vdb_done = False
        if not force and manifest_vdb_src.exists() and len(embedded_chunks) > 0:
            try:
                with open(manifest_vdb_src, "r", encoding="utf-8") as fp:
                    indexing_manifest = json.load(fp)
                pts = indexing_manifest.get("total_points_in_collection", 0)
                if pts >= len(embedded_chunks):
                    is_vdb_done = True
            except Exception:
                is_vdb_done = False

        if is_vdb_done:
            pts_cnt = indexing_manifest.get("total_points_in_collection", len(embedded_chunks))
            print(f"\n[PIPELINE: NODE 3 - VECTOR DB SKIPPED] Vector DB & BM25 collection already indexed ({pts_cnt} points). Skipping node.")
        else:
            indexing_manifest = run_auto_indexing(
                output_dir=f["vector_db"]["output"],
                force=force
            )

        # ---------------------------------------------------------------------
        # 4. Node 4: Query Parser
        # ---------------------------------------------------------------------
        qp_src = f["query_parser"]["output"] / "parsed_queries.json"
        is_qp_done = False
        if not force and qp_src.exists():
            try:
                with open(qp_src, "r", encoding="utf-8") as fp:
                    parsed_queries = json.load(fp)
                if parsed_queries:
                    is_qp_done = True
            except Exception:
                is_qp_done = False

        if is_qp_done:
            print(f"\n[PIPELINE: NODE 4 - QUERY PARSER SKIPPED] All {len(parsed_queries)} queries were already parsed in output. Skipping node.")
        else:
            parsed_queries = run_auto_query_parser(
                output_dir=f["query_parser"]["output"],
                force=force
            )

        # Handover parsed_queries.json to Node 5: Retrieval Input
        if qp_src.exists():
            ret_in = f["retrieval"]["input"] / "parsed_queries.json"
            try:
                with open(qp_src, "r", encoding="utf-8") as s_fp:
                    data = json.load(s_fp)
                with open(ret_in, "w", encoding="utf-8") as d_fp:
                    json.dump(data, d_fp, ensure_ascii=False, indent=2)
            except Exception:
                pass

        # ---------------------------------------------------------------------
        # 5. Node 5: Hybrid Retrieval
        # ---------------------------------------------------------------------
        ret_src = f["retrieval"]["output"] / "retrieval_candidates.json"
        is_ret_done = False
        if not force and ret_src.exists():
            try:
                with open(ret_src, "r", encoding="utf-8") as fp:
                    candidates = json.load(fp)
                if len(candidates) >= len(parsed_queries) and len(parsed_queries) > 0:
                    is_ret_done = True
            except Exception:
                is_ret_done = False

        if is_ret_done:
            print(f"\n[PIPELINE: NODE 5 - RETRIEVAL SKIPPED] All {len(candidates)} search candidates are present in output. Skipping node.")
        else:
            candidates = run_auto_retrieval(
                output_dir=f["retrieval"]["output"],
                top_k=self.config.HYBRID_TOP_K,
                force=force
            )

        # Handover retrieval_candidates.json to Node 6: Reranking Input
        if ret_src.exists():
            rerank_in = f["reranking"]["input"] / "candidates.json"
            try:
                with open(ret_src, "r", encoding="utf-8") as s_fp:
                    data = json.load(s_fp)
                with open(rerank_in, "w", encoding="utf-8") as d_fp:
                    json.dump(data, d_fp, ensure_ascii=False, indent=2)
            except Exception:
                pass

        # ---------------------------------------------------------------------
        # 6. Node 6: Cross-Encoder Reranking
        # ---------------------------------------------------------------------
        rerank_src = f["reranking"]["output"] / "reranked_chunks.json"
        is_rr_done = False
        if not force and rerank_src.exists():
            try:
                with open(rerank_src, "r", encoding="utf-8") as fp:
                    reranked = json.load(fp)
                if len(reranked) >= len(candidates) and len(candidates) > 0:
                    is_rr_done = True
            except Exception:
                is_rr_done = False

        if is_rr_done:
            print(f"\n[PIPELINE: NODE 6 - RERANKING SKIPPED] All {len(reranked)} reranked results are present in output. Skipping node.")
        else:
            reranked = run_auto_reranking(
                output_dir=f["reranking"]["output"],
                top_k=self.config.RERANK_TOP_K,
                force=force
            )

        # Handover reranked_chunks.json to Node 7: Generator Input
        if rerank_src.exists():
            gen_in = f["generator"]["input"] / "top_chunks.json"
            try:
                with open(rerank_src, "r", encoding="utf-8") as s_fp:
                    data = json.load(s_fp)
                with open(gen_in, "w", encoding="utf-8") as d_fp:
                    json.dump(data, d_fp, ensure_ascii=False, indent=2)
            except Exception:
                pass

        # ---------------------------------------------------------------------
        # 7. Node 7: Context Synthesizer & LLM Generator
        # ---------------------------------------------------------------------
        gen_ans_src = f["generator"]["output"] / "answers.json"
        is_gen_done = False
        if not force and gen_ans_src.exists():
            try:
                with open(gen_ans_src, "r", encoding="utf-8") as fp:
                    answers = json.load(fp)
                if len(answers) >= len(reranked) and len(reranked) > 0:
                    is_gen_done = True
            except Exception:
                is_gen_done = False

        if is_gen_done:
            print(f"\n[PIPELINE: NODE 7 - GENERATOR SKIPPED] All {len(answers)} generated answers are present in output. Skipping node.")
        else:
            answers = run_auto_generator(
                output_dir=f["generator"]["output"],
                force=force
            )

        # ---------------------------------------------------------------------
        # 8-12. Auxiliary Nodes: Guardrails, Tracing, Tasks, Skills, Tools
        # ---------------------------------------------------------------------
        guards = run_auto_guardrails(output_dir=f["guardrails"]["output"], force=force)
        traces = run_auto_tracing(output_dir=f["tracing"]["output"], force=force)
        tasks = run_auto_tasks(output_dir=f["tasks"]["output"], force=force)
        skills = run_auto_skills(output_dir=f["skills"]["output"], force=force)
        tools = run_auto_tools(output_dir=f["tools"]["output"], force=force)

        total_elapsed = round(time.time() - start_all, 2)
        print("\n" + "=" * 80)
        print("  ALL 12 MODULES COMPLETED PROCESSING AND SYNCHRONIZATION")
        print(f"  Execution time: {total_elapsed}s")
        print("=" * 80 + "\n")

        return {
            "status": "SUCCESS",
            "total_chunks": len(chunks),
            "embedded_chunks": len(embedded_chunks),
            "indexed_points": indexing_manifest.get("total_points_in_collection", 0),
            "parsed_queries": len(parsed_queries),
            "retrieval_candidates": len(candidates),
            "reranked_queries": len(reranked),
            "generated_answers": len(answers),
            "guardrails_checks": len(guards),
            "traces_recorded": len(traces),
            "tasks_executed": len(tasks),
            "skills_executed": len(skills),
            "tools_executed": len(tools),
            "elapsed_seconds": total_elapsed
        }

    def watch_inputs(self, poll_interval_sec: int = 3) -> None:
        """
        Continuous directory watcher daemon:
        Monitors input directories across nodes. When new data is placed into an
        input directory, the system automatically triggers execution to output.
        """
        print(f"\n[WATCHER DAEMON] Monitoring input directories in {self.base_dir}...")
        print("Files placed into any input directory will be processed automatically.")
        print("Press Ctrl+C to stop monitoring.\n")

        last_hashes = {}
        folders = self.get_node_folders()

        def compute_dir_signature(d: Path) -> str:
            if not d.exists():
                return ""
            files = sorted(list(d.rglob("*")))
            sig_parts = []
            for f in files:
                if f.is_file() and not f.name.startswith("."):
                    stat = f.stat()
                    sig_parts.append(f"{f.name}:{stat.st_size}:{stat.st_mtime}")
            return ";".join(sig_parts)

        # Initialize initial directory signatures
        for node_name, paths in folders.items():
            last_hashes[node_name] = compute_dir_signature(paths["input"])

        try:
            while True:
                time.sleep(poll_interval_sec)
                for node_name, paths in folders.items():
                    current_sig = compute_dir_signature(paths["input"])
                    prev_sig = last_hashes.get(node_name, "")
                    if current_sig != prev_sig:
                        last_hashes[node_name] = current_sig
                        print(f"\n[CHANGE DETECTED] New data in {paths['input']}!")
                        print(f"Triggering execution for Node: '{node_name}'...")
                        try:
                            self.run_node(node_name, force=False)
                            # If chunking, automatically trigger downstream pipeline
                            if node_name == "chunking":
                                print("Triggering downstream processing pipeline...")
                                self.run_full_pipeline(force=False)
                            elif node_name == "query_parser":
                                print("Handing over queries to retrieval, reranking, and generator nodes...")
                                self.run_node("retrieval", force=False)
                                self.run_node("reranking", force=False)
                                self.run_node("generator", force=False)
                        except Exception as e:
                            print(f"[NodePipeline] Error while processing Node '{node_name}': {e}")
        except KeyboardInterrupt:
            print("\n[NodePipeline] Watcher daemon stopped.")


def main():
    parser = argparse.ArgumentParser(description="AI Banking Assistant - Auto Folder-to-Folder Pipeline")
    parser.add_argument("--auto", action="store_true", help="Run automated end-to-end pipeline from input to output")
    parser.add_argument("--node", type=str, default=None, help="Run a specific single node (chunking, embedding, vector_db, query_parser, retrieval, reranking, generator...)")
    parser.add_argument("--watch", action="store_true", help="Run automatic input folder watcher daemon")
    parser.add_argument("--force", action="store_true", help="Clear state caches and re-run all nodes from scratch")
    parser.add_argument("--max-files", type=int, default=None, help="Limit maximum number of document files during chunking")
    args = parser.parse_args()

    pipeline = NodePipeline()

    if args.watch:
        pipeline.watch_inputs()
    elif args.node:
        pipeline.run_node(args.node, force=args.force)
    elif args.auto:
        pipeline.run_full_pipeline(force=args.force, max_files=args.max_files)
    else:
        # Default behavior: run full automated pipeline
        pipeline.run_full_pipeline(force=args.force, max_files=args.max_files)


if __name__ == "__main__":
    main()
