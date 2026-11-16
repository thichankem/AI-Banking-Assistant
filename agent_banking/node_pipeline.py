"""Folder-based knowledge indexing and isolated component diagnostics."""

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

from agent_banking.chunking.run_chunking import atomic_json, run_auto_chunking
from agent_banking.embedding.run_embedding import run_auto_embedding
from agent_banking.vector_db.run_vector_db import run_auto_indexing
from agent_banking.query_parser.run_query_parser import run_auto_query_parser
from agent_banking.retrieval.run_retrieval import run_auto_retrieval
from agent_banking.reranking.run_reranking import run_auto_reranking
from agent_banking.guardrails.run_guardrails import run_auto_guardrails
from agent_banking.tracing.run_tracing import run_auto_tracing
from agent_banking.skills.run_skills import run_auto_skills
from agent_banking.tools.run_tools import run_auto_tools


class NodePipeline:
    """
    Orchestrator for automated node execution from Input to Output with deduplication.
    """

    def __init__(self, base_dir: Path | None = None):
        if base_dir is not None:
            self.base_dir = Path(base_dir)
        else:
            self.base_dir = PROJECT_ROOT / "agent_banking"

    def get_node_folders(self) -> dict:
        """Get input and output directories for all pipeline nodes."""
        nodes = [
            "chunking", "embedding", "vector_db", "query_parser",
            "retrieval", "reranking", "guardrails",
            "tracing", "skills", "tools"
        ]
        res = {}
        for n in nodes:
            in_p = self.base_dir / n / "input"
            out_p = self.base_dir / n / "output"
            in_p.mkdir(parents=True, exist_ok=True)
            out_p.mkdir(parents=True, exist_ok=True)
            res[n] = {"input": in_p, "output": out_p}
        return res

    @staticmethod
    def _resolve_node_input(input_dir: Path, candidates: list[str]) -> Path:
        """Resolve an existing candidate input file inside input_dir, or default primary path."""
        for name in candidates:
            candidate = input_dir / name
            if candidate.exists():
                return candidate
        return input_dir / candidates[0]

    def _run_aux_node(self, runner, input_dir: Path, output_dir: Path, candidates: list[str], force: bool):
        """Invoke an auxiliary node runner, passing input_file if a local fixture exists."""
        aux_in = self._resolve_node_input(input_dir, candidates)
        if aux_in.exists():
            return runner(input_file=aux_in, output_dir=output_dir, force=force)
        return runner(output_dir=output_dir, force=force)

    def run_node(self, node_name: str, force: bool = False, max_files: int | None = None) -> dict:
        """Run an isolated single node from input to output with deduplication."""
        f = self.get_node_folders()
        clean_name = node_name.strip().lower()

        if clean_name in ("1", "chunking"):
            chunks = run_auto_chunking(
                f["chunking"]["input"],
                f["chunking"]["output"],
                force=force,
                max_files=max_files
            )
            return {"node": "chunking", "count": len(chunks)}
        elif clean_name in ("2", "embedding"):
            published = f["chunking"]["output"] / "chunking_manifest.json"
            if published.is_file():
                embed_in = published
            else:
                embed_in = self._resolve_node_input(
                    f["embedding"]["input"],
                    ["chunks.json", "chunking_manifest.json", "all_chunks.json"]
                )
            embeds = run_auto_embedding(
                input_file=embed_in,
                output_dir=f["embedding"]["output"],
                force=force
            )
            return {"node": "embedding", "count": len(embeds)}
        elif clean_name in ("3", "vector_db", "indexing"):
            vdb_in = self._resolve_node_input(f["vector_db"]["input"], ["embedded_chunks.json"])
            options = {}
            publication_dir = f["chunking"]["output"]
            if (publication_dir / "chunking_manifest.json").is_file():
                published_embeddings = f["embedding"]["output"] / "embedded_chunks.json"
                if published_embeddings.is_file():
                    vdb_in = published_embeddings
                if not vdb_in.is_file():
                    raise FileNotFoundError("Embed the current chunking publication before indexing")
                options = {"snapshot": True, "publication_dir": publication_dir}
            vdb = run_auto_indexing(
                input_file=vdb_in,
                output_dir=f["vector_db"]["output"],
                force=force,
                **options,
            )
            return {"node": "vector_db", "manifest": vdb}
        elif clean_name in ("4", "query_parser"):
            qp_in = self._resolve_node_input(
                f["query_parser"]["input"],
                ["queries.json", "latest_query.json", "queries.txt"]
            )
            qp = run_auto_query_parser(
                input_file=qp_in,
                output_dir=f["query_parser"]["output"],
                force=force
            )
            return {"node": "query_parser", "count": len(qp)}
        elif clean_name in ("5", "retrieval"):
            ret_in = self._resolve_node_input(f["retrieval"]["input"], ["parsed_queries.json"])
            ret = run_auto_retrieval(
                input_file=ret_in,
                output_dir=f["retrieval"]["output"],
                force=force
            )
            return {"node": "retrieval", "count": len(ret)}
        elif clean_name in ("6", "reranking"):
            rr_in = self._resolve_node_input(f["reranking"]["input"], ["candidates.json"])
            rerank = run_auto_reranking(
                input_file=rr_in,
                output_dir=f["reranking"]["output"],
                force=force
            )
            return {"node": "reranking", "count": len(rerank)}
        elif clean_name in ("8", "guardrails"):
            guards = self._run_aux_node(
                run_auto_guardrails,
                f["guardrails"]["input"],
                f["guardrails"]["output"],
                ["sample_queries.json", "queries.json"],
                force
            )
            return {"node": "guardrails", "count": len(guards)}
        elif clean_name in ("9", "tracing"):
            traces = self._run_aux_node(
                run_auto_tracing,
                f["tracing"]["input"],
                f["tracing"]["output"],
                ["sample_trace_inputs.json"],
                force
            )
            return {"node": "tracing", "count": len(traces)}
        elif clean_name in ("11", "skills"):
            skills = self._run_aux_node(
                run_auto_skills,
                f["skills"]["input"],
                f["skills"]["output"],
                ["sample_skill_inputs.json"],
                force
            )
            return {"node": "skills", "count": len(skills)}
        elif clean_name in ("12", "tools"):
            tools = self._run_aux_node(
                run_auto_tools,
                f["tools"]["input"],
                f["tools"]["output"],
                ["sample_tool_inputs.json"],
                force
            )
            return {"node": "tools", "count": len(tools)}
        else:
            print(f"[NodePipeline] Unrecognized node: '{node_name}'. Valid options: chunking, embedding, vector_db, query_parser, retrieval, reranking, guardrails, tracing, skills, tools")
            return {"error": "UNKNOWN_NODE"}

    def run_indexing_pipeline(self, force: bool = False, max_files: int | None = None) -> dict:
        """Publish chunks and update the embedding and search indexes."""
        start_all = time.time()
        print("\n" + "#" * 80)
        print("  UPDATING THE BANKING KNOWLEDGE INDEX")
        print("#" * 80)

        f = self.get_node_folders()

        # ---------------------------------------------------------------------
        # 1. Node 1: Chunking
        # ---------------------------------------------------------------------
        chunk_out_dir = f["chunking"]["output"]
        if max_files is not None:
            chunk_out_dir = chunk_out_dir / "diagnostics" / ("accuracy-first-" + str(max_files))

        chunks = run_auto_chunking(input_dir=f["chunking"]["input"], output_dir=f["chunking"]["output"],
                                   force=force, max_files=max_files)
        if max_files is not None:
            chunk_manifest = json.loads((chunk_out_dir / "chunking_manifest.json").read_text(encoding="utf-8"))
            failed_files = [record["file_name"] for record in chunk_manifest["records"] if record["status"] == "FAILED"]
            return {"status": "DIAGNOSTICS", "total_chunks": len(chunks), "failed_files": failed_files,
                    "chunking_output": str(chunk_out_dir)}
        atomic_json(f["embedding"]["input"] / "chunks.json", chunks)

        # ---------------------------------------------------------------------
        # 2. Node 2: Embedding
        # ---------------------------------------------------------------------
        embedded_chunks, chunk_manifest = run_auto_embedding(
            input_file=chunk_out_dir, output_dir=f["embedding"]["output"], force=force, return_manifest=True)
        failed_files = [record["file_name"] for record in chunk_manifest["records"] if record["status"] == "FAILED"]

        # Handover embedded_chunks.json to Node 3: Vector DB Input
        vdb_in = f["vector_db"]["input"] / "embedded_chunks.json"
        atomic_json(vdb_in, embedded_chunks)

        # ---------------------------------------------------------------------
        # 3. Node 3: Vector DB & BM25 Indexing
        # ---------------------------------------------------------------------
        indexing_manifest = run_auto_indexing(
            embedded_chunks=embedded_chunks, output_dir=f["vector_db"]["output"], force=force and not failed_files, snapshot=True,
            publication_dir=chunk_out_dir, publication_manifest=chunk_manifest)
        total_elapsed = round(time.time() - start_all, 2)
        print("\n" + "=" * 80)
        print("  KNOWLEDGE INDEX UPDATE FINISHED")
        print(f"  Execution time: {total_elapsed}s")
        print("=" * 80 + "\n")

        return {
            "status": "PARTIAL_SUCCESS" if failed_files else "SUCCESS",
            "failed_files": failed_files,
            "total_chunks": len(chunks),
            "embedded_chunks": len(embedded_chunks),
            "indexed_points": indexing_manifest.get("total_points_in_collection", 0),
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
                            if node_name == "chunking":
                                self.run_indexing_pipeline(force=False)
                            else:
                                self.run_node(node_name, force=False)
                        except Exception as e:
                            print(f"[NodePipeline] Error while processing Node '{node_name}': {e}")
        except KeyboardInterrupt:
            print("\n[NodePipeline] Watcher daemon stopped.")


def main():
    parser = argparse.ArgumentParser(description="AI Banking Assistant - Auto Folder-to-Folder Pipeline")
    parser.add_argument("--auto", action="store_true", help="Update the knowledge index")
    parser.add_argument("--node", type=str, default=None, help="Run a specific single node (chunking, embedding, vector_db, query_parser, retrieval, reranking, guardrails, tracing, skills, tools)")
    parser.add_argument("--watch", action="store_true", help="Run automatic input folder watcher daemon")
    parser.add_argument("--force", action="store_true", help="Clear state caches and re-run all nodes from scratch")
    parser.add_argument("--max-files", type=int, default=None, help="Limit maximum number of document files during chunking")
    args = parser.parse_args()

    pipeline = NodePipeline()

    if args.watch:
        pipeline.watch_inputs()
    elif args.node:
        pipeline.run_node(args.node, force=args.force, max_files=args.max_files)
    elif args.auto:
        pipeline.run_indexing_pipeline(force=args.force, max_files=args.max_files)
    else:
        # Default behavior: update the knowledge index
        pipeline.run_indexing_pipeline(force=args.force, max_files=args.max_files)


if __name__ == "__main__":
    main()
