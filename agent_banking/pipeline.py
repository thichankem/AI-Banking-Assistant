"""Banking knowledge ingestion, indexing, and retrieval services for ReAct."""

import sys
import time
import json
from pathlib import Path

# Ensure UTF-8 on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from agent_banking.core.config.settings import CoreConfig, get_core_messages
from agent_banking.core.environment import ROOT_DIR
from agent_banking.core.settings import Config
from .chunking.run_chunking import load_published_chunks
from .chunking.universal.extraction.sources import discover_documents, document_source_root, source_key
from .chunking.universal.pipeline import DLAChunker
from .chunking.universal.planning.schema import link_chunk_sequence, require_index_ready
from .embedding.embedder import BankingEmbedder
from .vector_db.qdrant_store import (
    QdrantBankingStore, QdrantVectorStore, mark_publication_chunks, publication_scope,
)
from .core.llm_rag_cache import rag_llm_available
from .tools.extensions.banking_query import ParsedQuery, parse_banking_query
from .tools.extensions.llm_query_planner import LLMQueryPlanner
from .retrieval.hybrid_retriever import HybridRetriever
from .reranking.rag_reranker import BankingReranker
from .tracing.tracer import BankingTracer

_RAG_CFG = CoreConfig.CORE_RAG_PIPELINE
_PATHS_CFG = CoreConfig.CORE_PATHS
_MESSAGES = get_core_messages("rag_pipeline", "pipeline_messages.yaml", "RAG_PIPELINE_MESSAGES")

_DEFAULT_CHUNK_SIZE_TOKENS = int(_RAG_CFG["default_chunk_size_tokens"])
_DEFAULT_CHUNK_MAX_TOKENS = int(_RAG_CFG["default_chunk_max_tokens"])
_DEFAULT_CHUNK_MIN_TOKENS = int(_RAG_CFG["default_chunk_min_tokens"])
_DEFAULT_CHUNK_SEARCH_LANGUAGE = str(_RAG_CFG["default_chunk_search_language"])
_EMBEDDING_CACHE_FILENAME = str(_RAG_CFG["embedding_cache_filename"])
_DEFAULT_QDRANT_SUBDIR = str(_RAG_CFG["default_qdrant_subdir"])
_DEFAULT_BATCH_SIZE = int(_RAG_CFG["ingest_batch_size"])
_CHUNK_FILE_PATTERN = str(_RAG_CFG["chunk_file_pattern"])
_MANIFEST_FILENAME = str(_RAG_CFG["chunking_manifest_filename"])
_DEFAULT_EXTENSIONS = list(_RAG_CFG["default_extensions"])
_EXCLUDED_ROLE = str(_RAG_CFG["excluded_retrieval_role"])


def _default_chunking_output_dir():
    target = ROOT_DIR
    for part in str(_PATHS_CFG["chunking_output_relpath"]).replace("\\", "/").split("/"):
        if part:
            target = target / part
    return target


class BankingRAGPipeline:
    """Own the banking knowledge index and its search components."""

    def __init__(self, config=None, use_memory=False):
        if config is not None:
            self.config = config
        else:
            self.config = Config()
        self.use_memory = use_memory

        print("=" * 70)
        print(_MESSAGES["init_banner"])
        print("=" * 70)

        # 1. Initialize DLA Chunker lazily (avoids ~1.65s cold startup when only querying)
        print("[1/4] Layout-Aware Document Chunker registered (lazy-init on ingest)...")
        self._dla_chunker = None

        # 2. Initialize Dense Embedding Model (integrated SQLite Cache)
        print(f"[2/4] Initializing Dense Embedding Model ({self.config.EMBEDDING_MODEL_1024})...")
        self.embedder = BankingEmbedder(
            dimension=self.config.EMBEDDING_DIM,
            model_name=self.config.EMBEDDING_MODEL_1024,
            embedding_url=self.config.EMBEDDING_URL,
            cache_db_path=str(self.config.STORAGE_DIR / _EMBEDDING_CACHE_FILENAME)
        )
        self.embedding_model = self.embedder

        # 3. Initialize Dense Vector Store & BM25 Hybrid Store
        if use_memory:
            store_mode = "In-Memory"
        else:
            store_mode = "Disk"
        print(f"[3/4] Initializing Vector & BM25 Store ({store_mode})...")
        selected_store = getattr(self.config, "ABANK_QDRANT_STORAGE_PATH", "")
        raw_qdrant_path = Path(selected_store or getattr(self.config, "QDRANT_STORAGE_PATH",
                                                        self.config.STORAGE_DIR / _DEFAULT_QDRANT_SUBDIR))
        if raw_qdrant_path.is_absolute():
            qdrant_storage_path = raw_qdrant_path
        else:
            qdrant_storage_path = self.config.ROOT_DIR / raw_qdrant_path
        self.vector_store = QdrantBankingStore(
            collection_name=self.config.QDRANT_COLLECTION,
            dimension=self.config.EMBEDDING_DIM,
            storage_path=str(qdrant_storage_path),
            use_memory=use_memory
        )

        # 4. Initialize Query Parser, Hybrid Retriever (RRF) & Reranker
        print(f"[4/4] Initializing Hybrid Retriever (RRF k={self.config.RRF_K})...")
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedding_model=self.embedder,
            rrf_k=self.config.RRF_K,
            dense_weight=self.config.DENSE_WEIGHT
        )

        print(f"[4/4] Initializing Reranker ({self.config.RERANKER_MODEL})...")
        self.reranker = BankingReranker(
            model_name=self.config.RERANKER_MODEL
        )

        use_live_planner = (not use_memory) and rag_llm_available("chunking.text")
        self.llm_query_planner = LLMQueryPlanner(use_llm=use_live_planner)

        self.tracer = BankingTracer(config=self.config)

        self.indexed_chunks = []
        print(_MESSAGES["ready_message"])

    def ingest_chunks(self, chunks, batch_size=None, *, return_index_keys=False):
        """
        Directly ingest a list of chunks into the system:
        1. Generate dense vector embeddings (using SQLite cache).
        2. Ingest concurrently into HNSW Vector Store and Okapi BM25 index.
        """
        if batch_size is None:
            batch_size = _DEFAULT_BATCH_SIZE
        chunks = list(chunks)
        require_index_ready(chunks)
        chunks = [chunk for chunk in chunks
                  if chunk.get("business_metadata", {}).get("retrieval", {}).get("role") != _EXCLUDED_ROLE]
        link_chunk_sequence(chunks)
        if not chunks:
            if return_index_keys:
                return 0, set()
            return 0

        t0 = time.time()
        print(f" -> [Ingest] Generating embeddings for {len(chunks)} chunks...")
        chunks_with_vec = self.embedder.embed_chunks(chunks, batch_size=batch_size)
        t_embed = (time.time() - t0) * 1000

        self.tracer.trace_embedding(
            num_texts=len(chunks),
            dimension=self.config.EMBEDDING_DIM,
            duration_ms=round(t_embed, 1),
            model_name=self.config.EMBEDDING_MODEL_1024,
            cache_hits=getattr(self.embedder, "last_cache_hits", 0)
        )

        t1 = time.time()
        print(" -> [Ingest] Ingesting into Vector & BM25 Store...")
        index_keys = self.vector_store.add_chunks(chunks_with_vec)
        t_store = (time.time() - t1) * 1000

        print(f"    + Embedding finished ({t_embed:.1f}ms) and storage finished ({t_store:.1f}ms).")
        self.indexed_chunks.extend(chunks)
        if return_index_keys:
            return len(chunks), index_keys
        return len(chunks)

    def ingest_published_chunks(self, output_dir=None, require_ready=True, batch_size=None):
        """
        Synchronize a verified chunking publication with Vector DB and BM25.
        Retain the indexed versions of sources that failed this publication.
        """
        if batch_size is None:
            batch_size = _DEFAULT_BATCH_SIZE
        if output_dir is not None:
            target_out = Path(output_dir)
        else:
            default_out = _default_chunking_output_dir()
            target_out = Path(getattr(getattr(self, "config", None), "CHUNKING_OUTPUT_DIR", default_out))

        chunks, manifest = load_published_chunks(target_out, require_ready=require_ready, return_manifest=True)
        require_index_ready(chunks)
        print(f"[*] Loaded {len(chunks)} verified published chunks from {target_out}")
        scope = publication_scope(target_out)
        searchable = mark_publication_chunks([
            chunk for chunk in chunks
            if chunk.get("business_metadata", {}).get("retrieval", {}).get("role") != _EXCLUDED_ROLE
        ], scope)
        link_chunk_sequence(searchable)
        count, index_keys = self.ingest_chunks(searchable, batch_size=batch_size, return_index_keys=True)
        sources = {record["rel_path"] for record in manifest["records"] if record["status"] != "FAILED"}
        failed_sources = {record["rel_path"] for record in manifest["records"] if record["status"] == "FAILED"}
        active = {chunk["chunk_id"] for chunk in searchable}
        self.vector_store.prune_chunks(active, sources, scope=scope, preserve_sources=failed_sources,
                                       active_index_keys=index_keys)
        return count

    def ingest_chunk_files(self, chunks_dir, file_pattern=None):
        """
        Ingest chunk JSON files in the specified directory into Vector DB.
        If chunks_dir contains a published v2.0 chunking_manifest.json and default
        pattern is used, delegates to load_published_chunks for integrity verification.
        """
        if file_pattern is None:
            file_pattern = _CHUNK_FILE_PATTERN
        c_dir = Path(chunks_dir)
        if not c_dir.exists():
            raise FileNotFoundError(_MESSAGES["chunking_dir_missing"].format(path=c_dir))

        if file_pattern == _CHUNK_FILE_PATTERN and (c_dir / _MANIFEST_FILENAME).exists():
            return self.ingest_published_chunks(c_dir)

        json_files = sorted(list(c_dir.glob(file_pattern)))
        print(f"[*] Found {len(json_files)} JSON files in {c_dir}")

        total_ingested = 0
        for idx, jf in enumerate(json_files, start=1):
            with open(jf, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                file_chunks = loaded
            elif isinstance(loaded, dict) and isinstance(loaded.get("chunks"), list):
                file_chunks = loaded["chunks"]
            else:
                continue
            print(f"  [{idx}/{len(json_files)}] {jf.name}: {len(file_chunks)} chunks")
            self.ingest_chunks(file_chunks)
            total_ingested += len(file_chunks)

        print(f"Successfully ingested a total of {total_ingested} chunks into Vector DB.")
        return total_ingested

    def ingest_documents(
        self,
        dataset_dir=None,
        max_files=None,
        extensions=None,
        use_dla=True,
        filenames=None
    ):
        """
        Ingestion & Dual-Indexing phase from source documents:
        1. Layout-aware chunking via Breadcrumbs (DLA Chunker).
        2. Parallel dual indexing into Dense Vector Store and Okapi BM25.
        """
        if dataset_dir:
            target_dir = Path(dataset_dir).resolve()
        else:
            target_dir = Path(self.config.DATASET_DIR).resolve()

        configured_input = getattr(getattr(self, "config", None), "CHUNKING_INPUT_DIR", None)
        source_root = document_source_root(target_dir, configured_input)

        print(f"[*] Starting ingestion phase from: {target_dir}")

        if not target_dir.exists():
            raise FileNotFoundError(_MESSAGES["dataset_dir_missing"].format(path=target_dir))

        if extensions:
            target_exts = extensions
        else:
            target_exts = list(_DEFAULT_EXTENSIONS)
            if configured_input and target_dir.is_relative_to(Path(configured_input).resolve()):
                target_exts.append(".xlsx")

        # Layout-Aware DLA Chunking
        t0 = time.time()
        if filenames:
            if isinstance(filenames, (str, Path)):
                filenames = [filenames]
            if not extensions and ".xlsx" not in target_exts and any(Path(name).suffix.lower() == ".xlsx" for name in filenames):
                target_exts.append(".xlsx")
            print(f" -> [Ingestion] Chunking {len(filenames)} specified files...")
            available = {}
            for path in discover_documents(target_dir, target_exts):
                for alias in (path.name, str(path), path.relative_to(target_dir).as_posix(),
                              path.relative_to(source_root).as_posix()):
                    available.setdefault(source_key(alias), set()).add(path)
            selected = {}
            for fname in filenames:
                matches = available.get(source_key(fname), set())
                if not matches:
                    raise FileNotFoundError(str(fname))
                if len(matches) != 1:
                    raise ValueError(_MESSAGES["ambiguous_doc_path"].format(path=fname))
                path = next(iter(matches))
                selected[path] = path.relative_to(source_root).as_posix()
            chunks = []
            for path, source in selected.items():
                chunks.extend(self.dla_chunker.chunk_file(path, source_name=source))
        else:
            print(" -> [Ingestion] Chunking via Layout-Aware Chunker...")
            chunks = self.dla_chunker.chunk_directory(
                target_dir,
                extensions=target_exts,
                max_files=max_files,
                source_root=source_root,
            )

        print(f"    + Extracted: {len(chunks)} chunks ({round((time.time() - t0)*1000, 1)}ms)")
        chunk_duration_ms = round((time.time() - t0) * 1000, 1)
        self.tracer.trace_chunking(
            file_path=str(target_dir),
            num_chunks=len(chunks),
            duration_ms=chunk_duration_ms,
            details={"extensions": target_exts, "max_files": max_files}
        )

        # Ingest into Vector DB and BM25
        return self.ingest_chunks(chunks)

    def parse_query(self, query):
        """
        Normalize domain acronyms (eKYC, TKTT, Smart OTP, Digibank) and extract keywords.
        """
        return parse_banking_query(query)

    def plan_query(self, query):
        """
        Analyze and expand banking query using configured cloud LLM with SQLite WAL cache.
        """
        planner = getattr(self, "llm_query_planner", None)
        if planner is None:
            return {"rewritten_query": str(query or ""), "sub_queries": [], "dense_weight": 0.5, "method": "none"}
        return planner.plan(query)

    def retrieve_candidates(
        self,
        parsed_query,
        top_k=None,
        filter_dict=None
    ):
        """
        Query concurrently across Dense Vector and BM25, then fuse via RRF k=60 to get top candidates.
        """
        if top_k is not None:
            k = top_k
        else:
            k = self.config.HYBRID_TOP_K

        return self.retriever.retrieve(
            query=parsed_query,
            top_k=k,
            candidate_k=self.config.HYBRID_CANDIDATE_K,
            filter_dict=filter_dict
        )

    def rerank_candidates(
        self,
        query_or_parsed,
        candidates,
        top_k=None
    ):
        """
        Use Cross-Encoder reranker to refine candidates to top most accurate chunks.
        """
        if top_k is not None:
            k = top_k
        else:
            k = self.config.RERANK_TOP_K

        if isinstance(query_or_parsed, ParsedQuery):
            raw_query = query_or_parsed.raw_query
        else:
            raw_query = str(query_or_parsed)

        top_reranked = self.reranker.rerank(
            query=raw_query,
            candidates=candidates,
            top_k=k
        )
        return top_reranked

    @property
    def dla_chunker(self):
        if self._dla_chunker is None:
            self._dla_chunker = DLAChunker(
                target_chunk_tokens=int(getattr(self.config, "CHUNK_SIZE_TOKENS", _DEFAULT_CHUNK_SIZE_TOKENS)),
                max_chunk_tokens=int(getattr(self.config, "CHUNK_MAX_TOKENS", _DEFAULT_CHUNK_MAX_TOKENS)),
                min_chunk_tokens=int(getattr(self.config, "CHUNK_MIN_TOKENS", _DEFAULT_CHUNK_MIN_TOKENS)),
                search_language=getattr(self.config, "CHUNK_SEARCH_LANGUAGE", _DEFAULT_CHUNK_SEARCH_LANGUAGE),
                config=self.config
            )
        return self._dla_chunker

    @dla_chunker.setter
    def dla_chunker(self, value):
        self._dla_chunker = value

    @property
    def chunker(self):
        return self.dla_chunker

    @chunker.setter
    def chunker(self, value):
        self._dla_chunker = value

    def close(self):
        """Release connection resources."""
        if getattr(self, "_dla_chunker", None) is not None and hasattr(self._dla_chunker, "close"):
            self._dla_chunker.close()
        if hasattr(self.vector_store, "close"):
            self.vector_store.close()
