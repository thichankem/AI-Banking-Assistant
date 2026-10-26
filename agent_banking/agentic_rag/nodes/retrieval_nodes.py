# -*- coding: utf-8 -*-
"""
Retrieval & Reranking Nodes for Banking Agentic RAG
======================================================================
Cung cấp các Node truy xuất tài liệu và tìm kiếm thông tin:
4. retrieve_vector_node: Truy xuất văn bản từ hệ thống Hybrid Retrieval sẵn có (tích hợp sub_questions và khử trùng lặp).
5. rerank_documents_node: Tái xếp hạng tài liệu bằng Cross-Encoder (bge-reranker-v2-m3) của dự án.
6. web_search_node: Tìm kiếm web bổ sung (Tavily / DuckDuckGo / Fallback) khi thiếu thông tin.

NGUYÊN TẮC:
1. 0% REGEX LAW: Tuyệt đối không dùng regex methods.
2. ZERO TERNARY: Ưu tiên if/else tường minh, dễ đọc.
3. REUSE EXISTING: Tái sử dụng trọn vẹn BankingRAGTool và BankingReranker.
"""

from __future__ import annotations
import os
import logging
from typing import Dict, Any, List, Optional

from langchain_core.documents import Document

from agent_banking.config import Config
from agent_banking.agentic_rag.state import GraphState

logger = logging.getLogger(__name__)

# Global singletons cho RAG Tool và Reranker để tránh khởi tạo lại nhiều lần
_GLOBAL_RAG_TOOL = None
_GLOBAL_RERANKER = None


def get_rag_tool(pipeline_instance=None):
    """
    Lấy hoặc khởi tạo singleton BankingRAGTool từ codebase hiện có.
    """
    global _GLOBAL_RAG_TOOL
    if pipeline_instance is not None:
        from agent_banking.tools.rag_tool import BankingRAGTool
        return BankingRAGTool(pipeline_instance=pipeline_instance)

    if _GLOBAL_RAG_TOOL is None:
        try:
            from agent_banking.tools.rag_tool import BankingRAGTool
            _GLOBAL_RAG_TOOL = BankingRAGTool()
        except Exception as e:
            logger.warning(f"Chưa thể kết nối trực tiếp BankingRAGTool: {e}")
            _GLOBAL_RAG_TOOL = None
    return _GLOBAL_RAG_TOOL


def get_reranker():
    """
    Lấy hoặc khởi tạo singleton BankingReranker từ codebase hiện có.
    """
    global _GLOBAL_RERANKER
    if _GLOBAL_RERANKER is None:
        try:
            from agent_banking.reranking.rag_reranker import BankingReranker
            _GLOBAL_RERANKER = BankingReranker()
        except Exception as e:
            logger.warning(f"Chưa thể khởi tạo BankingReranker: {e}")
            _GLOBAL_RERANKER = None
    return _GLOBAL_RERANKER


def retrieve_vector_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 4: Retrieve Vector Node.
    Kết nối với hệ thống Hybrid Retrieval (Dense Vector 1024-dim + BM25 Okapi với RRF).
    Nếu có danh sách sub_questions, thực hiện truy xuất cho từng câu hỏi phụ
    và hợp nhất kết quả, đồng thời khử trùng lặp (deduplication) các chunks.
    """
    logger.info("---NODE: RETRIEVE VECTOR---")

    sub_questions = state.get("sub_questions", [])
    if not sub_questions or len(sub_questions) == 0:
        raw_q = state.get("question")
        if raw_q is None:
            raw_q = state.get("original_question", "")
        queries_to_search = [str(raw_q).strip()]
    else:
        queries_to_search = sub_questions

    rag_tool = get_rag_tool()
    cfg = Config()
    top_k = cfg.HYBRID_TOP_K

    all_retrieved_docs: List[Document] = []
    seen_identifiers = set()

    for query_item in queries_to_search:
        if not query_item:
            continue
        logger.info(f"Đang truy xuất cho truy vấn: '{query_item}'")

        if rag_tool is not None:
            try:
                res = rag_tool.execute(query=query_item, top_k=top_k)
                raw_docs = res.get("documents", [])

                for idx, d in enumerate(raw_docs):
                    if isinstance(d, dict):
                        content = d.get("content", "")
                        source = d.get("source", "Tài liệu Vietcombank")
                        breadcrumb = d.get("breadcrumb", "")
                        chunk_id = d.get("chunk_id", f"{source}_{idx}")
                        score = d.get("score", 0.0)
                    else:
                        content = str(d)
                        source = "Tài liệu Vietcombank"
                        breadcrumb = ""
                        chunk_id = f"chunk_{idx}"
                        score = 0.0

                    # Khử trùng lặp dựa trên chunk_id và nội dung
                    dedup_key = f"{chunk_id}_{content[:100]}"
                    if dedup_key in seen_identifiers:
                        continue
                    seen_identifiers.add(dedup_key)

                    metadata = {
                        "chunk_id": chunk_id,
                        "source": source,
                        "breadcrumb": breadcrumb,
                        "score": score,
                        "datasource": "vectorstore",
                        "query_matched": query_item
                    }
                    doc_obj = Document(page_content=content, metadata=metadata)
                    all_retrieved_docs.append(doc_obj)
            except Exception as e:
                logger.error(f"Lỗi khi thực thi BankingRAGTool cho '{query_item}': {e}")
        else:
            logger.warning("BankingRAGTool chưa khả dụng, trả về danh sách rỗng.")

    logger.info(f"Tổng số tài liệu độc nhất thu được từ VectorStore: {len(all_retrieved_docs)}")

    return {
        "documents": all_retrieved_docs
    }


def rerank_documents_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 5: Rerank Documents Node.
    Tái sắp xếp lại danh sách documents bằng mô hình Cross-Encoder (bge-reranker-v2-m3).
    Lọc ra Top-K tài liệu sát với câu hỏi nhất trước khi chuyển sang bước chấm điểm/sinh lời giải.
    """
    logger.info("---NODE: RERANK DOCUMENTS---")

    docs = state.get("documents", [])
    if not docs or len(docs) == 0:
        logger.info("Không có tài liệu nào để tái sắp xếp.")
        return {"documents": []}

    question = state.get("question")
    if question is None:
        question = state.get("original_question", "")
    target_question = str(question).strip()

    cfg = Config()
    rerank_top_k = cfg.RERANK_TOP_K

    reranker = get_reranker()
    if reranker is None:
        logger.warning("Reranker chưa sẵn sàng, giữ nguyên thứ tự tài liệu ban đầu.")
        return {"documents": docs[:rerank_top_k]}

    try:
        # Chuẩn bị danh sách ứng viên cho BankingReranker
        candidates: List[Dict[str, Any]] = []
        for d in docs:
            candidates.append({
                "chunk": {
                    "content": d.page_content,
                    "source": d.metadata.get("source", ""),
                    "breadcrumb": d.metadata.get("breadcrumb", "")
                },
                "doc_object": d
            })

        reranked_items = reranker.rerank(
            query=target_question,
            candidates=candidates,
            top_k=rerank_top_k
        )

        reranked_docs: List[Document] = []
        for item in reranked_items:
            doc_obj = item.get("doc_object")
            if doc_obj is not None:
                doc_obj.metadata["rerank_score"] = item.get("rerank_score", 0.0)
                reranked_docs.append(doc_obj)
            else:
                chunk = item.get("chunk", {})
                content = chunk.get("content", "")
                source = chunk.get("source", "")
                metadata = {
                    "source": source,
                    "breadcrumb": chunk.get("breadcrumb", ""),
                    "rerank_score": item.get("rerank_score", 0.0),
                    "datasource": "vectorstore"
                }
                reranked_docs.append(Document(page_content=content, metadata=metadata))

        logger.info(f"Đã tái sắp xếp và giữ lại Top {len(reranked_docs)} tài liệu qua Cross-Encoder.")
        return {"documents": reranked_docs}
    except Exception as e:
        logger.error(f"Lỗi trong quá trình Rerank: {e}. Giữ nguyên Top-{rerank_top_k} ban đầu.")
        return {"documents": docs[:rerank_top_k]}


def web_search_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 6: Web Search Node.
    Tìm kiếm thông tin bổ sung bên ngoài khi kho tài liệu nội bộ không đủ hoặc khi được yêu cầu.
    Hỗ trợ Tavily API (nếu có API Key), DuckDuckGo Search và Mock Fallback an toàn.
    Kết quả tìm kiếm được bọc trong Document và nối tiếp vào danh sách documents trong state.
    """
    logger.info("---NODE: WEB SEARCH---")

    question = state.get("question")
    if question is None:
        question = state.get("original_question", "")
    target_query = str(question).strip()

    existing_docs = list(state.get("documents", []))
    web_docs: List[Document] = []

    # 1. Thử nghiệm tìm kiếm qua Tavily nếu có API Key
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if tavily_key:
        try:
            from tavily import TavilyClient
            t_client = TavilyClient(api_key=tavily_key)
            response = t_client.search(query=target_query, max_results=3)
            results = response.get("results", [])
            for res in results:
                title = res.get("title", "")
                url = res.get("url", "")
                content = res.get("content", "")
                doc = Document(
                    page_content=f"[{title}]\n{content}",
                    metadata={
                        "source": url,
                        "title": title,
                        "datasource": "web_search",
                        "engine": "tavily"
                    }
                )
                web_docs.append(doc)
            logger.info(f"Tavily tìm thấy {len(web_docs)} kết quả.")
        except Exception as e:
            logger.warning(f"Lỗi gọi Tavily API: {e}. Chuyển sang DuckDuckGo.")

    # 2. Nếu Tavily không tìm được hoặc không có key, dùng DuckDuckGo
    if len(web_docs) == 0:
        try:
            # Thử thư viện ddgs trước, sau đó thử duckduckgo_search
            ddg_class = None
            try:
                from ddgs import DDGS
                ddg_class = DDGS
            except ImportError:
                try:
                    from duckduckgo_search import DDGS
                    ddg_class = DDGS
                except ImportError:
                    pass

            if ddg_class is not None:
                with ddg_class() as ddgs:
                    ddg_results = list(ddgs.text(target_query, max_results=3))
                    for res in ddg_results:
                        title = res.get("title", "")
                        body = res.get("body", "")
                        href = res.get("href", "")
                        doc = Document(
                            page_content=f"[{title}]\n{body}",
                            metadata={
                                "source": href,
                                "title": title,
                                "datasource": "web_search",
                                "engine": "duckduckgo"
                            }
                        )
                        web_docs.append(doc)
                logger.info(f"DuckDuckGo tìm thấy {len(web_docs)} kết quả.")
        except Exception as e:
            logger.warning(f"Lỗi tìm kiếm DuckDuckGo: {e}")

    # 3. Fallback khi môi trường mạng ngoại tuyến
    if len(web_docs) == 0:
        fallback_content = (
            f"Thông tin tra cứu mở rộng cho chủ đề: '{target_query}'. "
            f"Hiện tại dịch vụ tìm kiếm trực tuyến chưa thể kết nối tới máy chủ bên ngoài. "
            f"Quý khách vui lòng tham khảo các kênh thông tin chính thức của Ngân hàng Vietcombank."
        )
        fallback_doc = Document(
            page_content=fallback_content,
            metadata={
                "source": "https://www.vietcombank.com.vn",
                "title": "Cổng thông tin điện tử Ngân hàng TMCP Ngoại thương Việt Nam",
                "datasource": "web_search_fallback",
                "engine": "local_fallback"
            }
        )
        web_docs.append(fallback_doc)

    combined_docs = existing_docs + web_docs
    logger.info(f"Tổng tài liệu sau khi bổ sung từ Web Search: {len(combined_docs)}")

    return {
        "documents": combined_docs,
        "web_search_needed": False
    }
