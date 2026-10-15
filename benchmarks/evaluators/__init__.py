"""
4-tier Evaluation Suite: OCR, Structure, Retrieval, and System Profiler.
"""
from .ocr_evaluator import OCREvaluator, compute_cer, compute_wer
from .structure_evaluator import StructureEvaluator
from .retrieval_evaluator import RetrievalEvaluator, BM25Retriever
from .system_profiler import SystemProfiler, profile_task

__all__ = [
    "OCREvaluator",
    "compute_cer",
    "compute_wer",
    "StructureEvaluator",
    "RetrievalEvaluator",
    "BM25Retriever",
    "SystemProfiler",
    "profile_task"
]
