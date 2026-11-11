import os
from agent_banking.config.environment import ROOT_DIR

retrieval_dir = ROOT_DIR / "dataset" / "retrieval_study" / "retrieval_dataset"
chunking_dir_candidate = ROOT_DIR / "agent_banking" / "chunking" / "chunking_dataset"

if chunking_dir_candidate.exists():
    chunking_dir = chunking_dir_candidate
else:
    chunking_dir = ROOT_DIR / "dataset" / "chunking_dataset"

if retrieval_dir.exists():
    active_dataset_dir = retrieval_dir
else:
    active_dataset_dir = chunking_dir


class PathConfig:
    # 1. Data paths
    ROOT_DIR = ROOT_DIR
    RETRIEVAL_DATASET_DIR = retrieval_dir
    CHUNKING_DATASET_DIR = chunking_dir
    DATASET_DIR = active_dataset_dir
    STORAGE_DIR = ROOT_DIR / "agent_banking" / "vector_db"
    TRACE_DIR = ROOT_DIR / "agent_banking" / "tracing" / "traces"
