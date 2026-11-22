# -*- coding: utf-8 -*-
"""
Global Configuration Aggregator and Path Configuration.
Module-specific parameters are maintained independently in each module's config folder.
Complies with 0% Regex Law, Zero Ternary Law.
"""

from agent_banking.core.config.settings import CoreConfig
from agent_banking.core.environment import ROOT_DIR


def _resolve_relpath(relpath):
    target = ROOT_DIR
    for part in str(relpath).replace("\\", "/").split("/"):
        if part:
            target = target / part
    return target


_paths = CoreConfig.CORE_PATHS

retrieval_dir = _resolve_relpath(_paths["retrieval_dataset_relpath"])
dataset_abank_dir = _resolve_relpath(_paths["dataset_abank_relpath"])
chunking_input_dir = _resolve_relpath(_paths["chunking_input_relpath"])
chunking_output_dir = _resolve_relpath(_paths["chunking_output_relpath"])
chunking_abank_dir = chunking_input_dir / _paths["chunking_abank_subdir"]

if not retrieval_dir.exists():
    retrieval_dir = dataset_abank_dir

if chunking_abank_dir.exists():
    chunking_dir = chunking_abank_dir
elif chunking_input_dir.exists():
    chunking_dir = chunking_input_dir
else:
    chunking_dir = dataset_abank_dir

if chunking_dir.exists():
    active_dataset_dir = chunking_dir
else:
    active_dataset_dir = dataset_abank_dir


class PathConfig:
    """Core path definitions for dataset, storage, and tracing."""
    ROOT_DIR = ROOT_DIR
    RETRIEVAL_DATASET_DIR = retrieval_dir
    CHUNKING_INPUT_DIR = chunking_input_dir
    CHUNKING_OUTPUT_DIR = chunking_output_dir
    CHUNKING_DATASET_DIR = chunking_dir
    DATASET_DIR = active_dataset_dir
    STORAGE_DIR = _resolve_relpath(_paths["storage_relpath"])
    TRACE_DIR = _resolve_relpath(_paths["trace_relpath"])


from agent_banking.chunking.config.settings import ChunkingConfig
from agent_banking.embedding.config.settings import EmbeddingConfig
from agent_banking.vector_db.config.settings import VectorDBConfig
from agent_banking.retrieval.config.settings import RetrievalConfig
from agent_banking.reranking.config.settings import RerankingConfig
from agent_banking.tracing.config.settings import TracingConfig
from agent_banking.guardrails.config.settings import GuardrailsConfig

from agent_banking.agent.config.settings import AgentConfig
from agent_banking.skills.config.settings import SkillsConfig
from agent_banking.tools.config.settings import ToolsConfig


class Config(
    PathConfig,
    CoreConfig,
    ChunkingConfig,
    EmbeddingConfig,
    VectorDBConfig,
    RetrievalConfig,
    RerankingConfig,
    TracingConfig,
    GuardrailsConfig,
    AgentConfig,
    SkillsConfig,
    ToolsConfig
):
    """Unified Config class aggregating all modular configurations."""
    pass
