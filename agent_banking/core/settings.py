# -*- coding: utf-8 -*-
"""
Global Configuration Aggregator and Path Configuration.
Module-specific parameters are maintained independently in each module's config folder.
Complies with 0% Regex Law, Zero Ternary Law, and No Typing Library.
"""

import os
from agent_banking.core.environment import ROOT_DIR

retrieval_dir = ROOT_DIR / "dataset" / "A_Bank_golden_400"
dataset_abank_dir = ROOT_DIR / "dataset" / "A_Bank"
chunking_input_dir = ROOT_DIR / "agent_banking" / "chunking" / "input"
chunking_output_dir = ROOT_DIR / "agent_banking" / "chunking" / "output"
chunking_abank_dir = chunking_input_dir / "A_Bank"

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
    STORAGE_DIR = ROOT_DIR / "agent_banking" / "vector_db"
    TRACE_DIR = ROOT_DIR / "agent_banking" / "tracing" / "traces"


from agent_banking.chunking.config.settings import ChunkingConfig
from agent_banking.embedding.config.settings import EmbeddingConfig
from agent_banking.vector_db.config.settings import VectorDBConfig
from agent_banking.query_parser.config.settings import QueryParserConfig
from agent_banking.retrieval.config.settings import RetrievalConfig
from agent_banking.reranking.config.settings import RerankingConfig
from agent_banking.generator.config.settings import ContextConfig, GeneratorConfig
from agent_banking.tracing.config.settings import TracingConfig
from agent_banking.guardrails.config.settings import GuardrailsConfig


class DecisionConfig:
    """Fallback defaults for optional decision settings (chunking uses core.models)."""
    ENABLE_DECISION = False
    DECISION_MODEL = os.getenv("DECISION_MODEL", "")
    DECISION_BASE_URL = os.getenv("DECISION_BASE_URL", os.getenv("BASE_URL", ""))
    DECISION_API_KEY = os.getenv("DECISION_API_KEY", "")
    DECISION_TIMEOUT = 8.0
    DECISION_CONFIDENCE_THRESHOLD = 0.70


from agent_banking.langgraph_agent.config.settings import AgentConfig
from agent_banking.skills.config.settings import SkillsConfig
from agent_banking.tasks.config.settings import TasksConfig
from agent_banking.tools.config.settings import ToolsConfig


class Config(
    PathConfig,
    ChunkingConfig,
    EmbeddingConfig,
    VectorDBConfig,
    QueryParserConfig,
    RetrievalConfig,
    RerankingConfig,
    ContextConfig,
    GeneratorConfig,
    TracingConfig,
    GuardrailsConfig,
    DecisionConfig,
    AgentConfig,
    SkillsConfig,
    TasksConfig,
    ToolsConfig
):
    """Unified Config class aggregating all modular configurations."""
    pass
