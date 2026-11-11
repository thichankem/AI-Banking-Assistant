# -*- coding: utf-8 -*-
"""
Global Configuration Aggregator and Path Configuration.
Module-specific parameters are maintained independently in each module's config folder.
Complies with 0% Regex Law, Zero Ternary Law, and No Typing Library.
"""

import os
from agent_banking.core.environment import ROOT_DIR

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
    """Core path definitions for dataset, storage, and tracing."""
    ROOT_DIR = ROOT_DIR
    RETRIEVAL_DATASET_DIR = retrieval_dir
    CHUNKING_DATASET_DIR = chunking_dir
    DATASET_DIR = active_dataset_dir
    STORAGE_DIR = ROOT_DIR / "agent_banking" / "vector_db"
    TRACE_DIR = ROOT_DIR / "agent_banking" / "tracing" / "traces"


from agent_banking.chunking.config.settings import ChunkingConfig
from agent_banking.embedding.config.settings import EmbeddingConfig
from agent_banking.vector_db.config.settings import VectorDBConfig
from agent_banking.retrieval.config.settings import RetrievalConfig
from agent_banking.reranking.config.settings import RerankingConfig
from agent_banking.generator.config.settings import ContextConfig, GeneratorConfig
from agent_banking.tracing.config.settings import TracingConfig
from agent_banking.guardrails.config.settings import GuardrailsConfig
from agent_banking.decision.config.settings import DecisionConfig
from agent_banking.langgraph_agent.config.settings import AgentConfig
from agent_banking.skills.config.settings import SkillsConfig
from agent_banking.tasks.config.settings import TasksConfig


class Config(
    PathConfig,
    ChunkingConfig,
    EmbeddingConfig,
    VectorDBConfig,
    RetrievalConfig,
    RerankingConfig,
    ContextConfig,
    GeneratorConfig,
    TracingConfig,
    GuardrailsConfig,
    DecisionConfig,
    AgentConfig,
    SkillsConfig,
    TasksConfig
):
    """Unified Config class aggregating all modular configurations."""
    pass
