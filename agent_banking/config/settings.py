"""Aggregate configuration; module settings live in their own folders."""
from agent_banking.config.common.settings import PathConfig
from agent_banking.config.chunking.settings import ChunkingConfig
from agent_banking.config.embedding.settings import EmbeddingConfig
from agent_banking.config.vector_db.settings import VectorDBConfig
from agent_banking.config.retrieval.settings import RetrievalConfig
from agent_banking.config.reranking.settings import RerankingConfig
from agent_banking.config.generator.settings import ContextConfig
from agent_banking.config.generator.settings import GeneratorConfig
from agent_banking.config.tracing.settings import TracingConfig
from agent_banking.config.guardrails.settings import GuardrailsConfig
from agent_banking.config.decision.settings import DecisionConfig
from agent_banking.config.langgraph_agent.settings import AgentConfig
from agent_banking.config.skills.settings import SkillsConfig
from agent_banking.config.tasks.settings import TasksConfig


class Config(PathConfig, ChunkingConfig, EmbeddingConfig, VectorDBConfig, RetrievalConfig, RerankingConfig, ContextConfig, GeneratorConfig, TracingConfig, GuardrailsConfig, DecisionConfig, AgentConfig, SkillsConfig, TasksConfig):
    pass
