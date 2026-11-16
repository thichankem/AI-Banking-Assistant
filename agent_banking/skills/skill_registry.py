"""Discover declarative banking skills and their supported execution modes."""

from pathlib import Path

from agent_banking.core.plugin_specs import check_parameters, load_specs, public_name
from agent_banking.core.settings import Config
from agent_banking.skills.base_skill import BaseSkill
from agent_banking.tools.tool_registry import BankingToolRegistry, ONCE_PER_TURN


SUPPORTED_MODES = {"rag_pipeline", "fact_checker_engine", "incident_dispatcher", "financial_dispatcher", "tool_call", "registered_tool"}


def mapped_arguments(execution: dict, params: dict) -> dict:
    mapping = execution.get("arg_mapping")
    if mapping is None:
        return params
    return {target: params[source] for target, source in mapping.items() if source in params}


class SkillRegistry:
    def __init__(self, pipeline_instance=None, config=None, skills_dir=None, tool_registry=None):
        self.config = config
        if self.config is None:
            self.config = Config()
        self.pipeline_instance = pipeline_instance
        self.skills_dir = Path(__file__).resolve().parent
        if skills_dir is not None:
            self.skills_dir = Path(skills_dir)
        self.tools = tool_registry
        if self.tools is None:
            self.tools = BankingToolRegistry(pipeline_instance=pipeline_instance)

    @staticmethod
    def _validate(spec: dict, tools: dict) -> None:
        check_parameters(spec.get("parameters_schema"))
        execution = spec.get("execution")
        if not isinstance(execution, dict) or not isinstance(execution.get("type"), str):
            raise ValueError("Missing skill execution mode")
        mode = execution["type"]
        if mode not in SUPPORTED_MODES:
            raise ValueError("Unknown skill execution mode")
        if mode == "tool_call" and execution.get("function") != "check_biometric_compliance_qd2345":
            raise ValueError("Use registered_tool for a new tool-backed skill")
        if mode != "registered_tool":
            return

        target = execution.get("tool")
        if not public_name(target) or target not in tools or target in ONCE_PER_TURN:
            raise ValueError("Skill target must be a registered read-only tool")
        mapping = execution.get("arg_mapping")
        accepted = tools[target]["parameters"].get("properties", {})
        available = spec["parameters_schema"].get("properties", {})
        if mapping is None:
            mapping = {name: name for name in available}
        if not isinstance(mapping, dict):
            raise ValueError("Invalid skill argument mapping")
        if any(not public_name(source) or source not in available or key not in accepted for key, source in mapping.items()):
            raise ValueError("Skill arguments must map between declared parameters")
        if not set(tools[target]["parameters"].get("required", [])).issubset(mapping):
            raise ValueError("Skill must map all required tool arguments")

    def snapshot(self, tools=None) -> dict:
        if tools is None:
            tools = self.tools.snapshot()
        return load_specs(self.skills_dir, "*_skill.json", "skill_id", lambda spec: self._validate(spec, tools))

    def get_skill(self, skill_id: str, *, definition=None) -> BaseSkill | None:
        if definition is None:
            definition = self.snapshot().get(skill_id)
        if definition is None:
            return None
        skill = BaseSkill.from_spec(definition, config=self.config, pipeline_instance=self.pipeline_instance)
        skill.tool_registry = self.tools
        return skill

    def list_skills(self) -> list[dict]:
        return [
            {
                "skill_id": name, "display_name": spec.get("display_name", name),
                "description": spec.get("description", ""), "version": spec.get("version", "2.0.0"),
                "category": spec.get("category", "GENERAL"), "capabilities": spec.get("capabilities", []),
                "triggers": spec.get("intent_triggers", [])
            }
            for name, spec in self.snapshot().items()
        ]
