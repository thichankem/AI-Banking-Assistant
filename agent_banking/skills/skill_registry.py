"""Discover Agent Skills and bind approved A-Bank capabilities."""

import logging
from pathlib import Path

from jsonschema.exceptions import SchemaError

from agent_banking.core.plugin_specs import check_parameters, confined_file, load_yaml, public_name, read_resource
from agent_banking.core.settings import Config
from agent_banking.skills.base_skill import BaseSkill
from agent_banking.skills.skill_format import read_skill, valid_name, validate_metadata
from agent_banking.tools.tool_registry import BankingToolRegistry, ONCE_PER_TURN, tool_input_schema


logger = logging.getLogger(__name__)
NATIVE_MODES = {
    "rag-knowledge-retrieval": "rag_pipeline",
    "fact-checking-audit": "fact_checker_engine",
    "incident-support-ticket": "incident_dispatcher",
    "financial-calculator": "financial_dispatcher",
    "compliance-qd2345-audit": "tool_call",
}
RUNTIME_ASSET = "assets/a-bank-runtime.yaml"


def resolve_skill(skills: dict, name: str) -> dict | None:
    """Accept prior snake_case IDs in CLI requests and saved turns."""
    return skills.get(name) or skills.get(name.replace("_", "-")) or skills.get(name.replace("-", "_"))


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
        if mode not in ("registered_tool", "guidance") and NATIVE_MODES.get(spec["skill_id"]) != mode:
            raise ValueError("Only built-in skills may use native adapters")
        if mode == "tool_call" and execution.get("function") != "check_biometric_compliance_qd2345":
            raise ValueError("Use registered_tool for a new tool-backed skill")
        if mode != "registered_tool":
            return

        target = execution.get("tool")
        if not public_name(target) or target not in tools or target in ONCE_PER_TURN:
            raise ValueError("Skill target must be a registered read-only tool")
        mapping = execution.get("arg_mapping")
        input_schema = tool_input_schema(tools[target])
        accepted = input_schema.get("properties", {})
        available = spec["parameters_schema"].get("properties", {})
        if mapping is None:
            mapping = {name: name for name in available}
        if not isinstance(mapping, dict):
            raise ValueError("Invalid skill argument mapping")
        if any(not public_name(source) or source not in available or key not in accepted for key, source in mapping.items()):
            raise ValueError("Skill arguments must map between declared parameters")
        if not set(input_schema.get("required", [])).issubset(mapping):
            raise ValueError("Skill must map all required tool arguments")

    def _package(self, directory: Path, tools: dict) -> dict:
        data, revision, _ = read_skill(confined_file(directory, "SKILL.md"))
        name = directory.name
        validate_metadata(data, name)
        metadata = data.get("metadata", {})
        spec = {
            "skill_id": name, "description": data["description"], "revision": revision,
            "display_name": metadata.get("a-bank-display-name", name),
            "category": metadata.get("a-bank-category", "GENERAL"),
            "version": metadata.get("version", "1.0.0"),
            "execution": {"type": "guidance"},
            "parameters_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        }
        if data.get("compatibility"):
            spec["compatibility"] = data["compatibility"]
        target = metadata.get("a-bank-tool")
        binding_path = directory / RUNTIME_ASSET
        if binding_path.exists() or binding_path.is_symlink():
            if target is not None:
                raise ValueError("Conflicting skill bindings")
            binding = load_yaml(read_resource(directory, RUNTIME_ASSET))
            if set(binding) - {"execution", "parameters_schema", "output_schema"}:
                raise ValueError("Unsupported runtime binding field")
            if not {"execution", "parameters_schema"} <= set(binding):
                raise ValueError("Runtime binding is incomplete")
            spec.update(binding)
        elif target is not None:
            if target not in tools:
                raise ValueError("Unknown tool binding")
            spec.update(execution={"type": "registered_tool", "tool": target}, parameters_schema=tool_input_schema(tools[target]))
        self._validate(spec, tools)
        return spec

    def snapshot(self, tools=None) -> dict:
        if tools is None:
            tools = self.tools.snapshot()
        root = self.skills_dir.resolve()
        if not root.is_dir():
            return {}
        skills = {}
        for directory in sorted(root.iterdir()):
            if not valid_name(directory.name) or not directory.is_dir() or directory.resolve() != directory:
                continue
            entry = directory / "SKILL.md"
            if not (entry.exists() or entry.is_symlink()):
                continue
            try:
                skills[directory.name] = self._package(directory, tools)
            except (OSError, ValueError, SchemaError, RecursionError) as error:
                logger.warning("Skipping skill %s (%s)", directory.name, type(error).__name__)
        return skills

    def _root(self, spec: dict) -> Path:
        name = spec["skill_id"]
        if not valid_name(name):
            raise ValueError("Invalid skill directory")
        path = self.skills_dir.resolve() / name
        if path.resolve(strict=True) != path or not path.is_dir():
            raise ValueError("Invalid skill directory")
        return path

    def instructions(self, spec: dict) -> dict:
        result = {"description": spec.get("description", ""), "category": spec.get("category", "GENERAL"),
                  "instructions": spec.get("instructions", ""), "references": [],
                  "parameters": spec.get("parameters_schema", {}), "executable": spec["execution"]["type"] != "guidance"}
        if "revision" not in spec:
            # Existing paused turns retain their original, validated snapshot.
            return result
        root = self._root(spec)
        data, _, body = read_skill(confined_file(root, "SKILL.md"), body=True, expected=spec["revision"])
        validate_metadata(data, root.name)
        result["instructions"] = body
        refs = root / "references"
        if refs.is_dir() and refs.resolve() == refs:
            result["references"] = [path.relative_to(root).as_posix() for path in sorted(refs.glob("*.md"))
                                    if path.is_file() and path.resolve() == path][:24]
        if spec.get("compatibility"):
            result["compatibility"] = spec["compatibility"]
        return result

    def reference(self, spec: dict, path: str, inventory: list[str]) -> str:
        if path not in inventory or not path.startswith("references/") or not path.endswith(".md"):
            raise ValueError("Skill reference is not in the activated inventory")
        return read_resource(self._root(spec), path).strip()

    def get_skill(self, skill_id: str, *, definition=None) -> BaseSkill | None:
        if definition is None:
            definition = resolve_skill(self.snapshot(), skill_id)
        if definition is None or definition["execution"]["type"] == "guidance":
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
