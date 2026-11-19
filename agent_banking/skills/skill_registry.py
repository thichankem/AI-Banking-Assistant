"""Discover Agent Skills and bind approved A-Bank capabilities."""

import logging
from pathlib import Path

from jsonschema.exceptions import SchemaError

from agent_banking.core.plugin_specs import check_parameters, confined_file, load_yaml, public_name, read_resource
from agent_banking.core.settings import Config
from agent_banking.skills.base_skill import BaseSkill
from agent_banking.skills.config.prompts import get_skill_prompt_override, get_skill_reference_override
from agent_banking.skills.config.settings import get_skill_runtime_config, get_skill_runtime_override
from agent_banking.skills.skill_format import read_skill, valid_name, validate_metadata
from agent_banking.tools.tool_registry import BankingToolRegistry, ONCE_PER_TURN, tool_input_schema


logger = logging.getLogger(__name__)
_SKILL_CFG = get_skill_runtime_config()
_REGISTRY_CFG = _SKILL_CFG["registry"]
_META_DEFAULTS = _SKILL_CFG["metadata_defaults"]
NATIVE_MODES = dict(_REGISTRY_CFG["native_modes"])
RUNTIME_ASSET = str(_REGISTRY_CFG["runtime_asset"])
ALLOWED_TOOL_CALL_FUNCTIONS = set(_REGISTRY_CFG["allowed_tool_call_functions"])
MAX_SKILL_REFERENCES = int(_REGISTRY_CFG["max_references"])
DEFAULT_CATEGORY = str(_META_DEFAULTS["category"])
DEFAULT_PACKAGE_VERSION = str(_META_DEFAULTS["registry_version"])
DEFAULT_LIST_VERSION = str(_META_DEFAULTS["version"])
_DEFAULT_SKILLS_DIR = Path(__file__).resolve().parent


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
        self.skills_dir = _DEFAULT_SKILLS_DIR
        if skills_dir is not None:
            self.skills_dir = Path(skills_dir)
        self.tools = tool_registry
        if self.tools is None:
            self.tools = BankingToolRegistry(pipeline_instance=pipeline_instance)

    def _uses_default_dir(self) -> bool:
        return self.skills_dir.resolve() == _DEFAULT_SKILLS_DIR

    @staticmethod
    def _validate(spec: dict, tools: dict) -> None:
        check_parameters(spec.get("parameters_schema"))
        execution = spec.get("execution")
        if not isinstance(execution, dict) or not isinstance(execution.get("type"), str):
            raise ValueError("Missing skill execution mode")
        mode = execution["type"]
        if mode not in ("registered_tool", "guidance") and NATIVE_MODES.get(spec["skill_id"]) != mode:
            raise ValueError("Only built-in skills may use native adapters")
        if mode == "tool_call" and execution.get("function") not in ALLOWED_TOOL_CALL_FUNCTIONS:
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
            "category": metadata.get("a-bank-category", DEFAULT_CATEGORY),
            "version": metadata.get("version", DEFAULT_PACKAGE_VERSION),
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

        if self._uses_default_dir():
            prompt_override = get_skill_prompt_override(name)
            for field_name in ("description", "display_name", "category", "version", "compatibility"):
                value = prompt_override.get(field_name)
                if isinstance(value, str) and value.strip():
                    spec[field_name] = value.strip()
            runtime_override = get_skill_runtime_override(name)
            if runtime_override:
                if set(runtime_override) - {"execution", "parameters_schema", "output_schema"}:
                    raise ValueError("Unsupported runtime binding field")
                spec.update(runtime_override)

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
        result = {"description": spec.get("description", ""), "category": spec.get("category", DEFAULT_CATEGORY),
                  "instructions": spec.get("instructions", ""), "references": [],
                  "parameters": spec.get("parameters_schema", {}), "executable": spec["execution"]["type"] != "guidance"}
        if "revision" not in spec:
            # Existing paused turns retain their original, validated snapshot.
            return result
        root = self._root(spec)
        data, _, body = read_skill(confined_file(root, "SKILL.md"), body=True, expected=spec["revision"])
        validate_metadata(data, root.name)
        result["instructions"] = body
        if self._uses_default_dir():
            prompt_override = get_skill_prompt_override(root.name)
            for field_name in ("description", "category", "compatibility", "instructions"):
                value = prompt_override.get(field_name)
                if isinstance(value, str) and value.strip():
                    result[field_name] = value.strip()
        refs = root / "references"
        if refs.is_dir() and refs.resolve() == refs:
            result["references"] = [path.relative_to(root).as_posix() for path in sorted(refs.glob("*.md"))
                                    if path.is_file() and path.resolve() == path][:MAX_SKILL_REFERENCES]
        if spec.get("compatibility") and "compatibility" not in result:
            result["compatibility"] = spec["compatibility"]
        return result

    def reference(self, spec: dict, path: str, inventory: list[str]) -> str:
        if path not in inventory or not path.startswith("references/") or not path.endswith(".md"):
            raise ValueError("Skill reference is not in the activated inventory")
        if self._uses_default_dir():
            override_text = get_skill_reference_override(spec["skill_id"], path)
            if override_text:
                return override_text
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
                "description": spec.get("description", ""), "version": spec.get("version", DEFAULT_LIST_VERSION),
                "category": spec.get("category", DEFAULT_CATEGORY), "capabilities": spec.get("capabilities", []),
                "triggers": spec.get("intent_triggers", [])
            }
            for name, spec in self.snapshot().items()
        ]
