"""Discover standard tool contracts and dispatch through local A-Bank policy."""

from ast import FunctionDef, parse
from importlib import import_module, invalidate_caches
from importlib.util import find_spec
import json
import logging
from pathlib import Path
import time

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from agent_banking.core.fallback import is_temporary_error
from agent_banking.core.plugin_specs import check_parameters, load_yaml, public_name, read_resource
from agent_banking.guardrails.tool_guardrail import (
    SENSITIVE_UNAPPROVED_ACTIONS, get_tool_guardrail
)
from agent_banking.tools.config.prompts import TOOL_TEMPLATES, get_tool_prompt_override
from agent_banking.tools.config.settings import get_tool_runtime_config, get_tool_runtime_override
from agent_banking.tools.extensions.legal_web_search import PROVIDERS, TOOL_NAME as LEGAL_WEB_SEARCH_TOOL, provider_key

_TYPE_CONVERTERS = {"float": float, "int": int, "str": str, "bool": bool}


def _build_builtin_args(raw_map: dict) -> dict:
    built = {}
    for tool_name, arg_specs in raw_map.items():
        tool_entry = {}
        for arg_name, spec in arg_specs.items():
            converter = _TYPE_CONVERTERS.get(spec.get("type"))
            alias = spec.get("alias")
            if alias is not None:
                tool_entry[arg_name] = (spec.get("default"), converter, alias)
            else:
                tool_entry[arg_name] = (spec.get("default"), converter)
        built[tool_name] = tool_entry
    return built


_REG_CFG = get_tool_runtime_config()["registry"]
BANKING_MODULE = str(_REG_CFG["banking_module"])
EXTENSIONS_MODULE = str(_REG_CFG["extensions_module"])
RAG_HANDLER = dict(_REG_CFG["rag_handler"])
BUILTIN_ARGS = _build_builtin_args(_REG_CFG["builtin_args"])
ONCE_PER_TURN = set(_REG_CFG["once_per_turn_tools"])
PUBLIC_FIELDS = tuple(_REG_CFG["public_fields"])
ANNOTATIONS = set(_REG_CFG["annotations"])
ALLOWED_RUNTIME_FIELDS = set(_REG_CFG["allowed_runtime_fields"])
DEFAULT_CATEGORY = str(_REG_CFG["default_category"])
DEFAULT_RAG_TOP_K = int(_REG_CFG["default_rag_top_k"])
_DEFAULT_TOOLS_DIR = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


def tool_input_schema(spec: dict) -> dict:
    if "inputSchema" in spec:
        return spec["inputSchema"]
    # Turns paused before the migration retain their serialized catalog.
    return spec["parameters"]


def tool_contract(spec: dict) -> dict:
    contract = {key: spec[key] for key in PUBLIC_FIELDS if key in spec}
    contract["inputSchema"] = tool_input_schema(spec)
    return contract


class ToolInputError(ValueError):
    pass


class ToolOutputError(ValueError):
    pass


def _legacy_value(value, convert):
    if convert is bool:
        if isinstance(value, str):
            value = value.strip().casefold()
            if value in ("true", "1"):
                return True
            if value in ("false", "0"):
                return False
        if type(value) in (bool, int) and value in (0, 1):
            return bool(value)
        raise ValueError("Invalid boolean argument")
    if convert is not None:
        return convert(value)
    return value


def is_extension(spec: dict) -> bool:
    return spec["handler"]["module"].startswith(EXTENSIONS_MODULE + ".")


class BankingToolRegistry:
    def __init__(self, tools_dir=None, pipeline_instance=None):
        self.tools_dir = _DEFAULT_TOOLS_DIR
        if tools_dir is not None:
            self.tools_dir = Path(tools_dir)
        self.pipeline_instance = pipeline_instance

    def _uses_default_dir(self) -> bool:
        return self.tools_dir.resolve() == _DEFAULT_TOOLS_DIR

    def _extension_file(self, spec: dict) -> Path:
        handler = spec["handler"]
        stem = handler["module"].removeprefix(EXTENSIONS_MODULE + ".")
        if not public_name(stem) or not public_name(handler.get("function")):
            raise ValueError("Invalid extension handler")
        parent = self.tools_dir / "extensions"
        path = parent / (stem + ".py")
        if parent.is_symlink() or path.is_symlink() or not path.is_file():
            raise ValueError("Extension source is missing")
        return path.resolve()

    def _validate(self, spec: dict) -> None:
        check_parameters(spec.get("inputSchema"))
        if spec["inputSchema"].get("additionalProperties") is not False:
            raise ValueError("Tool input must reject undeclared arguments")
        if "outputSchema" in spec:
            check_parameters(spec["outputSchema"])
        name = spec["name"]
        for key in ("description", "title"):
            if key == "title" and key not in spec:
                continue
            if not isinstance(spec.get(key), str) or not spec[key].strip():
                raise ValueError("Missing tool description or invalid title")
        annotations = spec.get("annotations", {})
        if not isinstance(annotations, dict) or set(annotations) - ANNOTATIONS or any(
            type(value) is not bool for value in annotations.values()
        ):
            raise ValueError("Invalid tool annotations")
        policy = spec.get("policy")
        if name in ONCE_PER_TURN:
            expected_access = "simulation"
        else:
            expected_access = "read_only"
        if not isinstance(policy, dict) or set(policy) - {"access", "workers"} or policy.get("access") != expected_access:
            raise ValueError("Invalid local policy")
        if "readOnlyHint" in annotations and annotations["readOnlyHint"] != (expected_access == "read_only"):
            raise ValueError("readOnlyHint disagrees with local policy")
        workers = policy.get("workers", [])
        if not isinstance(workers, list) or not all(public_name(role) for role in workers):
            raise ValueError("Invalid worker allowlist")
        if name in ONCE_PER_TURN and workers:
            raise ValueError("Simulated actions cannot be delegated")
        handler = spec.get("handler")
        if not isinstance(handler, dict):
            raise ValueError("Missing handler")
        if name == "search_banking_knowledge" and handler == RAG_HANDLER:
            return
        if name in BUILTIN_ARGS and handler == {"module": BANKING_MODULE, "function": name}:
            return
        if name in BUILTIN_ARGS or name == "search_banking_knowledge":
            raise ValueError("Built-in tool names are reserved")
        if not isinstance(handler.get("module"), str) or not is_extension(spec):
            raise ValueError("Handler is outside the extension namespace")
        if set(handler) != {"module", "function"}:
            raise ValueError("Extension handlers must be functions")
        if any(action in name.lower() for action in SENSITIVE_UNAPPROVED_ACTIONS):
            raise ValueError("Sensitive action is outside the extension policy")
        path = self._extension_file(spec)
        try:
            tree = parse(path.read_bytes(), filename=path.name)
        except SyntaxError as error:
            raise ValueError("Invalid extension source") from error
        if not any(isinstance(node, FunctionDef) and node.name == handler["function"] for node in tree.body):
            raise ValueError("Extension handler needs a top-level function")

    def snapshot(self) -> dict:
        root = self.tools_dir.resolve()
        if not root.is_dir():
            return {}
        tools = {}
        uses_default = self._uses_default_dir()
        for directory in sorted(root.iterdir()):
            if not public_name(directory.name) or not directory.is_dir() or directory.resolve() != directory:
                continue
            if not (directory / "tool.yaml").exists():
                continue
            if uses_default and directory.name == LEGAL_WEB_SEARCH_TOOL and not any(provider_key(name) for name in PROVIDERS):
                continue
            try:
                public = load_yaml(read_resource(directory, "tool.yaml"))
                local = load_yaml(read_resource(directory, "runtime.yaml"))
                if uses_default:
                    prompt_override = get_tool_prompt_override(directory.name)
                    if prompt_override:
                        public = {**public, **prompt_override}
                    runtime_override = get_tool_runtime_override(directory.name)
                    if runtime_override:
                        local = {**local, **runtime_override}
                if set(public) - set(PUBLIC_FIELDS) or public.get("name") != directory.name:
                    raise ValueError("Invalid tool name or public field")
                if set(local) - ALLOWED_RUNTIME_FIELDS:
                    raise ValueError("Invalid runtime binding field")
                spec = {**public, **local}
                json.dumps(spec, allow_nan=False)
                self._validate(spec)
                tools[spec["name"]] = spec
            except (OSError, TypeError, ValueError, SchemaError, RecursionError) as error:
                logger.warning("Skipping tool %s (%s)", directory.name, type(error).__name__)
        return tools

    def list_tools(self) -> list[dict]:
        return [
            {
                "name": name, "display_name": spec.get("title", name),
                "category": spec.get("category", DEFAULT_CATEGORY), "description": spec.get("description", "")
            }
            for name, spec in self.snapshot().items()
        ]

    def get_tool_definition(self, name: str) -> dict | None:
        return self.snapshot().get(name)

    def get_tool_specs(self) -> list[dict]:
        return [
            {"type": "function", "function": {
                "name": name, "description": spec.get("description", ""), "parameters": tool_input_schema(spec)
            }}
            for name, spec in self.snapshot().items()
        ]

    def get_tool_contracts(self) -> list[dict]:
        """Return public Tool objects without exposing Python handlers or policy."""
        return [tool_contract(spec) for spec in self.snapshot().values()]

    @staticmethod
    def _prepare_arguments(spec: dict, params: dict, *, legacy_batch: bool) -> dict:
        if not isinstance(params, dict):
            raise ToolInputError("Tool arguments must be an object")
        name = spec["name"]
        if legacy_batch and name in BUILTIN_ARGS:
            converted = {}
            try:
                for key, (default, convert, *aliases) in BUILTIN_ARGS[name].items():
                    if aliases:
                        fallback_val = params.get(aliases[0], default)
                    else:
                        fallback_val = default
                    value = params.get(key, fallback_val)
                    converted[key] = _legacy_value(value, convert)
            except (TypeError, ValueError, OverflowError) as error:
                raise ToolInputError("Invalid legacy tool arguments") from error
            params = converted
        schema = {**tool_input_schema(spec), "additionalProperties": False}
        try:
            json.dumps(params, allow_nan=False)
            valid = Draft202012Validator(schema).is_valid(params)
        except (ValueError, TypeError) as error:
            raise ToolInputError("Invalid JSON arguments") from error
        if not valid:
            raise ToolInputError("Tool arguments do not match inputSchema")
        guarded = get_tool_guardrail().guard_tool_call(name, params, registered_readonly=is_extension(spec))
        if not guarded["is_safe"]:
            raise ToolInputError("Tool guard rejected arguments")
        safe_params = guarded["sanitized_params"]
        if not Draft202012Validator(schema).is_valid(safe_params):
            raise ToolInputError("Sanitized arguments do not match inputSchema")
        return safe_params

    def _invoke(self, spec: dict, params: dict):
        name = spec["name"]
        handler = spec["handler"]
        if handler == RAG_HANDLER:
            from agent_banking.tools.rag_tool import BankingRAGTool

            excel = {}
            if "excel" in params:
                excel = {"excel": params["excel"]}
            return BankingRAGTool(pipeline_instance=self.pipeline_instance).execute(
                query=params.get("query", ""), top_k=params.get("top_k", DEFAULT_RAG_TOP_K),
                exact=params.get("exact", ""), source=params.get("source", ""), offset=params.get("offset"),
                page=params.get("page"), **excel,
            )
        if not is_extension(spec):
            return getattr(import_module(BANKING_MODULE), handler["function"])(**params)

        path = self._extension_file(spec)
        invalidate_caches()
        origin = find_spec(handler["module"])
        if origin is None or not origin.origin or Path(origin.origin).resolve() != path:
            raise ValueError("Unexpected extension module origin")
        module = import_module(handler["module"])
        function = getattr(module, handler["function"])
        if not callable(function) or getattr(function, "__module__", None) != module.__name__:
            raise ValueError("Handler must be defined in the extension module")
        return function(**params)

    def execute_tool(self, name: str, params: dict, *, definition=None, legacy_batch=False) -> dict:
        start = time.perf_counter()
        spec = definition
        if spec is None:
            spec = self.get_tool_definition(name)
        if spec is None or spec["name"] != name:
            msg_tpl = TOOL_TEMPLATES["tool_not_found_message"]
            return {"status": "ERROR", "message": msg_tpl.format(name=name), "latency_ms": 0.0}
        try:
            args = self._prepare_arguments(spec, params, legacy_batch=legacy_batch)
            data = self._invoke(spec, args)
            try:
                json.dumps(data, allow_nan=False)
            except (TypeError, ValueError, OverflowError) as error:
                raise ToolOutputError("Tool output must be JSON") from error
            if not isinstance(data, dict):
                raise ToolOutputError("Tool handler must return an object")
            if "outputSchema" in spec and not Draft202012Validator(spec["outputSchema"]).is_valid(data):
                raise ToolOutputError("Tool output does not match outputSchema")
            return {
                "status": "SUCCESS", "tool_name": name, "data": data,
                "latency_ms": round((time.perf_counter() - start) * 1000, 2)
            }
        except Exception as error:
            return {
                "status": "FAILED", "tool_name": name, "error": type(error).__name__,
                "temporary": is_temporary_error(error),
                "latency_ms": round((time.perf_counter() - start) * 1000, 2)
            }
