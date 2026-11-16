"""Discover tool specs and dispatch to project-owned banking handlers."""

from ast import FunctionDef, parse
from importlib import import_module, invalidate_caches
from importlib.util import find_spec
from pathlib import Path
import time

from jsonschema import Draft202012Validator

from agent_banking.core.plugin_specs import check_parameters, load_specs, public_name
from agent_banking.guardrails.tool_guardrail import (
    SENSITIVE_UNAPPROVED_ACTIONS, get_tool_guardrail
)


BANKING_MODULE = "agent_banking.tools.banking_api_tools"
EXTENSIONS_MODULE = "agent_banking.tools.extensions"
RAG_HANDLER = {"module": "agent_banking.tools.rag_tool", "class": "BankingRAGTool", "method": "execute"}

# Retain the fixture runner's aliases, coercions and defaults for existing tools.
BUILTIN_ARGS = {
    "calculate_loan_amortization": {
        "principal_vnd": (500000000.0, float), "annual_rate_percent": (7.5, float), "term_months": (12, int)
    },
    "calculate_savings_profit": {"amount_vnd": (100000000.0, float), "term_months": (6, int)},
    "get_interest_rates": {"deposit_type": ("vnd", None)},
    "get_foreign_exchange_rates": {"currency": ("USD", None)},
    "check_biometric_compliance_qd2345": {
        "single_tx_amount_vnd": (0.0, float, "transaction_amount"),
        "daily_cumulative_amount_vnd": (0.0, float, "daily_accumulated_amount"),
        "is_first_time_new_device": (False, bool, "is_new_device"),
        "has_verified_chip_cccd": (True, bool, "has_cccd_chip_nfc")
    },
    "create_support_ticket": {
        "customer_name": ("KHACH_HANG_HOTLINE", None, "user_contact"),
        "issue_type": ("nuot_the_atm", None),
        "description": ("Sự cố tiếp nhận qua AI Agent", None, "details"), "priority": ("NORMAL", None)
    },
    "parse_vietnamese_financial_params": {"query": ("", None)},
    "check_complain_count": {"complain_count": (0, int)},
    "transfer_to_agent": {
        "business_process": ("request_human_agent", str, "reason"),
        "customer_segment": ("individual", str, "department")
    },
    "rate_agent_csat": {
        "rating_score": (5, int, "rating"), "user_message": ("", str, "feedback"), "channel": ("web", str)
    },
    "deflect_competitor_comparison": {"competitor_name": ("", str), "query": ("", str, "product_type")}
}
ONCE_PER_TURN = {"create_support_ticket", "transfer_to_agent", "rate_agent_csat"}


def is_extension(spec: dict) -> bool:
    return spec["handler"]["module"].startswith(EXTENSIONS_MODULE + ".")


class BankingToolRegistry:
    def __init__(self, tools_dir=None, pipeline_instance=None):
        self.tools_dir = Path(__file__).resolve().parent
        if tools_dir is not None:
            self.tools_dir = Path(tools_dir)
        self.pipeline_instance = pipeline_instance

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
        check_parameters(spec.get("parameters"))
        name = spec["name"]
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
        policy = spec.get("policy")
        if not isinstance(policy, dict) or policy.get("access") != "read_only":
            raise ValueError("New extensions require a read-only policy")
        workers = policy.get("workers", [])
        if not isinstance(workers, list) or not all(public_name(role) for role in workers):
            raise ValueError("Invalid worker allowlist")
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
        return load_specs(self.tools_dir, "*.json", "name", self._validate)

    def list_tools(self) -> list[dict]:
        return [
            {
                "name": name, "display_name": spec.get("display_name", name),
                "category": spec.get("category", "GENERAL"), "description": spec.get("description", "")
            }
            for name, spec in self.snapshot().items()
        ]

    def get_tool_definition(self, name: str) -> dict | None:
        return self.snapshot().get(name)

    def get_tool_specs(self) -> list[dict]:
        return [
            {"type": "function", "function": {
                "name": name, "description": spec.get("description", ""), "parameters": spec["parameters"]
            }}
            for name, spec in self.snapshot().items()
        ]

    def _invoke(self, spec: dict, params: dict):
        name = spec["name"]
        handler = spec["handler"]
        if handler == RAG_HANDLER:
            from agent_banking.tools.rag_tool import BankingRAGTool

            return BankingRAGTool(pipeline_instance=self.pipeline_instance).execute(
                query=params.get("query", ""), top_k=params.get("top_k", 3)
            )
        if not is_extension(spec):
            arguments = {}
            for key, (default, convert, *aliases) in BUILTIN_ARGS[name].items():
                value = params.get(key, default)
                if key not in params and aliases:
                    value = params.get(aliases[0], default)
                if convert is not None:
                    value = convert(value)
                arguments[key] = value
            return getattr(import_module(BANKING_MODULE), handler["function"])(**arguments)

        result = get_tool_guardrail().guard_tool_call(name, params, registered_readonly=True)
        if not result["is_safe"]:
            raise ValueError("Tool guard rejected arguments")
        safe_params = result["sanitized_params"]
        schema = {**spec["parameters"], "additionalProperties": False}
        if not Draft202012Validator(schema).is_valid(safe_params):
            raise ValueError("Tool arguments do not match the schema")

        path = self._extension_file(spec)
        invalidate_caches()
        origin = find_spec(handler["module"])
        if origin is None or not origin.origin or Path(origin.origin).resolve() != path:
            raise ValueError("Unexpected extension module origin")
        module = import_module(handler["module"])
        function = getattr(module, handler["function"])
        if not callable(function) or getattr(function, "__module__", None) != module.__name__:
            raise ValueError("Handler must be defined in the extension module")
        return function(**safe_params)

    def execute_tool(self, name: str, params: dict, *, definition=None) -> dict:
        start = time.perf_counter()
        spec = definition
        if spec is None:
            spec = self.get_tool_definition(name)
        if spec is None or spec["name"] != name:
            return {"status": "ERROR", "message": f"Tool '{name}' not found in registry.", "latency_ms": 0.0}
        try:
            data = self._invoke(spec, params)
            if not isinstance(data, dict):
                raise ValueError("Tool handler must return an object")
            return {
                "status": "SUCCESS", "tool_name": name, "data": data,
                "latency_ms": round((time.perf_counter() - start) * 1000, 2)
            }
        except Exception as error:
            return {
                "status": "FAILED", "tool_name": name, "error": type(error).__name__,
                "latency_ms": round((time.perf_counter() - start) * 1000, 2)
            }
