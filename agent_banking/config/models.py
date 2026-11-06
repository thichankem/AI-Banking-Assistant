"""Resolve per-module model lists, with secrets referenced by environment name."""
from importlib import import_module
import os

from agent_banking.config.environment import load_environment
from agent_banking.config.fallback import ModelEndpoint, ModelFallbackError


def _value(spec, name, default=""):
    if name in spec:
        return spec[name]
    names = spec.get(name + "_env", ())
    if isinstance(names, str):
        names = (names,)
    for env_name in names:
        value = os.getenv(env_name)
        if value:
            return value
    return spec.get(name + "_default", default)


def get_model_chain(module, primary=None):
    load_environment()
    config_module = module
    chain_name = "MODELS"
    if module.startswith("chunking."):
        config_module = "chunking"
        chain_name = module.split(".", 1)[1].upper() + "_MODELS"
    specs = getattr(import_module("agent_banking.config." + config_module + ".models"), chain_name)
    result = []
    for index, spec in enumerate(specs):
        if not spec.get("enabled", True):
            continue
        values = {
            "name": spec.get("name", "candidate-" + str(index + 1)),
            "model": _value(spec, "model"),
            "url": _value(spec, "url"),
            "api_key": _value(spec, "api_key"),
            "timeout": float(spec.get("timeout", 60)),
            "attempts": int(spec.get("attempts", 1)),
            "dimension": int(spec.get("dimension", 0)),
        }
        if spec.get("append_path") and values["url"]:
            values["url"] = values["url"].rstrip("/") + spec["append_path"]
        if not values["url"] and spec.get("base_url_env"):
            base_url = os.getenv(spec["base_url_env"], spec.get("base_url_default", ""))
            if base_url:
                values["url"] = base_url.rstrip("/") + spec.get("path", "")
        if not result and primary:
            for key, value in primary.items():
                if value is not None:
                    values[key] = value
        result.append(ModelEndpoint(**values))
    if module == "embedding" and result:
        identity = (result[0].model, result[0].dimension)
        for candidate in result:
            if (candidate.model, candidate.dimension) != identity:
                raise ValueError("Embedding fallbacks must use the same model and dimension; changing vector space requires reindexing")
    return result


def chain_fingerprint(module, primary=None):
    """Non-secret identity for cache invalidation after changing model priority."""
    from hashlib import sha256
    import json
    chain = get_model_chain(module, primary)
    identity = [(item.model, item.url, item.dimension) for item in chain]
    return sha256(json.dumps(identity).encode("utf-8")).hexdigest()


def model_primary(module):
    chain = get_model_chain(module)
    if not chain:
        raise ModelFallbackError(module, [("configuration", "empty model chain")])
    return chain[0]
