# -*- coding: utf-8 -*-
"""
Resolve per-module model lists, with secrets referenced by environment name.
Loads model definitions from module-level config/models.py.
Complies with 0% Regex Law, Zero Ternary Law, and No Typing Library.
"""

from dataclasses import replace
from importlib import import_module
import math
import os

from agent_banking.core.config.settings import CoreConfig, get_core_messages
from agent_banking.core.environment import load_environment
from agent_banking.core.fallback import ModelEndpoint, ModelFallbackError

_MODELS_CFG = CoreConfig.CORE_MODELS
_MESSAGES = get_core_messages("models", "core_messages.yaml", "CORE_MODEL_MESSAGES")

ROUTING_VERSION = _MODELS_CFG["routing_version"]
ROUTED_MODULES = {
    key: tuple(value) for key, value in _MODELS_CFG["routed_modules"].items()
}
_MODULE_ALIASES = dict(_MODELS_CFG["module_aliases"])
_CHUNKING_PREFIX = _MODELS_CFG["chunking_module_prefix"]
_CHUNKING_CONFIG_MODULE = _MODELS_CFG["chunking_config_module"]
_DEFAULT_CHAIN_ATTR = _MODELS_CFG["default_chain_attr"]
_AUTO_ROUTING_ENV = _MODELS_CFG["auto_routing_env"]
_TRUTHY = tuple(_MODELS_CFG["truthy_values"])
_FALSY = tuple(_MODELS_CFG["falsy_values"])
_VALID_TOKEN_PARAMS = tuple(_MODELS_CFG["valid_token_parameters"])
_VALID_JSON_MODES = tuple(_MODELS_CFG["valid_json_modes"])
_VALID_IMAGE_DETAILS = tuple(_MODELS_CFG["valid_image_details"])
_VALID_REASONING_EFFORTS = tuple(_MODELS_CFG["valid_reasoning_efforts"])
_MIN_TEMP = float(_MODELS_CFG["min_temperature"])
_MAX_TEMP = float(_MODELS_CFG["max_temperature"])
_DEFAULT_TIMEOUT = float(_MODELS_CFG["default_candidate_timeout"])
_DEFAULT_ATTEMPTS = int(_MODELS_CFG["default_candidate_attempts"])
_DEFAULT_DIMENSION = int(_MODELS_CFG["default_candidate_dimension"])
_DEFAULT_PROFILE = _MODELS_CFG["default_profile"]
_DEEP_PROFILE = _MODELS_CFG["deep_profile"]


def auto_routing_enabled():
    return os.getenv(_AUTO_ROUTING_ENV, "").strip().lower() in _TRUTHY


def model_configured(candidate):
    return bool(candidate.model and candidate.url and candidate.api_key)


def _profile(endpoint, prefix, profile, protocol):
    def setting(name, default):
        return os.getenv(prefix + "_" + name, str(default)).strip()

    def enabled(name, default):
        value = setting(name, str(default).lower()).lower()
        if value in _TRUTHY:
            return True
        if value in _FALSY:
            return False
        raise ValueError(_MESSAGES["boolean_error"].format(prefix=prefix, name=name))

    values = {
        "profile": profile,
        "protocol": setting("PROTOCOL", protocol),
        "model": setting("MODEL", endpoint.model),
        "url": setting("BASE_URL", endpoint.url),
        "api_key": setting("API_KEY", endpoint.api_key),
    }
    if values["protocol"] != protocol:
        raise ValueError(_MESSAGES["protocol_error"].format(prefix=prefix))
    try:
        values.update(
            timeout=float(setting("TIMEOUT", endpoint.timeout)),
            attempts=int(setting("ATTEMPTS", endpoint.attempts)),
            context_window=int(setting("CONTEXT_WINDOW", endpoint.context_window)),
            max_output_tokens=int(setting("MAX_OUTPUT_TOKENS", endpoint.max_output_tokens)),
            image_token_reserve=int(setting("IMAGE_TOKEN_RESERVE", endpoint.image_token_reserve)),
        )
        if endpoint.temperature is None:
            default_temp = ""
        else:
            default_temp = endpoint.temperature
        temperature = setting("TEMPERATURE", default_temp)
        if temperature:
            values["temperature"] = float(temperature)
        else:
            values["temperature"] = None
    except ValueError:
        raise ValueError(_MESSAGES["numeric_error"].format(prefix=prefix)) from None
    if (
        not math.isfinite(values["timeout"])
        or values["timeout"] <= 0
        or values["attempts"] < 1
        or min(values["context_window"], values["max_output_tokens"], values["image_token_reserve"]) < 0
        or (values["temperature"] is not None and not _MIN_TEMP <= values["temperature"] <= _MAX_TEMP)
    ):
        raise ValueError(_MESSAGES["limit_error"].format(prefix=prefix))
    values.update(
        token_parameter=setting("TOKEN_PARAMETER", endpoint.token_parameter),
        json_mode=setting("JSON_MODE", endpoint.json_mode),
        send_temperature=enabled("SEND_TEMPERATURE", endpoint.send_temperature),
        image_detail=setting("IMAGE_DETAIL", endpoint.image_detail),
        reasoning_effort=setting("REASONING_EFFORT", ""),
    )
    if setting("THINKING", ""):
        values["thinking"] = enabled("THINKING", False)
    if (
        values["token_parameter"] not in _VALID_TOKEN_PARAMS
        or values["json_mode"] not in _VALID_JSON_MODES
        or values["image_detail"] not in _VALID_IMAGE_DETAILS
        or values["reasoning_effort"] not in _VALID_REASONING_EFFORTS
    ):
        raise ValueError(_MESSAGES["option_error"].format(prefix=prefix))
    return replace(endpoint, **values)


def _routed_chain(module, chain):
    prefix, protocol = ROUTED_MODULES[module]
    base = ModelEndpoint(_DEFAULT_PROFILE, "", "", "", protocol=protocol)
    if chain:
        base = chain[0]
    # Named profiles are applied after legacy primary overrides from existing callers.
    standard = _profile(base, prefix + "_" + _DEFAULT_PROFILE.upper(), _DEFAULT_PROFILE, protocol)
    result = [standard, *(replace(item, protocol=protocol) for item in chain[1:])]
    if os.getenv(prefix + "_" + _DEEP_PROFILE.upper() + "_MODEL", "").strip():
        deep = replace(
            base,
            name=_DEEP_PROFILE,
            model="",
            url=standard.url,
            api_key=standard.api_key,
            temperature=None,
            disable_thinking=False,
        )
        result.append(_profile(deep, prefix + "_" + _DEEP_PROFILE.upper(), _DEEP_PROFILE, protocol))
    for endpoint in result:
        if endpoint.context_window and endpoint.max_output_tokens >= endpoint.context_window:
            raise ValueError(_MESSAGES["output_vs_context_error"].format(prefix=prefix))
    return result


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


def get_model_chain(module, primary=None, *, routed=None):
    load_environment()
    if routed is None:
        routed = auto_routing_enabled()
    config_module = module
    chain_name = _DEFAULT_CHAIN_ATTR
    if module in _MODULE_ALIASES:
        canonical_alias = _MODULE_ALIASES[module]
        config_module = _CHUNKING_CONFIG_MODULE
        chain_name = canonical_alias.split(".", 1)[1].upper() + "_" + _DEFAULT_CHAIN_ATTR
    elif module.startswith(_CHUNKING_PREFIX):
        config_module = _CHUNKING_CONFIG_MODULE
        chain_name = module.split(".", 1)[1].upper() + "_" + _DEFAULT_CHAIN_ATTR
    elif module == "agent_review":
        if not routed:
            return []
        return _routed_chain(module, [])

    module_path = "agent_banking." + config_module + ".config.models"
    config = import_module(module_path)
    specs = getattr(config, chain_name)
    result = []
    for index, spec in enumerate(specs):
        if not spec.get("enabled", True):
            continue
        if config_module == _CHUNKING_CONFIG_MODULE and not config.endpoint_enabled(spec):
            continue
        values = {
            "name": spec.get("name", "candidate-" + str(index + 1)),
            "model": _value(spec, "model"),
            "url": _value(spec, "url"),
            "api_key": _value(spec, "api_key"),
            "timeout": float(spec.get("timeout", _DEFAULT_TIMEOUT)),
            "attempts": int(spec.get("attempts", _DEFAULT_ATTEMPTS)),
            "dimension": int(spec.get("dimension", _DEFAULT_DIMENSION)),
        }
        if spec.get("append_path") and values["url"]:
            values["url"] = values["url"].rstrip("/") + spec["append_path"]
        if not values["url"] and spec.get("base_url_env"):
            base_url = os.getenv(spec["base_url_env"], spec.get("base_url_default", ""))
            if base_url:
                values["url"] = base_url.rstrip("/") + spec.get("path", "")
        if config_module == _CHUNKING_CONFIG_MODULE:
            values.update(config.endpoint_options(spec))
        if not result and primary:
            values.update({key: value for key, value in primary.items() if value is not None})
        result.append(ModelEndpoint(**values))
    if module == "embedding" and result:
        identity = (result[0].model, result[0].dimension)
        for candidate in result:
            if (candidate.model, candidate.dimension) != identity:
                raise ValueError(_MESSAGES["embedding_fallback_error"])
    canonical = _MODULE_ALIASES.get(module, module)
    if routed and canonical in ROUTED_MODULES:
        if config_module == _CHUNKING_CONFIG_MODULE and not any(
            spec.get("enabled", True) and config.endpoint_enabled(spec) for spec in specs
        ):
            return []
        return _routed_chain(canonical, result)
    return result


def chain_fingerprint(module, primary=None, *, routed=None):
    """Non-secret identity for cache invalidation after changing model priority."""
    from hashlib import sha256
    import json
    if routed is None:
        routed = auto_routing_enabled()
    chain = get_model_chain(module, primary, routed=routed)
    identity = [(item.model, item.url, item.dimension) for item in chain]
    if module.startswith(_CHUNKING_PREFIX) or module in _MODULE_ALIASES:
        identity = [
            (
                item.model,
                item.url,
                item.dimension,
                item.max_output_tokens,
                item.token_parameter,
                item.json_mode,
                item.send_temperature,
                item.temperature,
                item.disable_thinking,
                item.image_detail,
                item.timeout,
                item.attempts,
            )
            for item in chain
        ]
    if routed and (module in ROUTED_MODULES or module in _MODULE_ALIASES):
        identity = [
            ROUTING_VERSION,
            [
                (
                    item.name,
                    item.model,
                    item.url,
                    item.timeout,
                    item.attempts,
                    item.profile,
                    item.protocol,
                    item.context_window,
                    item.max_output_tokens,
                    item.token_parameter,
                    item.json_mode,
                    item.send_temperature,
                    item.temperature,
                    item.disable_thinking,
                    item.thinking,
                    item.reasoning_effort,
                    item.image_detail,
                    item.image_token_reserve,
                )
                for item in chain
            ],
        ]
    return sha256(json.dumps(identity).encode("utf-8")).hexdigest()


def model_primary(module):
    chain = get_model_chain(module)
    if not chain:
        raise ModelFallbackError(module, [(_MESSAGES["empty_chain_source"], _MESSAGES["empty_chain_reason"])])
    if auto_routing_enabled() and (module in ROUTED_MODULES or module in _MODULE_ALIASES):
        return next((candidate for candidate in chain if model_configured(candidate)), chain[0])
    return chain[0]
