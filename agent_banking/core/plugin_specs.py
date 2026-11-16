"""Load declarative capabilities from trusted project directories."""

import json
import logging

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

logger = logging.getLogger(__name__)


def public_name(name) -> bool:
    return (
        isinstance(name, str) and name.isascii() and name.isidentifier()
        and not name.startswith("_") and len(name) <= 96
    )


def check_parameters(schema: dict) -> None:
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("Capability parameters must be an object schema")

    def check_refs(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("$ref", "$dynamicRef") and not str(item).startswith("#"):
                    raise ValueError("Only local schema references are supported")
                check_refs(item)
        elif isinstance(value, list):
            for item in value:
                check_refs(item)

    check_refs(schema)
    Draft202012Validator.check_schema(schema)


def load_specs(directory, pattern: str, name_key: str, validate) -> dict:
    specs = {}
    duplicates = set()
    for path in sorted(directory.glob(pattern)):
        if path.name.endswith("_manifest.json") or path.is_symlink() or not path.is_file():
            continue
        try:
            spec = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(spec, dict):
                raise ValueError("Expected a JSON object")
            name = spec.get(name_key, path.stem)
            if not public_name(name):
                raise ValueError("Invalid capability name")
            spec[name_key] = name
            validate(spec)
            if name in specs:
                duplicates.add(name)
                logger.warning("Duplicate capability %s in %s", name, directory.name)
            specs[name] = spec
        except (OSError, UnicodeError, ValueError, SchemaError) as error:
            logger.warning("Skipping %s (%s)", path.name, type(error).__name__)
    return {name: spec for name, spec in specs.items() if name not in duplicates}
