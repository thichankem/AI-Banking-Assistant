"""Load declarative capabilities from trusted project directories."""

from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

from agent_banking.core.config.settings import CoreConfig, get_core_messages

_SPEC_CFG = CoreConfig.CORE_PLUGIN_SPECS
_MESSAGES = get_core_messages("plugin_specs", "core_messages.yaml", "CORE_PLUGIN_MESSAGES")

MAX_PACKAGE_FILE = int(_SPEC_CFG["max_package_file_bytes"])
_MAX_PUBLIC_NAME_LENGTH = int(_SPEC_CFG["max_public_name_length"])
_JSON_SCHEMA_DRAFT = str(_SPEC_CFG["json_schema_draft"])


class PackageYamlLoader(yaml.SafeLoader):
    def compose_node(self, parent, index):
        if self.check_event(yaml.AliasEvent):
            raise ValueError(_MESSAGES["yaml_alias_error"])
        return super().compose_node(parent, index)

    def construct_mapping(self, node, deep=False):
        result = super().construct_mapping(node, deep=deep)
        if len(result) != len(node.value):
            raise ValueError(_MESSAGES["duplicate_yaml_error"])
        return result


def load_yaml(text: str) -> dict:
    try:
        data = yaml.load(text, Loader=PackageYamlLoader)
    except yaml.YAMLError as error:
        raise ValueError(_MESSAGES["invalid_yaml_error"]) from error
    if not isinstance(data, dict):
        raise ValueError(_MESSAGES["expected_mapping_error"])
    return data


def confined_file(root: Path, relative: str) -> Path:
    """The caller supplies a canonical root. Reject links and path traversal."""
    path = root / relative
    if not path.is_relative_to(root) or ".." in path.parts or path.resolve(strict=True) != path or not path.is_file():
        raise ValueError(_MESSAGES["outside_package_error"])
    return path


def read_resource(root: Path, relative: str) -> str:
    path = confined_file(root, relative)
    with path.open("rb") as stream:
        content = stream.read(MAX_PACKAGE_FILE + 1)
    if len(content) > MAX_PACKAGE_FILE:
        raise ValueError(_MESSAGES["resource_too_large_error"])
    return content.decode("utf-8-sig")


def public_name(name) -> bool:
    return (
        isinstance(name, str) and name.isascii() and name.isidentifier()
        and not name.startswith("_") and len(name) <= _MAX_PUBLIC_NAME_LENGTH
    )


def check_parameters(schema: dict) -> None:
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError(_MESSAGES["invalid_schema_type_error"])
    draft = _JSON_SCHEMA_DRAFT
    if schema.get("$schema", draft) not in (draft, draft + "#"):
        raise ValueError(_MESSAGES["unsupported_schema_draft_error"])

    def check_refs(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("$ref", "$dynamicRef") and not str(item).startswith("#"):
                    raise ValueError(_MESSAGES["external_ref_error"])
                check_refs(item)
        elif isinstance(value, list):
            for item in value:
                check_refs(item)

    check_refs(schema)
    Draft202012Validator.check_schema(schema)
