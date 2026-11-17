"""Load declarative capabilities from trusted project directories."""

from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

MAX_PACKAGE_FILE = 65536


class PackageYamlLoader(yaml.SafeLoader):
    def compose_node(self, parent, index):
        if self.check_event(yaml.AliasEvent):
            raise ValueError("YAML aliases are not supported in package definitions")
        return super().compose_node(parent, index)

    def construct_mapping(self, node, deep=False):
        result = super().construct_mapping(node, deep=deep)
        if len(result) != len(node.value):
            raise ValueError("Duplicate YAML field")
        return result


def load_yaml(text: str) -> dict:
    try:
        data = yaml.load(text, Loader=PackageYamlLoader)
    except yaml.YAMLError as error:
        raise ValueError("Invalid package YAML") from error
    if not isinstance(data, dict):
        raise ValueError("Expected a YAML mapping")
    return data


def confined_file(root: Path, relative: str) -> Path:
    """The caller supplies a canonical root. Reject links and path traversal."""
    path = root / relative
    if not path.is_relative_to(root) or ".." in path.parts or path.resolve(strict=True) != path or not path.is_file():
        raise ValueError("File is outside the package or is not a regular file")
    return path


def read_resource(root: Path, relative: str) -> str:
    path = confined_file(root, relative)
    with path.open("rb") as stream:
        content = stream.read(MAX_PACKAGE_FILE + 1)
    if len(content) > MAX_PACKAGE_FILE:
        raise ValueError("Package resource is too large")
    return content.decode("utf-8-sig")


def public_name(name) -> bool:
    return (
        isinstance(name, str) and name.isascii() and name.isidentifier()
        and not name.startswith("_") and len(name) <= 96
    )


def check_parameters(schema: dict) -> None:
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("Capability schema must describe an object")
    draft = "https://json-schema.org/draft/2020-12/schema"
    if schema.get("$schema", draft) not in (draft, draft + "#"):
        raise ValueError("Only JSON Schema Draft 2020-12 is supported")

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
