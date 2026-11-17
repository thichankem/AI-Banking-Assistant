"""Read Agent Skills packages from a confined, project-owned directory."""

import os
from pathlib import Path

from agent_banking.core.plugin_specs import MAX_PACKAGE_FILE, load_yaml
MAX_FRONTMATTER = 16384
FIELDS = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}


def valid_name(value) -> bool:
    return (isinstance(value, str) and 1 <= len(value) <= 64 and value == value.lower()
            and not value.startswith("-") and not value.endswith("-") and "--" not in value
            and all(char.isalnum() or char == "-" for char in value))


def read_skill(path: Path, *, body: bool = False, expected=None) -> tuple:
    with path.open(encoding="utf-8-sig") as stream:
        stat = os.fstat(stream.fileno())
        revision = [stat.st_size, stat.st_mtime_ns]
        if stat.st_size > MAX_PACKAGE_FILE or (expected is not None and revision != expected):
            raise ValueError("Skill is too large or changed since discovery")
        if stream.readline(MAX_FRONTMATTER).strip() != "---":
            raise ValueError("Missing YAML frontmatter")
        lines = []
        consumed = 0
        for line in stream:
            if line.strip() == "---":
                break
            consumed += len(line)
            if consumed > MAX_FRONTMATTER:
                raise ValueError("Skill frontmatter is too large")
            lines.append(line)
        else:
            raise ValueError("Unclosed YAML frontmatter")
        data = load_yaml("".join(lines))
        return data, revision, stream.read(MAX_PACKAGE_FILE).strip() if body else ""


def validate_metadata(data: dict, directory_name: str) -> None:
    if set(data) - FIELDS or data.get("name") != directory_name or not valid_name(directory_name):
        raise ValueError("Invalid Agent Skill name or frontmatter field")
    for key, maximum in (("description", 1024), ("compatibility", 500),
                         ("license", MAX_FRONTMATTER), ("allowed-tools", MAX_FRONTMATTER)):
        if key not in data and key != "description":
            continue
        value = data.get(key)
        if not isinstance(value, str) or not value.strip() or len(value) > maximum:
            raise ValueError("Invalid skill metadata text")
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()):
        raise ValueError("Skill metadata keys and values must be strings")


