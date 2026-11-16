"""
Banking Skill Registry (JSON-Driven Architecture).
Design:
- Automatically scans and loads skills from declarative JSON specification files (*_skill.json).
- Zero hardcoded skills list or schemas in Python code.
- Provides skill descriptions and schemas for the ReAct agent.

MANDATORY RULES:
1. NO __INIT__.PY: Direct module imports.
2. NO TYPING LIBRARY: Native Python types only.
3. ZERO TERNARY: Explicit if/else only.
4. 0% REGEX LAW: Absolutely no regular expressions.
5. KEEP IT SIMPLE: Clear, direct, readable code.
"""

import json
from pathlib import Path

from .base_skill import BaseSkill
from agent_banking.core.settings import Config


class SkillRegistry:
    """Central management hub for business skills automatically loaded from JSON."""

    def __init__(self, pipeline_instance=None, config: Config | None = None, skills_dir: str | Path | None = None):
        if config is not None:
            self.config = config
        else:
            self.config = Config()

        self.pipeline_instance = pipeline_instance
        self.skills: dict[str, BaseSkill] = {}

        if skills_dir is not None:
            self.skills_dir = Path(skills_dir)
        else:
            self.skills_dir = Path(__file__).resolve().parent

        self._load_skills_from_json_files()

    def _load_skills_from_json_files(self) -> None:
        """Scan and load all JSON configuration files in the skills directory."""
        if not self.skills_dir.exists():
            return

        json_files = sorted(list(self.skills_dir.glob("*_skill.json")))
        for jf in json_files:
            try:
                skill = BaseSkill.from_json_file(
                    json_path=str(jf),
                    config=self.config,
                    pipeline_instance=self.pipeline_instance
                )
                self.register(skill)
            except Exception as e:
                print(f"⚠️ Failed to load skill from {jf.name}: {e}")

    def register(self, skill: BaseSkill) -> None:
        """Register a skill into the registry."""
        self.skills[skill.metadata.skill_id] = skill

    def get_skill(self, skill_id: str) -> BaseSkill | None:
        """Retrieve skill instance by identifier."""
        return self.skills.get(skill_id)

    def list_skills(self) -> list[dict]:
        """Return summary records of all registered skills."""
        records = []
        for s in self.skills.values():
            meta = s.metadata
            records.append({
                "skill_id": meta.skill_id,
                "display_name": meta.display_name,
                "description": meta.description,
                "version": meta.version,
                "category": meta.category,
                "capabilities": meta.capabilities,
                "triggers": meta.intent_triggers
            })
        return records
