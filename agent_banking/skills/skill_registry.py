"""
Danh bạ Kỹ năng Nghiệp vụ Ngân hàng (Skill Registry - JSON-Driven Architecture).
Thiết kế tối giản, trực quan, dễ hiểu theo chuẩn Chunking:
- Tự động quét và nạp toàn bộ các Kỹ năng từ các tệp đặc tả JSON (*.json).
- Không hardcode danh sách kỹ năng hay schema trong code Python.
- Cung cấp mô tả kỹ năng cho Intent Router và cấu trúc tham số cho ReAct Engine.

NGUYÊN TẮC BẮT BUỘC:
1. KHÔNG DÙNG __INIT__.PY: Import trực tiếp file module.
2. KHÔNG DÙNG THƯ VIỆN TYPING: Thuần túy dùng kiểu gốc của Python.
3. ZERO TERNARY: Bắt buộc dùng if/else tường minh, không dùng ?: hoặc toán tử 3 ngôi.
4. 0% REGEX LAW: Tuyệt đối không dùng regex.
5. HẠN CHẾ TỐI ĐA THỪA THÃI: Code ngắn gọn, trực diện, dễ hiểu.
"""

from __future__ import annotations
import json
from pathlib import Path

from .base_skill import BaseSkill, SkillMetadata
from agent_banking.config import Config


class SkillRegistry:
    """Trung tâm quản lý danh mục Kỹ năng nghiệp vụ được nạp tự động từ JSON."""

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
        """Quét và nạp tất cả các tệp cấu hình JSON trong thư mục skills."""
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
                print(f"⚠️ Không thể nạp kỹ năng từ {jf.name}: {e}")

    def register(self, skill: BaseSkill) -> None:
        """Đăng ký một skill vào danh bạ."""
        self.skills[skill.metadata.skill_id] = skill

    def get_skill(self, skill_id: str) -> BaseSkill | None:
        """Lấy đối tượng skill theo mã định danh."""
        return self.skills.get(skill_id)

    def list_skills(self) -> list[dict]:
        """Trả về danh sách tóm tắt tất cả các kỹ năng."""
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

    def get_skills_prompt_description(self) -> str:
        """Xuất mô tả kỹ năng phục vụ Router phân loại Intent."""
        lines = []
        for s in self.skills.values():
            m = s.metadata
            lines.append(f"- [{m.skill_id}] ({m.display_name}): {m.description}")
        return "\n".join(lines)

    def get_skills_react_schema(self) -> str:
        """Xuất danh mục kỹ năng kèm tham số chi tiết phục vụ vòng lặp ReAct hoàn toàn từ JSON."""
        lines = []
        for s in self.skills.values():
            m = s.metadata
            lines.append(f"Kỹ năng: {m.skill_id} ({m.display_name})")
            lines.append(f"Mô tả: {m.description}")
            if m.react_schema_example:
                lines.append(f"Tham số JSON: {m.react_schema_example}")
            else:
                param_props = m.parameters_schema.get("properties", {})
                param_keys = list(param_props.keys())
                example_dict = {}
                for pk in param_keys:
                    example_dict[pk] = f"<{pk}>"
                lines.append(f"Tham số JSON: {json.dumps(example_dict, ensure_ascii=False)}")
            lines.append("")
        return "\n".join(lines)
