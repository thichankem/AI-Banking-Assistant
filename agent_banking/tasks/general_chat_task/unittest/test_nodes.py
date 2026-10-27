"""
Kiểm thử Đơn vị (Unit Test) từng Node kế thừa trong General Chat Task.
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_banking.tasks.general_chat_task.task import GeneralChatTask


def test_individual_nodes():
    task = GeneralChatTask()

    # 1. Test Node Input Guard
    safe_res = task.node_input_guard("Chào buổi sáng")
    assert safe_res["is_safe"] is True

    # 2. Test Node Output Format
    out_res = task.node_format_output(answer="Xin chào Quý khách!")
    assert out_res.status == "SUCCESS"
    assert "Xin chào" in out_res.answer

    print("✅ [UNIT TEST PASS] Toàn bộ các Node kế thừa của GeneralChatTask hoạt động chính xác!")


if __name__ == "__main__":
    test_individual_nodes()
