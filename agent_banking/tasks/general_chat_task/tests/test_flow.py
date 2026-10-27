"""
Kiểm thử Luồng Tích hợp (Integration Test) cho General Chat Task.
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_banking.tasks.general_chat_task.task import GeneralChatTask


def test_general_chat_flow():
    task = GeneralChatTask()
    res = task.execute("Xin chào bạn, bạn là ai?")

    assert res.task_id == "TASK_GENERAL_CHAT"
    assert res.status == "SUCCESS"
    assert len(res.execution_steps) >= 3
    assert "Vietcombank" in res.answer
    print("✅ [TEST FLOW PASS] GeneralChatTask thực thi thành công!")


if __name__ == "__main__":
    test_general_chat_flow()
