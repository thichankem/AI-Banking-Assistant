"""
Kiểm thử Luồng Tích hợp (Integration Test) cho ReAct Agent Task.
Quy chuẩn Chunking: 0% Regex, Zero Ternary, No Typing Library, No __init__.py.
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_banking.tasks.react_agent_task.task import ReActAgentTask


def test_react_agent_flow():
    task = ReActAgentTask()
    query = "Tôi muốn vay 100 triệu trong 12 tháng với lãi suất 9%/năm, tính giúp tôi tiền lãi tháng đầu tiên?"
    res = task.execute(query)

    assert res.task_id == "TASK_REACT_AGENT"
    assert res.status == "SUCCESS"
    assert len(res.execution_steps) >= 3
    assert len(res.answer) > 0
    print("✅ [TEST FLOW PASS] ReActAgentTask thực thi luồng thành công!")
    print(f"Câu trả lời: {res.answer[:150]}...")


if __name__ == "__main__":
    test_react_agent_flow()
