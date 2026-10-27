"""
Kiểm thử Luồng Tích hợp (Integration Test) cho Biometric Compliance Task.
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_banking.tasks.biometric_compliance_task.task import BiometricComplianceTask


def test_biometric_compliance_flow():
    task = BiometricComplianceTask()
    res = task.execute("Chuyển tiền 15 triệu có bắt buộc quét khuôn mặt sinh trắc học theo Quyết định 2345 không?")

    assert res.task_id == "TASK_BIOMETRIC_COMPLIANCE"
    assert res.status == "SUCCESS"
    assert len(res.execution_steps) >= 4
    print("✅ [TEST FLOW PASS] BiometricComplianceTask thực thi thành công!")


if __name__ == "__main__":
    test_biometric_compliance_flow()
