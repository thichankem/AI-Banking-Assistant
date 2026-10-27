"""
Kiểm thử Đơn vị (Unit Test) từng Node kế thừa trong Biometric Compliance Task.
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_banking.tasks.biometric_compliance_task.task import BiometricComplianceTask


def test_individual_nodes():
    task = BiometricComplianceTask()

    # 1. Test Node Parameter Extractor
    extracted = task.node_parameter_extractor("Chuyển 25 triệu đồng qua app")
    assert extracted.get("amount_vnd") == 25000000.0

    # 2. Test Node Banking Tool (Compliance QĐ 2345)
    comp_res = task.node_banking_tool("compliance_qd2345", {
        "transaction_amount": 15000000.0,
        "daily_accumulated_amount": 0.0,
        "is_new_device": False
    })
    assert comp_res.get("requires_biometric") is True

    # 3. Test Node Input Guard
    safe_res = task.node_input_guard("Quy định sinh trắc học")
    assert safe_res["is_safe"] is True

    print("✅ [UNIT TEST PASS] Toàn bộ các Node kế thừa của BiometricComplianceTask hoạt động chính xác!")


if __name__ == "__main__":
    test_individual_nodes()
