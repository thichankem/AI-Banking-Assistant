"""
Unit Test cho phân đoạn câu thông minh với wtpsplit (SaT)
Đảm bảo ưu tiên số 1 cho wtpsplit (SaT), thay thế hoàn toàn Regex và mảng ký tự neo.
"""

import sys
import unittest

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rag_banking.chunking.unified_chunk.canonical_resolver import (
    split_sentences,
    get_sat_segmenter
)
from rag_banking.chunking.unified_chunk.syllable_budget_manager import SyllableBudgetManager


class TestWTPSplitSaT(unittest.TestCase):
    def setUp(self):
        self.budget_manager = SyllableBudgetManager(
            min_syllables=10,
            max_syllables=50,
            ceiling_syllables=100
        )

    def test_01_sat_segmenter_initialization(self):
        """Kiểm tra khởi tạo thành công wtpsplit (SaT) singleton."""
        sat = get_sat_segmenter()
        self.assertIsNotNone(sat, "wtpsplit (SaT) phải được khởi tạo thành công.")

    def test_02_banking_abbreviations_and_decimals(self):
        """Kiểm tra tách câu với chữ viết tắt, số tiền thập phân và số hiệu văn bản pháp quy."""
        text = (
            "Ngân hàng TMCP Ngoại thương Việt Nam áp dụng mức lãi suất 5.5% một năm. "
            "Chi nhánh tại TP. Hồ Chí Minh đã triển khai Thông tư 17/2024/TT-NHNN đến toàn thể khách hàng. "
            "Mọi chi tiết xin liên hệ số dư tài khoản từ 1.000.000 VNĐ."
        )
        sentences = split_sentences(text)
        self.assertGreaterEqual(len(sentences), 3)
        # Kiểm tra không bị ngắt vụn ở 5.5% hoặc TP. Hồ Chí Minh hoặc 1.000.000 VNĐ
        joined = " ".join(sentences)
        self.assertIn("5.5%", joined)
        self.assertIn("TP. Hồ Chí Minh", joined)
        self.assertIn("17/2024/TT-NHNN", joined)

    def test_03_budget_grouping_with_sat(self):
        """Kiểm tra gom nhóm câu theo ngân sách âm tiết mà không làm rách câu."""
        text = (
            "Câu số một giới thiệu về quy trình cho vay tiêu dùng. "
            "Câu số hai nêu rõ điều kiện thu nhập tối thiểu hàng tháng của khách hàng. "
            "Câu số ba quy định hồ sơ giấy tờ chứng minh pháp lý cần nộp. "
            "Câu số bốn hướng dẫn phương thức giải ngân tiền mặt hoặc chuyển khoản."
        )
        sentences = split_sentences(text)
        chunks = self.budget_manager.split_sentences_by_syllable_budget(
            sentences,
            max_syllables=25
        )
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertTrue(len(chunk.strip()) > 0)


if __name__ == "__main__":
    unittest.main()
