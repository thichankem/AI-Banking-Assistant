"""
Unit & Integration Test cho Phân hệ DLA Chunking (rag_banking/chunking)
Kiểm tra toàn diện 2 luồng cốt lõi:
1. Test DOCXChunker trên tệp .docx (Heading hierarchy, FAQ, Paragraphs, Lists)
2. Test PDFChunker trên tệp .pdf (Digital text & OCR scan)
3. Test LayoutAwareChunker (DLAChunker) điều phối tự động theo đuôi tệp.
4. Kiểm định Schema toàn vẹn (Data Contract Integrity).
"""

import sys
import unittest
from pathlib import Path

# Đảm bảo UTF-8 cho console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_banking.config import Config
from rag_banking.chunking.docx_chunker.docx_chunker import DOCXChunker
from rag_banking.chunking.pdf_chunker.pdf_chunker import PDFChunker
from rag_banking.chunking.unified_chunk.layout_aware_chunker import (
    LayoutAwareChunker,
    DLAChunker
)


class TestDLAChunking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Config()
        cls.dataset_dir = cls.config.CHUNKING_DATASET_DIR
        cls.docx_dir = cls.dataset_dir / "docx"
        cls.pdf_dir = cls.dataset_dir / "pdf"

    def test_01_docx_chunking(self):
        """Kiểm tra DOCXChunker trên tệp Word thực tế của Vietcombank."""
        docx_files = list(self.docx_dir.glob("*.docx"))
        self.assertTrue(len(docx_files) > 0, "Không tìm thấy tệp .docx trong dataset!")

        sample_docx = docx_files[0]
        chunker = DOCXChunker(max_chunk_size=800, chunk_overlap=100)
        chunks = chunker.chunk_docx(sample_docx)

        self.assertGreater(len(chunks), 0, "DOCXChunker không sinh ra chunk nào!")

        # Kiểm tra tính toàn vẹn của Schema
        first_chunk = chunks[0]
        required_keys = ["chunk_id", "source", "chunk_type", "breadcrumb", "content", "word_count"]
        for key in required_keys:
            self.assertIn(key, first_chunk, f"DOCX chunk thiếu key: {key}")
        for forbidden in ["ref", "legal_ref", "raw_content", "char_length", "format"]:
            self.assertNotIn(forbidden, first_chunk, f"DOCX chunk không được chứa trường thừa: {forbidden}")

        self.assertEqual(first_chunk["source"], sample_docx.name)
        self.assertTrue(first_chunk["breadcrumb"], "DOCX chunk phải có breadcrumb!")
        self.assertGreater(first_chunk["word_count"], 0)
        print(f"\n[PASS] DOCX Chunker trích xuất thành công {len(chunks)} chunks từ {sample_docx.name}")

    def test_02_format_guardrail(self):
        """Kiểm tra cơ chế bảo vệ định dạng: Chỉ hỗ trợ DOCX và PDF."""
        dla = DLAChunker()
        unsupported_res = dla.chunk_file("test_matrix.csv")
        self.assertEqual(unsupported_res, [], "LayoutAwareChunker phải từ chối các định dạng không phải docx/pdf!")
        print("\n[PASS] Format Guardrail xác thực thành công: Chỉ hỗ trợ DOCX và PDF.")

    def test_03_pdf_chunking(self):
        """Kiểm tra PDFChunker trên tệp PDF thực tế (Digital hoặc OCR Scan)."""
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        self.assertTrue(len(pdf_files) > 0, "Không tìm thấy tệp .pdf trong dataset!")

        sample_pdf = pdf_files[0]
        chunker = PDFChunker(max_chunk_size=800, chunk_overlap=0)
        chunks = chunker.chunk_pdf(sample_pdf, max_pages=2)

        self.assertGreater(len(chunks), 0, "PDFChunker không sinh ra chunk nào!")

        first_chunk = chunks[0]
        self.assertEqual(first_chunk["source"], sample_pdf.name)
        self.assertIn(first_chunk["chunk_type"], ["pdf_text", "pdf_ocr_scan", "pdf_table"])
        for forbidden in ["ref", "legal_ref", "raw_content", "char_length", "format"]:
            self.assertNotIn(forbidden, first_chunk, f"PDF chunk không được chứa trường thừa: {forbidden}")
        self.assertTrue("page_numbers" in first_chunk or "location" in first_chunk or "Trang" in first_chunk.get("breadcrumb", ""))
        print(f"\n[PASS] PDF Chunker trích xuất thành công {len(chunks)} chunks từ {sample_pdf.name}")

    def test_04_unified_dla_chunker(self):
        """Kiểm tra LayoutAwareChunker (DLAChunker) tự động định tuyến theo extension (.docx, .pdf)."""
        dla = DLAChunker(max_chunk_size=800, chunk_overlap=100)

        # 1. Test định tuyến .docx
        docx_files = list(self.docx_dir.glob("*.docx"))
        if docx_files:
            c_docx = dla.chunk_file(docx_files[0])
            self.assertGreater(len(c_docx), 0)
            self.assertEqual(c_docx[0]["source"], docx_files[0].name)

        # 2. Test định tuyến .pdf
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        if pdf_files:
            c_pdf = dla.chunk_file(pdf_files[0], max_pdf_pages=2)
            self.assertGreater(len(c_pdf), 0)
            self.assertEqual(c_pdf[0]["source"], pdf_files[0].name)

        print("\n[PASS] Unified DLAChunker tự động định tuyến chính xác 2 định dạng cốt lõi: DOCX & PDF!")

    def test_05_document_structure_chunker_strategies(self):
        """Kiểm tra DocumentStructureChunker với 3 chiến lược: Structure, Hierarchical, Section-Length-Aware."""
        from rag_banking.chunking.unified_chunk.structure_chunker import DocumentStructureChunker

        sample_markdown = (
            "# QUY CHẾ DỊCH VỤ NGÂN HÀNG SỐ\n\n"
            "## Chương I. Quy định chung\n\n"
            "### Điều 1. Phạm vi điều chỉnh\n"
            "Quy chế này quy định về việc cung cấp và sử dụng dịch vụ ngân hàng điện tử.\n\n"
            "### Điều 2. Đối tượng áp dụng\n"
            "Áp dụng cho mọi khách hàng cá nhân và tổ chức mở tài khoản thanh toán tại ngân hàng.\n\n"
            "## Chương II. Hạn mức giao dịch\n\n"
            "### Điều 3. Hạn mức eKYC\n"
            "Hạn mức giao dịch chuyển tiền qua kênh ngân hàng số dành cho khách hàng xác thực eKYC "
            "không vượt quá 100.000.000 VND một tháng theo quy định của Ngân hàng Nhà nước.\n\n"
            "| Gói dịch vụ | Hạn mức ngày | Hạn mức giao dịch |\n"
            "| --- | --- | --- |\n"
            "| Cơ bản | 50.000.000 VND | 20.000.000 VND |\n"
            "| Nâng cao | 500.000.000 VND | 100.000.000 VND |\n"
        )

        chunker = DocumentStructureChunker(chunk_size=300, min_section_size=30, chunk_overlap=20)
        sub_sections = chunker.parse_structure(sample_markdown, source_name="quy_che_sample.docx")
        self.assertGreater(len(sub_sections), 0, "Không phân tích được văn bản cấu trúc thành sub-sections!")

        # 1. Test Strategy 1: Structure-based chunking
        s1_chunks = chunker.chunk_strategy_1_structure_based(sub_sections, source_name="quy_che_sample.docx")
        self.assertGreater(len(s1_chunks), 0)
        self.assertEqual(s1_chunks[0]["strategy"], "strategy_1_structure_based")
        has_table = any(c.get("chunk_type") == "table" for c in s1_chunks)
        self.assertTrue(has_table, "Strategy 1 phải tách riêng chunk bảng!")

        # 2. Test Strategy 2: Hierarchical chunking (Parent - Child)
        s2_chunks = chunker.chunk_strategy_2_hierarchical(sub_sections, source_name="quy_che_sample.docx")
        self.assertGreater(len(s2_chunks), 0)
        has_child = any(c.get("hierarchy_level") == "child" for c in s2_chunks)
        self.assertTrue(has_child, "Strategy 2 phải sinh ra child chunks!")

        # 3. Test Strategy 3: Section-Length-Aware (Anti-Fragmentation)
        s3_chunks = chunker.chunk_strategy_3_section_length_aware(sub_sections, source_name="quy_che_sample.docx")
        self.assertGreater(len(s3_chunks), 0)
        # Kiểm tra xem Điều 1 và Điều 2 (đều ngắn < 30 từ) có được gộp vào nhau không
        merged_breadcrumb_found = any("&" in c["breadcrumb"] for c in s3_chunks)
        self.assertTrue(merged_breadcrumb_found, "Strategy 3 phải tự động gộp các mục con ngắn để chống phân mảnh!")
        print(f"\n[PASS] DocumentStructureChunker hoàn thành xuất sắc cả 3 chiến lược: S1={len(s1_chunks)}, S2={len(s2_chunks)}, S3={len(s3_chunks)} chunks!")

    def test_06_enterprise_chunk_fields(self):
        """Kiểm tra các trường mở rộng Enterprise: location, parent_id, original_text, token_count, is_active, updated_time."""
        from rag_banking.chunking.unified_chunk.syllable_budget_manager import build_banking_chunk
        chunk = build_banking_chunk(
            chunk_id="test_01",
            source="test.pdf",
            chunk_type="pdf_text",
            breadcrumb="Bieu phi > The",
            section_title="The Infinite",
            content="Noi dung the tin dung",
            original_text="Noi dung the tin dung",
            parent_id="test_parent",
            location=[{"page": 1, "bbox": [10.0, 20.0, 100.0, 200.0]}]
        )
        self.assertIn("parent_id", chunk)
        self.assertEqual(chunk["parent_id"], "test_parent")
        self.assertIn("location", chunk)
        self.assertEqual(len(chunk["location"]), 1)
        self.assertEqual(chunk["location"][0]["page"], 1)
        self.assertIn("original_text", chunk)
        self.assertIn("token_count", chunk)
        self.assertGreater(chunk["token_count"], 0)
        self.assertIn("is_active", chunk)
        self.assertTrue(chunk["is_active"])
        self.assertIn("updated_time", chunk)
        self.assertGreater(chunk["updated_time"], 0)
        print("\n[PASS] Enterprise chunk fields (location, parent_id, original_text, token_count, is_active, updated_time) xác thực thành công!")


if __name__ == "__main__":
    unittest.main()
