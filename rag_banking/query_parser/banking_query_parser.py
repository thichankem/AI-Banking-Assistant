"""
Banking Query Parser (Bộ tiền xử lý và phân tích truy vấn ngân hàng)
Nhiệm vụ Bước 1:
1. Chuẩn hóa các từ viết tắt chuyên ngành ngân hàng (eKYC, TKTT, Smart OTP, VCB Digibank...).
2. Phát hiện các thực thể pháp lý (Điều X, Khoản Y).
3. Trích xuất từ khóa trọng tâm (Keywords) phục vụ thuật toán Sparse BM25.
4. Sinh câu truy vấn mở rộng (Normalized Query) phục vụ Dense Vector Embedding.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# Từ điển ánh xạ từ viết tắt chuyên ngành ngân hàng & pháp lý Vietcombank
# Sắp xếp các cụm dài lên trước để tránh thay thế chèn lặp từ con
BANKING_ACRONYMS: Dict[str, str] = {
    r"\bVCB\s*Digibank\b": "ngân hàng số VCB Digibank (Vietcombank Digibank)",
    r"\bSmart\s*OTP\b": "phương thức xác thực Smart OTP",
    r"\bSMS\s*Banking\b": "dịch vụ tin nhắn SMS Banking",
    r"\bPhone\s*Banking\b": "dịch vụ ngân hàng qua điện thoại Phone Banking",
    r"\beKYC\b": "định danh điện tử (eKYC)",
    r"\bTKTT\b": "tài khoản thanh toán (TKTT)",
    r"\bKHCN\b": "khách hàng cá nhân (KHCN)",
    r"\bKHDN\b": "khách hàng doanh nghiệp (KHDN)",
    r"\bCCCD\b": "căn cước công dân gắn chip (CCCD)",
    r"\bCMND\b": "chứng minh nhân dân (CMND)",
    r"\bTKTG\b": "tài khoản tiền gửi (TKTG)",
    r"\bNHĐT\b": "ngân hàng điện tử (NHĐT)",
    r"\bĐGD\b": "địa điểm giao dịch (ĐGD)",
    r"\bSTH\b": "sinh trắc học nhận diện khuôn mặt (STH)",
    r"\bTDQT\b": "thẻ tín dụng quốc tế (TDQT)",
    r"\bOTP\b": "mã xác thực một lần (OTP)",
    r"\bVNPAY\b": "ví điện tử VNPAY"
}

VIETNAMESE_STOPWORDS = {
    "là", "gì", "như", "thế", "nào", "sao", "cho", "tôi", "hỏi", "được",
    "không", "và", "của", "với", "ở", "tại", "về", "có", "hay", "khi",
    "muốn", "cần", "phải", "để", "những", "các", "này", "đó", "ra", "vào"
}

@dataclass
class ParsedQuery:
    raw_query: str
    normalized_query: str
    keywords: List[str]
    keywords_str: str
    detected_acronyms: List[Tuple[str, str]] = field(default_factory=list)
    legal_references: List[str] = field(default_factory=list)

class BankingQueryParser:
    def __init__(self, acronym_dict: Dict[str, str] = None):
        self.acronym_dict = acronym_dict or BANKING_ACRONYMS

    def normalize_acronyms(self, query: str) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Chuẩn hóa và mở rộng các từ viết tắt chuyên ngành ngân hàng.
        Sử dụng cơ chế single-pass placeholder để chống chèn lặp.
        """
        detected = []
        text = query
        placeholders = {}
        counter = 0

        # Thay thế bằng placeholder để không bị thay thế lồng nhau
        for pattern, replacement in self.acronym_dict.items():
            matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
            if matches:
                for m in reversed(matches):
                    original_term = m.group(0)
                    detected.append((original_term, replacement))
                    ph_key = f"__ACRONYM_PH_{counter}__"
                    placeholders[ph_key] = replacement
                    text = text[:m.start()] + ph_key + text[m.end():]
                    counter += 1

        # Phục hồi placeholder về dạng định nghĩa mở rộng
        for ph_key, rep in placeholders.items():
            text = text.replace(ph_key, rep)

        text = re.sub(r"\s+", " ", text).strip()
        return text, detected

    def extract_legal_entities(self, query: str) -> List[str]:
        """Trích xuất các thực thể pháp lý nếu có trong câu hỏi: Điều X, Khoản Y"""
        refs = []
        dieu_match = re.findall(r"Điều\s+\d+[a-zA-Z]?", query, re.IGNORECASE)
        khoan_match = re.findall(r"Khoản\s+\d+", query, re.IGNORECASE)
        refs.extend(dieu_match)
        refs.extend(khoan_match)
        return list(dict.fromkeys(refs))

    def extract_keywords(self, query: str, normalized_query: str) -> List[str]:
        """
        Trích xuất từ khóa trọng tâm (Keywords) phục vụ BM25:
        Kết hợp các thực thể chuyên ngành từ câu gốc và câu chuẩn hóa.
        """
        words = re.findall(r"[\w]+", normalized_query.lower())
        filtered = [w for w in words if w not in VIETNAMESE_STOPWORDS and len(w) > 1]

        special_tokens = []
        for pat in [r"ekyc", r"tktt", r"smart\s*otp", r"vcb\s*digibank", r"điều\s*\d+", r"khoản\s*\d+"]:
            m = re.findall(pat, query.lower())
            special_tokens.extend(m)

        keywords = []
        for tok in special_tokens + filtered:
            clean_tok = tok.strip()
            if clean_tok and clean_tok not in keywords:
                keywords.append(clean_tok)

        return keywords

    def parse(self, query: str) -> ParsedQuery:
        """
        Thực thi toàn bộ Bước 1: Query Parser
        """
        clean_query = query.strip()
        normalized_query, detected_acronyms = self.normalize_acronyms(clean_query)
        legal_refs = self.extract_legal_entities(clean_query)
        keywords = self.extract_keywords(clean_query, normalized_query)

        return ParsedQuery(
            raw_query=clean_query,
            normalized_query=normalized_query,
            keywords=keywords,
            keywords_str=" ".join(keywords),
            detected_acronyms=detected_acronyms,
            legal_references=legal_refs
        )
