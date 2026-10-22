"""
Banking Query Parser (Bước 1: Tiền Xử Lý & Phân Tích Truy Vấn Ngân Hàng)
Tuân thủ nguyên tắc:
1. 0% REGEX LAW: Tuyệt đối không sử dụng `import re` hay regex methods.
2. CHUẨN HÓA UNICODE NFC: Đảm bảo tiếng Việt từ mọi bộ gõ (Unikey, macOS, mobile) đều về chuẩn NFC.
3. PHÂN TÍCH TỪ VỰNG THUẦN TÚY: Mở rộng từ viết tắt qua ranh giới ký tự cơ sở (str.find, isalnum).
4. TRÍCH XUẤT TỪ KHÓA BM25: Lọc từ dừng (Stopwords) qua tra cứu tập hợp Set O(1) hiệu năng cao.
"""

import unicodedata
from dataclasses import dataclass, field

# Từ điển ánh xạ từ viết tắt chuyên ngành ngân hàng & pháp lý Vietcombank
DEFAULT_BANKING_ACRONYMS = {
    "vcb digibank": "ngân hàng số VCB Digibank (Vietcombank Digibank)",
    "smart otp": "phương thức xác thực Smart OTP",
    "sms banking": "dịch vụ tin nhắn SMS Banking",
    "phone banking": "dịch vụ ngân hàng qua điện thoại Phone Banking",
    "ekyc": "định danh điện tử (eKYC)",
    "tktt": "tài khoản thanh toán (TKTT)",
    "khcn": "khách hàng cá nhân (KHCN)",
    "khdn": "khách hàng doanh nghiệp (KHDN)",
    "cccd": "căn cước công dân gắn chip (CCCD)",
    "cmnd": "chứng minh nhân dân (CMND)",
    "tktg": "tài khoản tiền gửi (TKTG)",
    "nhđt": "ngân hàng điện tử (NHĐT)",
    "đgd": "địa điểm giao dịch (ĐGD)",
    "sth": "sinh trắc học nhận diện khuôn mặt (STH)",
    "tdqt": "thẻ tín dụng quốc tế (TDQT)",
    "otp": "mã xác thực một lần (OTP)",
    "vnpay": "ví điện tử VNPAY"
}

VIETNAMESE_STOPWORDS = {
    "là", "gì", "như", "thế", "nào", "sao", "cho", "tôi", "hỏi", "được",
    "không", "và", "của", "với", "ở", "tại", "về", "có", "hay", "khi",
    "muốn", "cần", "phải", "để", "những", "các", "này", "đó", "ra", "vào",
    "ạ", "vậy", "giúp", "với", "xin", "làm", "xem"
}

@dataclass
class ParsedQuery:
    raw_query: str
    normalized_query: str
    keywords: list
    keywords_str: str
    detected_acronyms: list = field(default_factory=list)
    legal_references: list = field(default_factory=list)

class BankingQueryParser:
    """
    Bộ tiền xử lý truy vấn ngân hàng (0% Regex, Unicode NFC Native).
    """

    def __init__(self, acronym_dict=None):
        raw_dict = acronym_dict or DEFAULT_BANKING_ACRONYMS
        # Chuẩn hóa toàn bộ khóa trong từ điển về Unicode NFC chữ thường
        self.acronym_dict = {
            unicodedata.normalize("NFC", k.strip().lower()): v
            for k, v in raw_dict.items()
        }
        # Sắp xếp các cụm từ dài lên trước để ưu tiên khớp cụm trước từ đơn lẻ
        self.sorted_acronyms = sorted(
            self.acronym_dict.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )

    def normalize_acronyms(self, query):
        """
        Chuẩn hóa và mở rộng các từ viết tắt chuyên ngành ngân hàng (0% Regex).
        Kiểm tra ranh giới từ qua isalnum() để tránh thay thế nhầm chuỗi con.
        """
        text = unicodedata.normalize("NFC", query)
        detected = []

        for phrase, replacement in self.sorted_acronyms:
            p_len = len(phrase)
            start_pos = 0

            while True:
                lower_text = text.lower()
                idx = lower_text.find(phrase, start_pos)
                if idx == -1:
                    break

                # Kiểm tra ranh giới từ trước và sau
                before_ok = (idx == 0 or not text[idx - 1].isalnum())
                after_ok = (idx + p_len == len(text) or not text[idx + p_len].isalnum())

                if before_ok and after_ok:
                    original_token = text[idx : idx + p_len]
                    detected.append((original_token, replacement))
                    text = text[:idx] + replacement + text[idx + p_len:]
                    start_pos = idx + len(replacement)
                else:
                    start_pos = idx + 1

        # Chuẩn hóa khoảng trắng cơ bản
        normalized = " ".join(text.split())
        return normalized, detected

    @staticmethod
    def extract_legal_entities(query):
        """
        Trích xuất các thực thể pháp lý nếu có trong câu hỏi (0% Regex):
        Duyệt chuỗi nhận diện các cụm 'Điều X', 'Khoản Y' tự nhiên.
        """
        words = query.split()
        refs = []
        num_words = len(words)

        for i in range(num_words):
            clean_word = "".join(c for c in words[i] if c.isalnum()).lower()
            if clean_word == "điều" and i + 1 < num_words:
                next_word = "".join(c for c in words[i + 1] if c.isalnum())
                if any(c.isdigit() for c in next_word):
                    refs.append(f"Điều {next_word}")
            elif clean_word == "khoản" and i + 1 < num_words:
                next_word = "".join(c for c in words[i + 1] if c.isalnum())
                if any(c.isdigit() for c in next_word):
                    refs.append(f"Khoản {next_word}")

        # Khử trùng lặp bảo toàn thứ tự
        return list(dict.fromkeys(refs))

    @staticmethod
    def extract_keywords(normalized_query):
        """
        Trích xuất từ khóa trọng tâm (Keywords) phục vụ BM25 (0% Regex):
        1. Làm sạch ký tự dấu câu sang khoảng trắng.
        2. Tách từ và loại bỏ Stopwords qua Set Lookup O(1).
        3. Khử trùng lặp bảo toàn thứ tự xuất hiện.
        """
        text_nfc = unicodedata.normalize("NFC", normalized_query)
        # Chuyển đổi mọi ký tự không phải chữ cái/số thành khoảng trắng
        cleaned_chars = []
        for c in text_nfc:
            if c.isalnum() or c.isspace():
                cleaned_chars.append(c)
            else:
                cleaned_chars.append(" ")
        cleaned_text = "".join(cleaned_chars)

        words = cleaned_text.lower().split()
        keywords = []

        for w in words:
            if len(w) > 1 and w not in VIETNAMESE_STOPWORDS and w not in keywords:
                keywords.append(w)

        return keywords

    def parse(self, query):
        """
        Thực thi toàn bộ Bước 1: Query Parser (0% Regex)
        """
        clean_query = " ".join(unicodedata.normalize("NFC", query.strip()).split())
        normalized_query, detected_acronyms = self.normalize_acronyms(clean_query)
        legal_refs = self.extract_legal_entities(clean_query)
        keywords = self.extract_keywords(normalized_query)

        return ParsedQuery(
            raw_query=clean_query,
            normalized_query=normalized_query,
            keywords=keywords,
            keywords_str=" ".join(keywords),
            detected_acronyms=detected_acronyms,
            legal_references=legal_refs
        )
