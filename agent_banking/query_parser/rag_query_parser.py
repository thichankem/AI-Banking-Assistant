"""
Banking Query Parser & Preprocessing Module.
========================================================================
Core Responsibilities:
1. Normalize Vietnamese Unicode to NFC composed standard (resolving NFD and typing glitches).
2. Expand Vietnamese banking acronyms (eKYC, Smart OTP, VCB Digibank, TKTT, STH, NAPAS 247...).
3. Extract legal entities (Articles, Clauses, Points, Circulars, Decisions, Decrees, Laws) with 0% Regex.
4. Detect financial transaction amounts & standardize currency units (10tr -> 10,000,000 VND, 500k, 2 billion).
5. Classify query intent (12 Intents Taxonomy) & route domain (8 Domains) without substring collisions.
6. Extract key BM25 keywords, eliminate decimal fragmentation digits, and filter stopwords.
7. Construct expanded queries optimized for dense vector search models.
8. Optional Semantic LLM Routing Fallback for complex queries.

MANDATORY RULES (GEMINI.md & AGENTS.md):
1. 0% REGEX LAW: String methods only, no regex imports or functions.
2. ZERO TERNARY: Explicit if/else statements only, no ternary operators.
3. NO TYPING LIBRARY: Native Python types only, no typing module imports.
4. NO __INIT__.PY: Direct module imports.
"""

import os
import json
import unicodedata
from dataclasses import dataclass, field

# Vietnamese banking acronym and terminology expansion dictionary (70+ items)
DEFAULT_BANKING_ACRONYMS = {
    # Digital channels, mobile apps, and authentication methods
    "vcb digibank": "ngân hàng số VCB Digibank",
    "vietinbank ipay": "ngân hàng số VietinBank iPay",
    "ipay": "ngân hàng số VietinBank iPay",
    "bidv smartbanking": "ngân hàng số BIDV SmartBanking",
    "smartbanking": "ngân hàng số BIDV SmartBanking",
    "techcombank mobile": "ngân hàng số Techcombank Mobile",
    "vpbank neo": "ngân hàng số VPBank NEO",
    "neo": "ngân hàng số VPBank NEO",
    "mbbank app": "ứng dụng ngân hàng số MBBank",
    "mb bank app": "ứng dụng ngân hàng số MBBank",
    "mb app": "ứng dụng ngân hàng số MBBank",
    "ocb omni": "ngân hàng số OCB OMNI",
    "omni": "ngân hàng số OCB OMNI",
    "acb one": "ngân hàng số ACB ONE",
    "acb online": "ngân hàng số ACB ONE",
    "digimi": "ngân hàng số Digimi của BVBank",
    "smart otp": "phương thức xác thực Smart OTP",
    "smartotp": "phương thức xác thực Smart OTP",
    "soft otp": "phương thức xác thực Smart OTP",
    "softotp": "phương thức xác thực Smart OTP",
    "sms banking": "dịch vụ tin nhắn SMS Banking",
    "phone banking": "dịch vụ ngân hàng qua điện thoại Phone Banking",
    "internet banking": "ngân hàng điện tử Internet Banking",
    "mobile banking": "ứng dụng di động Mobile Banking",
    "ebanking": "ngân hàng điện tử (E-Banking)",
    "e-banking": "ngân hàng điện tử (E-Banking)",
    "ibanking": "ngân hàng điện tử Internet Banking",
    "nhđt": "ngân hàng điện tử (NHĐT)",
    "đgd": "địa điểm giao dịch (ĐGD)",
    "pgd": "phòng giao dịch (PGD)",
    "atm": "máy rút tiền tự động (ATM)",
    "cdm": "máy nộp và rút tiền tự động (CDM)",
    "pos": "máy chấp nhận thanh toán thẻ (POS)",
    "otp": "mã xác thực một lần (OTP)",
    "fido": "phương thức xác thực không mật khẩu FIDO",
    "fido2": "phương thức xác thực không mật khẩu FIDO2",
    "passkey": "phương thức khóa bảo mật Passkey",

    # Identity verification and biometrics
    "ekyc": "định danh điện tử (eKYC)",
    "kyc": "định danh khách hàng (KYC)",
    "sth": "sinh trắc học nhận diện khuôn mặt (STH)",
    "cccd": "căn cước công dân gắn chip (CCCD)",
    "cmnd": "chứng minh nhân dân (CMND)",
    "faceid": "nhận diện khuôn mặt (FaceID)",
    "face id": "nhận diện khuôn mặt (FaceID)",
    "nfc": "kết nối không dây tầm ngắn đọc chip CCCD (NFC)",
    "vneid": "ứng dụng định danh điện tử quốc gia VNeID",

    # Accounts, customer categories, and financial management
    "tktt": "tài khoản thanh toán (TKTT)",
    "tktg": "tài khoản tiền gửi (TKTG)",
    "khcn": "khách hàng cá nhân (KHCN)",
    "khdn": "khách hàng doanh nghiệp (KHDN)",
    "cif": "mã định danh khách hàng (CIF)",
    "casa": "tiền gửi không kỳ hạn (CASA)",
    "stk": "sổ tiết kiệm (STK)",
    "sdkd": "số dư khả dụng (SĐKD)",
    "sđkd": "số dư khả dụng (SĐKD)",
    "hmgd": "hạn mức giao dịch (HMGD)",
    "thấu chi": "thấu chi tài khoản",
    "vãng lai": "tài khoản thanh toán vãng lai",
    "tai khoan vang lai": "tài khoản thanh toán vãng lai",

    # Cards, payments, and e-wallets
    "tdqt": "thẻ tín dụng quốc tế (TDQT)",
    "tdnd": "thẻ tín dụng nội địa (TDNĐ)",
    "tdnđ": "thẻ tín dụng nội địa (TDNĐ)",
    "gnqt": "thẻ ghi nợ quốc tế (GNQT)",
    "gnnd": "thẻ ghi nợ nội địa (GNNĐ)",
    "gnnđ": "thẻ ghi nợ nội địa (GNNĐ)",
    "visa": "thẻ quốc tế Visa",
    "mastercard": "thẻ quốc tế Mastercard",
    "jcb": "thẻ quốc tế JCB",
    "cvv": "mã bảo mật thẻ quốc tế (CVV)",
    "cvc": "mã bảo mật thẻ quốc tế (CVC)",
    "pin": "mã PIN bảo mật thẻ",
    "napas 247": "chuyển tiền nhanh liên ngân hàng NAPAS 247",
    "napas247": "chuyển tiền nhanh liên ngân hàng NAPAS 247",
    "napas": "chuyển tiền liên ngân hàng NAPAS",
    "vietqr": "mã thanh toán quốc gia VietQR",
    "vnpay-qr": "thanh toán mã VNPAY-QR",
    "vnpayqr": "thanh toán mã VNPAY-QR",
    "vnpay": "ví điện tử và cổng thanh toán VNPAY",
    "qr pay": "thanh toán qua mã QR",
    "qr code": "mã phản hồi nhanh QR",
    "payoo": "cổng thanh toán Payoo",
    "momo": "ví điện tử MoMo",
    "zalopay": "ví điện tử ZaloPay",
    "zalo pay": "ví điện tử ZaloPay",
    "viettelmoney": "tiền di động và ví điện tử Viettel Money",
    "viettel money": "tiền di động và ví điện tử Viettel Money",
    "viettelpay": "tiền di động và ví điện tử Viettel Money",
    "viettel pay": "tiền di động và ví điện tử Viettel Money",
    "shopeepay": "ví điện tử ShopeePay",
    "shopee pay": "ví điện tử ShopeePay",
    "airpay": "ví điện tử ShopeePay",
    "apple pay": "phương thức thanh toán Apple Pay",
    "applepay": "phương thức thanh toán Apple Pay",
    "google pay": "phương thức thanh toán Google Pay",
    "googlepay": "phương thức thanh toán Google Pay",
    "samsung pay": "phương thức thanh toán Samsung Pay",
    "samsungpay": "phương thức thanh toán Samsung Pay",

    # Commercial transactions, international trade, credit, and guarantees
    "unc": "ủy nhiệm chi (UNC)",
    "swift": "mã định danh chuyển tiền quốc tế SWIFT/BIC",
    "bic": "mã định danh ngân hàng BIC",
    "iban": "số tài khoản ngân hàng quốc tế IBAN",
    "ttqt": "thanh toán quốc tế (TTQT)",
    "l/c": "thư tín dụng chứng từ (L/C)",
    "lc": "thư tín dụng chứng từ (L/C)",
    "bg": "bảo lãnh ngân hàng (BG)",
    "tsbđ": "tài sản bảo đảm (TSBĐ)",
    "tsbd": "tài sản bảo đảm (TSBĐ)",
    "bđs": "bất động sản (BĐS)",
    "bds": "bất động sản (BĐS)",
    "vld": "vốn lưu động (VLĐ)",
    "vlđ": "vốn lưu động (VLĐ)",
    "dư nợ": "dư nợ tín dụng",
    "nợ xấu": "nợ xấu phân loại CIC",

    # Interest rates and fees
    "ls": "lãi suất (LS)",
    "lscv": "lãi suất cho vay (LSCV)",
    "lstg": "lãi suất tiền gửi (LSTG)",
    "lshd": "lãi suất huy động (LSHĐ)",

    # Business and taxation
    "mst": "mã số thuế (MST)",
    "đkkd": "đăng ký kinh doanh (ĐKKD)",
    "dkkd": "đăng ký kinh doanh (ĐKKD)",
    "bhxh": "bảo hiểm xã hội (BHXH)",
    "vat": "thuế giá trị gia tăng (VAT)",

    # Credit and legal regulations
    "tctd": "tổ chức tín dụng (TCTD)",
    "cic": "trung tâm thông tin tín dụng quốc gia (CIC)",
    "bhtg": "bảo hiểm tiền gửi (BHTG)",
    "nhnn": "Ngân hàng Nhà nước Việt Nam (NHNN)",
    "sbv": "Ngân hàng Nhà nước Việt Nam (SBV)",
    "qđ 2345": "Quyết định 2345/QĐ-NHNN về xác thực sinh trắc học",
    "qd 2345": "Quyết định 2345/QĐ-NHNN về xác thực sinh trắc học",
    "tt 17": "Thông tư 17/2024/TT-NHNN về mở tài khoản thanh toán",
    "tt17": "Thông tư 17/2024/TT-NHNN về mở tài khoản thanh toán",
    "tt 18": "Thông tư 18/2024/TT-NHNN về hoạt động thẻ ngân hàng",
    "tt18": "Thông tư 18/2024/TT-NHNN về hoạt động thẻ ngân hàng",
    "nđ 52": "Nghị định 52/2024/NĐ-CP về thanh toán không dùng tiền mặt",
    "nd 52": "Nghị định 52/2024/NĐ-CP về thanh toán không dùng tiền mặt",
    "luật tctd": "Luật Các tổ chức tín dụng",
    "aml": "phòng chống rửa tiền (AML)",
    "fatca": "đạo luật tuân thủ thuế đối với tài khoản nước ngoài (FATCA)",
    "pep": "người có chức vụ chính trị và người liên quan (PEP)",

    # Common commercial banks (acronyms)
    "vcb": "Vietcombank",
    "tcb": "Techcombank",
    "bidv": "BIDV",
    "ctg": "VietinBank",
    "stb": "Sacombank",
    "vpb": "VPBank",
    "tpb": "TPBank",
    "mbb": "MBBank",
    "acb": "ACB",
    "vib": "VIB",
    "shb": "SHB",
    "msb": "MSB",
    "ocb": "OCB",
    "lpb": "LPBank",
    "ssb": "SeABank",
    "bab": "Bac A Bank",
    "klb": "Kienlongbank",
    "vab": "VietABank",
    "nab": "Nam A Bank",
    "bvb": "BaoViet Bank",
    "sgb": "Saigonbank",
    "pgb": "PGBank",
    "eib": "Eximbank",
    "ncb": "NCB",
    "abb": "ABBank",
    "vbb": "VietBank"
}

# Selected Vietnamese stopwords set
VIETNAMESE_STOPWORDS = {
    "là", "gì", "như", "thế", "nào", "sao", "cho", "tôi", "hỏi", "được",
    "không", "và", "của", "với", "ở", "tại", "về", "có", "hay", "khi",
    "muốn", "cần", "phải", "để", "những", "các", "này", "đó", "ra", "vào",
    "ạ", "vậy", "giúp", "xin", "làm", "xem", "ai", "đâu", "bao", "nhiêu",
    "em", "mình", "bạn", "dạ", "vâng", "nhé", "nha", "đấy", "ơi", "ad",
    "admin", "chi", "thì", "mà", "nhưng", "hoặc", "đến", "từ", "qua",
    "trên", "dưới", "trong", "ngoài", "do", "bởi", "vì", "nên", "mọi",
    "tất", "cả", "kia", "mỗi", "từng", "theo", "căn", "cứ", "hãy",
    "chào", "phiền"
}

# Recognition map for commercial banks, financial institutions, and payment intermediaries (35+ organizations)
BANK_RECOGNITION_MAP = {
    # Big 4
    "vietcombank": "Vietcombank",
    "vcb": "Vietcombank",
    "vietinbank": "VietinBank",
    "ctg": "VietinBank",
    "bidv": "BIDV",
    "agribank": "Agribank",
    "vba": "Agribank",

    # Joint stock commercial banks
    "techcombank": "Techcombank",
    "tcb": "Techcombank",
    "mbbank": "MBBank",
    "mb": "MBBank",
    "mbb": "MBBank",
    "vpbank": "VPBank",
    "vpb": "VPBank",
    "sacombank": "Sacombank",
    "stb": "Sacombank",
    "acb": "ACB",
    "tpbank": "TPBank",
    "tpb": "TPBank",
    "hdbank": "HDBank",
    "hdb": "HDBank",
    "vib": "VIB",
    "shb": "SHB",
    "msb": "MSB",
    "maritimebank": "MSB",
    "ocb": "OCB",
    "phuongdong": "OCB",
    "lpbank": "LPBank",
    "lienvietpostbank": "LPBank",
    "lpb": "LPBank",
    "seabank": "SeABank",
    "ssb": "SeABank",
    "bacabank": "Bac A Bank",
    "bac a bank": "Bac A Bank",
    "bab": "Bac A Bank",
    "kienlongbank": "Kienlongbank",
    "klb": "Kienlongbank",
    "vietabank": "VietABank",
    "viet a bank": "VietABank",
    "vab": "VietABank",
    "namabank": "Nam A Bank",
    "nam a bank": "Nam A Bank",
    "nab": "Nam A Bank",
    "baovietbank": "BaoViet Bank",
    "baoviet bank": "BaoViet Bank",
    "baoviet": "BaoViet Bank",
    "bvb": "BaoViet Bank",
    "saigonbank": "Saigonbank",
    "sgb": "Saigonbank",
    "pgbank": "PGBank",
    "pgb": "PGBank",
    "pvcombank": "PVcomBank",
    "eximbank": "Eximbank",
    "eib": "Eximbank",
    "ncb": "NCB",
    "quocdan": "NCB",
    "abbank": "ABBank",
    "anbinhbank": "ABBank",
    "abb": "ABBank",
    "bvbank": "BVBank",
    "banviet": "BVBank",
    "vietbank": "VietBank",
    "vbb": "VietBank",
    "scb": "SCB",

    # Foreign banks and regulatory authorities
    "nhnn": "NHNN",
    "ngan hang nha nuoc": "NHNN",
    "sbv": "NHNN",
    "state bank": "NHNN",
    "shinhan": "Shinhan Bank",
    "shinhan bank": "Shinhan Bank",
    "woori": "Woori Bank",
    "woori bank": "Woori Bank",
    "hsbc": "HSBC",
    "standard chartered": "Standard Chartered",
    "uob": "UOB",

    # Payment intermediaries and e-wallets
    "napas": "NAPAS",
    "momo": "MoMo",
    "zalopay": "ZaloPay",
    "zalo pay": "ZaloPay",
    "viettelmoney": "ViettelMoney",
    "viettel money": "ViettelMoney",
    "viettelpay": "ViettelMoney",
    "viettel pay": "ViettelMoney",
    "vnpay": "VNPAY",
    "shopeepay": "ShopeePay",
    "shopee pay": "ShopeePay",
    "payoo": "Payoo"
}


@dataclass
class ParsedQuery:
    """
    Banking query parse result contract.
    Maintains 100% backward compatibility with Retrieval, Tracer, and Pipeline modules.
    """
    raw_query: str
    normalized_query: str
    keywords: list
    keywords_str: str
    detected_acronyms: list = field(default_factory=list)
    legal_references: list = field(default_factory=list)
    financial_amounts: list = field(default_factory=list)
    banks_mentioned: list = field(default_factory=list)
    intent: str = "hoi_dap_chung"
    sub_intent: str = "tra_cuu_thong_tin"
    domain: str = "ngan_hang_chung"
    suggested_filters: dict = field(default_factory=dict)
    expanded_query: str = ""

    @property
    def entities(self):
        """Backward-compatible alias for detected_acronyms."""
        return self.detected_acronyms

    def to_dict(self):
        """Export dictionary structure for JSON serialization."""
        return {
            "raw_query": self.raw_query,
            "normalized_query": self.normalized_query,
            "keywords": self.keywords,
            "keywords_str": self.keywords_str,
            "detected_acronyms": self.detected_acronyms,
            "legal_references": self.legal_references,
            "financial_amounts": self.financial_amounts,
            "banks_mentioned": self.banks_mentioned,
            "intent": self.intent,
            "sub_intent": self.sub_intent,
            "domain": self.domain,
            "suggested_filters": self.suggested_filters,
            "expanded_query": self.expanded_query,
            "entities": self.detected_acronyms
        }


class BankingQueryParser:
    """
    Next-generation banking query preprocessor and analyzer (0% Regex, Native Unicode NFC).
    Achieves linear O(N) performance, preventing bottlenecks on queries up to 100,000 characters.
    """

    def __init__(self, acronym_dict=None, use_llm_fallback=False):
        if acronym_dict is not None:
            raw_dict = acronym_dict
        else:
            raw_dict = DEFAULT_BANKING_ACRONYMS

        self.use_llm_fallback = use_llm_fallback

        # Normalize all dictionary keys to lowercase Unicode NFC
        self.acronym_dict = {}
        for k, v in raw_dict.items():
            norm_k = unicodedata.normalize("NFC", k.strip().lower())
            self.acronym_dict[norm_k] = v

        # Sort longer phrases first to prioritize multi-word matches over single words
        self.sorted_acronyms = sorted(
            self.acronym_dict.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )

    def normalize_acronyms(self, query):
        """
        Normalize and expand banking acronyms (0% Regex).
        Explicit word-boundary validation prevents substring collisions.
        Example: 'ekyc' -> 'electronic Know Your Customer (eKYC)'
        """
        if not query:
            return "", []

        text = unicodedata.normalize("NFC", query)
        lower_text = text.lower()
        t_len = len(text)
        matches = []

        occupied = bytearray(t_len)

        for phrase, replacement in self.sorted_acronyms:
            p_len = len(phrase)
            if phrase not in lower_text:
                continue

            pos = 0
            while pos < t_len:
                idx = lower_text.find(phrase, pos)
                if idx == -1:
                    break

                end_idx = idx + p_len

                # 1. Check leading word boundary
                before_ok = False
                if idx == 0:
                    before_ok = True
                else:
                    prev_char = text[idx - 1]
                    if not prev_char.isalnum():
                        if prev_char in ("-", "/") and idx >= 2 and text[idx - 2].isalnum():
                            before_ok = False
                        else:
                            before_ok = True

                # 2. Check trailing word boundary
                after_ok = False
                if end_idx == t_len:
                    after_ok = True
                else:
                    next_char = text[end_idx]
                    if not next_char.isalnum():
                        if next_char in ("-", "/") and end_idx + 1 < t_len and text[end_idx + 1].isalnum():
                            after_ok = False
                        else:
                            after_ok = True

                if before_ok and after_ok:
                    # Check if interval [idx, end_idx) overlaps with previously matched spans (O(1) via bytearray)
                    if b"\x01" not in occupied[idx:end_idx]:
                        occupied[idx:end_idx] = b"\x01" * p_len
                        matches.append((idx, end_idx, phrase, replacement))
                        pos = end_idx
                    else:
                        pos = idx + 1
                else:
                    pos = idx + 1

        if not matches:
            clean_words = text.split()
            normalized = " ".join(clean_words)
            return normalized, []

        # Sort replacement spans by initial appearance order
        matches.sort(key=lambda x: x[0])

        parts = []
        detected = []
        last_pos = 0

        for start_idx, end_idx, phrase, replacement in matches:
            parts.append(text[last_pos:start_idx])
            parts.append(replacement)
            original_token = text[start_idx:end_idx]
            detected.append((original_token, replacement))
            last_pos = end_idx

        parts.append(text[last_pos:])
        joined_text = "".join(parts)
        clean_words = joined_text.split()
        normalized = " ".join(clean_words)
        return normalized, detected

    @staticmethod
    def extract_legal_entities(query):
        """
        Extract banking legal entities and state regulatory citations (0% Regex):
        Recognizes: Articles, Clauses, Points, Chapters, Sections, Appendices, Circulars, Decisions, Decrees, Laws.
        Supports both abbreviated and full citation formats (e.g. TT 17, QD 2345/QD-NHNN, ND 52/2024/ND-CP).
        """
        if not query:
            return []

        text = unicodedata.normalize("NFC", query)
        words = text.split()
        refs = []
        num_words = len(words)

        for i in range(num_words):
            raw_w = words[i]
            clean_token = raw_w.strip(",.;:()[]{}\"'“”‘’!?")
            clean_lower = clean_token.lower()

            # 1. Article X (e.g., Article 15, Article 3a, Article 12) and Article X and Y
            if clean_lower in ("điều", "dieu"):
                if i + 1 < num_words:
                    next_raw = words[i + 1].strip(",.;:()[]{}\"'“”‘’!?")
                    has_digit = False
                    for ch in next_raw:
                        if ch.isdigit():
                            has_digit = True
                            break
                    if has_digit:
                        refs.append(f"Điều {next_raw}")

                    # Handle multi-article citation: e.g. "Article 15 and 16"
                    if i + 3 < num_words:
                        sep_word = words[i + 2].strip(",.;:()[]{}\"'“”‘’!?").lower()
                        if sep_word in ("và", "va", "đến", "den", "-"):
                            comp_raw = words[i + 3].strip(",.;:()[]{}\"'“”‘’!?")
                            comp_digit = False
                            for ch in comp_raw:
                                if ch.isdigit():
                                    comp_digit = True
                                    break
                            if comp_digit:
                                refs.append(f"Điều {comp_raw}")

            # 2. Clause Y (e.g., Clause 3, Clause 2) and Clause Y and Z
            elif clean_lower in ("khoản", "khoan"):
                if i + 1 < num_words:
                    next_raw = words[i + 1].strip(",.;:()[]{}\"'“”‘’!?")
                    has_digit = False
                    for ch in next_raw:
                        if ch.isdigit():
                            has_digit = True
                            break
                    if has_digit:
                        refs.append(f"Khoản {next_raw}")

                    # Handle multi-clause citation: e.g. "Clause 2 and 3"
                    if i + 3 < num_words:
                        sep_word = words[i + 2].strip(",.;:()[]{}\"'“”‘’!?").lower()
                        if sep_word in ("và", "va", "đến", "den", "-"):
                            comp_raw = words[i + 3].strip(",.;:()[]{}\"'“”‘’!?")
                            comp_digit = False
                            for ch in comp_raw:
                                if ch.isdigit():
                                    comp_digit = True
                                    break
                            if comp_digit:
                                refs.append(f"Khoản {comp_raw}")

            # 3. Point Z (e.g., Point a, Point b, Point 1)
            elif clean_lower in ("điểm", "diem"):
                if i + 1 < num_words:
                    next_raw = words[i + 1].strip(",.;:()[]{}\"'“”‘’!?")
                    if len(next_raw) <= 3 and next_raw.isalnum():
                        refs.append(f"Điểm {next_raw}")

            # 4. Chapter N (e.g., Chapter II, Chapter 2)
            elif clean_lower in ("chương", "chuong"):
                if i + 1 < num_words:
                    next_raw = words[i + 1].strip(",.;:()[]{}\"'“”‘’!?")
                    if len(next_raw) <= 5:
                        refs.append(f"Chương {next_raw}")

            # 5. Section M (e.g., Section 1, Section I)
            elif clean_lower in ("mục", "muc"):
                if i + 1 < num_words:
                    next_raw = words[i + 1].strip(",.;:()[]{}\"'“”‘’!?")
                    if len(next_raw) <= 5 and next_raw.isalnum():
                        refs.append(f"Mục {next_raw}")

            # 6. Appendix K (e.g., Appendix 01, Appendix 1)
            elif clean_lower in ("phụ", "phu"):
                if i + 1 < num_words:
                    w2 = words[i + 1].strip(",.;:()[]{}\"'“”‘’!?").lower()
                    if w2 in ("lục", "luc"):
                        if i + 2 < num_words:
                            next_raw = words[i + 2].strip(",.;:()[]{}\"'“”‘’!?")
                            if next_raw and len(next_raw) <= 10:
                                refs.append(f"Phụ lục {next_raw}")

            # 7. Circular (e.g., Circular 17, TT 17, TT17, TT-17, TT39/2016/TT-NHNN)
            elif clean_lower in ("thông", "thong"):
                if i + 1 < num_words:
                    w2 = words[i + 1].strip(",.;:()[]{}\"'“”‘’!?").lower()
                    if w2 in ("tư", "tu"):
                        target_idx = i + 2
                        if target_idx < num_words:
                            w3 = words[target_idx].strip(",.;:()[]{}\"'“”‘’!?").lower()
                            if w3 in ("số", "so"):
                                target_idx = target_idx + 1
                        if target_idx < num_words:
                            ref_code = words[target_idx].strip(",.;:()[]{}\"'“”‘’!?")
                            has_digit = False
                            for ch in ref_code:
                                if ch.isdigit():
                                    has_digit = True
                                    break
                            if has_digit:
                                refs.append(f"Thông tư {ref_code}")
            elif clean_lower in ("tt", "tt-nhnn"):
                target_idx = i + 1
                if target_idx < num_words:
                    w2 = words[target_idx].strip(",.;:()[]{}\"'“”‘’!?").lower()
                    if w2 in ("số", "so"):
                        target_idx = target_idx + 1
                if target_idx < num_words:
                    ref_code = words[target_idx].strip(",.;:()[]{}\"'“”‘’!?")
                    has_digit = False
                    for ch in ref_code:
                        if ch.isdigit():
                            has_digit = True
                            break
                    if has_digit:
                        refs.append(f"Thông tư {ref_code}")
            elif clean_lower.startswith("tt") and len(clean_token) > 2:
                # Check concatenated formats: TT17, TT39/2016/TT-NHNN, TT-17, TT.17, TT/17
                code_part = ""
                if clean_token[2].isdigit():
                    code_part = clean_token[2:]
                elif clean_token[2] in ("-", ".", "/", "_") and len(clean_token) > 3:
                    if clean_token[3].isdigit():
                        code_part = clean_token[3:]
                if code_part:
                    refs.append(f"Thông tư {code_part}")

            # 8. Decision (e.g., Decision 2345, QD 2345, QD2345, QD-2345, QD2345/QD-NHNN)
            elif clean_lower in ("quyết", "quyet"):
                if i + 1 < num_words:
                    w2 = words[i + 1].strip(",.;:()[]{}\"'“”‘’!?").lower()
                    if w2 in ("định", "dinh"):
                        target_idx = i + 2
                        if target_idx < num_words:
                            w3 = words[target_idx].strip(",.;:()[]{}\"'“”‘’!?").lower()
                            if w3 in ("số", "so"):
                                target_idx = target_idx + 1
                        if target_idx < num_words:
                            ref_code = words[target_idx].strip(",.;:()[]{}\"'“”‘’!?")
                            has_digit = False
                            for ch in ref_code:
                                if ch.isdigit():
                                    has_digit = True
                                    break
                            if has_digit:
                                refs.append(f"Quyết định {ref_code}")
            elif clean_lower in ("qđ", "qd", "qđ-nhnn", "qd-nhnn"):
                target_idx = i + 1
                if target_idx < num_words:
                    w2 = words[target_idx].strip(",.;:()[]{}\"'“”‘’!?").lower()
                    if w2 in ("số", "so"):
                        target_idx = target_idx + 1
                if target_idx < num_words:
                    ref_code = words[target_idx].strip(",.;:()[]{}\"'“”‘’!?")
                    has_digit = False
                    for ch in ref_code:
                        if ch.isdigit():
                            has_digit = True
                            break
                    if has_digit:
                        refs.append(f"Quyết định {ref_code}")
            elif (clean_lower.startswith("qđ") or clean_lower.startswith("qd")) and len(clean_token) > 2:
                # Check concatenated formats: QD2345, QD2345/QD-NHNN, QD-2345, QD.2345
                code_part = ""
                if clean_token[2].isdigit():
                    code_part = clean_token[2:]
                elif clean_token[2] in ("-", ".", "/", "_") and len(clean_token) > 3:
                    if clean_token[3].isdigit():
                        code_part = clean_token[3:]
                if code_part:
                    refs.append(f"Quyết định {code_part}")

            # 9. Decree (e.g., Decree 52, ND 52, ND52, ND-52, ND52/2024/ND-CP)
            elif clean_lower in ("nghị", "nghi"):
                if i + 1 < num_words:
                    w2 = words[i + 1].strip(",.;:()[]{}\"'“”‘’!?").lower()
                    if w2 in ("định", "dinh"):
                        target_idx = i + 2
                        if target_idx < num_words:
                            w3 = words[target_idx].strip(",.;:()[]{}\"'“”‘’!?").lower()
                            if w3 in ("số", "so"):
                                target_idx = target_idx + 1
                        if target_idx < num_words:
                            ref_code = words[target_idx].strip(",.;:()[]{}\"'“”‘’!?")
                            has_digit = False
                            for ch in ref_code:
                                if ch.isdigit():
                                    has_digit = True
                                    break
                            if has_digit:
                                refs.append(f"Nghị định {ref_code}")
            elif clean_lower in ("nđ", "nd", "nđ-cp", "nd-cp"):
                target_idx = i + 1
                if target_idx < num_words:
                    w2 = words[target_idx].strip(",.;:()[]{}\"'“”‘’!?").lower()
                    if w2 in ("số", "so"):
                        target_idx = target_idx + 1
                if target_idx < num_words:
                    ref_code = words[target_idx].strip(",.;:()[]{}\"'“”‘’!?")
                    has_digit = False
                    for ch in ref_code:
                        if ch.isdigit():
                            has_digit = True
                            break
                    if has_digit:
                        refs.append(f"Nghị định {ref_code}")
            elif (clean_lower.startswith("nđ") or clean_lower.startswith("nd")) and len(clean_token) > 2:
                # Check concatenated formats: ND52, ND52/2024/ND-CP, ND-52, ND.52
                code_part = ""
                if clean_token[2].isdigit():
                    code_part = clean_token[2:]
                elif clean_token[2] in ("-", ".", "/", "_") and len(clean_token) > 3:
                    if clean_token[3].isdigit():
                        code_part = clean_token[3:]
                if code_part:
                    refs.append(f"Nghị định {code_part}")

            # 10. Directive (e.g., Directive 01, CT 01, CT01, CT-01)
            elif clean_lower in ("chỉ", "chi"):
                if i + 1 < num_words:
                    w2 = words[i + 1].strip(",.;:()[]{}\"'“”‘’!?").lower()
                    if w2 in ("thị", "thi"):
                        if i + 2 < num_words:
                            ref_code = words[i + 2].strip(",.;:()[]{}\"'“”‘’!?")
                            has_digit = False
                            for ch in ref_code:
                                if ch.isdigit():
                                    has_digit = True
                                    break
                            if has_digit:
                                refs.append(f"Chỉ thị {ref_code}")
            elif clean_lower.startswith("ct") and len(clean_token) > 2:
                code_part = ""
                if clean_token[2].isdigit():
                    code_part = clean_token[2:]
                elif clean_token[2] in ("-", ".", "/") and len(clean_token) > 3:
                    if clean_token[3].isdigit():
                        code_part = clean_token[3:]
                if code_part:
                    refs.append(f"Chỉ thị {code_part}")

        # Check key statutory law and ordinance catalog
        lower_q = text.lower()
        if "luật các tổ chức tín dụng" in lower_q or "luật tctd" in lower_q or "luật các tctd" in lower_q or "32/2024/qh15" in lower_q:
            refs.append("Luật Các tổ chức tín dụng")
        if "luật phòng chống rửa tiền" in lower_q or "luật phòng, chống rửa tiền" in lower_q or "phòng chống rửa tiền" in lower_q or "phòng, chống rửa tiền" in lower_q or "14/2022/qh15" in lower_q or "aml" in lower_q:
            refs.append("Luật Phòng chống rửa tiền")
        if "luật ngân hàng nhà nước" in lower_q or "luật nhnn" in lower_q or "46/2010/qh12" in lower_q:
            refs.append("Luật Ngân hàng Nhà nước")
        if "luật giao dịch điện tử" in lower_q or "20/2023/qh15" in lower_q or "giao dịch điện tử" in lower_q:
            refs.append("Luật Giao dịch điện tử")
        if "luật các công cụ chuyển nhượng" in lower_q or "công cụ chuyển nhượng" in lower_q or "luật cccn" in lower_q:
            refs.append("Luật Các công cụ chuyển nhượng")
        if "luật bảo hiểm tiền gửi" in lower_q or "bảo hiểm tiền gửi" in lower_q or "luật bhtg" in lower_q:
            refs.append("Luật Bảo hiểm tiền gửi")
        if "luật căn cước" in lower_q or "26/2023/qh15" in lower_q:
            refs.append("Luật Căn cước")
        if "pháp lệnh ngoại hối" in lower_q or "phap lenh ngoai hoi" in lower_q:
            refs.append("Pháp lệnh Ngoại hối")

        # Deduplicate while preserving original order of appearance
        unique_refs = []
        for r in refs:
            if r not in unique_refs:
                unique_refs.append(r)
        return unique_refs

    @staticmethod
    def _parse_amount_value(num_str, multiplier):
        """
        Convert numeric string (including decimals 1.5, 2.5 or thousands separators 10,000,000)
        combined with multiplier ('k', 'm', 'tr', 'million', 'billion') to float.
        """
        clean = num_str.strip()
        if not clean:
            return None

        dot_count = clean.count(".")
        comma_count = clean.count(",")

        # Multiple thousands separators (e.g., 10.000.000 or 10,000,000)
        if dot_count > 1 or comma_count > 1:
            raw_digits = clean.replace(".", "").replace(",", "")
            if raw_digits.isdigit():
                return int(raw_digits) * multiplier
            return None

        # Exactly one dot or comma separator
        if dot_count == 1 or comma_count == 1:
            sep = "."
            if comma_count == 1:
                sep = ","
            parts = clean.split(sep)
            if len(parts) == 2:
                left = parts[0]
                right = parts[1]
                if left.isdigit() and right.isdigit():
                    # Decimal combined with multiplier (e.g., 1.5 million, 2.5 billion, 0.5k)
                    if multiplier > 1 and len(right) <= 2:
                        val_float = float(f"{left}.{right}")
                        return int(round(val_float * multiplier))
                    # Three-digit thousands separator without large multiplier (e.g., 500.000 d)
                    if len(right) == 3 and multiplier == 1:
                        return int(left + right)
                    if multiplier > 1:
                        val_float = float(f"{left}.{right}")
                        return int(round(val_float * multiplier))

        # Plain integer without separators
        if clean.isdigit():
            return int(clean) * multiplier

        return None

    @staticmethod
    def extract_financial_amounts(query):
        """
        Extract financial amounts & currency formats in banking queries (0% Regex):
        Recognizes attached unit formats (10tr, 1.5tr, 500k, 2 ty) and spaced numbers
        (20.000.000 VND, 50 trieu dong).
        Returns list of dicts with float amount and standardized string representation.
        """
        if not query:
            return []

        words = query.lower().split()
        amounts = []
        num_words = len(words)

        for i in range(num_words):
            w = words[i].strip(",.;:()[]{}\"'“”‘’!?")
            if not w:
                continue

            # Case 1: Number concatenated with unit (e.g., 10tr, 1.5tr, 500k, 2ty, 2.5ty, 50m, 20000000vnd)
            cut_idx = -1
            for ci, char in enumerate(w):
                if not char.isdigit() and char not in (".", ","):
                    cut_idx = ci
                    break
            if cut_idx > 0:
                raw_num_part = w[:cut_idx]
                unit_part = w[cut_idx:].strip()
                multiplier = 0
                if unit_part in ("tr", "trieu", "m"):
                    multiplier = 1_000_000
                elif unit_part in ("k", "nghin", "ngan"):
                    multiplier = 1_000
                elif unit_part in ("ty", "b"):
                    multiplier = 1_000_000_000
                elif unit_part in ("vnd", "vnđ", "dong", "đồng", "đ"):
                    multiplier = 1

                if multiplier > 0:
                    num_val = BankingQueryParser._parse_amount_value(raw_num_part, multiplier)
                    if num_val is not None and num_val > 0:
                        if multiplier == 1:
                            if num_val >= 1000:
                                formatted_str = f"{num_val:,}".replace(",", ".") + " VNĐ"
                                amounts.append({
                                    "raw": w,
                                    "value": num_val,
                                    "formatted": formatted_str
                                })
                        else:
                            formatted_str = f"{num_val:,}".replace(",", ".") + " VNĐ"
                            amounts.append({
                                "raw": w,
                                "value": num_val,
                                "formatted": formatted_str
                            })

            # Case 2: Number separated from unit (e.g., "10 million", "1.5 million", "500 thousand", "2.5 billion", "20,000,000 VND")
            if i + 1 < num_words:
                next_w = words[i + 1].strip(",.;:()[]{}\"'“”‘’!?")
                multiplier = 0
                if next_w in ("triệu", "trieu", "tr"):
                    multiplier = 1_000_000
                elif next_w in ("tỷ", "ty"):
                    multiplier = 1_000_000_000
                elif next_w in ("nghìn", "ngàn", "nghin", "ngan", "k"):
                    multiplier = 1_000
                elif next_w in ("đ", "dong", "đồng", "vnd", "vnđ"):
                    multiplier = 1

                if multiplier > 0:
                    num_val = BankingQueryParser._parse_amount_value(w, multiplier)
                    if num_val is not None and num_val > 0:
                        if multiplier == 1:
                            if num_val >= 1000:
                                raw_str = f"{w} {next_w}"
                                formatted_str = f"{num_val:,}".replace(",", ".") + " VNĐ"
                                amounts.append({
                                    "raw": raw_str,
                                    "value": num_val,
                                    "formatted": formatted_str
                                })
                        else:
                            raw_str = f"{w} {next_w}"
                            formatted_str = f"{num_val:,}".replace(",", ".") + " VNĐ"
                            amounts.append({
                                "raw": raw_str,
                                "value": num_val,
                                "formatted": formatted_str
                            })

        # Deduplicate by numeric value
        unique_amounts = []
        seen_vals = set()
        for a in amounts:
            val = a["value"]
            if val not in seen_vals:
                seen_vals.add(val)
                unique_amounts.append(a)
        return unique_amounts

    @staticmethod
    def extract_bank_mentions(query):
        """
        Identify commercial bank and payment intermediary names mentioned in the query
        (e.g., Vietcombank, VCB, Techcombank, TCB, MB, MoMo, ZaloPay, NAPAS...).
        """
        if not query:
            return []

        lower_q = query.lower()
        banks = []
        q_len = len(lower_q)

        for code, bname in BANK_RECOGNITION_MAP.items():
            pos = 0
            p_len = len(code)
            while pos < q_len:
                idx = lower_q.find(code, pos)
                if idx == -1:
                    break

                before_ok = False
                if idx == 0:
                    before_ok = True
                else:
                    prev_c = lower_q[idx - 1]
                    if not prev_c.isalnum():
                        before_ok = True

                after_ok = False
                if idx + p_len == q_len:
                    after_ok = True
                else:
                    next_c = lower_q[idx + p_len]
                    if not next_c.isalnum():
                        after_ok = True

                if before_ok and after_ok:
                    if bname not in banks:
                        banks.append(bname)
                    break
                else:
                    pos = idx + 1

        return banks

    @staticmethod
    def classify_intent_and_domain(normalized_query, detected_acronyms=None, legal_refs=None):
        """
        Classify intent (12 Intents Taxonomy) and banking domain (8 Domains) (0% Regex):
        Applies keyword and context matching with strict precedence rules.
        """
        lower_q = normalized_query.lower()

        # Tokenize individual words for independent matching, preventing substring collisions
        words_list = lower_q.split()
        words_set = set()
        for w in words_list:
            clean_w = w.strip(",.;:()[]{}\"'“”‘’!?")
            if clean_w:
                words_set.add(clean_w)

        intent = "hoi_dap_chung"
        sub_intent = "tra_cuu_thong_tin"
        domain = "ngan_hang_chung"
        suggested_filters = {}

        # 1. PRIORITY 1: Report fraud, scams, money loss, emergency card lock (bao_mat_bao_cao_gian_lan)
        is_emergency_scam = False
        scam_phrases = (
            "lừa đảo", "lua dao", "gian lận", "gian lan", "bị lừa", "bi lua",
            "mất tiền", "mat tien", "lộ mật khẩu", "lo mat khau", "hack",
            "mạo danh", "chiếm đoạt", "kẻ gian", "ke gian", "giao dịch lạ"
        )
        for p in scam_phrases:
            if p in lower_q:
                is_emergency_scam = True
                break

        if is_emergency_scam:
            intent = "bao_mat_bao_cao_gian_lan"
            domain = "an_toan_bao_mat"
            suggested_filters["priority"] = "urgent"
            if "khóa" in lower_q or "khoa" in lower_q:
                sub_intent = "xu_ly_su_co_khan_cap"
            elif "lừa đảo" in lower_q or "bị lừa" in lower_q or "lua dao" in lower_q:
                sub_intent = "bao_cao_lua_dao"
            else:
                sub_intent = "canh_bao_bao_mat"

        # 2. Biometrics & eKYC (QD 2345, FaceID, chip-based citizen ID NFC)
        elif any(k in lower_q for k in ("sinh trắc học", "sinh trac hoc", "khuôn mặt", "faceid", "cccd", "gắn chip", "quét nfc", "ekyc", "định danh điện tử", "2345")):
            intent = "sinh_trac_hoc_ekyc"
            domain = "an_toan_bao_mat"
            suggested_filters["domain"] = "sinh_trac_hoc"
            if "cài đặt" in lower_q or "hướng dẫn" in lower_q or "cách" in lower_q:
                sub_intent = "huong_dan_cai_dat"
            elif "hạn mức" in lower_q or "bao nhiêu" in lower_q:
                sub_intent = "kiem_tra_han_muc"
            elif "lỗi" in lower_q or "loi" in lower_q or "không quét" in lower_q:
                sub_intent = "khac_phuc_loi_nfc"
            else:
                sub_intent = "tra_cuu_quy_dinh"

        # 3. Fee schedule lookup (Fee Schedule - suggests chunk_type: table)
        elif any(k in lower_q for k in ("biểu phí", "bieu phi", "bảng phí", "bang phi", "phí dịch vụ", "phi dich vu", "phí thường niên", "phi thuong nien", "phí duy trì", "phi duy tri")):
            intent = "tra_cuu_bieu_phi"
            domain = "bieu_phi_dich_vu"
            suggested_filters["chunk_type"] = "table"
            if "thường niên" in lower_q or "thuong nien" in lower_q:
                sub_intent = "phi_thuong_nien"
            elif "chuyển tiền" in lower_q or "chuyen tien" in lower_q or "swift" in lower_q:
                sub_intent = "phi_chuyen_tien"
            elif "duy trì" in lower_q or "duy tri" in lower_q:
                sub_intent = "phi_duy_tri"
            else:
                sub_intent = "bieu_phi_chung"

        # 4. Statement and transaction history lookup
        elif any(k in lower_q for k in ("sao kê", "sao ke", "in sao kê", "sao kê điện tử", "lịch sử giao dịch", "lich su giao dich", "biên lai")) and not any(k in lower_q for k in ("mở tài khoản", "mo tai khoan")):
            intent = "tra_cuu_sao_ke"
            domain = "dich_vu_tai_khoan"
            suggested_filters["domain"] = "sao_ke"
            if "điện tử" in lower_q or "dien tu" in lower_q or "online" in lower_q:
                sub_intent = "sao_ke_dien_tu"
            elif "tra soát" in lower_q or "tra soat" in lower_q:
                sub_intent = "tra_soat_giao_dich"
            else:
                sub_intent = "in_sao_ke"

        # 5. Bank account (open payment account, account number, lock account)
        # Use independent token lookup via words_set to prevent 'momo' from triggering 'mo'
        elif any(k in lower_q for k in ("tài khoản", "tai khoan", "tktt", "tktg", "số tài khoản", "so tai khoan", "cif", "casa", "vãng lai", "vang lai")):
            has_explicit_transfer = any(k in lower_q for k in ("chuyển tiền", "chuyen tien", "chuyển khoản", "chuyen khoan", "napas", "vietqr", "vnpay", "bắn tiền", "ban tien", "ủy nhiệm chi", "unc"))
            is_opening_or_management = False
            if "mở" in words_set or "mo" in words_set or "đóng" in words_set or "dong" in words_set or "khóa" in words_set or "khoa" in words_set:
                is_opening_or_management = True
            elif any(k in lower_q for k in ("mở tài khoản", "mo tai khoan", "thủ tục", "thu tuc", "giấy tờ", "giay to", "điều kiện", "dieu kien", "đăng ký", "dang ky", "quản lý tài khoản")):
                is_opening_or_management = True

            if has_explicit_transfer and not is_opening_or_management:
                intent = "chuyen_tien_thanh_toan"
                domain = "thanh_toan_chuyen_tien"
                if "hạn mức" in lower_q or "tối đa" in lower_q or "bao nhiêu" in lower_q:
                    sub_intent = "kiem_tra_han_muc"
                elif "phí" in lower_q or "biểu phí" in lower_q:
                    sub_intent = "tra_cuu_bieu_phi"
                    suggested_filters["chunk_type"] = "table"
                else:
                    sub_intent = "huong_dan_chuyen_tien"
            else:
                intent = "tai_khoan_ngan_hang"
                domain = "dich_vu_tai_khoan"
                if "mở" in words_set or "mo" in words_set or any(k in lower_q for k in ("mở tài khoản", "mo tai khoan", "thủ tục", "thu tuc", "giấy tờ", "giay to", "điều kiện", "dieu kien", "đăng ký", "dang ky")):
                    sub_intent = "thu_tuc_mo_tai_khoan"
                elif "đóng" in words_set or "dong" in words_set or "khóa" in words_set or "khoa" in words_set or "hủy" in lower_q:
                    sub_intent = "dong_khoa_tai_khoan"
                elif "số dư" in lower_q or "so du" in lower_q:
                    sub_intent = "tra_cuu_so_du"
                elif "phí" in lower_q or "biểu phí" in lower_q:
                    sub_intent = "tra_cuu_bieu_phi"
                    suggested_filters["chunk_type"] = "table"
                else:
                    sub_intent = "quan_ly_tai_khoan"

        # 6. Bank cards (ATM card, credit card, debit card, swallowed card, PIN, CVV)
        elif any(k in lower_q for k in ("thẻ", "the atm", "thẻ tín dụng", "thẻ ghi nợ", "nuốt thẻ", "mã pin", "cvv", "cvc", "khóa thẻ", "khoa the", "mở thẻ")):
            intent = "the_ngan_hang"
            domain = "dich_vu_the"
            if "nuốt thẻ" in lower_q or "mất thẻ" in lower_q or "khóa thẻ" in lower_q:
                sub_intent = "xu_ly_su_co_khan_cap"
            elif "mở thẻ" in lower_q or "điều kiện" in lower_q or "hồ sơ" in lower_q:
                sub_intent = "huong_dan_mo_the"
            elif "mã pin" in lower_q or "đổi pin" in lower_q or "doi pin" in lower_q:
                sub_intent = "doi_ma_pin"
            elif "phí" in lower_q or "biểu phí" in lower_q:
                sub_intent = "tra_cuu_bieu_phi"
                suggested_filters["chunk_type"] = "table"
            else:
                sub_intent = "tra_cuu_dich_vu_the"

        # 7. Loans and credit (home loan, consumer loan, debt balance, bad debt, CIC, loan rates)
        # Must precede savings to accurately handle queries regarding loan/mortgage interest rates
        elif any(k in lower_q for k in ("vay", "thế chấp", "the chap", "tín chấp", "tin chap", "giải ngân", "giai ngan", "dư nợ", "du no", "nợ xấu", "no xau", "cic", "trả góp", "tra gop", "mua nhà", "mua nha", "mua xe")):
            intent = "vay_von_tin_dung"
            domain = "tin_dung_cho_vay"
            if "thủ tục" in lower_q or "hồ sơ" in lower_q:
                sub_intent = "ho_so_vay_von"
            elif "lãi suất" in lower_q or "lai suat" in lower_q or "bao nhiêu" in lower_q:
                sub_intent = "lai_suat_vay"
                suggested_filters["chunk_type"] = "table"
            elif "thế chấp" in lower_q or "the chap" in lower_q:
                sub_intent = "vay_the_chap"
            elif "tín chấp" in lower_q or "tin chap" in lower_q:
                sub_intent = "vay_tin_chap"
            elif "cic" in lower_q:
                sub_intent = "tra_cuu_cic"
            else:
                sub_intent = "tu_van_goi_vay"

        # 8. Savings & deposits (interest rates, passbook, settlement)
        elif any(k in lower_q for k in ("tiết kiệm", "tiet kiem", "tiền gửi", "tien gui", "lãi suất", "lai suat", "tất toán", "tat toan", "kỳ hạn", "ky han")):
            intent = "tiet_kiem_tien_gui"
            domain = "huy_dong_von"
            if "lãi suất" in lower_q or "bao nhiêu %" in lower_q or "bảng" in lower_q or "biểu" in lower_q:
                sub_intent = "tra_cuu_lai_suat"
                suggested_filters["chunk_type"] = "table"
            elif "tất toán" in lower_q or "rút" in lower_q:
                sub_intent = "tat_toan_tiet_kiem"
            else:
                sub_intent = "huong_dan_gui_tiet_kiem"

        # 9. Digital banking & mobile app (App, forgot password, Smart OTP installation)
        elif any(k in lower_q for k in ("app", "ứng dụng", "ung dung", "digibank", "ipay", "smartbanking", "neo", "omni", "quên mật khẩu", "quen mat khau", "đăng nhập", "dang nhap", "lỗi app", "loi app")):
            intent = "ngan_hang_so"
            domain = "ngan_hang_dien_tu"
            if "quên mật khẩu" in lower_q or "quen mat khau" in lower_q:
                sub_intent = "quen_mat_khau"
            elif "smart otp" in lower_q or "smartotp" in lower_q or "soft otp" in lower_q:
                sub_intent = "dang_ky_smart_otp"
            elif "cài đặt" in lower_q or "cai dat" in lower_q:
                sub_intent = "cai_dat_app"
            else:
                sub_intent = "loi_giao_dich_app"

        # 10. Money transfer & payments (NAPAS 247, VietQR, VNPAY, UNC, MoMo, ZaloPay)
        elif any(k in lower_q for k in ("chuyển tiền", "chuyen tien", "chuyển khoản", "chuyen khoan", "napas", "vietqr", "vnpay", "thanh toán", "thanh toan", "lệnh chuyển", "ủy nhiệm chi", "unc", "bắn tiền", "ban tien", "momo", "zalopay")):
            intent = "chuyen_tien_thanh_toan"
            domain = "thanh_toan_chuyen_tien"
            if "hạn mức" in lower_q or "tối đa" in lower_q or "bao nhiêu" in lower_q:
                sub_intent = "kiem_tra_han_muc"
            elif "napas" in lower_q:
                sub_intent = "chuyen_tien_napas"
            elif "vietqr" in lower_q or "qr" in lower_q:
                sub_intent = "thanh_toan_qr"
            elif "phí" in lower_q or "biểu phí" in lower_q:
                sub_intent = "tra_cuu_bieu_phi"
                suggested_filters["chunk_type"] = "table"
            else:
                sub_intent = "huong_dan_chuyen_tien"

        # 11. Legal documents & compliance (Articles, Clauses, Circulars, Decisions, Laws, Ordinances)
        elif (legal_refs is not None and len(legal_refs) > 0) or any(k in lower_q for k in ("thông tư", "nghị định", "quyết định", "điều khoản", "luật", "pháp lệnh")):
            intent = "phap_ly_tuan_thu"
            domain = "phap_ly_tuan_thu"
            if "aml" in lower_q or "rửa tiền" in lower_q:
                sub_intent = "quy_dinh_aml"
            elif "tctd" in lower_q:
                sub_intent = "luat_tctd"
            else:
                sub_intent = "tra_cuu_van_ban_phap_ly"

        # 12. Customer support & hotline (Hotline, transaction dispute, complaint, branch)
        elif any(k in lower_q for k in ("tổng đài", "tong dai", "hotline", "khiếu nại", "khieu nai", "tra soát", "tra soat", "chi nhánh", "chi nhanh", "phòng giao dịch", "phong giao dich", "giờ làm việc", "gio lam viec")):
            intent = "ho_tro_khach_hang"
            domain = "cham_soc_khach_hang"
            if "hotline" in lower_q or "tổng đài" in lower_q:
                sub_intent = "lien_he_tong_dai"
            elif "chi nhánh" in lower_q or "phòng giao dịch" in lower_q or "atm" in lower_q:
                sub_intent = "dia_chi_pgd_atm"
            elif "giờ" in lower_q:
                sub_intent = "gio_lam_viec"
            else:
                sub_intent = "khieu_nai_tra_soat"

        return intent, sub_intent, domain, suggested_filters

    @staticmethod
    def extract_keywords(normalized_query):
        """
        Extract salient keywords for BM25 retrieval (0% Regex):
        1. Identifies decimal numbers to filter out fragmented noise digits.
        2. Tokenizes on whitespace and converts to lowercase.
        3. Removes Vietnamese stopwords.
        4. Preserves critical domain keywords and numeric values.
        """
        if not normalized_query:
            return []

        text_nfc = unicodedata.normalize("NFC", normalized_query)

        # Identify digits created by splitting decimal separators (e.g., 1.5, 2.5, 10.5)
        decimal_split_digits = set()
        t_len = len(text_nfc)
        for i, ch in enumerate(text_nfc):
            if ch in (".", ",") and i > 0 and i + 1 < t_len:
                if text_nfc[i - 1].isdigit() and text_nfc[i + 1].isdigit():
                    left_start = i - 1
                    while left_start > 0 and text_nfc[left_start - 1].isdigit():
                        left_start = left_start - 1
                    left_val = text_nfc[left_start:i]

                    right_end = i + 1
                    while right_end + 1 < t_len and text_nfc[right_end + 1].isdigit():
                        right_end = right_end + 1
                    right_val = text_nfc[i + 1:right_end + 1]

                    if len(right_val) <= 2:
                        decimal_split_digits.add(left_val)
                        decimal_split_digits.add(right_val)

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
            if len(w) > 1 or w.isdigit():
                if w not in VIETNAMESE_STOPWORDS:
                    # Remove isolated digits resulting from decimal splitting that add noise to BM25
                    if w in decimal_split_digits and w.isdigit():
                        continue
                    if w not in keywords:
                        keywords.append(w)

        return keywords

    def _semantic_llm_fallback(self, query):
        """
        Optional Semantic LLM Routing Fallback (R3):
        Triggered only when rule-based classifier yields 'hoi_dap_chung' on complex queries.
        Uses chat completion API configured in environment (if available).
        Completely safe: handles exceptions and returns None when offline or unconfigured.
        """
        from agent_banking.config.chat import FallbackChatClient
        from agent_banking.config.fallback import ModelFallbackError

        try:
            client = FallbackChatClient("query_parser")
            system_msg = (
                "Bạn là bộ phân loại ý định ngân hàng. Hãy phân tích câu hỏi người dùng và trả về JSON: "
                "{\"intent\": \"...\", \"sub_intent\": \"...\", \"domain\": \"...\"}. "
                "Các intent hợp lệ: sinh_trac_hoc_ekyc, xu_ly_su_co_khan_cap, tai_khoan_ngan_hang, "
                "tra_cuu_sao_ke, chuyen_tien_thanh_toan, vay_von_tin_dung, tiet_kiem_tien_gui, "
                "the_ngan_hang, ngan_hang_so, phap_ly_tuan_thu, tra_cuu_bieu_phi, ho_tro_khach_hang."
            )

            def parse_response(response):
                content = response.choices[0].message.content.strip()
                start_brace = content.find("{")
                end_brace = content.rfind("}")
                if start_brace < 0 or end_brace <= start_brace:
                    raise ValueError("Missing query classification JSON")
                data = json.loads(content[start_brace:end_brace + 1])
                if not isinstance(data, dict) or not all(isinstance(data.get(key), str) and data[key] for key in ("intent", "sub_intent", "domain")):
                    raise ValueError("Invalid query classification schema")
                return data["intent"], data["sub_intent"], data["domain"]

            resp = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": query}
                ],
                temperature=0.1,
                max_tokens=150,
                validate=parse_response,
            )
            return parse_response(resp)
        except ModelFallbackError:
            raise
        except Exception:
            pass
        return None, None, None

    def parse(self, query, use_llm=False):
        """
        Executes complete Step 1: Query Parser (0% Regex, Zero Ternary, Native Unicode NFC).
        Takes raw user query string -> Returns fully standardized ParsedQuery object.
        """
        if not query:
            clean_query = ""
        else:
            clean_query = " ".join(unicodedata.normalize("NFC", str(query).strip()).split())

        if not clean_query:
            return ParsedQuery(
                raw_query="",
                normalized_query="",
                keywords=[],
                keywords_str="",
                detected_acronyms=[],
                legal_references=[],
                financial_amounts=[],
                banks_mentioned=[],
                intent="hoi_dap_chung",
                sub_intent="tra_cuu_thong_tin",
                domain="ngan_hang_chung",
                suggested_filters={},
                expanded_query=""
            )

        # 1. Normalize acronyms
        normalized_query, detected_acronyms = self.normalize_acronyms(clean_query)

        # 2. Extract legal entities (merged from original and expanded text)
        refs_raw = self.extract_legal_entities(clean_query)
        refs_norm = self.extract_legal_entities(normalized_query)
        legal_refs = []
        for r in refs_raw + refs_norm:
            if r not in legal_refs:
                legal_refs.append(r)

        # 3. Extract BM25 keywords
        keywords = self.extract_keywords(normalized_query)
        keywords_str = " ".join(keywords)

        # 4. Extract monetary amounts and mentioned banks
        financial_amounts = self.extract_financial_amounts(clean_query)
        banks_mentioned = self.extract_bank_mentions(clean_query)

        # 5. Classify intent and domain
        intent, sub_intent, domain, suggested_filters = self.classify_intent_and_domain(
            normalized_query=normalized_query,
            detected_acronyms=detected_acronyms,
            legal_refs=legal_refs
        )

        # Optional Semantic LLM Fallback (R3) when rule-based returns generic FAQ
        should_call_llm = False
        if use_llm:
            should_call_llm = True
        elif self.use_llm_fallback:
            should_call_llm = True

        if should_call_llm and intent == "hoi_dap_chung" and len(clean_query.split()) >= 4:
            llm_intent, llm_sub, llm_dom = self._semantic_llm_fallback(clean_query)
            if llm_intent:
                intent = llm_intent
                sub_intent = llm_sub
                domain = llm_dom

        # 6. Construct expanded query for dense search (with asymmetric 'query: ' prefix)
        expanded_parts = [normalized_query]
        if legal_refs:
            for ref in legal_refs:
                if ref.lower() not in normalized_query.lower():
                    expanded_parts.append(ref)
        combined_text = " ".join(expanded_parts).strip()
        if not combined_text.startswith("query: "):
            expanded_query = f"query: {combined_text}"
        else:
            expanded_query = combined_text

        return ParsedQuery(
            raw_query=clean_query,
            normalized_query=normalized_query,
            keywords=keywords,
            keywords_str=keywords_str,
            detected_acronyms=detected_acronyms,
            legal_references=legal_refs,
            financial_amounts=financial_amounts,
            banks_mentioned=banks_mentioned,
            intent=intent,
            sub_intent=sub_intent,
            domain=domain,
            suggested_filters=suggested_filters,
            expanded_query=expanded_query
        )


# Backward compatibility aliases
RAGQueryParser = BankingQueryParser
QueryParser = BankingQueryParser
