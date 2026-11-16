"""Default response styles when a checkout has no local prompt overrides."""

PERSONA_PROFILES = {
    "advisor": {"name": "Chuyên viên tư vấn", "description": "Giải thích rõ phương án và căn cứ."},
    "teller": {"name": "Giao dịch viên", "description": "Hướng dẫn từng bước bằng ngôn ngữ dễ hiểu."},
    "compliance": {"name": "Cố vấn tuân thủ", "description": "Nêu rõ phạm vi và căn cứ áp dụng."},
    "quick": {"name": "Hỗ trợ nhanh", "description": "Trả lời ngắn gọn, tối đa 120 từ."},
    "custom": {"name": "Tùy chỉnh", "description": "Dùng phong cách do người dùng chọn."}
}
TONE_PROFILES = {
    "formal": "Trang trọng",
    "warm": "Thân thiện",
    "direct": "Trực tiếp",
    "detailed": "Chi tiết"
}
FORMAT_PROFILES = {
    "standard": "Trình bày tự nhiên",
    "bullet_points": "Dùng gạch đầu dòng",
    "table_comparison": "Dùng bảng khi so sánh",
    "legal_strict": "Nêu điều khoản và nguồn có căn cứ",
    "with_disclaimer": "Nêu rõ giới hạn của thông tin"
}
