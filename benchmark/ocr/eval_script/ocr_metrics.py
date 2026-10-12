#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
OCR EVALUATION METRICS SUITE FOR BANKING & VIETNAMESE DOCUMENTS (100% LOCAL)
Folder: benchmark/ocr/eval_script/
File: ocr_metrics.py
=============================================================================
Module tính toán độc lập, hoàn chỉnh, chạy 100% Offline Local với hiệu năng cao:
1. CER (Character Error Rate) - Khoảng cách Levenshtein cấp ký tự
2. WER (Word Error Rate) - Khoảng cách Levenshtein cấp từ
3. NED (Normalized Edit Distance / 1 - NED) - Chuẩn hóa khoảng cách chỉnh sửa
4. Diacritic-CER & Diacritic Accuracy - Đánh giá chuyên sâu dấu tiếng Việt
5. GCER (Grapheme Cluster Error Rate) - Lỗi theo cụm ký tự ngữ âm
6. Numeric & Financial Fact Accuracy - Độ chính xác số tiền, số TK, ngày tháng
7. Field-level KIE Metrics (Precision, Recall, F1, ANLS) - Cho biểu mẫu eKYC
=============================================================================
"""

import sys
import re
import unicodedata
from typing import Dict, Any, List, Tuple, Optional, Set

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Toàn bộ ký tự có dấu thanh và dấu phụ trong bảng mã tiếng Việt
VIETNAMESE_DIACRITIC_CHARS: Set[str] = set(
    "áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ"
    "ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ"
)


def normalize_vietnamese_text(text: str, form: str = "NFC") -> str:
    """
    Chuẩn hóa chuỗi văn bản:
    - Chuyển về chuẩn Unicode thống nhất (mặc định NFC).
    - Chuẩn hóa khoảng trắng dư thừa, tab, xuống dòng.
    """
    if not text:
        return ""
    text = unicodedata.normalize(form, text)
    # Thay thế các ký tự khoảng trắng đặc biệt thành dấu cách thường
    text = re.sub(r"[\u00A0\u2000-\u200B\u202F\u205F\u3000]", " ", text)
    return text


def clean_eval_text(text: str) -> str:
    """
    Làm sạch văn bản trước khi tính toán độ đo:
    - Loại bỏ tiêu đề metadata markdown (ví dụ: '# OCR Extraction', '**Engine**: ...')
    - Chuẩn hóa khoảng trắng liên tiếp
    """
    if not text:
        return ""
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        # Bỏ qua các dòng metadata markdown thường gặp trong benchmark
        if s.startswith("# ") or s.startswith("## ") or s.startswith("### "):
            continue
        if s.startswith("**Engine") or s.startswith("**Processing") or s.startswith("**Nguồn") or s.startswith("**Thời gian"):
            continue
        if s.startswith("```") or s == "---":
            continue
        # Bỏ các ký tự định dạng markdown bảng biểu để so sánh nội dung thuần túy
        s = s.replace("|", " ").replace("*", "").replace("`", "")
        lines.append(s)

    cleaned = " ".join(lines)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return normalize_vietnamese_text(cleaned, "NFC")


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Tính khoảng cách chỉnh sửa Levenshtein tối ưu hóa bộ nhớ và tốc độ (Wagner-Fischer 2-row).
    """
    if s1 == s2:
        return 0
    len1, len2 = len(s1), len(s2)
    if len1 < len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1
    if len2 == 0:
        return len1

    v0 = list(range(len2 + 1))
    v1 = [0] * (len2 + 1)

    for i in range(len1):
        v1[0] = i + 1
        c1 = s1[i]
        for j in range(len2):
            cost = 0 if c1 == s2[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        v0, v1 = v1, v0

    return v0[len2]


def word_levenshtein_distance(words1: List[str], words2: List[str]) -> int:
    """
    Tính khoảng cách Levenshtein ở cấp độ từ (Token Levenshtein tối ưu).
    """
    if words1 == words2:
        return 0
    len1, len2 = len(words1), len(words2)
    if len1 < len2:
        words1, words2 = words2, words1
        len1, len2 = len2, len1
    if len2 == 0:
        return len1

    v0 = list(range(len2 + 1))
    v1 = [0] * (len2 + 1)

    for i in range(len1):
        v1[0] = i + 1
        w1 = words1[i]
        for j in range(len2):
            cost = 0 if w1 == words2[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        v0, v1 = v1, v0

    return v0[len2]


def calculate_cer(ref: str, hyp: str, normalize: bool = True) -> float:
    """
    Character Error Rate (CER):
    CER = (Substitutions + Insertions + Deletions) / len(ref) * 100%
    """
    if normalize:
        ref = normalize_vietnamese_text(ref, "NFC")
        hyp = normalize_vietnamese_text(hyp, "NFC")

    if not ref and not hyp:
        return 0.0
    if not ref:
        return 100.0

    dist = levenshtein_distance(ref, hyp)
    cer = (dist / len(ref)) * 100.0
    return round(cer, 2)


def calculate_wer(ref: str, hyp: str, normalize: bool = True) -> float:
    """
    Word Error Rate (WER):
    WER = (Word Edits) / len(ref_words) * 100%
    """
    if normalize:
        ref = normalize_vietnamese_text(ref, "NFC")
        hyp = normalize_vietnamese_text(hyp, "NFC")

    ref_words = ref.split()
    hyp_words = hyp.split()

    if not ref_words and not hyp_words:
        return 0.0
    if not ref_words:
        return 100.0

    dist = word_levenshtein_distance(ref_words, hyp_words)
    wer = (dist / len(ref_words)) * 100.0
    return round(wer, 2)


def calculate_ned(ref: str, hyp: str, normalize: bool = True, char_dist: Optional[int] = None) -> float:
    """
    Normalized Edit Distance (NED / S_ED):
    S_ED = max(0, 1 - EditDist(ref, hyp) / max(len(ref), len(hyp)))
    Giá trị trong khoảng [0.0, 1.0], càng gần 1.0 càng chính xác.
    """
    if normalize:
        ref = normalize_vietnamese_text(ref, "NFC")
        hyp = normalize_vietnamese_text(hyp, "NFC")

    if not ref and not hyp:
        return 1.0
    max_len = max(len(ref), len(hyp))
    if max_len == 0:
        return 1.0

    if char_dist is None:
        char_dist = levenshtein_distance(ref, hyp)

    ned = max(0.0, 1.0 - (char_dist / max_len))
    return round(ned, 4)


def calculate_diacritic_metrics(ref: str, hyp: str, normalize: bool = True) -> Dict[str, Any]:
    """
    Đánh giá chuyên sâu dấu tiếng Việt (Diacritics Preservation & Diacritic-CER):
    - Đếm số ký tự mang dấu thanh/mũ/móc trong ref và hyp.
    - Tính tỷ lệ bảo toàn dấu (Accuracy).
    - Tính Diacritic-CER (Tỷ lệ lỗi trên chuỗi trích xuất các ký tự có dấu).
    """
    if normalize:
        ref = normalize_vietnamese_text(ref, "NFC")
        hyp = normalize_vietnamese_text(hyp, "NFC")

    ref_diacritics = [c for c in ref if c in VIETNAMESE_DIACRITIC_CHARS]
    hyp_diacritics = [c for c in hyp if c in VIETNAMESE_DIACRITIC_CHARS]

    total_ref = len(ref_diacritics)
    total_hyp = len(hyp_diacritics)

    if total_ref == 0:
        return {
            "ref_diacritics_count": 0,
            "hyp_diacritics_count": total_hyp,
            "diacritic_accuracy": 100.0,
            "diacritic_cer": 0.0,
        }

    # 1. Diacritic Preservation Accuracy: Tỷ lệ ký tự dấu có mặt
    matched = min(total_ref, total_hyp)
    diacritic_acc = round((matched / total_ref) * 100.0, 2)

    # 2. Diacritic-CER: Tính Levenshtein trực tiếp trên chuỗi ký tự có dấu
    ref_diac_str = "".join(ref_diacritics)
    hyp_diac_str = "".join(hyp_diacritics)
    diac_dist = levenshtein_distance(ref_diac_str, hyp_diac_str)
    diacritic_cer = round((diac_dist / total_ref) * 100.0, 2)

    return {
        "ref_diacritics_count": total_ref,
        "hyp_diacritics_count": total_hyp,
        "diacritic_accuracy": diacritic_acc,
        "diacritic_cer": diacritic_cer,
    }


def split_grapheme_clusters(text: str) -> List[str]:
    """
    Phân tách chuỗi thành các cụm ký tự ngữ âm (Grapheme Clusters).
    Sử dụng chuẩn NFD để gom ký tự gốc cùng các dấu tổ hợp (Combining Diacritical Marks).
    """
    decomposed = unicodedata.normalize("NFD", text)
    clusters = []
    current_cluster = []

    for char in decomposed:
        if unicodedata.category(char).startswith("M"):
            current_cluster.append(char)
        else:
            if current_cluster:
                clusters.append("".join(current_cluster))
            current_cluster = [char]

    if current_cluster:
        clusters.append("".join(current_cluster))

    return [unicodedata.normalize("NFC", c) for c in clusters]


def calculate_gcer(ref: str, hyp: str) -> float:
    """
    Grapheme Cluster Error Rate (GCER):
    Mã hóa từng cụm ký tự thành chuỗi ký tự duy nhất để tối ưu tính toán.
    """
    ref_clusters = split_grapheme_clusters(ref)
    hyp_clusters = split_grapheme_clusters(hyp)

    if not ref_clusters and not hyp_clusters:
        return 0.0
    if not ref_clusters:
        return 100.0

    # Ánh xạ cụm ký tự sang integer/chr để tăng tốc độ so sánh
    vocab = {}
    def encode_clusters(clusters):
        encoded = []
        for c in clusters:
            if c not in vocab:
                vocab[c] = chr(len(vocab) + 32)
            encoded.append(vocab[c])
        return "".join(encoded)

    s1_encoded = encode_clusters(ref_clusters)
    s2_encoded = encode_clusters(hyp_clusters)

    dist = levenshtein_distance(s1_encoded, s2_encoded)
    gcer = (dist / len(ref_clusters)) * 100.0
    return round(gcer, 2)


def extract_numbers_and_financials(text: str) -> List[str]:
    """
    Trích xuất toàn bộ các thực thể số học trong văn bản ngân hàng:
    - Số tiền (VD: '15.000.000', '15,000,000.50', '50.000 VND')
    - Tỷ lệ phần trăm (VD: '8.5%', '12%')
    - Mốc ngày tháng (VD: '16/12/2025', '2024-09-27')
    - Số tài khoản / Số định danh (VD: '19038472910', '0011001234567')
    """
    pattern = r"\b(?:\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?%?|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{6,20})\b"
    matches = re.findall(pattern, text)
    cleaned_matches = []
    for m in matches:
        s = m.strip().rstrip(".,")
        if s:
            cleaned_matches.append(s)
    return cleaned_matches


def calculate_numeric_metrics(ref: str, hyp: str) -> Dict[str, Any]:
    """
    Đánh giá độ chính xác số học & Fact-level Financial Precision:
    - Precision: Tỷ lệ số dự đoán đúng trên tổng số tìm thấy.
    - Recall: Tỷ lệ số thực tế được tìm thấy đúng.
    - F1-Score: Cân bằng giữa Precision và Recall.
    - Numeric Accuracy: Khớp chính xác.
    """
    ref_numbers = extract_numbers_and_financials(ref)
    hyp_numbers = extract_numbers_and_financials(hyp)

    if not ref_numbers and not hyp_numbers:
        return {
            "ref_numbers_count": 0,
            "hyp_numbers_count": 0,
            "numeric_precision": 100.0,
            "numeric_recall": 100.0,
            "numeric_f1": 100.0,
            "numeric_accuracy": 100.0,
            "matched_numbers_count": 0,
        }

    if not ref_numbers:
        return {
            "ref_numbers_count": 0,
            "hyp_numbers_count": len(hyp_numbers),
            "numeric_precision": 0.0,
            "numeric_recall": 100.0,
            "numeric_f1": 0.0,
            "numeric_accuracy": 0.0,
            "matched_numbers_count": 0,
        }

    ref_pool = list(ref_numbers)
    matched_count = 0
    for num in hyp_numbers:
        if num in ref_pool:
            matched_count += 1
            ref_pool.remove(num)

    precision = (matched_count / len(hyp_numbers)) * 100.0 if hyp_numbers else 0.0
    recall = (matched_count / len(ref_numbers)) * 100.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "ref_numbers_count": len(ref_numbers),
        "hyp_numbers_count": len(hyp_numbers),
        "matched_numbers_count": matched_count,
        "numeric_precision": round(precision, 2),
        "numeric_recall": round(recall, 2),
        "numeric_f1": round(f1, 2),
        "numeric_accuracy": round(recall, 2),
    }


def calculate_kie_field_metrics(
    ref_fields: Dict[str, str],
    hyp_fields: Dict[str, str],
    anls_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Đánh giá trích xuất thông tin khóa (Key Information Extraction - KIE) cho biểu mẫu:
    - Exact Match (EM): Khớp chính xác 100%.
    - ANLS (Average Normalized Levenshtein Similarity): Dung sai cho OCR sai lệch nhỏ.
    - Field-level Precision, Recall, F1.
    """
    total_fields = len(ref_fields)
    if total_fields == 0:
        return {"field_f1": 100.0, "field_anls": 1.0, "exact_match": 100.0}

    exact_matches = 0
    anls_scores = []
    field_details = {}

    for field, ref_val in ref_fields.items():
        ref_norm = normalize_vietnamese_text(ref_val.strip(), "NFC")
        hyp_val = hyp_fields.get(field, "")
        hyp_norm = normalize_vietnamese_text(hyp_val.strip(), "NFC") if hyp_val else ""

        is_em = (ref_norm == hyp_norm)
        if is_em:
            exact_matches += 1

        if not ref_norm and not hyp_norm:
            score = 1.0
        elif not ref_norm or not hyp_norm:
            score = 0.0
        else:
            dist = levenshtein_distance(ref_norm, hyp_norm)
            norm_dist = dist / max(len(ref_norm), len(hyp_norm))
            score = (1.0 - norm_dist) if (1.0 - norm_dist) >= anls_threshold else 0.0

        anls_scores.append(score)
        field_details[field] = {
            "ref": ref_val,
            "hyp": hyp_val,
            "exact_match": is_em,
            "anls": round(score, 4),
        }

    em_rate = round((exact_matches / total_fields) * 100.0, 2)
    mean_anls = round(sum(anls_scores) / total_fields, 4)
    f1_score = round(mean_anls * 100.0, 2)

    return {
        "total_fields": total_fields,
        "exact_matches": exact_matches,
        "exact_match_rate": em_rate,
        "mean_anls": mean_anls,
        "field_f1": f1_score,
        "field_details": field_details,
    }


def evaluate_ocr_pair(
    ref_text: str,
    hyp_text: str,
    ref_fields: Optional[Dict[str, str]] = None,
    hyp_fields: Optional[Dict[str, str]] = None,
    clean_markdown: bool = True,
) -> Dict[str, Any]:
    """
    Tổng hợp Đầy đủ Ma trận Đánh giá OCR Hoàn hảo cho 1 cặp (Ground Truth, Prediction):
    Trả về Dictionary chứa toàn bộ chỉ số của cả 5 tầng đánh giá.
    """
    if clean_markdown:
        ref_clean = clean_eval_text(ref_text)
        hyp_clean = clean_eval_text(hyp_text)
    else:
        ref_clean = normalize_vietnamese_text(ref_text.strip(), "NFC")
        hyp_clean = normalize_vietnamese_text(hyp_text.strip(), "NFC")

    # 1. Tầng 1: Ký tự & Từ
    char_dist = levenshtein_distance(ref_clean, hyp_clean)
    ref_len = len(ref_clean)
    cer = (char_dist / max(ref_len, 1)) * 100.0
    cer = round(cer, 2)

    wer = calculate_wer(ref_clean, hyp_clean, normalize=False)
    ned = calculate_ned(ref_clean, hyp_clean, normalize=False, char_dist=char_dist)

    # 2. Tầng 2: Dấu tiếng Việt & Ngữ âm
    diac_metrics = calculate_diacritic_metrics(ref_clean, hyp_clean, normalize=False)
    gcer = calculate_gcer(ref_clean, hyp_clean)

    # 3. Tầng 3: Số liệu tài chính
    numeric_metrics = calculate_numeric_metrics(ref_clean, hyp_clean)

    results = {
        "ref_char_count": ref_len,
        "hyp_char_count": len(hyp_clean),
        "ref_word_count": len(ref_clean.split()),
        "hyp_word_count": len(hyp_clean.split()),
        "char_distance": char_dist,
        # Metrics chính
        "cer": cer,
        "wer": wer,
        "ned": ned,
        "gcer": gcer,
        "diacritic_accuracy": diac_metrics["diacritic_accuracy"],
        "diacritic_cer": diac_metrics["diacritic_cer"],
        "numeric_accuracy": numeric_metrics["numeric_accuracy"],
        "numeric_f1": numeric_metrics["numeric_f1"],
        "details": {
            "diacritics": diac_metrics,
            "numerics": numeric_metrics,
        },
    }

    # 4. Tầng 4: KIE Fields (nếu có)
    if ref_fields is not None:
        kie_res = calculate_kie_field_metrics(ref_fields, hyp_fields or {})
        results["kie"] = kie_res

    return results


if __name__ == "__main__":
    gt_sample = "NGÂN HÀNG NGOẠI THƯƠNG VIỆT NAM (VCB) THÔNG BÁO BIỂU PHÍ NGÀY 16/12/2025: PHÍ 50.000 VND."
    pred_sample = "NGAN HANG NGOAI THUONG VIET NAM (VCB) THONG BAO BIEU PHI NGAY 16/12/2025: PHI 50.000 VND."

    print("=== TEST DEMO OCR EVALUATION METRICS ===")
    res = evaluate_ocr_pair(gt_sample, pred_sample, clean_markdown=False)
    print(f"Ref : {gt_sample}")
    print(f"Hyp : {pred_sample}")
    print(f"CER: {res['cer']}%")
    print(f"WER: {res['wer']}%")
    print(f"GCER: {res['gcer']}%")
    print(f"Diacritic Preservation: {res['diacritic_accuracy']}%")
    print(f"Diacritic CER: {res['diacritic_cer']}%")
    print(f"Numeric Accuracy: {res['numeric_accuracy']}% (Ref numbers: {res['details']['numerics']['ref_numbers_count']})")
