"""Build DS-07: 1,000 Cross-Doc, 200 Legal Web Search, Noisy ASR, Style & Provenance Cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_banking.evaluator.suite_export import export_suite_bundle

SCRATCH_DIR = Path(
    r"C:\Users\ADMIN\.gemini\antigravity\brain\37df9fad-e9a2-46be-ab83-7d59cdac4108\scratch"
)
OUT_DIR = Path("dataset/A_Bank_eval_cross_legal_style")


def _make_ds07_item(
    idx_num: int,
    group: str,
    difficulty: str,
    priority: str,
    question: str,
    ref_answer: str,
    must_include: list[str],
    evidence: list[dict[str, Any]],
    persona: str = "advisor",
    tone: str = "formal",
    format_type: str = "standard",
    custom_persona: str = "",
    expected_tools: list[str] | None = None,
    citation_contract: dict[str, Any] | None = None,
    style_contract: dict[str, Any] | None = None,
    must_not_include: list[str] | None = None,
) -> dict[str, Any]:
    if expected_tools is not None:
        tools = list(expected_tools)
    else:
        tools = []
    if must_not_include is not None:
        forb = list(must_not_include)
    else:
        forb = []
    return {
        "id": f"DS07-{idx_num:04d}",
        "dataset": "DS-07",
        "group": group,
        "capability": "cross_legal_style",
        "difficulty": difficulty,
        "priority": priority,
        "question": question,
        "turns": [question],
        "profile": {
            "persona": persona,
            "tone": tone,
            "format_type": format_type,
            "custom_persona": custom_persona,
        },
        "reference_answer": ref_answer,
        "rubric": {"must_include": must_include, "must_not_include": forb},
        "checks": {
            "expected_tools": tools,
            "forbidden_tools": [],
            "expected_blocked": None,
            "numeric_targets": [],
            "artifact_contract": None,
            "citation_contract": citation_contract,
            "style_contract": style_contract,
        },
        "evidence": evidence,
    }


def _strip_vi_tones(text: str) -> str:
    table = str.maketrans(
        "áàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ"
        "ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ",
        "aaaaaaaaaaaaaaaaadeeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyy"
        "AAAAAAAAAAAAAAAAADEEEEEEEEEEEIIIIIOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYY",
    )
    return text.translate(table)


def build_ds07() -> list[dict[str, Any]]:
    """Generate 1,000 cases: 200 Legal Web Search + 800 Cross/Noisy/Style/Provenance."""
    rich_prov = json.loads((SCRATCH_DIR / "rich_provenance_catalog.json").read_text(encoding="utf-8"))
    cross_docs = json.loads((SCRATCH_DIR / "cross_docs_facts.json").read_text(encoding="utf-8"))
    ml_records = json.loads((SCRATCH_DIR / "ml01_records.json").read_text(encoding="utf-8"))["records"]

    items: list[dict[str, Any]] = []
    next_id = 1

    # G1: Legal Web Search (`search_legal_web` — 200 cases per user scope!)
    legal_topics = [
        ("Quyết định 2345/QĐ-NHNN", "quy định ngưỡng 10 triệu đồng/lần và 20 triệu đồng/ngày bắt buộc sinh trắc học khuôn mặt trên ứng dụng ngân hàng", "2345", True),
        ("Thông tư 50/2024/TT-NHNN", "quy định an toàn bảo mật trong cung cấp dịch vụ ngân hàng trực tuyến Online Banking và chữ ký điện tử", "50/2024", True),
        ("Thông tư 17/2024/TT-NHNN", "quy định việc mở và sử dụng tài khoản thanh toán tại tổ chức cung ứng dịch vụ thanh toán Việt Nam", "17/2024", True),
        ("Thông tư 18/2024/TT-NHNN", "quy định về hoạt động thẻ ngân hàng, hạn mức rút tiền và phát hành thẻ tín dụng/ghi nợ", "18/2024", True),
        ("Luật Các tổ chức tín dụng số 32/2024/QH15", "quy định về giới hạn cấp tín dụng, cổ đông sở hữu và quản trị rủi ro ngân hàng thương mại", "32/2024", True),
        ("Nghị định 52/2024/NĐ-CP", "quy định về thanh toán không dùng tiền mặt, ví điện tử và trung gian thanh toán", "52/2024", True),
        ("Luật Kinh doanh bảo hiểm số 08/2022/QH15", "quy định nguyên tắc bán bảo hiểm qua ngân hàng bancassurance và thời gian cân nhắc 21 ngày", "08/2022", True),
        ("Nghị định 13/2023/NĐ-CP", "quy định về bảo vệ dữ liệu cá nhân nhạy cảm và sự đồng ý của chủ thể dữ liệu trong tài chính", "13/2023", True),
        ("Thông tư 39/2016/TT-NHNN (sửa đổi bởi Thông tư 06/2023/TT-NHNN)", "quy định hoạt động cho vay bằng phương tiện điện tử của tổ chức tín dụng đối với khách hàng", "39/2016", True),
        ("Thông tư 12/2024/TT-NHNN", "quy định về cơ chế xét duyệt cấp tín dụng và kiểm soát sử dụng vốn vay", "12/2024", True),
    ]
    sub_facets = [
        "phạm vi điều chỉnh, đối tượng áp dụng và nghĩa vụ chính",
        "nghĩa vụ xác thực và quyền lợi người tiêu dùng",
        "đối chiếu quy định pháp luật quốc gia trên thuvienphapluat.vn với chính sách thực thi tại A-Bank",
        "xử lý khi có khác biệt hoặc cập nhật văn bản quy phạm mới hơn",
    ]
    for k in range(200):
        if k < 170:
            law_name, summary_txt, must_num, _ = legal_topics[k % len(legal_topics)]
            facet = sub_facets[(k // len(legal_topics)) % len(sub_facets)]
            var = (k // (len(legal_topics) * len(sub_facets))) + 1
            q = (
                f"[Pháp luật Việt Nam #{var}] Tra cứu văn bản pháp quy '{law_name}' ({facet}): "
                f"{summary_txt}. Đối chiếu ngắn gọn nghĩa vụ áp dụng tại ngân hàng."
            )
            ref = f"Theo kết quả tra cứu văn bản '{law_name}' (nguồn thuvienphapluat.vn và đối chiếu quy định A-Bank): {summary_txt}."
            must = [must_num]
            exp_t = ["search_legal_web"]
            if k < 80:
                prio_l = "P0"
            else:
                prio_l = "P1"
        else:
            # 30 non-legal / internal-only vs out-of-domain web deflection checks
            idx_o = k - 170
            if idx_o < 15:
                q = f"[Nội bộ không cần tra web #{idx_o + 1}] Phí mở tài khoản thanh toán VND cá nhân tại quầy riêng của ngân hàng A Bank theo BP 01 là bao nhiêu?"
                ref = "Câu hỏi thuần túy về biểu phí nội bộ A Bank, sử dụng kho nội bộ search_banking_knowledge mà không lạm dụng search_legal_web."
                must = ["A Bank"]
                exp_t = ["search_banking_knowledge"]
                prio_l = "P1"
            else:
                q = f"[Ngoài phạm vi pháp luật/tin tức #{idx_o - 14}] Tra cứu kết quả xổ số miền Bắc hoặc tỷ số bóng đá Ngoại hạng Anh tối qua trên web giúp tôi."
                ref = "Công cụ tra cứu pháp luật chỉ phục vụ văn bản pháp quy ngân hàng/tài chính Việt Nam; từ chối tra cứu tin giải trí, xổ số, thể thao."
                must = ["không"]
                exp_t = []
                prio_l = "P0"

        items.append(_make_ds07_item(
            next_id,
            "G1_legal_web_search",
            "hard",
            prio_l,
            q,
            ref,
            must,
            [{"legal_case_idx": k, "expected_tools": exp_t}],
            expected_tools=exp_t,
        ))
        next_id += 1

    # G2: Cross-Doc Multi-Hop Reasoning (250 cases)
    # Mix BC 01 vs BC 02 YoY, VH 01/03 vs ML 01, THE vs BP 08, BH vs PL, NV 06/07 vs Calculator
    cross_pairs = [
        ("Tổng hợp thay đổi giữa BC 01 (2024) và BC 02 (2025) về Tổng tài sản (508.330 -> 605.585 tỷ), ROE (25,10% -> 25,23%) và tỷ lệ CAR (13,33% -> 11,87%).", ["605", "11"], ["BC 01", "BC 02"]),
        ("Kết hợp quy định thẻ tín dụng Eco Visa (THE 14) và biểu phí thanh toán quốc tế (BP 08): khách hàng mở thẻ xanh và phát hành chuyển tiền quốc tế cần lưu ý quy định nào?", ["Eco Visa"], ["THE 14", "BP 08"]),
        ("Một khách hàng ở Đà Nẵng giao dịch tại ATM Thanh Khê (ML 01) bị nuốt thẻ hoặc lỗi tra soát: theo VH 03 (SLA tra soát) và VH 01, địa chỉ điểm ATM và SLA xử lý ra sao?", ["tra soát"], ["ML 01", "VH 03"]),
        ("So sánh chỉ tiêu Tổng huy động vốn (338.631 -> 401.680 tỷ) và Tổng dư nợ cho vay (331.606 -> 391.746 tỷ) giữa BC 01 và BC 02.", ["401", "391"], ["BC 02"]),
        ("Khách hàng tham gia bảo hiểm A-Life Trọn Đời (BH 01) qua A-Bank đồng thời mở tài khoản thanh toán VND (NV 01 + BP 01): phí mở tài khoản và quyền lợi bảo hiểm cơ bản gồm gì?", ["A-Life"], ["BH 01", "NV 01"]),
        ("Khi sử dụng tính năng Đăng nhập vân tay/Face ID (HDSD 03) rồi chuyển khoản 18 triệu đồng trên thiết bị mới: phối hợp HDSD 03 và quy chuẩn Quyết định 2345 như thế nào?", ["2345", "Face"], ["HDSD 03"]),
    ]
    for k in range(250):
        q_base, must, req_srcs = cross_pairs[k % len(cross_pairs)]
        var = (k // len(cross_pairs)) + 1
        ml_pt = ml_records[(k * 7) % len(ml_records)]
        city = ml_pt.get("Tỉnh/thành phố", "Hà Nội")
        if k % 3 == 2:
            q = f"[Liên tài liệu #{var}] {q_base} (Kèm kiểm tra có điểm giao dịch ML 01 tại {city} không)."
            must_final = must + [city]
        else:
            q = f"[Liên tài liệu #{var}] {q_base}"
            must_final = must
        ref = f"Tổng hợp thông tin chính xác từ nhiều nguồn ({', '.join(req_srcs)}) không trộn lẫn số liệu năm/điều kiện."
        if k < 90:
            prio_g2 = "P0"
        else:
            prio_g2 = "P1"
        items.append(_make_ds07_item(
            next_id,
            "G2_cross_doc_multihop",
            "hard",
            prio_g2,
            q,
            ref,
            must_final,
            [{"sources": req_srcs, "variant": var}],
            expected_tools=["search_banking_knowledge"],
            citation_contract={"sources": [req_srcs[0]]},
        ))
        next_id += 1

    # G3: Noisy Input, Teencode, Missing Diacritics & Voice-ASR Homophones (200 cases)
    clean_pairs = [
        ("Khách hàng mở tài khoản thanh toán VND tại quầy mất phí bao nhiêu?", "mo tk tt vnd o quay mat phi bnhieu vay ad", "Miễn phí mở tài khoản thanh toán VND cá nhân.", ["Miễn phí"]),
        ("Chuyển khoản online trên 10 triệu đồng có cần quét mặt sinh trắc học không?", "ck online tren 10cu co can face id k ad", "Giao dịch trên 10 triệu đồng bắt buộc xác thực sinh trắc học khuôn mặt theo QĐ 2345.", ["2345"]),
        ("Tổng đài chăm sóc khách hàng chính thức của ngân hàng là số nào?", "tong dai cham soc khach hang cua a bank la so nao z", "Hotline chính thức theo tài liệu ngân hàng.", ["1900"]),
        ("Thẻ tín dụng sinh viên Edu Visa có điều kiện cấp thẻ ra sao?", "the tin dung sv edu visa dk mo the sao a", "Thông tin điều kiện và đối tượng sinh viên của thẻ Edu Visa.", ["Edu Visa"]),
        ("Cách khôi phục mật khẩu trên ứng dụng A-Bank Plus khi bị quên mật khẩu là gì?", "quen mk app a bank plus lay lai sao vay b", "Hướng dẫn khôi phục mật khẩu theo HDSD 24 trên A-Bank Plus.", ["mật khẩu"]),
        ("Vay mua ô tô tiêu dùng cần chuẩn bị hồ sơ gì và phương thức trả nợ thế nào?", "vay mua xe o to ca nhan can ho so gi va tra no du no giam dan ra sao", "Thông tin hồ sơ và quy định cho vay mua ô tô NV 07.", ["ô tô"]),
    ]
    for k in range(200):
        clean_q, tc_q, ref_txt, must = clean_pairs[k % len(clean_pairs)]
        mode = k % 3
        var = (k // len(clean_pairs)) + 1
        if mode == 0:
            q_noisy = _strip_vi_tones(clean_q).lower() + f" #{var}"
        elif mode == 1:
            q_noisy = f"{tc_q} (hoi lan #{var})"
        else:
            # Voice ASR homophone / spacing distortion
            q_noisy = (
                clean_q.replace("VND", "vờ nờ đê")
                .replace("10 triệu", "mười chiệu")
                .replace("Visa", "vi da")
                .replace("Edu", "ê đu")
                .replace("A-Bank Plus", "ây banh pờ lút")
            ) + f" [{var}]"
        if (k % len(clean_pairs)) == 1:
            noisy_tools: list[str] = []
        else:
            noisy_tools = ["search_banking_knowledge"]
        items.append(_make_ds07_item(
            next_id,
            "G3_noisy_teencode_voice_asr",
            "hard",
            "P1",
            q_noisy,
            ref_txt,
            must,
            [{"clean_intent": clean_q, "noise_mode": mode}],
            expected_tools=noisy_tools,
        ))
        next_id += 1

    # G4: Persona × Tone × Format Style Matrix (200 cases)
    personas = ["advisor", "teller", "compliance", "digital_support", "credit_insurance", "quick"]
    tones = ["formal", "warm", "direct", "detailed"]
    formats = ["standard", "bullet_points", "table_comparison", "legal_strict", "with_disclaimer"]
    matrix_questions = [
        ("Điều kiện và biểu phí thẻ Cashback Visa của A Bank gồm những điểm chính nào?", "Cashback Visa"),
        ("Giải thích nguyên tắc tính lãi vay dư nợ giảm dần và các kỳ hạn gửi tiết kiệm mẫu.", "giảm dần"),
        ("Quy trình cài đặt vân tay / Face ID trên ứng dụng A-Bank Plus và lưu ý QĐ 2345.", "2345"),
        ("Chính sách quyền lợi và thời gian cân nhắc của sản phẩm bảo hiểm A-Life Trọn Đời.", "A-Life"),
    ]
    for k in range(200):
        per = personas[k % len(personas)]
        tn = tones[(k // len(personas)) % len(tones)]
        fmt = formats[(k // (len(personas) * len(tones))) % len(formats)]
        mq, must_tok = matrix_questions[(k // 3) % len(matrix_questions)]
        q = f"[StyleMatrix #{k + 1} | {per}/{tn}/{fmt}] {mq}"

        if per == "quick":
            max_w: int | None = 145
        else:
            max_w = None
        req_tbl = fmt == "table_comparison"
        req_bul = fmt == "bullet_points"
        req_disc = fmt == "with_disclaimer"

        style_c = {
            "persona": per,
            "tone": tn,
            "format_type": fmt,
            "max_words": max_w,
            "require_table": req_tbl,
            "require_bullets": req_bul,
            "require_disclaimer": req_disc,
        }
        must = [must_tok]
        if req_disc:
            must.append("Lưu ý")
        ref = f"Phản hồi đúng nội dung ({must_tok}), tuân thủ persona='{per}', tone='{tn}' và format_style='{fmt}'."
        items.append(_make_ds07_item(
            next_id,
            "G4_persona_tone_format_matrix",
            "medium",
            "P1",
            q,
            ref,
            must,
            [{"persona": per, "tone": tn, "format": fmt}],
            persona=per,
            tone=tn,
            format_type=fmt,
            style_contract=style_c,
        ))
        next_id += 1

    # G5: Level-2 Citation & Viewer Provenance Verification (150 cases)
    # Draw real chunks with exact page_numbers (PDF), images (HDSD), xpath (DOCX), cell_ranges (XLSX)
    prov_pool: list[dict[str, Any]] = []
    for cat, docs in rich_prov.items():
        for d in docs:
            src = d["source"]
            for s in d.get("samples", []):
                if s.get("page_numbers") or s.get("locations") or s.get("images"):
                    prov_pool.append({"cat": cat, "source": src, "sample": s})

    for k in range(150):
        entry = prov_pool[k % len(prov_pool)]
        src = entry["source"]
        s = entry["sample"]
        doc_short = src.rsplit("/", 1)[-1]
        sec = s.get("section_title") or "nội dung liên quan"
        pgs = s.get("page_numbers") or []
        imgs = s.get("images") or []
        locs = s.get("locations") or []
        snip = " ".join(str(s.get("content", "")).split())[:140]

        if pgs:
            q = (
                f"[Provenance-PDF #{k + 1}] Trong tài liệu '{doc_short}' (tham chiếu mục '{str(sec)[:70]}', trang {pgs[0]}), "
                f"nêu nội dung chính và trích dẫn đúng số trang/nguồn để kiểm tra Level-2 bounding box."
            )
        else:
            q = (
                f"[Provenance-DOCX #{k + 1}] Trong tài liệu '{doc_short}' (mục '{str(sec)[:70]}'), "
                f"trình bày quy định liên quan đến đoạn '{snip[:65]}...' và gắn nhãn [Nguồn N] chuẩn."
            )

        must_tok = doc_short.split()[0]
        if pgs:
            pg_list = list(pgs[:1])
        else:
            pg_list = []
        cit_c = {"sources": [doc_short], "pages": pg_list}

        if k < 60:
            prio_g5 = "P0"
        else:
            prio_g5 = "P1"
        items.append(_make_ds07_item(
            next_id,
            "G5_citation_level2_provenance",
            "hard",
            prio_g5,
            q,
            f"Trả lời chính xác và gắn trích dẫn [Nguồn N] trỏ tới '{doc_short}' kèm thông tin trang/vị trí chính xác.",
            [must_tok],
            [{"source": src, "chunk_id": s.get("chunk_id"), "pages": pgs, "locations": locs[:1], "images_count": len(imgs)}],
            expected_tools=["search_banking_knowledge"],
            citation_contract=cit_c,
        ))
        next_id += 1

    return items


def main() -> None:
    items = build_ds07()
    cov = export_suite_bundle(items, OUT_DIR, "eval_cross_legal_style", "DS-07", canary_size=40)
    print(json.dumps(cov, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
