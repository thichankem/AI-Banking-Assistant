"""Build DS-01: 1,000 Excel Operations & Multi-Version as_of Evaluation Cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill

from agent_banking.evaluator.suite_export import export_suite_bundle

SCRATCH_DIR = Path(
    r"C:\Users\ADMIN\.gemini\antigravity\brain\37df9fad-e9a2-46be-ab83-7d59cdac4108\scratch"
)
OUT_DIR = Path("dataset/A_Bank_eval_excel_ops")
FIXTURES_DIR = OUT_DIR / "fixtures"

VERSIONED_ITEMS = [
    ("DV01", "Chuyển tiền nhanh Napas 24/7 trong nước qua App", "Miễn phí", 0, "Miễn phí", 0),
    ("DV02", "Chuyển tiền liên ngân hàng tại quầy cùng tỉnh", "0,03% (min 15.000đ, max 500.000đ)", 15000, "0,02% (min 10.000đ, max 300.000đ)", 10000),
    ("DV03", "Chuyển tiền liên ngân hàng tại quầy khác tỉnh", "0,05% (min 20.000đ, max 1.000.000đ)", 20000, "0,04% (min 15.000đ, max 800.000đ)", 15000),
    ("DV04", "Phí quản lý tài khoản thanh toán cá nhân (số dư dưới ngưỡng)", "11.000đ/tháng", 11000, "5.500đ/tháng", 5500),
    ("DV05", "Phí cấp lại mã PIN thẻ ghi nợ nội địa tại quầy", "20.000đ/lần", 20000, "10.000đ/lần", 10000),
    ("DV06", "Phí sao kê tài khoản giấy tại quầy (trong 12 tháng gần nhất)", "10.000đ/bản", 10000, "5.000đ/bản", 5000),
    ("DV07", "Phí xác nhận số dư tài khoản song ngữ", "50.000đ/bản", 50000, "30.000đ/bản", 30000),
    ("DV08", "Phí kiểm đếm tiền mặt VND tại quầy (từ 500 triệu trở lên)", "0,03% (min 30.000đ)", 30000, "0,02% (min 20.000đ)", 20000),
    ("DV09", "Phí đổi tiền không đủ tiêu chuẩn lưu thông", "3% tổng giá trị", 30000, "Miễn phí (theo quy định NHNN)", 0),
    ("DV10", "Phí phát hành séc trắng (quyển 10 tờ)", "55.000đ/quyển", 55000, "40.000đ/quyển", 40000),
    ("DV11", "Phí đình chỉ thanh toán séc bị mất", "100.000đ/tờ", 100000, "80.000đ/tờ", 80000),
    ("DV12", "Phí rút tiền mặt tại quầy bằng VND trong vòng 2 ngày nộp tiền", "0,05% (min 20.000đ)", 20000, "0,04% (min 15.000đ)", 15000),
    ("DV13", "Phí chuyển tiền quốc tế đi bằng điện SWIFT (SHA)", "0,22% (min 15 USD, max 350 USD)", 15, "0,18% (min 12 USD, max 300 USD)", 12),
    ("DV14", "Điện phí SWIFT chuyển tiền quốc tế đi", "12 USD/điện", 12, "10 USD/điện", 10),
    ("DV15", "Phí tra soát giao dịch chuyển tiền quốc tế theo yêu cầu KH", "25 USD/lần", 25, "20 USD/lần", 20),
    ("DV16", "Phí tất toán khoản vay tiêu dùng trước hạn năm đầu", "3% dư nợ trả trước", 3, "2% dư nợ trả trước", 2),
    ("DV17", "Phí thẩm định hồ sơ tài sản bảo đảm bất động sản ngoài địa bàn", "2.000.000đ/hồ sơ", 2000000, "1.500.000đ/hồ sơ", 1500000),
    ("DV18", "Phí xuất mượn hồ sơ tài sản bảo đảm theo nhu cầu cá nhân", "200.000đ/lần", 200000, "150.000đ/lần", 150000),
    ("DV19", "Phí nhắn tin SMS Banking chủ động tài khoản cá nhân", "15.000đ/tháng", 15000, "12.000đ/tháng", 12000),
    ("DV20", "Phí phát hành thẻ tín dụng chuẩn A-Bank Classic", "100.000đ/thẻ", 100000, "Miễn phí phát hành năm đầu", 0),
    ("DV21", "Phí thường niên thẻ tín dụng hạng Vàng", "500.000đ/năm", 500000, "400.000đ/năm", 400000),
    ("DV22", "Phí ứng tiền mặt thẻ tín dụng tại ATM trong nước", "4% (min 80.000đ)", 80000, "3,5% (min 60.000đ)", 60000),
    ("DV23", "Phí chuyển đổi ngoại tệ khi chi tiêu thẻ quốc tế", "2,8% giá trị giao dịch", 2.8, "2,2% giá trị giao dịch", 2.2),
    ("DV24", "Phí giải ngân vốn vay doanh nghiệp qua ủy nhiệm chi", "50.000đ/món", 50000, "Miễn phí trên kênh Digital", 0),
    ("DV25", "Phí mở L/C nhập khẩu ký quỹ dưới 100% theo quý", "0,35%/quý (min 50 USD)", 50, "0,28%/quý (min 40 USD)", 40),
]


def _write_versioned_xlsx_fixtures() -> dict[str, Any]:
    """Create 3 clean-room multi-version Excel fixtures and revisions metadata."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    header_fill = PatternFill(start_color="0F2240", end_color="0F2240", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True, size=10)

    specs = [
        ("ABank_BieuPhi_PhienBan_2024_v1.xlsx", "2024-01-01", "v1.0-2024", "2024-01-01", "2025-06-30", False),
        ("ABank_BieuPhi_PhienBan_2025_v2.xlsx", "2025-07-01", "v2.0-2025", "2025-07-01", None, False),
        ("ABank_BieuPhi_BanNhap_ChuaCoNgay.xlsx", "", "draft-unknown", None, None, True),
    ]
    for fname, eff_str, bver, eff_from, eff_to, is_ambig in specs:
        fpath = FIXTURES_DIR / fname
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "BieuPhi"
        if not is_ambig:
            ws.append(["NGÂN HÀNG TMCP A (A-BANK) - BIỂU PHÍ DỊCH VỤ PHỤC VỤ KIỂM THỬ AS_OF"])
            ws.append([f"Phiên bản: {bver}", f"Ngày hiệu lực: {eff_str}"])
        else:
            ws.append(["NGÂN HÀNG TMCP A (A-BANK) - DỰ THẢO BIỂU PHÍ (CHƯA RÕ NGÀY HIỆU LỰC)"])
            ws.append(["Phiên bản: nháp chưa ban hành", "Ngày hiệu lực: chưa xác định"])
        ws.append(["STT", "Mã dịch vụ", "Tên khoản mục phí", "Mức phí áp dụng", "Giá trị chuẩn hóa", "Ngày hiệu lực"])
        for col_idx in range(1, 7):
            c = ws.cell(row=3, column=col_idx)
            c.fill = header_fill
            c.font = header_font
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

        for idx, row_item in enumerate(VERSIONED_ITEMS, start=1):
            code, name, fee_v1, num_v1, fee_v2, num_v2 = row_item
            if bver == "v1.0-2024":
                fee_text = fee_v1
                num_val = num_v1
            elif bver == "v2.0-2025":
                fee_text = fee_v2
                num_val = num_v2
            else:
                fee_text = "Đang rà soát (" + fee_v2 + ")"
                num_val = num_v2
            ws.append([idx, code, name, fee_text, num_val, eff_str])

        for col_idx, width in enumerate([6, 12, 46, 36, 16, 14], start=1):
            ws.column_dimensions[openpyxl.utils.get_column_letter(col_idx)].width = width
        wb.save(fpath)

    revisions_clean: list[dict[str, Any]] = []
    for fname, eff_str, bver, eff_from, eff_to, is_ambig in specs:
        if is_ambig:
            date_status = "unknown"
        else:
            date_status = "known"
        revisions_clean.append({
            "file": fname,
            "business_version": bver,
            "effective_from": eff_from,
            "effective_to": eff_to,
            "effective_date_status": date_status,
        })

    manifest = {"revisions": revisions_clean, "service_count": len(VERSIONED_ITEMS)}
    (FIXTURES_DIR / "revisions.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _clean_text(val: Any) -> str:
    return " ".join(str(val or "").split())


def _rewrite_item_clean(
    idx_num: int,
    group: str,
    difficulty: str,
    priority: str,
    question: str,
    ref_answer: str,
    must_include: list[str],
    evidence: list[dict[str, Any]],
    expected_tools: list[str] | None = None,
    numeric_targets: list[dict[str, Any]] | None = None,
    sources: list[str] | None = None,
    must_not_include: list[str] | None = None,
) -> dict[str, Any]:
    if expected_tools is not None:
        tools = list(expected_tools)
    else:
        tools = ["search_banking_knowledge"]
    if must_not_include is not None:
        forb = list(must_not_include)
    else:
        forb = []
    if numeric_targets is not None:
        nums = list(numeric_targets)
    else:
        nums = []
    if sources is not None:
        cit = {"sources": list(sources)}
    else:
        cit = None
    return {
        "id": f"DS01-{idx_num:04d}",
        "dataset": "DS-01",
        "group": group,
        "capability": "excel_ops",
        "difficulty": difficulty,
        "priority": priority,
        "question": question,
        "turns": [question],
        "profile": {"persona": "advisor", "tone": "formal", "format_type": "standard", "custom_persona": ""},
        "reference_answer": ref_answer,
        "rubric": {
            "must_include": must_include,
            "must_not_include": forb,
        },
        "checks": {
            "expected_tools": tools,
            "forbidden_tools": [],
            "expected_blocked": None,
            "numeric_targets": nums,
            "artifact_contract": None,
            "citation_contract": cit,
            "style_contract": None,
        },
        "evidence": evidence,
    }


def _parse_money_vnd(text: str) -> float | None:
    low = text.lower().replace("vnđ", "").replace("vnd", "").replace("đ", "").strip()
    digits: list[str] = []
    for ch in low:
        if ch.isdigit() or ch in ".,":
            digits.append(ch)
        elif digits:
            break
    tok = "".join(digits).strip(".,")
    if not tok:
        return None
    if "." in tok and "," not in tok:
        parts = tok.split(".")
        if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
            tok = tok.replace(".", "")
    elif "," in tok:
        tok = tok.replace(",", ".")
    try:
        return float(tok)
    except ValueError:
        return None


def _parse_pct(text: str) -> float | None:
    idx = text.find("%")
    if idx <= 0:
        return None
    head = text[:idx].strip()
    chars: list[str] = []
    for ch in reversed(head):
        if ch.isdigit() or ch in ".,":
            chars.append(ch)
        else:
            break
    tok = "".join(reversed(chars)).replace(",", ".")
    if not tok:
        return None
    try:
        return float(tok)
    except ValueError:
        return None


def build_ds01() -> list[dict[str, Any]]:
    """Build 1,000 grounded DS-01 cases and multi-version XLSX test fixtures."""
    _write_versioned_xlsx_fixtures()
    ml_records = json.loads((SCRATCH_DIR / "ml01_records.json").read_text(encoding="utf-8"))["records"]
    bp_rows = json.loads((SCRATCH_DIR / "bp_all_rows.json").read_text(encoding="utf-8"))

    items: list[dict[str, Any]] = []
    next_id = 1

    # G1: ML01 multi-filter & pagination (300 cases)
    # 1a: Exact individual branch/ATM/CDM/PGD rows across all 1260 records (step 4 -> 220 cases)
    ml_src = "ML 01 Danh sách mạng lưới Chi nhánh Phòng giao dịch ATM CDM.xlsx"
    sample_indices = list(range(5, len(ml_records), 5))
    query_styles = [
        ("Mã Core và địa chỉ", "core_addr"),
        ("số điện thoại và tọa độ vị trí", "phone_pos"),
        ("địa chỉ chính xác và thuộc quận/phường nào", "addr_dist"),
        ("loại điểm giao dịch và mã tỉnh/thành phố", "type_prov"),
    ]
    for k in range(220):
        rec = ml_records[sample_indices[k]]
        stt = rec.get("STT", "")
        name = _clean_text(rec.get("Tên ATM/CDM/Chi nhánh/Phòng GD Ngân hàng"))
        prov = _clean_text(rec.get("Tỉnh/thành phố"))
        dist = _clean_text(rec.get("Quận/huyện/thị xã/thành phố"))
        loc_type = _clean_text(rec.get("Loại"))
        addr = _clean_text(rec.get("Địa chỉ"))
        phone = _clean_text(rec.get("SĐT"))
        core = _clean_text(rec.get("Mã Core"))
        pos = _clean_text(rec.get("Vị trí"))
        pcode = _clean_text(rec.get("Mã Tỉnh/thành phố"))

        label, mode = query_styles[k % len(query_styles)]
        q = f"Trong danh sách mạng lưới A Bank (STT {stt}), cho biết {label} của điểm '{name}' tại {dist}, {prov}."
        if mode == "core_addr":
            if core:
                core_text = core
                must = [core, addr]
            else:
                core_text = "không có (trống)"
                must = [addr]
            ref = f"Theo dòng STT {stt} tại ML 01: '{name}' ({loc_type}) ở {addr}; Mã Core: {core_text}."
        elif mode == "phone_pos":
            ref = f"Theo ML 01 (STT {stt}): '{name}' có số điện thoại liên hệ {phone} và tọa độ vị trí {pos}."
            must = [phone, pos]
        elif mode == "addr_dist":
            ref = f"Theo ML 01 (STT {stt}): '{name}' đặt tại {addr}, thuộc địa bàn {dist}, {prov}."
            must = [addr, dist]
        else:
            ref = f"Theo ML 01 (STT {stt}): '{name}' là loại '{loc_type}', thuộc {prov} (Mã Tỉnh/TP: {pcode})."
            must = [loc_type, pcode, prov]

        if k % 2 == 0:
            diff = "medium"
        else:
            diff = "hard"
        if k < 60:
            prio = "P0"
        else:
            prio = "P1"

        items.append(_rewrite_item_clean(
            next_id,
            "G1_ml01_filter_pagination",
            diff,
            prio,
            q,
            ref,
            must,
            [{"source": ml_src, "stt": stt, "province": prov, "type": loc_type}],
            sources=["ML 01"],
        ))
        next_id += 1

    # 1b: Multi-condition district/province/type filters & counts (80 cases)
    grouped_ml: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for rec in ml_records:
        key = (_clean_text(rec.get("Tỉnh/thành phố")), _clean_text(rec.get("Loại")))
        grouped_ml.setdefault(key, []).append(rec)

    filter_combos = sorted(grouped_ml.items(), key=lambda x: (-len(x[1]), x[0]))
    combo_idx = 0
    while next_id <= 300:
        (prov, loc_type), matches = filter_combos[combo_idx % len(filter_combos)]
        combo_idx += 1
        page_num = (combo_idx // len(filter_combos)) + 1
        total_m = len(matches)
        first_name = _clean_text(matches[0].get("Tên ATM/CDM/Chi nhánh/Phòng GD Ngân hàng"))
        first_addr = _clean_text(matches[0].get("Địa chỉ"))
        if page_num == 1:
            q = (
                f"Hãy lọc bảng mạng lưới ML 01 cho biết tỉnh/thành phố '{prov}' có tổng cộng bao nhiêu điểm loại "
                f"'{loc_type}' và nêu tên cùng địa chỉ điểm đầu tiên trong danh sách đó."
            )
            ref = (
                f"Theo ML 01, tại {prov} có {total_m} điểm loại '{loc_type}'. "
                f"Điểm đầu tiên là '{first_name}' tại {first_addr}."
            )
            must = [str(total_m), first_name, prov]
            nums = [{"value": float(total_m), "tolerance_pct": 0.5, "label": "ml01_filtered_count"}]
        else:
            offset_idx = min(2, len(matches) - 1)
            picked = matches[offset_idx]
            p_name = _clean_text(picked.get("Tên ATM/CDM/Chi nhánh/Phòng GD Ngân hàng"))
            p_dist = _clean_text(picked.get("Quận/huyện/thị xã/thành phố"))
            p_addr = _clean_text(picked.get("Địa chỉ"))
            q = (
                f"Trong nhóm điểm giao dịch '{loc_type}' tại '{prov}' theo ML 01 (sau khi phân trang tới mục thứ {offset_idx + 1}), "
                f"hãy nêu thông tin tên điểm và địa chỉ chi tiết tại địa bàn {p_dist}."
            )
            ref = f"Theo ML 01, mục thứ {offset_idx + 1} loại '{loc_type}' tại {prov} là '{p_name}' ({p_dist}), địa chỉ: {p_addr}."
            must = [p_name, p_dist]
            nums = []

        if next_id <= 250:
            prio_b = "P0"
        else:
            prio_b = "P1"

        items.append(_rewrite_item_clean(
            next_id,
            "G1_ml01_filter_pagination",
            "hard",
            prio_b,
            q,
            ref,
            must,
            [{"source": ml_src, "province": prov, "type": loc_type, "count": total_m}],
            numeric_targets=nums,
            sources=["ML 01"],
        ))
        next_id += 1

    # G2: BP01..10 Fee Units & Percentage/Min/Max Cap Arithmetic (300 cases)
    # 2a: Direct fact lookup with sheet disambiguation (100 cases)
    clean_bp = [r for r in bp_rows if len(r.get("cells", [])) >= 4 and _clean_text(r["cells"][1])]
    step_bp = max(1, len(clean_bp) // 130)
    bp_picked = clean_bp[::step_bp][:100]
    for r in bp_picked:
        cells = r["cells"]
        fname = r["file"]
        sheet = r["sheet"]
        row_no = r["row"]
        item_name = _clean_text(cells[1])
        c2 = _clean_text(cells[2])
        c3 = _clean_text(cells[3])
        if len(cells) > 4:
            c4 = _clean_text(cells[4])
        else:
            c4 = ""
        if len(cells) > 5:
            c5 = _clean_text(cells[5])
        else:
            c5 = ""
        if sheet == "Bieu phi KHDN":
            seg = "Khách hàng doanh nghiệp / Tổ chức (sheet Bieu phi KHDN)"
            rate_str = c2
            min_str = c3
            max_str = c4
        else:
            seg = f"Khách hàng cá nhân (sheet {sheet}, mã/mức {c2})"
            rate_str = c3
            min_str = c4
            max_str = c5

        q = f"Theo tài liệu '{fname}' (sheet '{sheet}', dòng {row_no}), mức phí áp dụng cho khoản mục '{item_name[:140]}' ({seg}) là bao nhiêu?"
        parts = [f"Mức phí: {rate_str}"]
        must = [rate_str]
        if sheet != "Bieu phi KHDN" and c2 and len(c2) <= 20:
            parts.append(f"Mã phí: {c2}")
            must.append(c2)
        if min_str:
            parts.append(f"Tối thiểu: {min_str}")
            must.append(min_str)
        if max_str:
            parts.append(f"Tối đa: {max_str}")
            must.append(max_str)
        ref = f"Căn cứ '{fname}' sheet '{sheet}' (dòng {row_no}) cho mục '{item_name[:120]}': " + "; ".join(parts) + "."
        items.append(_rewrite_item_clean(
            next_id,
            "G2_bp_fee_units_calcs",
            "medium",
            "P1",
            q,
            ref,
            must,
            [{"source": fname, "sheet": sheet, "row": row_no, "cells": cells}],
            sources=[fname],
        ))
        next_id += 1

    # 2b: Percentage fee calculation with min/max caps (200 cases)
    pct_rows: list[dict[str, Any]] = []
    for r in clean_bp:
        cells = r["cells"]
        sheet = r["sheet"]
        if sheet == "Bieu phi KHDN":
            rate_tok = _clean_text(cells[2])
            min_tok = _clean_text(cells[3])
            if len(cells) > 4:
                max_tok = _clean_text(cells[4])
            else:
                max_tok = ""
            code_tok = "KHDN"
        else:
            code_tok = _clean_text(cells[2])
            rate_tok = _clean_text(cells[3])
            if len(cells) > 4:
                min_tok = _clean_text(cells[4])
            else:
                min_tok = ""
            if len(cells) > 5:
                max_tok = _clean_text(cells[5])
            else:
                max_tok = ""
        pct = _parse_pct(rate_tok)
        if pct is not None and pct > 0.0 and pct < 10.0 and "USD" not in min_tok.upper():
            min_v = _parse_money_vnd(min_tok)
            max_v = _parse_money_vnd(max_tok)
            if min_v is not None and min_v >= 1000:
                pct_rows.append({
                    "file": r["file"],
                    "sheet": sheet,
                    "row": r["row"],
                    "name": _clean_text(cells[1]),
                    "code": code_tok,
                    "pct": pct,
                    "min_fee": min_v,
                    "max_fee": max_v,
                })

    amounts_vnd = [
        30_000_000,
        55_000_000,
        80_000_000,
        140_000_000,
        250_000_000,
        420_000_000,
        750_000_000,
        1_250_000_000,
        2_500_000_000,
        4_800_000_000,
        8_000_000_000,
        15_000_000_000,
    ]
    p_idx = 0
    while next_id <= 600:
        row_info = pct_rows[p_idx % len(pct_rows)]
        amt = amounts_vnd[(p_idx // max(1, len(pct_rows))) % len(amounts_vnd)] + (p_idx // max(1, len(pct_rows) * len(amounts_vnd))) * 1_000_000
        p_idx += 1
        raw_fee = amt * (row_info["pct"] / 100.0)
        applied_fee = max(raw_fee, float(row_info["min_fee"]))
        if raw_fee < float(row_info["min_fee"]):
            cap_note = "áp dụng mức tối thiểu"
        else:
            cap_note = "tính theo tỷ lệ phần trăm"
        if row_info["max_fee"] is not None and applied_fee > float(row_info["max_fee"]):
            applied_fee = float(row_info["max_fee"])
            cap_note = "đã chạm trần phí tối đa"

        amt_m = amt // 1_000_000
        fee_fmt = f"{int(round(applied_fee)):,}".replace(",", ".")
        q = (
            f"Khách hàng thực hiện giao dịch số tiền {amt_m} triệu đồng theo khoản mục "
            f"'{row_info['name'][:110]}' (mã {row_info['code']}, dòng {row_info['row']}, tài liệu '{row_info['file']}', sheet '{row_info['sheet']}'). "
            f"Tính chính xác số tiền phí phải trả sau khi xét sàn tối thiểu và trần tối đa."
        )
        ref = (
            f"Với mức phí {row_info['pct']}% cho số tiền {amt_m} triệu đồng ({cap_note}), "
            f"tối thiểu {int(row_info['min_fee']):,}đ: phí thực thu là {fee_fmt} VNĐ."
        )
        if next_id <= 500:
            prio_g2 = "P0"
        else:
            prio_g2 = "P1"
        items.append(_rewrite_item_clean(
            next_id,
            "G2_bp_fee_units_calcs",
            "hard",
            prio_g2,
            q,
            ref,
            [fee_fmt, str(row_info["pct"]).replace(".", ",")],
            [{"source": row_info["file"], "sheet": row_info["sheet"], "row": row_info["row"]}],
            numeric_targets=[{"value": float(round(applied_fee)), "tolerance_pct": 0.5, "label": "capped_fee_vnd"}],
            sources=[row_info["file"]],
        ))
        next_id += 1

    # G3: Multi-Version as_of Versioning (200 cases)
    as_of_dates_v1 = ["2024-01-01", "2024-03-01", "2024-04-15", "2024-07-10", "2024-09-30", "2025-01-15", "2025-03-20", "2025-06-30"]
    as_of_dates_v2 = ["2025-07-01", "2025-08-15", "2025-10-01", "2025-11-20", "2026-01-05", "2026-02-10", "2026-06-15", "2026-10-01"]
    pre_dates = ["2022-05-10", "2022-11-20", "2023-01-15", "2023-06-01", "2023-09-01", "2023-12-31"]
    v_cursor = 0
    while next_id <= 800:
        code, name, fee_v1, num_v1, fee_v2, num_v2 = VERSIONED_ITEMS[v_cursor % len(VERSIONED_ITEMS)]
        mode = (v_cursor // len(VERSIONED_ITEMS)) % 4
        date_slot = (v_cursor // (len(VERSIONED_ITEMS) * 4))
        case_seq = v_cursor + 1
        v_cursor += 1
        if mode == 0:
            d = as_of_dates_v1[date_slot % len(as_of_dates_v1)]
            q = f"[Fixture #{case_seq} | as_of={d}] Tra cứu mức phí áp dụng tại ngày {d} cho dịch vụ {code} - {name} trong bộ biểu phí phiên bản."
            ref = f"Tại ngày as_of={d} (thuộc hiệu lực phiên bản v1.0-2024 từ 2024-01-01 đến 2025-06-30), phí của {code} ({name}) là {fee_v1}."
            must = [code, fee_v1]
            nums = [{"value": float(num_v1), "tolerance_pct": 0.5, "label": "v1_fee"}]
            srcs = ["ABank_BieuPhi_PhienBan_2024_v1.xlsx"]
        elif mode == 1:
            d = as_of_dates_v2[date_slot % len(as_of_dates_v2)]
            q = f"[Fixture #{case_seq} | as_of={d}] Tra cứu mức phí áp dụng tại ngày {d} cho dịch vụ {code} - {name} trong bộ biểu phí phiên bản."
            ref = f"Tại ngày as_of={d} (thuộc hiệu lực phiên bản v2.0-2025 từ 2025-07-01), phí của {code} ({name}) là {fee_v2}."
            must = [code, fee_v2]
            nums = [{"value": float(num_v2), "tolerance_pct": 0.5, "label": "v2_fee"}]
            srcs = ["ABank_BieuPhi_PhienBan_2025_v2.xlsx"]
        elif mode == 2:
            d = pre_dates[date_slot % len(pre_dates)]
            q = f"[Fixture #{case_seq} | as_of={d}] Tại ngày {d} (trước ngày hiệu lực v1 2024-01-01), dịch vụ {code} ({name}) áp dụng mức phí nào?"
            ref = f"Chưa có phiên bản biểu phí hiệu lực tại thời điểm as_of={d} (phiên bản sớm nhất v1.0 bắt đầu từ 2024-01-01); hệ thống không được suy đoán số phí."
            must = ["2024-01-01", "chưa có"]
            nums = []
            srcs = []
        else:
            q = f"[Fixture #{case_seq} | unknown-effective-date] So sánh dịch vụ {code} ({name}) giữa bản v1 (2024-01-01), bản v2 (2025-07-01) và bản nháp chưa rõ ngày hiệu lực."
            ref = f"Phiên bản v1.0 (2024-01-01): {fee_v1}; phiên bản v2.0 (2025-07-01): {fee_v2}. Bản nháp có ngày hiệu lực chưa xác định (unknown) nên cần cảnh báo không dùng làm căn cứ chính thức."
            must = [fee_v1, fee_v2, "chưa xác định"]
            nums = []
            srcs = ["ABank_BieuPhi_PhienBan_2024_v1.xlsx", "ABank_BieuPhi_PhienBan_2025_v2.xlsx"]

        if mode in (2, 3):
            prio_g3 = "P0"
        else:
            prio_g3 = "P1"
        items.append(_rewrite_item_clean(
            next_id,
            "G3_as_of_versioning",
            "hard",
            prio_g3,
            q,
            ref,
            must,
            [{"fixture_dir": str(FIXTURES_DIR).replace("\\", "/"), "code": code, "mode": mode}],
            numeric_targets=nums,
            sources=srcs,
        ))
        next_id += 1

    # G4: Excel Boundary & Error Handling (100 cases)
    fake_provinces = ["Tỉnh Bầu Trời", "Quận Atlantis", "Thành phố Mặt Trăng", "Huyện Sao Hỏa", "Tỉnh Kỳ Lân"]
    fake_codes = ["MA_PHI_KHONG_TON_TAI", "XYZ999CN", "AB8889999", "FEE_404_VN", "CORE_000000"]
    atm_rows = [r for r in ml_records if _clean_text(r.get("Loại")) == "ATM" and not _clean_text(r.get("Mã Core"))]
    g4_cursor = 0
    while next_id <= 900:
        bmode = g4_cursor % 4
        if bmode == 0:
            fp = fake_provinces[g4_cursor % len(fake_provinces)]
            q = f"Tìm giúp tôi địa chỉ và Mã Core Chi nhánh A Bank tại '{fp}' (STT {(g4_cursor % 50) + 1}) trong ML 01."
            ref = f"Không tìm thấy chi nhánh hoặc điểm giao dịch nào của A Bank tại '{fp}' trong danh sách mạng lưới ML 01."
            must = ["không tìm thấy", fp]
            srcs = []
        elif bmode == 1:
            fc = fake_codes[g4_cursor % len(fake_codes)]
            q = f"Tra cứu mức phí chính xác cho mã phí '{fc}' với giao dịch {(g4_cursor + 1) * 10} triệu đồng trong biểu phí A Bank."
            ref = f"Không có mã phí '{fc}' trong hệ thống biểu phí của A Bank. Vui lòng kiểm tra lại đúng tên dịch vụ hoặc mã phí."
            must = ["không", fc]
            srcs = []
        elif bmode == 2:
            rec = atm_rows[g4_cursor % len(atm_rows)]
            aname = _clean_text(rec.get("Tên ATM/CDM/Chi nhánh/Phòng GD Ngân hàng"))
            astt = rec.get("STT", "")
            q = f"Điểm '{aname}' (STT {astt}) trong ML 01 có Mã Core riêng là gì, hay ô Mã Core đang để trống (is_blank)?"
            ref = f"Theo ML 01 (STT {astt}), điểm '{aname}' thuộc loại ATM nên cột Mã Core để trống (blank), không có Mã Core."
            must = ["trống", "ATM"]
            srcs = ["ML 01"]
        else:
            q = f"Hãy tra cứu dòng STT {9000 + g4_cursor} trong bảng mạng lưới ML 01 của A Bank và cho biết số điện thoại chi nhánh đó."
            ref = f"Dữ liệu ML 01 chỉ có 1.260 dòng dữ liệu; dòng STT {9000 + g4_cursor} vượt quá phạm vi bảng nên không có thông tin."
            must = ["không", "1.260"]
            srcs = ["ML 01"]
        g4_cursor += 1
        items.append(_rewrite_item_clean(
            next_id,
            "G4_excel_boundary_errors",
            "hard",
            "P0",
            q,
            ref,
            must,
            [{"source": ml_src, "boundary_mode": bmode}],
            sources=srcs,
            must_not_include=["AB0099999", "Giả định Mã Core"],
        ))
        next_id += 1

    # G5: Mixed Excel Objects — Footnotes (1)..(12) & KHCN vs KHDN Sheet Contrast (100 cases)
    notes_rows: list[dict[str, Any]] = []
    for r in bp_rows:
        if r["sheet"] == "Lưu ý sử dụng Biểu phí" and len(r["cells"]) >= 2:
            c0 = _clean_text(r["cells"][0])
            c1 = _clean_text(r["cells"][1])
            if c0.startswith("(") and c0.endswith(")") and len(c1) > 30:
                notes_rows.append({"file": r["file"], "tag": c0, "text": c1, "row": r["row"]})
    # deduplicate notes by (tag, text)
    uniq_notes: list[dict[str, Any]] = []
    seen_tags: set[str] = set()
    for nr in notes_rows:
        if nr["tag"] not in seen_tags:
            seen_tags.add(nr["tag"])
            uniq_notes.append(nr)

    g5_cursor = 0
    bp_docs_list = [f"BP 0{i}" for i in range(1, 8)]
    khdn_rows = [r for r in clean_bp if r["sheet"] == "Bieu phi KHDN"]
    while next_id <= 1000:
        if g5_cursor % 2 == 0 and uniq_notes:
            note = uniq_notes[g5_cursor % len(uniq_notes)]
            variant_idx = g5_cursor // max(1, len(uniq_notes)) + 1
            bp_hint = bp_docs_list[g5_cursor % len(bp_docs_list)]
            q = (
                f"[Biến thể #{variant_idx}] Trong sheet 'Lưu ý sử dụng Biểu phí' thuộc bộ biểu phí ({bp_hint}), "
                f"ký hiệu ghi chú {note['tag']} được quy định cách hiểu và tính phí như thế nào?"
            )
            ref = f"Theo Phụ lục 03 - Lưu ý sử dụng Biểu phí ({bp_hint}), ký hiệu {note['tag']} quy định: {note['text']}"
            must = [note["tag"], note["text"][:35]]
            ev = [{"source": note["file"], "sheet": "Lưu ý sử dụng Biểu phí", "row": note["row"], "tag": note["tag"]}]
            srcs = []
        else:
            kr = khdn_rows[g5_cursor % len(khdn_rows)]
            k_item = _clean_text(kr["cells"][1])
            k_rate = _clean_text(kr["cells"][2])
            q = (
                f"Đối chiếu với sheet 'Bieu phi KHDN' trong {kr['file']} (dòng {kr['row']}, Thông báo 12768/2025/TB-ABank.HĐKD), "
                f"khoản mục '{k_item[:110]}' áp dụng cho Khách hàng Doanh nghiệp có mức phí là bao nhiêu?"
            )
            ref = (
                f"Theo sheet 'Bieu phi KHDN' (Thông báo 12768/2025/TB-ABank.HĐKD, dòng {kr['row']}), "
                f"mức phí cho '{k_item[:100]}' là {k_rate}."
            )
            must = [k_rate, "Doanh nghiệp"]
            ev = [{"source": kr["file"], "sheet": "Bieu phi KHDN", "row": kr["row"]}]
            srcs = [kr["file"]]

        g5_cursor += 1
        if g5_cursor % 2 == 0:
            diff_g5 = "medium"
        else:
            diff_g5 = "hard"
        items.append(_rewrite_item_clean(
            next_id,
            "G5_mixed_excel_objects",
            diff_g5,
            "P1",
            q,
            ref,
            must,
            ev,
            sources=srcs,
        ))
        next_id += 1

    return items


def main() -> None:
    items = build_ds01()
    cov = export_suite_bundle(items, OUT_DIR, "eval_excel_ops", "DS-01", canary_size=35)
    print(json.dumps(cov, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
