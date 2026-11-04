"""
Module Tự Động Phân Tích Câu Hỏi (Auto Query Parser Node with Deduplication).
=============================================================================
Đặc điểm:
1. Đọc câu hỏi từ input/ (queries.json, latest_query.json, queries.txt) hoặc qua tham số CLI --query.
2. Tính toán SHA-256 hash của từng câu hỏi.
3. Kiểm tra StateTracker: Tái sử dụng kết quả đã phân tích (SKIP), loại bỏ 100% xử lý trùng lặp.
4. Chuẩn hóa viết tắt, bóc tách thực thể pháp lý, nhận diện số tiền, phân loại ý định & trích xuất từ khóa BM25.
5. Xuất kết quả chuẩn: parsed_queries.json, latest_parsed.json, parsing_manifest.json.

NGUYÊN TẮC BẮT BUỘC (GEMINI.md & AGENTS.md):
1. KHÔNG DÙNG __INIT__.PY: Import trực tiếp file module.
2. KHÔNG DÙNG THU VIEN TYPING: Thuần túy dùng kiểu nguyên bản của Python.
3. ZERO TERNARY: Bắt buộc dùng if/else tường minh từng dòng, không dùng toán tử 3 ngôi.
4. 0% REGEX LAW: Tuyệt đối không dùng regex hay import re.
"""

from __future__ import annotations
import sys
import json
import time
import argparse
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.query_parser.rag_query_parser import BankingQueryParser, ParsedQuery
from agent_banking.state_tracker import StateTracker


def run_auto_query_parser(
    input_file=None,
    output_dir=None,
    force=False,
    single_query=None
):
    """
    Tự động phân tích câu hỏi từ input sang output kèm cơ chế chống trùng lặp StateTracker.
    """
    in_dir = PROJECT_ROOT / "agent_banking" / "query_parser" / "input"
    in_dir.mkdir(parents=True, exist_ok=True)

    if output_dir is not None:
        out_path = Path(output_dir)
    else:
        out_path = PROJECT_ROOT / "agent_banking" / "query_parser" / "output"
    out_path.mkdir(parents=True, exist_ok=True)

    state_file = out_path / ".state_tracker.json"
    tracker = StateTracker(state_file)
    if force:
        tracker.clear()

    # 1. Xác định danh sách truy vấn đầu vào
    query_list = []

    if single_query is not None and str(single_query).strip():
        query_list.append(str(single_query).strip())
    else:
        target_in_file = None
        if input_file is not None:
            p = Path(input_file)
            if p.exists():
                target_in_file = p
        else:
            candidates = [
                in_dir / "queries.json",
                in_dir / "latest_query.json",
                in_dir / "queries.txt"
            ]
            for c in candidates:
                if c.exists():
                    target_in_file = c
                    break

        if target_in_file is not None and target_in_file.exists():
            if target_in_file.suffix.lower() == ".json":
                with open(target_in_file, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, str):
                                if item.strip():
                                    query_list.append(item.strip())
                            elif isinstance(item, dict):
                                q = item.get("question", item.get("query", ""))
                                if q:
                                    query_list.append(str(q).strip())
                    elif isinstance(data, dict):
                        if "queries" in data and isinstance(data["queries"], list):
                            for item in data["queries"]:
                                if item:
                                    query_list.append(str(item).strip())
                        elif "query" in data and data["query"]:
                            query_list.append(str(data["query"]).strip())
            else:
                with open(target_in_file, "r", encoding="utf-8") as fp:
                    for line in fp:
                        l = line.strip()
                        if l:
                            query_list.append(l)

    # Nếu chưa có câu hỏi nào, tạo mẫu mặc định thực tế cho ngân hàng
    if not query_list:
        query_list = [
            "Hạn mức giao dịch mở tài khoản eKYC tối đa bao nhiêu một ngày?",
            "Bị nuốt thẻ ATM tại cây ATM thì liên hệ tổng đài số nào?",
            "Chuyển tiền NAPAS 247 hạn mức một lần chuyển là bao nhiêu?",
            "Theo Quyết định 2345/QĐ-NHNN, chuyển trên 10 triệu đồng có bắt buộc quét sinh trắc học FaceID không?",
            "Lãi suất tiền gửi tiết kiệm trực tuyến tại VCB Digibank hiện tại là bao nhiêu?"
        ]
        default_file = in_dir / "queries.json"
        with open(default_file, "w", encoding="utf-8") as fp:
            json.dump(query_list, fp, ensure_ascii=False, indent=2)

    print(f"\n{'='*75}")
    print(f"▶ [MODULE 4: QUERY PARSER] Input: {len(query_list)} queries -> Output: {out_path}")
    print(f"{'='*75}")

    # 2. Đọc kết quả cũ đã phân tích để tái sử dụng
    prev_parsed = {}
    prev_file = out_path / "parsed_queries.json"
    if prev_file.exists() and not force:
        try:
            with open(prev_file, "r", encoding="utf-8") as fp:
                prev_list = json.load(fp)
                for item in prev_list:
                    rq = item.get("raw_query", "")
                    if rq:
                        prev_parsed[rq] = item
            if len(prev_parsed) >= len(query_list) and len(query_list) > 0:
                q_hash_map = {}
                for q in query_list:
                    clean_q = q.strip()
                    q_hash_map[clean_q] = tracker.compute_text_hash(clean_q)
                if tracker.are_all_items_processed(q_hash_map):
                    print(f"  ⏭️ [SKIP ALL] Tất cả {len(query_list)} queries đã được phân tích trước đó. Bỏ qua toàn bộ trong 0.0s!")
                    print(f"✅ [Query Parser Hoàn Tất] Đã xuất {len(prev_list)} queries vào: {prev_file} (Bỏ qua: {len(query_list)}, Mới: 0) trong 0.0s")
                    return prev_list
        except Exception:
            prev_parsed = {}

    parser = BankingQueryParser()
    parsed_results = []
    skipped_count = 0
    new_count = 0
    intents_breakdown = {}
    start_t = time.time()

    for idx, q in enumerate(query_list, start=1):
        clean_q = q.strip()
        q_hash = tracker.compute_text_hash(clean_q)
        is_done = tracker.is_processed(clean_q, q_hash)

        if is_done and clean_q in prev_parsed:
            item = dict(prev_parsed[clean_q])
            item["id"] = idx
            parsed_results.append(item)
            skipped_count += 1
            it = item.get("intent", "hoi_dap_chung")
            intents_breakdown[it] = intents_breakdown.get(it, 0) + 1
            print(f"  [{idx:02d}] [SKIP] '{clean_q[:42]}...' (Đã phân tích)")
        else:
            t0 = time.time()
            parsed = parser.parse(clean_q)
            dt_ms = round((time.time() - t0) * 1000, 2)

            res = parsed.to_dict()
            res["id"] = idx

            parsed_results.append(res)
            tracker.record_processed(clean_q, q_hash, {
                "keywords": parsed.keywords_str,
                "intent": parsed.intent
            })
            new_count += 1
            it = parsed.intent
            intents_breakdown[it] = intents_breakdown.get(it, 0) + 1
            print(f"  [{idx:02d}] [NEW]  '{clean_q[:38]}...' -> Intent: {parsed.intent} | Từ khóa: {parsed.keywords_str[:35]} ({dt_ms}ms)")

    tracker.save()
    total_elapsed = round(time.time() - start_t, 4)

    # 3. Xuất các tệp kết quả
    out_file = out_path / "parsed_queries.json"
    if single_query is not None and prev_parsed:
        # Hợp nhất câu hỏi đơn lẻ vào tập queries sẵn có thay vì xóa sạch tệp batch
        merged_dict = dict(prev_parsed)
        for res_item in parsed_results:
            rq = res_item.get("raw_query")
            if rq:
                merged_dict[rq] = res_item
        merged_list = []
        for midx, mitem in enumerate(merged_dict.values(), start=1):
            item_copy = dict(mitem)
            item_copy["id"] = midx
            merged_list.append(item_copy)
        with open(out_file, "w", encoding="utf-8") as fp:
            json.dump(merged_list, fp, ensure_ascii=False, indent=2)
    else:
        with open(out_file, "w", encoding="utf-8") as fp:
            json.dump(parsed_results, fp, ensure_ascii=False, indent=2)

    if parsed_results:
        latest_file = out_path / "latest_parsed.json"
        with open(latest_file, "w", encoding="utf-8") as fp:
            json.dump(parsed_results[-1], fp, ensure_ascii=False, indent=2)

    # Xuất Manifest theo quy chuẩn hệ thống (tương đương indexing_manifest.json)
    manifest_file = out_path / "parsing_manifest.json"
    manifest_data = {
        "status": "success",
        "module": "query_parser",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "received_count": len(query_list),
        "new_count": new_count,
        "skipped_count": skipped_count,
        "total_parsed": len(parsed_results),
        "execution_time_sec": total_elapsed,
        "intents_breakdown": intents_breakdown,
        "output_file": str(out_file)
    }
    with open(manifest_file, "w", encoding="utf-8") as fp:
        json.dump(manifest_data, fp, ensure_ascii=False, indent=2)

    print(f"✅ [Query Parser Hoàn Tất] Đã xuất {len(parsed_results)} queries vào: {out_file} (Mới: {new_count}, Bỏ qua: {skipped_count}) trong {total_elapsed}s")
    return parsed_results


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Auto Query Parser Pipeline with Deduplication")
    cli_parser.add_argument("--force", action="store_true", help="Chạy lại toàn bộ và xóa sạch cache state tracker")
    cli_parser.add_argument("--input", type=str, default=None, help="Đường dẫn tệp input json/txt tùy chọn")
    cli_parser.add_argument("--output", type=str, default=None, help="Thư mục xuất kết quả tùy chọn")
    cli_parser.add_argument("--query", type=str, default=None, help="Phân tích trực tiếp một câu hỏi cụ thể")
    args = cli_parser.parse_args()

    run_auto_query_parser(
        input_file=args.input,
        output_dir=args.output,
        force=args.force,
        single_query=args.query
    )
