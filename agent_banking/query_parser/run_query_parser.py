"""
Module Tu Dong Phan Tich Cau Hoi (Auto Query Parser Node with Deduplication).
=============================================================================
Dac diem:
1. Doc cau hoi tu query_parser/input/queries.json (hoac latest_query.json / queries.txt).
2. Tinh toan SHA-256 hash cua tung cau hoi.
3. Kiem tra StateTracker: Cau hoi nao da phan tich thi tai su dung (SKIP).
4. Chi phan tich (chuan hoa tu viet tat, tach tu khoa) nhung cau hoi moi.
5. Xuat ket qua: parsed_queries.json, latest_parsed.json.

NGUYEN TAC BAT BUOC:
1. KHONG DUNG __INIT__.PY.
2. KHONG DUNG THU VIEN TYPING.
3. ZERO TERNARY: if/else tuong minh.
4. 0% REGEX LAW.
"""

from __future__ import annotations
import sys
import json
import time
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
    input_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    force: bool = False
) -> list[dict]:
    """
    Tu dong phan tich cau hoi tu input sang output kem co che chong trung lap.
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

    # 1. Tim nguon queries dau vao
    query_list: list[str] = []
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
                            query_list.append(item.strip())
                        elif isinstance(item, dict):
                            q = item.get("question", item.get("query", ""))
                            if q:
                                query_list.append(q.strip())
                elif isinstance(data, dict):
                    if "queries" in data:
                        for item in data["queries"]:
                            query_list.append(str(item).strip())
                    elif "query" in data:
                        query_list.append(str(data["query"]).strip())
        else:
            with open(target_in_file, "r", encoding="utf-8") as fp:
                for line in fp:
                    l = line.strip()
                    if l:
                        query_list.append(l)

    if not query_list:
        query_list = [
            "Hạn mức giao dịch mở tài khoản eKYC tối đa bao nhiêu một ngày?",
            "Bị nuốt thẻ ATM tại cây ATM thì liên hệ tổng đài số nào?",
            "Chuyển tiền NAPAS 247 hạn mức một lần chuyển là bao nhiêu?"
        ]
        # Luu mac dinh vao in_dir de nguoi dung tien theo doi
        with open(in_dir / "queries.json", "w", encoding="utf-8") as fp:
            json.dump(query_list, fp, ensure_ascii=False, indent=2)

    print(f"\n{'='*75}")
    print(f"▶ [MODULE 4: QUERY PARSER] Input: {len(query_list)} queries -> Output: {out_path}")
    print(f"{'='*75}")

    # 2. Doc ket qua cu da phan tich de tai su dung
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
                    q_id = tracker.compute_text_hash(clean_q)[:16]
                    q_hash_map[q_id] = tracker.compute_text_hash(clean_q)
                if tracker.are_all_items_processed(q_hash_map):
                    print(f"  ⏭️ [SKIP ALL] Tat ca {len(query_list)} queries da duoc phan tich truoc do. Bo qua toan bo trong 0.0s!")
                    print(f"✅ [Query Parser Hoan Tat] Da xuat {len(prev_list)} queries vao: {prev_file} (Bo qua: {len(query_list)}, Moi: 0) trong 0.0s")
                    return prev_list
        except Exception:
            prev_parsed = {}

    parser = BankingQueryParser()
    parsed_results: list[dict] = []
    skipped_count = 0
    new_count = 0
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
            print(f"  [{idx:02d}] [SKIP] '{clean_q[:40]}...' (Da phan tich)")
        else:
            t0 = time.time()
            parsed: ParsedQuery = parser.parse(clean_q)
            dt_ms = round((time.time() - t0) * 1000, 1)

            res = {
                "id": idx,
                "raw_query": parsed.raw_query,
                "normalized_query": parsed.normalized_query,
                "detected_acronyms": parsed.detected_acronyms,
                "keywords": parsed.keywords,
                "keywords_str": parsed.keywords_str
            }
            parsed_results.append(res)
            tracker.record_processed(clean_q, q_hash, {"keywords": parsed.keywords_str})
            new_count += 1
            print(f"  [{idx:02d}] [NEW]  '{clean_q[:40]}...' -> Tu khoa: {parsed.keywords_str} ({dt_ms}ms)")

    tracker.save()
    total_elapsed = round(time.time() - start_t, 2)

    # 3. Xuat ket qua
    out_file = out_path / "parsed_queries.json"
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(parsed_results, fp, ensure_ascii=False, indent=2)

    if parsed_results:
        latest_file = out_path / "latest_parsed.json"
        with open(latest_file, "w", encoding="utf-8") as fp:
            json.dump(parsed_results[-1], fp, ensure_ascii=False, indent=2)

    print(f"✅ [Query Parser Hoan Tat] Da xuat {len(parsed_results)} queries vao: {out_file} (Moi: {new_count}, Bo qua: {skipped_count}) trong {total_elapsed}s")
    return parsed_results


if __name__ == "__main__":
    run_auto_query_parser()
