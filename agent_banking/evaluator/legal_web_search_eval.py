"""Compare legal search providers on the same public queries without invoking an LLM."""

import json
from statistics import median
from time import perf_counter, sleep

from agent_banking.core.search_text import search_key
from agent_banking.tools.extensions.legal_web_search import PROVIDERS, provider_key, search_provider

# Each case has a Vietnamese TVPL document ID and words useful for the requested provision.
CASES = (
    ("Thông tư 17/2024/TT-NHNN Điều 16 thỏa thuận mở tài khoản thanh toán", "615776", ("thoa thuan", "tai khoan thanh toan")),
    ("Thông tư 18/2024/TT-NHNN xử lý tra soát khiếu nại giao dịch thẻ", "615582", ("tra soat", "khieu nai")),
    ("Nghị định 52/2024/NĐ-CP Điều 8 hành vi bị cấm khi mở tài khoản thanh toán", "427855", ("bi cam", "tai khoan thanh toan")),
    ("Thông tư sửa đổi Thông tư 17/2024/TT-NHNN năm 2025 về mở tài khoản thanh toán", "672167", ("25 2025 tt nhnn", "tai khoan thanh toan")),
    ("Luật Các tổ chức tín dụng 2024 giới hạn cấp tín dụng Điều 136", "577203", ("gioi han cap tin dung", "dieu 136")),
    ("Luật Phòng chống rửa tiền 2022 nhận biết khách hàng Điều 9", "519327", ("nhan biet khach hang", "dieu 9")),
    ("Luật Bảo vệ dữ liệu cá nhân số 91/2025/QH15 xử lý dữ liệu trong lĩnh vực ngân hàng", "625628", ("du lieu ca nhan", "ngan hang")),
)


def _evaluate(provider: str) -> dict:
    details = []
    for query, document_id, terms in CASES:
        started = perf_counter()
        result = search_provider(provider, query)
        elapsed = perf_counter() - started
        top = result["documents"][:4]
        rank = next((i for i, item in enumerate(top, 1) if item["source"].endswith(f"-{document_id}.aspx")), None)
        coverage = 0
        if rank:
            passage = " " + search_key(top[rank - 1]["content"]) + " "
            coverage = sum(f" {term} " in passage for term in terms)
        details.append({
            "query": query, "expected_document_id": document_id, "status": result["status"], "rank": rank,
            "evidence_terms": coverage, "expected_evidence_terms": len(terms), "latency_ms": round(elapsed * 1000),
        })
        if result["status"] in {"not_configured", "monthly_limit", "quota_unavailable", "http_401", "http_403", "http_429", "http_432", "http_433", "http_434"}:
            break
        if len(details) < len(CASES):
            sleep(1.1)
    successes = [item["latency_ms"] for item in details if item["status"] in ("ok", "no_documents")]
    return {
        "cases": details,
        "hit_at_4": sum(item["rank"] is not None for item in details),
        "evidence_terms": sum(item["evidence_terms"] for item in details),
        "reciprocal_rank": round(sum(1 / item["rank"] for item in details if item["rank"]) / len(CASES), 3),
        "median_ms": median(successes) if successes else None,
    }


def main() -> None:
    results = {name: _evaluate(name) for name in PROVIDERS if provider_key(name)}
    ranked = []
    for name, result in results.items():
        if len(result["cases"]) == len(CASES) and result["hit_at_4"]:
            ranked.append(((result["hit_at_4"], result["evidence_terms"], result["reciprocal_rank"], -result["median_ms"]), name))
    print(json.dumps({
        "case_count": len(CASES),
        "not_configured": [name for name in PROVIDERS if name not in results],
        "providers": results,
        "recommended_primary": max(ranked)[1] if len(results) == len(PROVIDERS) and ranked else None,
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
