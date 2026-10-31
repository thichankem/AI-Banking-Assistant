"""
Bộ Quản Lý Danh Mục Công Cụ Ngân Hàng (Banking Tool Registry - JSON-Driven Architecture).
Thiết kế tối giản, trực quan, dễ hiểu theo chuẩn Chunking:
- Quét và nạp toàn bộ định nghĩa công cụ từ các tệp JSON (*.json).
- Xuất định dạng chuẩn OpenAI Tool Calling / Anthropic Tool Use / LangGraph.
- Điều phối thực thi an toàn kèm theo kiểm tra tham số.

NGUYÊN TẮC BẮT BUỘC:
1. KHÔNG DÙNG __INIT__.PY: Import trực tiếp file module.
2. KHÔNG DÙNG THƯ VIỆN TYPING: Thuần túy dùng kiểu gốc của Python.
3. ZERO TERNARY: Bắt buộc dùng if/else tường minh, không dùng ?: hoặc toán tử 3 ngôi.
4. 0% REGEX LAW: Tuyệt đối không dùng regex.
5. HẠN CHẾ TỐI ĐA THỪA THÃI: Code ngắn gọn, trực diện, dễ hiểu.
"""

from __future__ import annotations
import json
import time
from pathlib import Path


class BankingToolRegistry:
    """Quản lý tập trung các công cụ nghiệp vụ ngân hàng được khai báo bằng JSON."""

    def __init__(self, tools_dir: str | Path | None = None, pipeline_instance=None):
        if tools_dir is not None:
            self.tools_dir = Path(tools_dir)
        else:
            self.tools_dir = Path(__file__).resolve().parent

        self.pipeline_instance = pipeline_instance
        self.tools_definitions: dict[str, dict] = {}
        self._load_tools_from_json()

    def _load_tools_from_json(self) -> None:
        """Nạp tất cả các tệp công cụ JSON."""
        if not self.tools_dir.exists():
            return

        json_files = list(self.tools_dir.glob("*.json"))
        for jf in json_files:
            if jf.name == "tools_manifest.json":
                continue
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    tool_def = json.load(f)
                tool_name = tool_def.get("name", jf.stem)
                self.tools_definitions[tool_name] = tool_def
            except Exception as e:
                print(f"⚠️ Không thể nạp tool từ {jf.name}: {e}")

    def list_tools(self) -> list[dict]:
        """Trả về danh sách tóm tắt toàn bộ công cụ."""
        result = []
        for name, t in self.tools_definitions.items():
            result.append({
                "name": name,
                "display_name": t.get("display_name", name),
                "category": t.get("category", "GENERAL"),
                "description": t.get("description", "")
            })
        return result

    def get_tool_definition(self, name: str) -> dict | None:
        """Lấy toàn bộ đặc tả JSON của một công cụ."""
        return self.tools_definitions.get(name)

    def get_openai_tool_specs(self) -> list[dict]:
        """Xuất danh sách công cụ theo chuẩn OpenAI Function / Tool Calling."""
        specs = []
        for name, t in self.tools_definitions.items():
            specs.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object", "properties": {}})
                }
            })
        return specs

    def execute_tool(self, name: str, params: dict) -> dict:
        """Thực thi công cụ theo tên và tham số truyền vào."""
        start_t = time.perf_counter()
        tool_def = self.tools_definitions.get(name)
        if not tool_def:
            return {
                "status": "ERROR",
                "message": f"Không tìm thấy công cụ '{name}' trong danh bạ JSON.",
                "latency_ms": 0.0
            }

        try:
            if name == "search_banking_knowledge":
                from agent_banking.tools.rag_tool import BankingRAGTool
                rag_tool = BankingRAGTool(pipeline_instance=self.pipeline_instance)
                q = params.get("query", "")
                k = params.get("top_k", 3)
                result = rag_tool.execute(query=q, top_k=k)
            elif name == "calculate_loan_amortization":
                from agent_banking.tools.banking_api_tools import calculate_loan_amortization
                p = float(params.get("principal_vnd", 500000000.0))
                r = float(params.get("annual_rate_percent", 7.5))
                m = int(params.get("term_months", 12))
                result = calculate_loan_amortization(principal_vnd=p, annual_rate_percent=r, term_months=m)
            elif name == "calculate_savings_profit":
                from agent_banking.tools.banking_api_tools import calculate_savings_profit
                amt = float(params.get("amount_vnd", 100000000.0))
                m = int(params.get("term_months", 6))
                result = calculate_savings_profit(amount_vnd=amt, term_months=m)
            elif name == "get_interest_rates":
                from agent_banking.tools.banking_api_tools import get_interest_rates
                dep = params.get("deposit_type", "vnd")
                result = get_interest_rates(deposit_type=dep)
            elif name == "get_foreign_exchange_rates":
                from agent_banking.tools.banking_api_tools import get_foreign_exchange_rates
                curr = params.get("currency", "USD")
                result = get_foreign_exchange_rates(currency=curr)
            elif name == "check_biometric_compliance_qd2345":
                from agent_banking.tools.banking_api_tools import check_biometric_compliance_qd2345
                stx = float(params.get("single_tx_amount_vnd", 0.0))
                dtx = float(params.get("daily_cumulative_amount_vnd", 0.0))
                ndev = bool(params.get("is_first_time_new_device", False))
                chip = bool(params.get("has_verified_chip_cccd", True))
                result = check_biometric_compliance_qd2345(
                    single_tx_amount_vnd=stx,
                    daily_cumulative_amount_vnd=dtx,
                    is_first_time_new_device=ndev,
                    has_verified_chip_cccd=chip
                )
            elif name == "create_support_ticket":
                from agent_banking.tools.banking_api_tools import create_support_ticket
                cname = params.get("customer_name", "KHACH_HANG_HOTLINE")
                itype = params.get("issue_type", "nuot_the_atm")
                desc = params.get("description", "Sự cố tiếp nhận qua AI Agent")
                prio = params.get("priority", "NORMAL")
                result = create_support_ticket(customer_name=cname, issue_type=itype, description=desc, priority=prio)
            elif name == "parse_vietnamese_financial_params":
                from agent_banking.tools.banking_api_tools import parse_vietnamese_financial_params
                q = params.get("query", "")
                result = parse_vietnamese_financial_params(query=q)
            elif name == "check_complain_count":
                from agent_banking.tools.banking_api_tools import check_complain_count
                cnt = int(params.get("complain_count", 0))
                result = check_complain_count(complain_count=cnt)
            elif name == "transfer_to_agent":
                from agent_banking.tools.banking_api_tools import transfer_to_agent
                dep = params.get("department", "CSKH_CHUNG")
                reason = params.get("reason", "Khách hàng yêu cầu hỗ trợ")
                result = transfer_to_agent(department=dep, reason=reason)
            elif name == "rate_agent_csat":
                from agent_banking.tools.banking_api_tools import rate_agent_csat
                rate = int(params.get("rating", 5))
                fb = params.get("feedback", "")
                result = rate_agent_csat(rating=rate, feedback=fb)
            elif name == "deflect_competitor_comparison":
                from agent_banking.tools.banking_api_tools import deflect_competitor_comparison
                comp = params.get("competitor_name", "")
                prod = params.get("product_type", "the_va_tai_khoan")
                result = deflect_competitor_comparison(competitor_name=comp, product_type=prod)
            else:
                result = {"status": "ERROR", "message": f"Chưa cài đặt handler cho tool '{name}'"}

            latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
            return {
                "status": "SUCCESS",
                "tool_name": name,
                "latency_ms": latency_ms,
                "data": result
            }
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
            return {
                "status": "FAILED",
                "tool_name": name,
                "latency_ms": latency_ms,
                "error": str(exc)
            }
