# universal/vision_client.py
# Cloud Provider Qwen2.5-VL-7B-Instruct Client for Universal Document Layout Analysis
# Complies with GEMINI.md: 0% regex, no typing library, zero ternary, zero __init__.py

import os
import sys
import json
import time
import base64
import urllib.request
import urllib.error
from pathlib import Path


def _ensure_env_loaded():
    """
    Load environment variables from .env if API_KEY is missing.
    Scans parent directories up to 6 levels.
    Strictly follows 0% Regex Law and Zero Ternary Law.
    """
    api_key = os.getenv("API_KEY")
    if api_key:
        return

    current_path = Path(__file__).resolve().parent
    for _ in range(6):
        env_candidate = current_path / ".env"
        if env_candidate.exists():
            try:
                with open(env_candidate, "r", encoding="utf-8") as file_pointer:
                    for line in file_pointer:
                        clean_line = line.strip()
                        if clean_line:
                            if not clean_line.startswith("#"):
                                if "=" in clean_line:
                                    parts = clean_line.split("=", 1)
                                    key = parts[0].strip()
                                    val = parts[1].strip().strip("'\"")
                                    os.environ.setdefault(key, val)
                break
            except Exception:
                pass
        current_path = current_path.parent


_ensure_env_loaded()


class QwenVisionClient:
    """
    Client for Qwen2.5-VL-7B-Instruct on Cloud Provider API.
    Sends base64 document page images to /v1/chat/completions and parses
    structured visual layouts (headings, paragraphs, tables, images).
    """

    SYSTEM_PROMPT = (
        "You are an expert universal document layout analysis model.\n"
        "Analyze the document page image and extract all layout elements.\n"
        "Return a valid JSON object strictly adhering to this format:\n"
        "{\n"
        '  "headings": [{"level": int, "text": str}],\n'
        '  "elements": [\n'
        '    {\n'
        '      "type": "paragraph" | "table" | "image" | "heading",\n'
        '      "text": str,\n'
        '      "table_data": [["col1", "col2"], ["val1", "val2"]] or null,\n'
        '      "image_info": {"caption": str, "description": str} or null\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- 'level' in headings is integer 1 (document title), 2 (main section), 3 (subsection), 4 (clause).\n"
        "- 'elements' must list all content items in natural reading order.\n"
        "- For tables, 'text' is formatted markdown table and 'table_data' is a 2D array of strings.\n"
        "- For images, 'text' is a concise description and 'image_info' contains caption and details.\n"
        "- Output ONLY raw JSON. Do NOT wrap in markdown code blocks like ```json ... ```."
    )

    def __init__(
        self,
        base_url=None,
        api_key=None,
        model=None,
        timeout=60,
        max_retries=3,
        backoff_factor=1.5,
    ):
        """
        Initialize QwenVisionClient with Cloud Provider API configuration.
        """
        if base_url:
            self.base_url = base_url.rstrip("/")
        else:
            env_base = os.getenv("BASE_URL", "https://api.example.com/v1")
            self.base_url = env_base.rstrip("/")

        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("API_KEY", "")

        if model:
            self.model = model
        else:
            self.model = os.getenv("VISION_MODEL", "Qwen2.5-VL-7B-Instruct")

        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def analyze_page_layout(self, image_bytes):
        """
        Analyze page layout from raw PNG/JPEG image bytes.
        Returns layout dict conforming to target schema:
        {
            "headings": [{"level": int, "text": str}],
            "elements": [
                {
                    "type": "paragraph" | "table" | "image" | "heading",
                    "text": str,
                    "table_data": list | None,
                    "image_info": dict | None
                }
            ]
        }
        """
        if not image_bytes:
            return {"headings": [], "elements": []}

        if not isinstance(image_bytes, (bytes, bytearray)):
            raise TypeError("image_bytes must be bytes or bytearray")

        if len(image_bytes) == 0:
            return {"headings": [], "elements": []}

        payload = self._build_payload(image_bytes)
        raw_response_text = self._execute_request_with_retry(payload)
        layout_dict = self._parse_and_validate_layout(raw_response_text)
        return layout_dict

    def _build_payload(self, image_bytes):
        """
        Build OpenAI-compatible JSON payload with base64 image data URL.
        """
        encoded_b64 = base64.b64encode(image_bytes).decode("ascii")
        user_content = [
            {
                "type": "text",
                "text": "Extract all headings and layout elements from this document page into the specified JSON format.",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encoded_b64}",
                },
            },
        ]

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.1,
            "max_tokens": 4096,
        }
        return payload

    def _execute_request_with_retry(self, payload):
        """
        Execute HTTP request via urllib.request with exponential backoff retry.
        Handles transient 429, 5xx, timeouts, and network connection drops.
        """
        if not self.api_key:
            raise ValueError("Cloud Provider API_KEY is empty or not configured.")

        endpoint = f"{self.base_url}/chat/completions"
        json_bytes = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "UniversalDocxChunker/1.0",
        }

        last_error = None
        attempt = 0

        while attempt <= self.max_retries:
            req = urllib.request.Request(
                endpoint, data=json_bytes, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    resp_bytes = resp.read()
                    resp_text = resp_bytes.decode("utf-8")
                    data = json.loads(resp_text)
                    choices = data.get("choices", [])
                    if choices:
                        message = choices[0].get("message", {})
                        content = message.get("content", "")
                        return content
                    else:
                        raise ValueError(f"Empty choices in API response: {resp_text}")

            except urllib.error.HTTPError as http_err:
                last_error = http_err
                status_code = http_err.code

                # Immediate raise for client auth or configuration errors
                if status_code == 401:
                    raise RuntimeError("Cloud Provider API authentication failed: 401 Unauthorized")
                elif status_code == 403:
                    raise RuntimeError("Cloud Provider API access forbidden: 403 Forbidden")
                elif status_code == 400:
                    err_msg = ""
                    try:
                        err_msg = http_err.read().decode("utf-8")
                    except Exception:
                        pass
                    raise RuntimeError(f"Cloud Provider API Bad Request (400): {err_msg}")

                # Retry on rate limit (429) or server errors (500, 502, 503, 504)
                if status_code == 429 or status_code >= 500:
                    if attempt < self.max_retries:
                        sleep_time = self.backoff_factor * (2 ** attempt)
                        time.sleep(sleep_time)
                        attempt += 1
                        continue
                    else:
                        raise RuntimeError(
                            f"Cloud Provider API failed after {self.max_retries} retries with status {status_code}"
                        )
                else:
                    raise RuntimeError(f"Cloud Provider API HTTP error {status_code}: {http_err.reason}")

            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as net_err:
                last_error = net_err
                if attempt < self.max_retries:
                    sleep_time = self.backoff_factor * (2 ** attempt)
                    time.sleep(sleep_time)
                    attempt += 1
                    continue
                else:
                    raise RuntimeError(
                        f"Cloud Provider API connection failed after {self.max_retries} retries: {net_err}"
                    )

            except Exception as unk_err:
                last_error = unk_err
                if attempt < self.max_retries:
                    sleep_time = self.backoff_factor * (2 ** attempt)
                    time.sleep(sleep_time)
                    attempt += 1
                    continue
                else:
                    raise RuntimeError(f"Unexpected error calling Cloud Provider API: {unk_err}")

        raise RuntimeError(f"Failed to execute vision API call: {last_error}")

    def _clean_trailing_commas(self, s):
        """
        Zero-regex trailing comma removal before closing brackets/braces.
        Handles dirty commas like `{"k": "v",}` and `[1,  ]`.
        """
        out = s
        for close_char in ["}", "]"]:
            idx = 0
            while True:
                close_idx = out.find(close_char, idx)
                if close_idx == -1:
                    break
                p = close_idx - 1
                while p >= 0 and out[p] in " \t\r\n":
                    p -= 1
                if p >= 0 and out[p] == ",":
                    comma_idx = p
                    out = out[:comma_idx] + out[comma_idx + 1:]
                    idx = close_idx
                else:
                    idx = close_idx + 1
        return out

    def _try_parse_json(self, raw_str):
        """
        Try parsing JSON directly, then with trailing comma cleanup, then outer braces.
        """
        if not raw_str:
            return None

        # 1. Direct parse attempt
        try:
            return json.loads(raw_str)
        except Exception:
            pass

        # 2. Cleaned trailing commas attempt
        cleaned = self._clean_trailing_commas(raw_str)
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        # 3. Outer braces attempt
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            try:
                return json.loads(cleaned[first_brace:last_brace + 1])
            except Exception:
                pass

        # 4. Outer brackets attempt
        first_bracket = cleaned.find("[")
        last_bracket = cleaned.rfind("]")
        if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
            try:
                return json.loads(cleaned[first_bracket:last_bracket + 1])
            except Exception:
                pass

        return None

    def _parse_and_validate_layout(self, raw_text):
        """
        Extract and validate JSON layout from LLM output.
        Zero Regex Law: uses only str.find, str.rfind, str.strip, str.replace.
        Zero Ternary Law: uses explicit if/else statements.
        """
        if raw_text is None:
            return {"headings": [], "elements": []}

        text = str(raw_text).strip()
        if not text:
            return {"headings": [], "elements": []}

        extracted_text = text

        # 1. Strip Markdown code fences without regex
        if "```json" in text:
            first_pos = text.find("```json")
            last_pos = text.rfind("```json")

            # Check if multiple ```json blocks exist
            if last_pos != first_pos:
                start_candidate = last_pos + 7
                end_candidate = text.rfind("```")
                if end_candidate > start_candidate:
                    candidate_sub = text[start_candidate:end_candidate].strip()
                    parsed_cand = self._try_parse_json(candidate_sub)
                    if parsed_cand is not None:
                        extracted_text = candidate_sub

            if extracted_text == text:
                start_pos = first_pos + 7
                end_pos = text.rfind("```")
                if end_pos > start_pos:
                    extracted_text = text[start_pos:end_pos].strip()
                else:
                    extracted_text = text[start_pos:].strip()

        elif "```" in text:
            start_pos = text.find("```") + 3
            newline_pos = text.find("\n", start_pos)
            if newline_pos != -1:
                start_pos = newline_pos + 1
            end_pos = text.rfind("```")
            if end_pos > start_pos:
                extracted_text = text[start_pos:end_pos].strip()
            else:
                extracted_text = text[start_pos:].strip()

        parsed_data = self._try_parse_json(extracted_text)

        # Fallback to original text if extracted text failed
        if parsed_data is None:
            parsed_data = self._try_parse_json(text)

        # Deceptive braces recovery: find JSON starting near '"headings"' or '"elements"'
        if parsed_data is None:
            key_idx = text.find('"headings"')
            if key_idx == -1:
                key_idx = text.find('"elements"')
            if key_idx != -1:
                cand_start = text.rfind("{", 0, key_idx)
                cand_end = text.rfind("}")
                if cand_start != -1 and cand_end > cand_start:
                    sub_str = text[cand_start:cand_end + 1]
                    parsed_data = self._try_parse_json(sub_str)

        if parsed_data is None:
            # Safe fallback if LLM returned non-JSON text
            return {
                "headings": [],
                "elements": [
                    {
                        "type": "paragraph",
                        "text": text,
                        "table_data": None,
                        "image_info": None,
                    }
                ],
            }

        # Normalize and validate output structure
        headings = []
        elements = []

        if isinstance(parsed_data, dict):
            raw_headings = parsed_data.get("headings", [])
            if isinstance(raw_headings, list):
                for h_item in raw_headings:
                    if isinstance(h_item, dict):
                        raw_level = h_item.get("level", 1)
                        try:
                            level_int = int(raw_level)
                        except Exception:
                            level_int = 1
                        if level_int < 1:
                            level_int = 1
                        elif level_int > 6:
                            level_int = 6
                        h_text = str(h_item.get("text", "")).strip()
                        headings.append({"level": level_int, "text": h_text})

            raw_elements = parsed_data.get("elements", [])
            if isinstance(raw_elements, list):
                for el_item in raw_elements:
                    if isinstance(el_item, dict):
                        el_type = str(el_item.get("type", "paragraph")).lower().strip()
                        if el_type not in ["heading", "paragraph", "table", "image"]:
                            el_type = "paragraph"
                        el_text = str(el_item.get("text", "")).strip()

                        table_data = el_item.get("table_data")
                        if not isinstance(table_data, list):
                            table_data = None

                        image_info = el_item.get("image_info")
                        if not isinstance(image_info, dict):
                            image_info = None

                        elements.append({
                            "type": el_type,
                            "text": el_text,
                            "table_data": table_data,
                            "image_info": image_info,
                        })

            # Auto-extract headings if top-level headings was empty but elements contains headings
            if not headings:
                for el in elements:
                    if el["type"] == "heading":
                        headings.append({"level": 1, "text": el["text"]})

        elif isinstance(parsed_data, list):
            for item in parsed_data:
                if isinstance(item, dict):
                    el_type = str(item.get("type", "paragraph")).lower().strip()
                    if el_type not in ["heading", "paragraph", "table", "image"]:
                        el_type = "paragraph"
                    el_text = str(item.get("text", "")).strip()

                    table_data = item.get("table_data")
                    if not isinstance(table_data, list):
                        table_data = None

                    image_info = item.get("image_info")
                    if not isinstance(image_info, dict):
                        image_info = None

                    if el_type == "heading":
                        raw_level = item.get("level", 1)
                        try:
                            level_int = int(raw_level)
                        except Exception:
                            level_int = 1
                        headings.append({"level": level_int, "text": el_text})

                    elements.append({
                        "type": el_type,
                        "text": el_text,
                        "table_data": table_data,
                        "image_info": image_info,
                    })

        return {
            "headings": headings,
            "elements": elements,
        }
