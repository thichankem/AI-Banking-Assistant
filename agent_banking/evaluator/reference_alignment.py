"""Audit reference answers against current A-Bank publications, outside the agent."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
from pathlib import Path

from agent_banking.core.search_text import search_key
from agent_banking.evaluator.config.settings import EvaluatorConfig, get_evaluator_prompt
from agent_banking.evaluator.tuning_eval import load_records, select_cases

_RA_CFG = EvaluatorConfig.REFERENCE_ALIGNMENT
_DOC_CODE_PREFIXES = set(_RA_CFG["doc_code_prefixes"])
_SOURCE_ROOT_PREFIX = str(_RA_CFG["source_root_prefix"])
AUDIT_PROMPT = get_evaluator_prompt("REFERENCE_AUDIT_PROMPT", "reference_audit.md")


def _path(value):
    return str(value).replace("\\", "/").removeprefix(_SOURCE_ROOT_PREFIX).casefold()


def _code(value):
    fields = search_key(_path(value).rsplit("/", 1)[-1]).split(maxsplit=2)
    if len(fields) >= 2 and fields[0] in _DOC_CODE_PREFIXES and fields[1].isdigit():
        return fields[0], int(fields[1])
    return None


def resolve_source(label, sources):
    """Resolve current filenames, recording when the benchmark uses an older name."""
    matches = [source for source in sources if _path(source) == _path(label)]
    if not matches:
        name = _path(label).rsplit("/", 1)[-1]
        matches = [source for source in sources if _path(source).rsplit("/", 1)[-1] == name]
    if len(matches) == 1:
        return matches[0], "filename"
    if len(matches) > 1:
        return None, "ambiguous"
    code = _code(label)
    if code is not None:
        matches = [source for source in sources if _code(source) == code]
        if len(matches) == 1:
            return matches[0], "same_document_code"
    if len(matches) > 1:
        return None, "ambiguous"
    return None, "unresolved"


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_publication_path(raw_path):
    path = Path(raw_path)
    if path.is_dir():
        return path
    if not path.is_absolute():
        rel_cand = _PROJECT_ROOT / path
        if rel_cand.is_dir():
            return rel_cand
    norm = str(raw_path).replace("\\", "/")
    pos = norm.find("/agent_banking/")
    if pos >= 0:
        mapped = _PROJECT_ROOT / norm[pos + 1:]
        if mapped.is_dir():
            return mapped
    return path


def source_documents(inventory):
    documents = {}
    for publication in inventory["publications"]:
        for chunk in json.loads((_resolve_publication_path(publication["publication"]) / "all_chunks.json").read_bytes()):
            documents.setdefault(chunk["source"], []).append(chunk)
    if set(documents) != set(inventory["source_hashes"]):
        raise ValueError("Publications do not cover the inventory sources")
    return documents


def inspect_case(case, documents, searchable):
    labels = case.get("evidence") or [{"source_path": case["document_title"]}]
    rows = []
    for evidence in labels:
        label = evidence.get("source_path") or case["document_title"]
        source, match = resolve_source(label, documents)
        row = {"label": label, "source": source, "match": match}
        for field in ("section", "quote"):
            value = search_key(str(evidence.get(field) or ""))
            if value:
                row[field + "_found"] = any(" " + value + " " in text for text in searchable.get(source, []))
        rows.append(row)
    return {"id": case["id"], "group": case["group"], "source_anchors": rows}


def _passages(case, sources, documents):
    terms = set(search_key(case["question"]).split())
    focus = set()
    for anchor in case.get("evidence", []):
        focus.update(search_key(str(anchor.get("section") or "") + " " + str(anchor.get("target_artifact") or "")).split())
    max_chars = int(_RA_CFG["passage_max_chars"])
    top_k = int(_RA_CFG["passage_top_k"])
    focus_weight = int(_RA_CFG["focus_weight"])
    options = []
    for source in sorted(sources):
        for chunk in documents[source]:
            section = chunk.get("section_title", "")
            lines = chunk.get("content", "").splitlines()
            if not lines:
                continue
            words = set(search_key(section + " " + chunk["content"]).split())
            score = len(words.intersection(terms)) + focus_weight * len(words.intersection(focus))
            if score:
                options.append((score, {"source": source, "section": section, "chunk_id": chunk["chunk_id"],
                                        "text": chunk["content"][:max_chars]}))
    options.sort(key=lambda item: -item[0])
    return [item[1] for item in options[:top_k]]


def triage_reviews(cases, audit, documents, baseline, path, workers):
    from agent_banking.agent.model_protocol import StructuredModel, object_schema
    from agent_banking.core.chat import FallbackChatClient
    from agent_banking.core.fallback import ModelFallbackError

    schema = object_schema({
        "verdict": {"enum": ["conflict", "supported", "inconclusive"]},
        "finding": {"type": "string"},
        "source_quote": {"type": "string"},
        "reference_quote": {"type": "string"},
        "chunk_id": {"type": "string"},
    }, ["verdict", "finding"])
    client = FallbackChatClient(str(_RA_CFG["judge_module"]), routed=bool(_RA_CFG["judge_routed"]))
    model = StructuredModel(client, client.candidates[0].model)
    saved = load_records(path, repair_tail=True)
    pending = []
    for case in cases:
        record = baseline.get(case["id"], {})
        previous = saved.get(case["id"], {})
        if record.get("status") == "FAIL" and (not previous or previous.get("status") == "model_error"):
            pending.append(case)

    finding_max = int(_RA_CFG["finding_max_chars"])
    judge_max_tokens = int(_RA_CFG["judge_max_tokens"])
    judge_role = str(_RA_CFG["judge_role"])

    def inspect(case):
        cid = case["id"]
        sources = {row["source"] for row in audit[cid]["source_anchors"] if row["source"]}
        passages = _passages(case, sources, documents)
        if not passages:
            return {"id": cid, "status": "no_source_excerpt"}
        reference = case["reference_answer"]
        try:
            reply = model.request(
                AUDIT_PROMPT,
                {"question": case["question"], "reference": reference, "evidence": passages},
                [schema],
                max_tokens=judge_max_tokens,
                concise=True,
                role=judge_role,
            )
        except ModelFallbackError as error:
            cause = getattr(error, "last_error", None)
            return {"id": cid, "status": "model_error", "error_type": type(cause).__name__}
        result = {"id": cid, "status": reply["verdict"], "finding": reply["finding"][:finding_max]}
        if reply["verdict"] == "conflict":
            source_quote = reply.get("source_quote", "").strip()
            reference_quote = reply.get("reference_quote", "").strip()
            match = next((item for item in passages if item["chunk_id"] == reply.get("chunk_id")
                          and source_quote and source_quote in item["text"]), None)
            if match and reference_quote and reference_quote in reference:
                result.update(status="conflict_candidate", source=match["source"], section=match["section"],
                              chunk_id=match["chunk_id"], source_quote=source_quote, reference_quote=reference_quote)
            else:
                result["status"] = "inconclusive"
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for record in pool.map(inspect, pending):
            saved[record["id"]] = record
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return dict(Counter(row["status"] for row in saved.values()))


def validate_corrections(path, cases, inventory, source_root=None):
    data = json.loads(path.read_bytes())
    known = {case["id"]: case for case in cases}
    corrections = {}
    for row in data["corrections"]:
        cid = row["id"]
        if cid not in known or cid in corrections:
            raise ValueError("Unknown or duplicate reference correction: " + cid)
        case = known[cid]
        if sha256(case["question"].encode("utf-8")).hexdigest() != row["question_sha256"]:
            raise ValueError("Question changed since review: " + cid)
        ref_hash = sha256(case["reference_answer"].encode("utf-8")).hexdigest()
        if ref_hash != row["original_reference_sha256"] and ref_hash != sha256(row["reference_answer"].encode("utf-8")).hexdigest():
            raise ValueError("Reference changed since review: " + cid)
        if inventory["source_hashes"].get(row["source"]) != row["source_sha256"]:
            raise ValueError("Source changed since review: " + cid)
        if source_root is not None:
            source = (source_root / row["source"]).resolve()
            if not source.is_relative_to(source_root.resolve()) or sha256(source.read_bytes()).hexdigest() != row["source_sha256"]:
                raise ValueError("Current source bytes do not match reviewed source: " + cid)
        if not all(row.get(key) for key in ("locator", "finding", "reference_answer", "must_include")):
            raise ValueError("Incomplete reviewed correction: " + cid)
        corrections[cid] = row
    return corrections


def regrade(path, records, cases, corrections, workers, provenance):
    from agent_banking.core.chat import FallbackChatClient
    from agent_banking.evaluator.master_785_runner import call_llm_judge

    client = FallbackChatClient(str(_RA_CFG["judge_module"]), routed=bool(_RA_CFG["judge_routed"]))
    eligible = [case for case in cases if case["id"] in corrections
                and records.get(case["id"], {}).get("status") in ("PASS", "FAIL")]
    checkpoint = path.with_suffix(".checkpoint.jsonl")
    run_key = sha256(json.dumps(provenance, sort_keys=True).encode("utf-8")).hexdigest()
    saved = load_records(checkpoint, repair_tail=True)
    expected = {case["id"] for case in eligible}
    if any(cid not in expected or row.get("run_key") != run_key for cid, row in saved.items()):
        raise ValueError("Answers, references or judge changed; choose a fresh alignment output path")
    pending = [case for case in eligible if saved.get(case["id"], {}).get("source_aligned_status") not in ("PASS", "FAIL")]

    max_attempts = int(_RA_CFG["regrade_max_attempts"])
    error_prefixes = tuple(_RA_CFG["judge_error_prefixes"])

    def grade(case):
        cid = case["id"]
        record = records[cid]
        correction = corrections[cid]
        for _ in range(max_attempts):
            judge = call_llm_judge(client, case["question"], correction["reference_answer"],
                                   correction["must_include"], record["response"]["answer"])
            status = judge["verdict"]
            if not judge.get("feedback", "").startswith(error_prefixes):
                break
            status = "JUDGE_ERROR"
        return {"id": cid, "original_status": record["status"], "source_aligned_status": status, "run_key": run_key,
                "judge": judge, "source": correction["source"], "locator": correction["locator"]}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for completed in as_completed([pool.submit(grade, case) for case in pending]):
            row = completed.result()
            saved[row["id"]] = row
            with checkpoint.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(row["id"], row["source_aligned_status"], flush=True)
    output = [saved[case["id"]] for case in eligible]
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return dict(Counter(row["source_aligned_status"] for row in output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--triage", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--workers", type=int, default=int(_RA_CFG["default_workers"]))
    parser.add_argument("--corrections", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--regrade-results", type=Path)
    args = parser.parse_args()
    if args.triage and not args.baseline:
        parser.error("--triage requires --baseline")
    if args.regrade_results and not args.corrections:
        parser.error("--regrade-results requires --corrections")
    cases = select_cases(
        Path(__file__).resolve().parents[2],
        int(_RA_CFG["default_limit"]),
        int(_RA_CFG["default_seed"]),
    )
    inventory = json.loads(args.inventory.read_bytes())
    documents = source_documents(inventory)
    searchable = {source: [" " + search_key(chunk.get("breadcrumb", "") + " " + chunk.get("content", "")) + " "
                           for chunk in chunks] for source, chunks in documents.items()}
    rows = {case["id"]: inspect_case(case, documents, searchable) for case in cases}
    summary = {"cases": len(rows), "documents": len(documents),
               "inventory_sha256": sha256(args.inventory.read_bytes()).hexdigest(), "anchor_matches": dict(Counter(
        item["match"] for case in rows.values() for item in case["source_anchors"])), "case_audit": list(rows.values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.corrections:
        corrections = validate_corrections(args.corrections, cases, inventory, args.source_root)
        summary["manual_corrections"] = sorted(corrections)
        summary["corrections_sha256"] = sha256(args.corrections.read_bytes()).hexdigest()
    if args.triage:
        summary["triage"] = triage_reviews(cases, rows, documents, load_records(args.baseline), args.triage, args.workers)
    if args.regrade_results:
        from agent_banking.core.models import chain_fingerprint

        summary["regraded_results_sha256"] = sha256(args.regrade_results.read_bytes()).hexdigest()
        summary["regrade_judge_model_chain"] = chain_fingerprint(
            str(_RA_CFG["judge_module"]),
            routed=bool(_RA_CFG["judge_routed"]),
        )
        provenance = {key: summary[key] for key in ("corrections_sha256", "regraded_results_sha256",
                                                    "regrade_judge_model_chain")}
        provenance["judge_helper_sha256"] = sha256(Path(__file__).with_name("master_785_runner.py").read_bytes()).hexdigest()
        summary["source_aligned_grades"] = regrade(args.output.with_suffix(".regraded.json"),
                                                  load_records(args.regrade_results), cases, corrections, args.workers,
                                                  provenance)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "case_audit"}))


if __name__ == "__main__":
    main()
