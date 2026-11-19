# Agent and knowledge tuning from the 785-case evaluation

The historical `outputs/eval_785_20261117/results.jsonl` contains 223 PASS,
539 FAIL and 23 ERROR results (28.41% PASS). The latest record for each ID is
used when comparing results. Golden answers and rubrics are evaluation inputs
only; they are never added to the knowledge index or agent context.

The tuned run completed the same 785 cases with 336 PASS and 449 FAIL
(42.80% PASS, +113 passes). The benchmark reused the historical judge helper
and model. Three transient judge errors were graded again from the saved
agent responses. The comparison includes 174 FAIL-to-PASS changes, six
ERROR-to-PASS changes and 67 PASS-to-FAIL changes.

| Group | Cases | Historical PASS | Tuned PASS |
| --- | ---: | ---: | ---: |
| Cards | 200 | 83 | 86 |
| Compliance | 200 | 83 | 104 |
| App | 195 | 49 | 69 |
| Specialized | 190 | 8 | 77 |

Across 762 cases scored in both runs, mean judge accuracy rose from 34.15 to
48.81, faithfulness from 37.11 to 48.27, and completeness from 32.70 to 47.26.
The files `full_candidate.jsonl`, `full_candidate_comparison.json`, and
`full_candidate_changed_cases.json` in `outputs/tuning_20261117` contain the
per-case results, matched comparison, and IDs for gains and regressions.

A follow-up fixed the legacy reviewer retry that dropped the thinking toggle
on its third attempt. The toggle stays off unless an earlier request showed
that the provider rejected it. In the complete benchmark, 56 cases ended with
a review truncated at 6,000 output tokens. A separate 12-case probe sampled
eight such failures and four pass controls. All eight in the first group got
past the reviewer outage, two changed to PASS, and all four controls stayed
PASS. The agent test suite passed 111 tests. `review_retry_comparison.json`
records the probe. The 336/785 result above describes the prior full run.

The original index held 2,627 chunks from 104 DOCX sources and no PDF sources.
The current source corpus consists of 78 DOCX and 52 PDF documents. Rebuilding
from those source files also removes dependence on obsolete app DOCX versions.
The verified candidate index contains 6,472 chunks from all 130 sources, with
embedded-image OCR enabled for all 78 DOCX files. No document has unavailable
image OCR and no PDF page failed extraction. There are 61 partially uncertain
PDF pages and 263 affected passages; warnings are attached locally to those
passages and must not be interpreted as fully verified transcription.

Controlled development comparisons used the same balanced 24 questions, the
same complete index and the unchanged Flash judge. Historical results for
these exact questions were 3 PASS. These small comparisons select configuration;
they are not the full 785-case result or a statistical claim about model quality.

| Agent configuration | PASS / 24 | ERROR |
| --- | ---: | ---: |
| DeepSeek-V4-Flash, 64 KB context | 10 | 0 |
| Azure:gpt-5.6-terra, native completion tokens | 5 | 1 |
| GLM-5.2, JSON mode off | 9 | 5 |
| GLM-5.2, JSON mode on | 8 | 5 |
| DeepSeek-V4-Flash, 128 KB context | 10 | 0 |

Retain Flash and the 64 KB context default. Increasing context did not improve
the pass count on this sample. Azure used `max_completion_tokens` without
temperature; credentials and provider URLs always came from private local
configuration. Profile tests changed only child-process environment settings.
The JSONL files, profile settings and metadata remain under
`outputs/tuning_20261117/compare_*`.

Changes to runtime behavior:

- The cloud reranker receives the candidate pool, rather than five truncated
  RRF results. `HYBRID_CANDIDATE_K` supports an environment override; its default
  is 15. An independent retrieval/reranking sweep on the covered cloud cases
  selected 15 ahead of 30, 60 and 80; increasing the pool can add false positives.
  The RAG skill inherits the configured pool instead of silently shrinking it
  to five candidates.
- Query-parser guesses about table format are soft hints. Explicit publication
  scopes remain enforced. Narrative exemptions and eligibility remain searchable.
- A search can be restricted to a discovered `source`. With `offset`, it reads
  ordered indexed passages without another embedding or reranking request.
  `source_inventory` and `next_offset` indicate what can be read next. Ambiguous
  basenames require the complete source path; relaxed retrieval cannot cross
  the requested source or publication scope.
- New search results receive context priority while citation labels remain
  stable. The evidence cap is 72 passages. Table context includes surrounding
  source text when available.
- Independent review can ground a numbered draft line against a numbered
  source line. Literal claim excerpts remain supported. Cited evidence is kept
  intact in both model-routing modes; unrelated search observations cannot
  evict it. Coverage, citations and evidence-pointer checks remain mandatory.
  Citation repair now receives the specific validation error. Read-only RAG
  searches are explicitly authorized in the agent prompt; loading a skill is
  not treated as having retrieved evidence.
- Benign insurance-exclusion questions and policy identifiers beside PIN/OTP
  are distinguished from harmful instructions and actual credential values.
- `core/cloud_rate_limit.py` coordinates request pacing across chat, OCR and
  worker processes using SQLite. The default is 120 requests/minute per model
  and provider. `CLOUD_REQUESTS_PER_MINUTE` changes the rate; zero disables this
  local limiter. The observed provider limit was 150 RPM for DeepSeek-V4-Flash.
  Temporary structured-request failures retry within the existing three-call
  budget, honoring numeric `Retry-After` values up to 60 seconds.

Run the following commands from the repository root. Publications and the
candidate index are isolated from the active index. The model, embedding,
reranking and vision calls use the configured cloud providers and caches.

```powershell
.\.venv\Scripts\python.exe -m agent_banking.evaluator.rebuild_knowledge --output agent_banking/outputs/tuning_20261117/knowledge --extensions .pdf --workers 3
.\.venv\Scripts\python.exe -m agent_banking.evaluator.rebuild_knowledge --output agent_banking/outputs/tuning_20261117/knowledge --extensions .docx --ocr-docx-images --workers 2
.\.venv\Scripts\python.exe -m agent_banking.evaluator.index_knowledge --publications agent_banking/outputs/tuning_20261117/knowledge --index agent_banking/vector_db/qdrant_tuned --require-complete --require-docx-images
.\.venv\Scripts\python.exe -m agent_banking.evaluator.tuning_eval --index agent_banking/vector_db/qdrant_tuned --output agent_banking/outputs/tuning_20261117/reproduced_candidate.jsonl --limit 785 --workers 8
.\.venv\Scripts\python.exe -m pytest agent_banking/agent/tests agent_banking/chunking/tests -q
```

Document builds resume from complete per-document publications. Indexing checks
publication integrity and the current source-file hashes, batches new embeddings,
and prunes replaced revisions within their original publication scopes.
`knowledge_inventory.json` records coverage, source hashes and partial OCR pages.
Source reconstruction and `index_ready` are not guarantees of perfect visual
transcription; partially read pages and ambiguous source tables still need audit.

Evaluations use fresh session IDs and write separate JSONL and summary files.
Their metadata fixes the question IDs, dataset hashes, code, model chains and
index inventory. Use a new output filename when any of those changes. Summaries
compare paired verdicts with the historical results using the same judge helper.
The runner releases each single-turn checkpoint when the chat call finishes.
`--retry-errors` processes technical errors only and reuses the saved answer
for judge errors. Use the same question selection and index when resuming.
The completed run resumed after 160 records to release idle checkpoints and
reduce concurrency from 12 to 8. The metadata and runtime hashes were verified
at resume and on completion. The unit and integration suite passed 250 tests.
`diagnostic*` and `candidate_partial*` runs are development checks, not a complete
785-case result. The fresh baseline run during OCR also encountered provider
rate limits, so its error-heavy score is not the primary comparison baseline.

Some references disagree with the current source documents. For example,
THE-010 requires a 40-million supplementary-card fee-waiver threshold that is
absent from the current THE 02 document's searchable text and fee table. The
document's embedded image is an issuance workflow, not another fee schedule.
Keep such disagreements visible in evaluation results; do not change runtime
facts or the judge threshold merely to turn those cases into PASS.

Two additional disagreements were checked by opening the actual DOCX images:
THE-052's current Figure 1 is a spending-allocation donut chart, while its golden
answer describes a three-level limit tree. PL-052's current Figure 3 labels
layer 4 as Smart OTP transaction signing, while its golden answer requires
WAF/API Gateway. `source_reference_audit.json` records the source and image
hashes and preview filenames. These findings are case-specific; an audit PASS
and a judge FAIL alone do not prove that a golden answer is wrong.

All 130 canonical documents under `dataset/A_Bank` were compared byte-for-byte
by SHA-256 against their copies under `chunking/input/A_Bank`; no differences
were found. Source documents and golden datasets remain unchanged.
Fifteen app cases have labels with older filenames. HDSD 22 and 26 use ASC in
the current corpus where the benchmark labels say ABS. HDSD 29 uses Thịnh Vượng
where the benchmark label says Phát Lộc. `source_label_coverage.json` lists the
affected cases. A filename mismatch warrants checking the content and does not
by itself establish an incorrect reference answer.

The verified tuned collection is active at `vector_db/qdrant_db`. The original
directory is backed up as `vector_db/qdrant_pre_tuning_20261118`, and the isolated
benchmark index remains at `vector_db/qdrant_tuned`. The local switch preserves
the additional collection in the default directory. To run against the old
store, set `ABANK_QDRANT_STORAGE_PATH` to the backup directory's absolute path
before starting the agent process. Remove that override to use the default
tuned store again.

## Follow-up on review, document context and reference alignment

The reviewer can attach up to four specific issues to a revision. A valid
draft-line number or retrieved source label accompanies each issue when the
reviewer identifies one. The next action receives those issues together with
the draft. A bounded recovery can return a supported part of the answer when
ordinary revisions are exhausted. An independent review, citation validation,
and the output guard must pass before recovery can publish an answer.

Ranked table and workflow results may carry up to three additional linked
passages. The links stay within the same document, publication scope and
section. App guide context identifies instruction text, illustrative screens
and published document contact information. Table context reserves space for
complete rows and includes available units, headings and explanatory text.
Use the result's source label with a scoped query for a more precise search,
or use a source label and offset to read adjacent chunks in document order.

The source audit resolved all 785 reference source labels against the current
130-document inventory. There were 770 filename matches and 15 unique matches
by document code after a rename. Automated triage of the 449 prior failed
answers' references found 145 provisionally supported, 68 potential conflicts
with literal excerpts from both texts, and 236 inconclusive. Every flagged
conflict received a manual disposition. Additional discrepancies were found
when checking changes between the two benchmark runs.

`evaluator/reference_corrections.json` contains 70 manually reviewed references
and six dispositions without a correction. Each correction binds the original
question and reference and the reviewed source to SHA-256 digests. A locator,
reason and revised scoring criteria accompany each correction. Some questions
invoke external legal authority or internally inconsistent source sections.
Their dispositions record the unresolved scope. The original benchmark and
the agent's knowledge corpus remain independent of this file.

Compare both benchmark runs using the same references. Use
`reference_alignment` to audit the corpus and separately grade
saved answers against the same reviewed corrections. The existing judge helper
and configured judge model score those saved answers. The main verdict JSONL
remains the score against the original references.

Source-aligned regrading writes each completed verdict to a checkpoint before
building the final JSON report. Reuse the same output path to finish an
interrupted run or retry judge errors. Cached verdicts bind to the saved
answers, correction file, judge helper and model chain. A change to any of
those inputs requires a fresh output path.

`phase2_baseline_source_alignment.regraded.json` contains the 70 regraded
answers from `full_candidate.jsonl`. The saved answers scored 53 PASS and
17 FAIL against the reviewed references, including 49 FAIL-to-PASS changes
and four retained passes. Substituting these 70 verdicts yields 385 PASS
across the 785 baseline answers. The score against the original references
is 336 PASS. Both numbers describe the same set of saved answers.

The next complete run, `phase2_full.jsonl`, produced a verdict for every one
of the same 785 IDs with the same index and judge configuration. Technical
judge failures were retried against saved answers. The paired comparison is
in `phase2_comparison.json`.

| Scoring references | Previous PASS | Phase 2 PASS | Cases |
| --- | ---: | ---: | ---: |
| Original benchmark | 336 | 326 | 785 |
| Only the reviewed references | 53 | 54 | 70 |
| Original benchmark with the same 70 replacements | 385 | 374 | 785 |

Within the original benchmark, Phase 2 added 55 passes and lost 65. Final
review unavailability dropped from 65 to one, while final citation failures
rose from 17 to 58. THE-065 passed after the independent recovery review.
The other responses verified with limitations are listed in the per-case
results. These metrics distinguish availability from answer quality.

The follow-up patch keeps up to five linked passages for an app workflow and
keeps the ordinary three-passage limit elsewhere. During a procedural search,
repeating filled-in screen templates give way to the instructions and published
footer. Queries about the displayed screen can still inspect the example.
Recovery may cite multiple passages in one document, with the same limit of
three distinct documents. It now names a limitation only when an item remains
unverified. The agent and chunking suites passed 264 tests. The subsequent
`phase3_full.jsonl` run prioritizes the 21 selected failure and recovery cases
and 24 same-verdict controls in `phase3_selection.json`.

On the 21 selected cases, the previous run passed four, Phase 2 passed two,
and the follow-up passed five. APP-004, APP-087 and APP-185 changed from
FAIL to PASS relative to Phase 2. On the 24 controls, the pass counts were
12, 12 and 13. Two controls improved while APP-099 regressed. The cases and
paired verdicts are preserved in `phase3_targeted_comparison.json`. The
selection emphasizes diagnosed failures and provides a development check.

```powershell
.\.venv\Scripts\python.exe -m agent_banking.evaluator.reference_alignment --inventory agent_banking/vector_db/qdrant_tuned/knowledge_inventory.json --source-root agent_banking/chunking/input --corrections agent_banking/evaluator/reference_corrections.json --output agent_banking/outputs/tuning_20261117/reproduced_alignment.json --regrade-results agent_banking/outputs/tuning_20261117/phase2_full.jsonl --workers 6
```


---

## V2 Upgrade & Regression Recovery (2026-11-18)

### Root Causes Fixed

1. **Short Document Code Lookup (`agent_banking/tools/rag_tool.py`)**
   - `_source_chunks` previously required exact full path, basename, or complete filename stem, returning `[]` for short document codes (`"VH 01"`, `"BC 02"`, `"HDSD 07"`, etc.). Added normalized code-prefix / unique-token fallback (`search_key`) while preserving ambiguity detection (`ValueError` on `> 1` sources).
   - `_context_section` now groups OCR step variant `"bdoc "` (`BĐỒC 1`) under the same parent workflow alongside `"buoc "`.

2. **BM25F Document Title Indexing, Source-Diverse Hybrid RRF & Cross-Encoder Reranking (`vector_db/qdrant_store.py`, `retrieval/hybrid_retriever.py`, `reranking/rag_reranker.py`, `retrieval/config/settings.py`, `retrieval/config/settings/config.yaml`, `skills/rag-knowledge-retrieval/assets/a-bank-runtime.yaml`)**
   - `_build_bm25_index` indexes `document_name` and `source` filename stem alongside breadcrumbs (requires no re-embedding).
   - `HybridRetriever.retrieve` pulls wider dense/BM25 legs (`leg_k >= 48` when unfiltered) and source-diversifies candidate pools (`<= 3` chunks per source) with `hybrid_candidate_k` raised from `15` to `30`.
   - `BankingReranker.rerank` sends `"top_n": len(candidates)` instead of truncating at `max(top_k, 5)`, prepends distinct document labels in `_extract_text`, adds high-precision `_metadata_score_bonus` (doc code match, calendar year alignment, distinctive product tokens, UI/App cues on `hdsd`), sorts by unclamped scores before `[0.0, 1.0]` clamping, deduplicates near-identical `(source, content[:160])` chunks, and applies a per-source cap (`<= 3` when `top_k >= 5`).

3. **Guardrail PII False Positives (`guardrails/rag_guardrails.py`)**
   - `_mask_pii` no longer masks comma/semicolon-separated 4-digit MCC lists (`8062, 5912, 8211, 8220`) or sequential calendar years (`2021 2022 2023 2024`) as `[CARD_NUMBER_REDACTED]`.
   - Official A-Bank / A-Life / regulator contact email domains (`@abank.vn`, `@abank.com.vn`, `@a-bank.vn`, `@alife.com.vn`, `.gov.vn`, etc.) are preserved instead of being turned into `[EMAIL_REDACTED]`.

4. **Agent Context Retention, Pagination Deduplication, Citation Parsing & Review Calibration (`agent/model_protocol.py`, `agent/react.py`, `agent/deep.py`)**
   - `AGENT_CONTEXT_BYTES` default raised from `65536` to `131072`; `_compact_observations` compacts `related_documents` alongside `documents` and preserves `retrieval_quality` (`next_offset`, `source_inventory`, `source_chunk_count`).
   - `react.py` deduplicates sequential `offset` reads by normalized `(source, offset, top_k, exact)` regardless of `query` wording, returns explicit `next_offset` guidance, and supports grouped citation markers such as `[Nguồn 1, Nguồn 2]` and `[Nguồn 1; 2]`.
   - `model_protocol.py:_parse` rejects empty or punctuation-only `final.answer` payloads (such as `"."` or whitespace) for clean JSON retry/repair.
   - `deep.py` calibrates `REVIEW_PROMPT` / `PARTIAL_REVIEW_PROMPT` (preventing false revisions on `ABank`/`A-Bank` naming variants, multi-source lines, and honest limitation statements), raises `partial` review max tokens to `4500`, instructs partial recovery on the 3-document cap, and avoids crashing `validate_review` with `Citation missing claim grounding` on `revise` verdicts or when valid claims already ground the answer.

### Verification

- **Unit & Regression Tests**:
  ```powershell
  .\.venv\Scripts\python.exe -m pytest agent_banking/agent/tests agent_banking/chunking/tests -q
  ```
  Result: `282 passed` (includes new regressions in `agent_banking/agent/tests/test_tuning_regressions.py`).

- **785-Case Benchmark (`outputs/tuning_20261117/v2_full.jsonl`)**:
  ```powershell
  .\.venv\Scripts\python.exe -m agent_banking.evaluator.tuning_eval --baseline agent_banking/outputs/tuning_20261117/full_candidate.jsonl --output agent_banking/outputs/tuning_20261117/v2_full.jsonl --limit 785 --workers 8
  ```
  - **V2 (`v2_full.jsonl`)**: `419 / 785 PASS` (`53.38%`, `0 ERROR`, `0 JUDGE_ERROR`, `mean_seconds = 37.79`)
    - Cards (`the`): `109 / 200` (P1: `86`, P2: `87`)
    - Compliance (`compliance`): `124 / 200` (P1: `104`, P2: `100`)
    - App Digital (`app`): `97 / 195` (P1: `69`, P2: `57`)
    - Specialized (`specialized`): `89 / 190` (P1: `77`, P2: `82`)
    - Paired vs Original (`223 / 785`): `229 FAIL->PASS`, `7 ERROR->PASS`, `183 PASS->PASS`, `40 PASS->FAIL`
    - Paired vs Phase 1 (`336 / 785`): `116 FAIL->PASS`, `33 PASS->FAIL` (`+83` net)
    - Paired vs Phase 2 (`326 / 785`): `125 FAIL->PASS`, `32 PASS->FAIL` (`+93` net)
    - Agent telemetry (`V2` vs `P2`): `audit_status=PASS` `756` vs `598`, `FALLBACK_ANSWER` `25` vs `185`, `revise` `221` vs `668`, `tool (ERROR)` calls `83` vs `940`
  - **With 70 Source-Aligned References (`outputs/tuning_20261117/v2_source_alignment.regraded.json`)**: `470 / 785 PASS` (`59.87%`, vs P1 `385 / 785` and P2 `374 / 785`; `64 / 70` PASS on corrected items).
