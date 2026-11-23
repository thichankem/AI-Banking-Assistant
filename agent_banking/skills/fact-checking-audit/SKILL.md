---
name: fact-checking-audit
description: Check draft answers or the adequacy of A-Bank evidence. Use when
  the answer involves multiple conditions, table values, or conflicting sources.
compatibility: Operational procedures use authorized tools within A-Bank.
metadata:
  version: 2.0.0
  a-bank-category: AUDIT
  a-bank-display-name: Hallucination Control & Document Audit
---

# Hallucination Control & Document Audit

## Instructions

- Use `grade_documents` with `query` to verify that documents cover the inquiry.
- Use `audit_answer` with `draft_answer` to cross-check the draft. The runtime retrieves documents from sources gathered in the turn.
- Verify subjects, conditions, numbers, units, and source labels. If points are missing, state precisely what data needs further lookup.
- When sources conflict under identical conditions, preserve both sources and state the inconclusive portion.

Audit results help refine the draft. The final review step still verifies the answer prior to submission.
