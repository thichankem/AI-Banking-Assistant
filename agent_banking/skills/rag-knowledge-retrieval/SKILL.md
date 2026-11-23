---
name: rag-knowledge-retrieval
description: Retrieve evidence from the indexed A-Bank repository. Use to verify regulations,
  fee schedules, limits, procedures, application guides, or data tables.
compatibility: Operational procedures use authorized tools within A-Bank.
metadata:
  version: 2.0.0
  a-bank-category: RAG
  a-bank-display-name: Bank Regulation Knowledge Retrieval (Agentic RAG)
---

# Bank Regulation Knowledge Retrieval (Agentic RAG)

## Instructions

1. Create a self-contained `query` with the subject, inquiry, and known conditions. Use `exact` to bind names or codes when necessary.
2. Read results matching the exact product name and section. If results are empty or off-target, rephrase, add clearer codes, or split the inquiry by subject.
   For a numbered PDF page, send `page` and `source` to `search_banking_knowledge`. If it lacks the requested fact, search by topic within the same source and cite the actual page.
3. When using tables, cross-check columns, units, product rows, and footnotes. Attach the source label from the current turn immediately after the claim.
4. When encountering conflicting sources, check the conditions of each source before concluding. Consult [evidence review guidance](references/evidence-check.md) if needed.

Only use retrieved chunks. A-Bank documents lacking dates remain valid per internal repository conventions. Do not infer validity from content version codes.
