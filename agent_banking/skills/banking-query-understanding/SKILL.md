---
name: banking-query-understanding
description: Analyze Vietnamese banking questions about A-Bank with abbreviations, missing tone marks, typos,
  or ambiguous meanings. Use before searching if the subject or parameters are unclear.
compatibility: Operational procedures use authorized tools within A-Bank.
metadata:
  version: 1.0.0
  a-bank-category: KNOWLEDGE
  a-bank-display-name: Clarify Banking Query
  a-bank-tool: analyze_banking_query
---

# Clarify Banking Query

## Instructions

1. Call `analyze_banking_query` with the original query. Review the normalized form, keywords, and candidate meanings.
2. Use previous turns to determine the target subject being discussed. Preserve amounts, terms, service codes, and product names stated by the customer.
3. If two candidate meanings lead to different answers, ask back using interactive option cards. If sufficiently clear, create a self-contained query with the subject and conditions.

For example, only expand "STK" (account number/savings book) to a single meaning when the query or context supports it. Analysis results are suggestions for retrieval, not authoritative banking grounds.
