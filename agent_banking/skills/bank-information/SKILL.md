---
name: bank-information
description: Look up working hours, support channels, addresses, and general information of A-Bank.
  Use when customers want to know where or when to contact.
compatibility: Operational procedures use authorized tools within A-Bank.
metadata:
  version: 1.0.0
  a-bank-category: General Information
  a-bank-display-name: bank-information
  a-bank-tool: search_banking_knowledge
---

# bank-information

## Instructions

1. Identify the channel or location requested by the customer. If the answer depends on a region or request type, ask back for that specific information.
2. Locate sources using `search_banking_knowledge`, combining channel name with support type. Distinguish between help in A-Bank Plus, physical branches, and the hotline.
3. Return the channel, access method, and service hours recorded in the same source. Attach sources immediately after usable information.

If the source does not state hours or contact numbers, clearly indicate the unverified portion. Do not use information from other banks or numbers from simulation tools.
