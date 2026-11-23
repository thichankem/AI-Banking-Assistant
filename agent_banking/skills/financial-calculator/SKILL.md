---
name: financial-calculator
description: Calculate loan amortization using declining balance and simulate savings interest.
  Use when customers request financial calculations or examples of interest rates, exchange rates, or terms.
compatibility: Operational procedures use authorized tools within A-Bank.
metadata:
  version: 2.0.0
  a-bank-category: CALCULATION
  a-bank-display-name: Financial Calculations & Loan Amortization Schedule
---

# Financial Calculations & Loan Amortization Schedule

## Instructions

1. Confirm parameters affecting calculations. For loan repayment, principal, annual interest rate, and term in months are required. For savings, principal and term are required.
2. Select the appropriate `operation`. Consult the [operations reference](references/operations.md) if parameter verification is needed.
3. Explain results in VND and state assumptions clearly. For loans, distinguish between first-month payment, total interest, and total amount payable.
4. Mark sample interest rates, exchange rates, and savings yields as simulated. If customers ask for official published tables, look up A-Bank documents for the exact product and conditions.

For example, convert "12 tháng" to `term_months: 12`. Preserve interest rates and amounts stated by the customer.
