# General Ledger Close and Payment Reconciliation

Document owner: Finance Control. Classification: Internal - Finance and Site Reliability Engineering.

## Daily close

The general ledger records the bank's financial position by account, legal entity and currency. The daily accounting close uses a cut-off of 21:00 local time for postings. After the cut-off, overnight batches post interest accruals, fees and settled payments, and the close is completed once all sub-ledger reconciliations pass.

When the close completes, the ledger publishes a balance snapshot to the table gl.daily_balances. The snapshot target is 01:30 UTC. The IRRBB engine, the liquidity engine and the regulatory reporting data mart all read this snapshot, so a late snapshot delays risk and regulatory outputs across the bank.

## Payment reconciliation

The payment reconciliation job starts at 00:45 UTC. It compares the total of settlement.settled_transactions for each general ledger cash account with the movements posted to that account for the same value date.

The reconciliation tolerance is $50,000 per cash account per day. A difference above the tolerance is recorded as a reconciliation break and escalated to Finance Control. Breaks below the tolerance are logged and cleared within five business days.

If any cash account has an open break above the tolerance at 01:15 UTC, the close is placed on hold. A close hold stops publication of gl.daily_balances until the break is resolved or a Finance Control manager approves publication with a documented adjustment.

## Dependencies

The general ledger close depends on several upstream processes. The most time-critical dependency is the nightly settlement load into settlement.settled_transactions, because payment reconciliation cannot start until that load has finished. Other dependencies include the card scheme fee files, the loan sub-ledger interest accrual batch and the foreign exchange revaluation job.

Downstream, three processes depend directly on gl.daily_balances: the IRRBB engine at 02:30 UTC, the liquidity engine at 03:00 UTC and the regulatory reporting data mart at 04:00 UTC.

## Roles and escalation

Finance Control owns reconciliation policy and approves any manual adjustment. The ledger operations team, part of Site Reliability Engineering, runs the overnight batches and is the first responder for job failures.

Escalation follows three levels. The ledger operations engineer investigates job failures immediately. If a close hold lasts more than 60 minutes, the Finance Control duty manager is paged. If gl.daily_balances is not published by 05:00 UTC, the Head of Financial Control and the regulatory reporting manager are informed, because regulatory deadlines may be at risk.

## Manual adjustments

A manual adjustment can be used to publish balances while a break is investigated. Every adjustment must reference the break identifier, the expected correcting entry and the approver. Adjustments are reversed automatically on the next business day and must be replaced by the correcting entry. Internal Audit reviews all adjustments above $1 million each month.
