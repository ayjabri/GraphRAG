# Postmortem INC-2291: Failed Nightly Settlement Load

Document owner: Site Reliability Engineering. Classification: Restricted - SRE and Finance only until the Internal Audit review completes.

## Summary

On 29 October 2026 the nightly settlement load stl_nightly_load failed and the day's settlements were not written to settlement.settled_transactions on time. The failure caused a reconciliation break in the general ledger, delayed the daily close and made the daily liquidity monitoring report late. Severity was rated SEV-2.

## Timeline (UTC)

- 23:15 - stl_nightly_load started as scheduled.
- 23:42 - The load rejected the settlement file because it contained an unexpected column named fx_rate_source.
- 23:44 - The payments on-call engineer received the critical alert.
- 00:45 - Payment reconciliation started with 312,000 settlements missing from the warehouse.
- 01:15 - Reconciliation reported a break of $2.3 million across four cash accounts and the general ledger close was placed on hold.
- 02:30 - The IRRBB engine waited 30 minutes for gl.daily_balances and then ran on the previous day's balances, marking its results as stale.
- 04:20 - A corrected file without the new column was loaded.
- 06:10 - Reconciliation passed and gl.daily_balances was published, 4 hours and 40 minutes after its 01:30 target.
- 13:05 - The daily liquidity monitoring report was submitted, 65 minutes after the 12:00 deadline.

## Root cause

A release of the payments hub added the column fx_rate_source to the settlement file. The change was not registered through the data contract process, so the warehouse schema was not updated and consumers were not notified. Because the load validates files against an exact column list, the whole file was rejected.

Pre-production testing ran against a copy of the warehouse that had been refreshed after the release branch was cut, so it already contained the new column and the test passed.

## Impact

The root cause of the $2.3 million reconciliation break was the missing settlements, not an accounting error. No customer balances were affected and no payments were lost.

Downstream impacts were the general ledger close hold, the stale IRRBB run that could not be used for limit monitoring that day, the late liquidity engine run, and the late submission of the daily liquidity monitoring report. The regulatory reporting manager notified the supervisor at 11:40, before the deadline.

## Actions

1. Payments Engineering will block releases that change the settlement file unless the data contract registry has been updated (owner: Payments Engineering, due 30 November 2026).
2. The pre-production warehouse will be refreshed from production before each release test (owner: Data Engineering, due 15 November 2026).
3. The IRRBB engine will page Market Risk when it falls back to stale balances (owner: Market Risk Technology, due 15 December 2026).
4. Ledger operations will receive the load SLA-miss alert directly instead of waiting for reconciliation to fail (owner: SRE, due 15 November 2026).

## Audit status

Internal Audit opened review IA-2026-118 on 3 November 2026. This postmortem stays restricted to SRE and Finance until the review completes, because it contains the reconciliation break amounts.
