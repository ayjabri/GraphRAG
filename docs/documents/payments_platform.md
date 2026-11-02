# Payments Platform and Settlement Data

Document owner: Payments Engineering. Classification: Internal - Engineering and Data teams.

## Platform overview

The payments hub processes card, faster payments and wire transactions for retail and business customers. On an average business day it settles about 1.2 million transactions with a total value of roughly $4.8 billion. Settlement is final once the hub receives confirmation from the relevant scheme or correspondent bank.

The system of record for completed payments is the table settlement.settled_transactions. Each row represents one settled transaction and carries the transaction identifier, value date, amount, currency, customer account, scheme, and the general ledger cash account that the settlement posts to.

## Nightly settlement load

During the day, settlements are written to the operational database of the payments hub. The batch job stl_nightly_load copies the full day of settled transactions into the enterprise data warehouse. The job starts at 23:15 UTC and must finish by 00:30 UTC, which is its service level agreement.

The load validates every file against the registered schema before writing. A file that contains an unexpected column, a missing mandatory column or a type mismatch is rejected as a whole, and no partial data is written. On rejection the job raises a critical alert to the payments on-call engineer.

Typical load duration is 35 minutes. The job is idempotent, so it can be rerun safely after a fix; a rerun replaces the data for the affected value date.

## Downstream consumers

Several critical processes depend on settlement.settled_transactions:

- General ledger reconciliation starts at 00:45 UTC and compares settled transactions with the cash accounts in the general ledger.
- Fraud analytics scores the previous day's settlements at 01:00 UTC.
- Customer statement generation reads the table at 04:00 UTC.
- Regulatory reporting uses settlement data indirectly through the general ledger balances.

Because general ledger reconciliation starts only 15 minutes after the load SLA, any delay in stl_nightly_load quickly affects the ledger close and everything downstream of it.

## Schema ownership and data contracts

Payments Engineering owns the schema of settlement.settled_transactions. Any schema change requires at least 10 business days of notice to registered consumers through the data contract process. The contract registry lists the columns, types and allowed values that each consumer depends on.

Schema changes are tested in the pre-production warehouse against the registered contract before release. Adding a column is a breaking change for this table because the load validates files against an exact column list.

## Monitoring and support

The payments on-call rota covers the platform 24 hours a day. Dashboards track load start and end times, row counts against the hub's own totals, and rejected file counts. A row count difference of more than 0.1 percent between the hub and the warehouse raises a high-severity alert.

Incidents affecting settlement data must be raised in the incident management tool within 15 minutes of detection, and the general ledger operations team must be informed when the load is expected to miss its SLA.
