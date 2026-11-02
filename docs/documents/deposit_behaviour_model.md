# Non-Maturity Deposit Behavioural Model

Document owner: Treasury ALM. Classification: Internal - Treasury and Risk only. Model ID: ALM-NMD-04. Last validated in June 2026 by Model Risk Management.

## Purpose and scope

The non-maturity deposit (NMD) behavioural model converts contractually overnight deposit balances into an expected repricing and run-off profile. Retail current accounts, retail instant-access savings and small business operational balances are in scope. Term deposits are excluded because their cash flows are contractual. The model output is the main behavioural input to the IRRBB measurement engine and to the liquidity stress framework.

At 30 September 2026 the in-scope balance was $18.4 billion, of which $11.2 billion sat in retail instant-access savings, $5.6 billion in retail current accounts and $1.6 billion in small business operational accounts.

## Stable and core balance segmentation

Balances are first split into volatile and stable portions using the lowest observed balance over a rolling 12-month window, adjusted for seasonality. The stable portion is then split into core and non-core. The core portion is the part of stable balances that is unlikely to reprice even when market rates change significantly.

For retail instant-access savings, the model sets the core share at 62 percent of stable balances. For retail current accounts the core share is 81 percent, reflecting the low rate sensitivity of transactional balances. Small business operational balances use a core share of 55 percent.

Non-core balances are treated as repricing overnight. Volatile balances are treated as overnight for both interest rate and liquidity purposes.

## Repricing betas

The deposit beta measures how much of a change in the policy rate is passed through to the customer rate. The model uses a repricing beta of 0.45 for retail instant-access savings. Retail current accounts use a beta of 0.10 and small business operational balances use a beta of 0.30. Betas are estimated by regressing monthly changes in product rates on changes in the policy rate over the last ten years, with a lag of up to three months.

Betas are asymmetric in practice. Customer rates follow rising policy rates more slowly than they follow falling rates, so Treasury applies a floor of zero to customer rates in down-rate scenarios and does not assume negative deposit rates.

## Core run-off profile and maturity caps

Core balances are slotted into the 19 standard time buckets used by the IRRBB engine with an exponential decay profile. The monthly decay rate is 1.6 percent for retail instant-access savings and 1.1 percent for retail current accounts.

The average maturity of core retail deposits is capped at 5 years, and the average maturity of core small business balances is capped at 4 years. These caps follow the supervisory IRRBB standard. The current calibration gives an average core maturity of 3.4 years for retail instant-access savings and 4.6 years for retail current accounts.

| Product | Core share | Repricing beta | Monthly decay |
| --- | --- | --- | --- |
| Retail instant-access savings | 62% | 0.45 | 1.6% |
| Retail current accounts | 81% | 0.10 | 1.1% |
| Small business operational | 55% | 0.30 | 2.0% |

## Calibration, governance and outputs

The model is recalibrated quarterly using ten years of month-end balance and rate history from the general ledger product hierarchy. Recalibrations that move the core share of any product by more than 5 percentage points require approval from the Asset and Liability Committee (ALCO).

Each business day, the model publishes a repricing and run-off profile per product to the table alm.nmd_profiles by 02:00 UTC. The IRRBB engine reads this table in its 02:30 UTC run. If the table is not refreshed, the IRRBB engine falls back to the previous profile and raises a data-quality warning.

Known limitations include a limited history of rapid rate rises, the use of a single beta per product rather than per customer segment, and the sensitivity of the stable balance estimate to one-off large withdrawals. Model Risk Management requires a back-test of predicted against actual balance run-off every quarter, with escalation if the error exceeds 10 percent for two consecutive quarters.
