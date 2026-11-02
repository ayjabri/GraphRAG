# Regulatory Reporting and Data Lineage

Document owner: Regulatory Reporting (Finance). Classification: Internal - Finance, Risk and Data Engineering.

## Reports in scope

The regulatory reporting team produces three recurring liquidity and interest rate returns. The daily liquidity monitoring report must be submitted to the supervisor by 12:00 local time on the next business day. The monthly liquidity coverage ratio (LCR) return is due on the 15th calendar day after month-end. The quarterly IRRBB supervisory return is due 30 calendar days after quarter-end.

## Data lineage

Every reported figure must be traceable from the submitted cell back to its source system. The lineage for the liquidity reports starts with settlement.settled_transactions in the payments platform, flows through payment reconciliation into gl.daily_balances, and then through the liquidity engine into the reporting data mart.

The quarterly IRRBB supervisory return takes EVE and NII results from irrbb.eve_nii_results. Those results depend on gl.daily_balances for positions and on alm.nmd_profiles for deposit behaviour, so the IRRBB return inherits the lineage of both the general ledger and the deposit model.

The lineage is recorded in the data catalogue at column level. Each data mart table lists its upstream tables, the transformation job that produced it, and the business owner who signs off its content.

## Data quality controls

Before submission, the reporting data mart runs completeness, accuracy and timeliness checks. Completeness checks compare record counts with source systems. Accuracy checks reconcile reported balances with gl.daily_balances within a tolerance of 0.1 percent. Timeliness checks confirm that each input carries the expected business date.

A failed check blocks the report from the submission queue. The reporting manager may override a failed check only with a documented rationale, and every override is reported to the Chief Financial Officer in the monthly controls summary.

## Late or incorrect submissions

If a report cannot be submitted by its deadline, the regulatory reporting manager must notify the supervisor before the deadline passes, stating the cause and the expected submission time. A late submission is recorded as a regulatory breach in the compliance register.

Resubmissions are required when a material error is found after submission. An error is material if it changes a reported ratio by more than 0.5 percentage points or any reported amount by more than $10 million.

## Responsibilities

Data Engineering maintains the reporting data mart and the transformation jobs. Regulatory Reporting owns the report templates, the controls and the submission. Market Risk signs off the IRRBB figures before the quarterly return is submitted.

The reporting framework follows the BCBS 239 principles for risk data aggregation, which require accurate, complete and timely data with clear ownership.
