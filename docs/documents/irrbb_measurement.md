# IRRBB Measurement Methodology: EVE and NII

Document owner: Market Risk (IRRBB team). Classification: Internal - Treasury, Risk and Finance.

## Overview

Interest rate risk in the banking book (IRRBB) is measured with two complementary metrics. Economic value of equity (EVE) sensitivity captures the change in the present value of banking book assets, liabilities and off-balance-sheet positions under a rate shock. Net interest income (NII) sensitivity captures the change in projected earnings over the next 12 months under the same shocks.

Both metrics are produced daily by the IRRBB engine, which runs at 02:30 UTC after the general ledger has published its daily balance snapshot.

## Inputs and data lineage

The IRRBB engine uses three inputs. Position balances come from the general ledger table gl.daily_balances, which is published by 01:30 UTC after the daily close. Behavioural profiles for non-maturity deposits come from alm.nmd_profiles, produced by the Treasury deposit model. Discount and forward curves come from the Treasury market data service, snapped at 17:00 New York time.

The engine checks that gl.daily_balances carries the current business date before it starts. If the snapshot is late, the engine waits up to 30 minutes and then runs on the previous day's balances, marking every output as stale.

## Rate shock scenarios

The bank applies the six supervisory standard shock scenarios: parallel up, parallel down, steepener, flattener, short rates up and short rates down. For USD the parallel shock size is 200 basis points, the short rate shock is 300 basis points and the long rate shock is 150 basis points. A post-shock floor applies so that rates do not fall below minus 100 basis points at the shortest maturity, rising linearly to zero at 50 years.

## EVE calculation

For each scenario, the engine projects cash flows for every position, slots them into the 19 standard time buckets and discounts them with the shocked risk-free curve. Commercial margins are excluded from the cash flows, and equity is excluded from liabilities.

The change in EVE for a scenario is the present value under the shocked curve minus the present value under the base curve. The supervisory outlier test compares the largest decline in EVE across the six scenarios with Tier 1 capital. A bank is an outlier if that decline exceeds 15 percent of Tier 1 capital.

At 30 September 2026, Tier 1 capital was $9.2 billion, so the outlier threshold was $1.38 billion. The largest EVE decline was $1.02 billion under the parallel up scenario, equal to 11.1 percent of Tier 1 capital, so the bank was not an outlier. The internal early-warning limit is 12 percent of Tier 1 capital, and ALCO is alerted when usage passes 90 percent of that limit.

## NII calculation

NII sensitivity uses a constant balance sheet over a 12-month horizon. Maturing positions are replaced with the same product at the shocked rate, and non-maturity deposits reprice according to the betas in alm.nmd_profiles.

At 30 September 2026, a parallel up shock of 200 basis points increased 12-month NII by $145 million. A parallel down shock of 200 basis points reduced 12-month NII by $212 million, mainly because deposit rates are floored at zero while asset yields fall.

## Outputs and controls

The engine writes results to irrbb.eve_nii_results. The quarterly IRRBB supervisory return and the monthly ALCO risk pack are built from this table.

Daily controls compare total engine balances with gl.daily_balances and flag any difference above 0.5 percent. Results produced from stale inputs must not be used for limit monitoring until they are rerun on current data. The IRRBB team signs off each quarter-end run before the supervisory return is submitted.
