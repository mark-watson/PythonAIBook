---
type: Playbook
title: Revenue Drop Investigation
description: Step-by-step guide for on-call analysts to diagnose an unexpected drop in daily revenue.
tags: [revenue, incident, investigation, on-call]
timestamp: 2026-06-05T14:00:00Z
owner: analytics-oncall@example.com
related_metrics: [metrics/daily_revenue, metrics/customer_ltv]
---

# Revenue Drop Investigation Playbook

Use this playbook when the [daily_revenue](../metrics/daily_revenue.md) metric
shows a significant unexpected decline (> 10 % day-over-day or > 2 σ below the
30-day rolling average).

## Step 1 — Confirm the drop is real (not a data issue)

1. Check the pipeline run log: did the nightly aggregation complete successfully?
2. Verify row counts in [sales_events](../tables/sales_events.md) for the affected day.
   A count near zero almost always indicates a pipeline or CDC failure, not a
   true business drop.
3. If counts look normal, proceed to Step 2.

## Step 2 — Isolate by dimension

Run the revenue query broken down by:

- **Store** — Is the drop isolated to one location (power outage, closure)?
- **Category** — Is a specific product category affected?
- **Payment method** — A payment processor outage shows up as a shift in method mix.
- **Hour of day** — A mid-day cliff suggests a system outage window.

## Step 3 — Check for external signals

- Review the incident board for ongoing POS system outages.
- Check weather and calendar (holidays, local events) for the affected stores.
- Consult the merchandising team for planned promotions that may have ended.

## Step 4 — Escalate if needed

If the root cause is not identified within 30 minutes, escalate to:

| Symptom | Team to page |
|---|---|
| Pipeline / data quality | `#data-engineering-oncall` |
| POS system outage | `#it-ops-oncall` |
| Payment processor | `#payments-oncall` |
| Genuine business drop | Notify `VP of Retail` via the standard business escalation path |
