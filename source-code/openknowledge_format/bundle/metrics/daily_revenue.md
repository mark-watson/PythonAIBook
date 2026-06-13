---
type: Metric
title: Daily Revenue
description: Total net revenue (after discounts, excluding returns) aggregated per calendar day per store.
resource: bigquery://retail-analytics/metrics/daily_revenue
tags: [revenue, daily, finance, KPI]
timestamp: 2026-06-10T00:00:00Z
owner: finance-analytics@example.com
sla: available by 03:00 UTC each morning
---

# Daily Revenue

**Daily Revenue** is the primary top-line financial KPI for the retail platform.
It answers the question: *"How much did we sell today?"*

## Definition

```
daily_revenue =
  SUM(
    sales_events.quantity
    * sales_events.unit_price_usd
    * (1 - sales_events.discount_pct / 100)
  )
WHERE
  sales_events.quantity > 0          -- exclude returns
  GROUP BY DATE(sales_events.event_ts), sales_events.store_id
```

## Grain

One row per `(date, store_id)`.

## Source Tables

- [tables/sales_events](../tables/sales_events.md) — primary fact source

## Important Caveats

- Returns (negative `quantity`) are **excluded**. A separate "Net Revenue After
  Returns" metric accounts for them.
- Currency is always USD. Multi-currency stores are converted at the daily
  closing rate before loading.
- If `daily_revenue` appears to drop sharply, first check the
  [revenue drop investigation playbook](../playbooks/revenue_drop_investigation.md).

## Refresh Schedule

Loaded by the nightly aggregation pipeline at 02:30 UTC.
The SLA guarantees availability by **03:00 UTC**.
