---
type: Metric
title: Customer Lifetime Value (LTV)
description: Predicted total net revenue a customer will generate over their entire relationship with the retailer.
resource: bigquery://retail-analytics/metrics/customer_ltv
tags: [ltv, customer, predictive, ML, KPI]
timestamp: 2026-06-10T00:00:00Z
owner: data-science@example.com
refresh_frequency: weekly (every Monday 06:00 UTC)
---

# Customer Lifetime Value (LTV)

**Customer LTV** is a forward-looking estimate of the total net revenue the
retailer expects to earn from a given customer over their entire relationship.
It is used to prioritize marketing spend and segment customers.

## Definition

LTV is computed by a machine-learning model (BG/NBD + Gamma-Gamma) trained on
historical purchase sequences:

```
customer_ltv (12-month horizon) =
  predicted_purchase_frequency_12m
  * predicted_avg_order_value_usd
  * gross_margin_factor
```

Where:
- `predicted_purchase_frequency_12m` — BG/NBD model output
- `predicted_avg_order_value_usd` — Gamma-Gamma model output
- `gross_margin_factor` — category-weighted blended margin (currently 0.42)

## Grain

One row per `customer_id`. Customers with fewer than 2 lifetime transactions
are excluded (insufficient signal).

## Source Tables

- [tables/customers](../tables/customers.md)
- [tables/sales_events](../tables/sales_events.md)

## Caveats

- Predictions are **not real-time**. Use the timestamp column to check staleness.
- New customers (< 30 days old) receive a cohort-average LTV as a prior until
  sufficient data accumulates.
- LTV does not account for expected churn — multiply by `(1 - churn_risk_score)`
  from the [customers](../tables/customers.md) table for a risk-adjusted view.
